#!/usr/bin/env python3
"""
Prepare 10 fixed input tensors per model and save as .npy files.

Directory layout:
  inputs/minilm/<i>.npy          int64 (1,32)
  inputs/modernbert/<i>.npy      int64 (1,128)
  inputs/gpt2/<i>.npy            int64 (1,64)
  inputs/vit/<i>.npy             float32 (1,3,224,224)
  inputs/fashionclip/image_<i>.npy  float32 (1,3,224,224)
  inputs/fashionclip/text_<i>.npy   int64  (1,77)

All 10 tensors per model are committed to the replication package.
"""

import os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, CLIPTokenizer

REPO_ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUTS_DIR = os.path.join(REPO_ROOT, "inputs")
HF_CACHE   = os.path.join(REPO_ROOT, ".hf_cache")

for sub in ("minilm", "modernbert", "gpt2", "vit", "fashionclip"):
    os.makedirs(os.path.join(INPUTS_DIR, sub), exist_ok=True)


# ---------------------------------------------------------------------------
# 1. all-MiniLM-L6-v2  (STS-B, seq_len=32)
# ---------------------------------------------------------------------------
def prepare_minilm():
    print("Preparing minilm inputs...")
    tok = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2", cache_dir=HF_CACHE)
    ds = load_dataset("sentence-transformers/stsb", split="validation",
                      cache_dir=HF_CACHE)
    samples = [row["sentence1"] for row in ds.select(range(10))]
    for i, s in enumerate(samples):
        enc = tok(s, max_length=32, truncation=True, padding="max_length",
                  return_tensors="np")
        np.save(os.path.join(INPUTS_DIR, "minilm", f"{i}.npy"),
                enc["input_ids"].astype(np.int64))
    print(f"  Saved 10 minilm tensors shape (1,32) int64")


# ---------------------------------------------------------------------------
# 2. ModernBERT-base  (Wikipedia, seq_len=128)
# ---------------------------------------------------------------------------
def prepare_modernbert():
    print("Preparing modernbert inputs...")
    tok = AutoTokenizer.from_pretrained(
        "answerdotai/ModernBERT-base", cache_dir=HF_CACHE)
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train",
                      streaming=True, cache_dir=HF_CACHE)
    samples = []
    for row in ds:
        samples.append(row["text"])
        if len(samples) >= 10:
            break
    for i, s in enumerate(samples):
        enc = tok(s, max_length=128, truncation=True, padding="max_length",
                  return_tensors="np")
        np.save(os.path.join(INPUTS_DIR, "modernbert", f"{i}.npy"),
                enc["input_ids"].astype(np.int64))
    print(f"  Saved 10 modernbert tensors shape (1,128) int64")


# ---------------------------------------------------------------------------
# 3. GPT-2  (WikiText-103, seq_len=64)
# ---------------------------------------------------------------------------
def prepare_gpt2():
    print("Preparing gpt2 inputs...")
    tok = AutoTokenizer.from_pretrained("openai-community/gpt2", cache_dir=HF_CACHE)
    tok.pad_token = tok.eos_token
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test",
                      cache_dir=HF_CACHE)
    samples = []
    for row in ds:
        text = row["text"].strip()
        if len(text) > 20:
            samples.append(text)
        if len(samples) >= 10:
            break
    for i, s in enumerate(samples):
        enc = tok(s, max_length=64, truncation=True, padding="max_length",
                  return_tensors="np")
        np.save(os.path.join(INPUTS_DIR, "gpt2", f"{i}.npy"),
                enc["input_ids"].astype(np.int64))
    print(f"  Saved 10 gpt2 tensors shape (1,64) int64")


# ---------------------------------------------------------------------------
# 4. vit-face-expression  (FER-2013 via clip-benchmark, 224x224 float32)
# ---------------------------------------------------------------------------
def prepare_vit():
    print("Preparing vit inputs...")
    ds = load_dataset("clip-benchmark/wds_fer2013", split="test",
                      cache_dir=HF_CACHE)
    saved = 0
    for row in ds:
        img = row.get("jpg") or row.get("image") or row.get("img")
        if img is None:
            continue
        img = img.convert("RGB").resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0   # (224,224,3)
        arr = arr.transpose(2, 0, 1)[np.newaxis, :]      # (1,3,224,224)
        np.save(os.path.join(INPUTS_DIR, "vit", f"{saved}.npy"), arr)
        saved += 1
        if saved >= 10:
            break
    print(f"  Saved {saved} vit tensors shape (1,3,224,224) float32")


# ---------------------------------------------------------------------------
# 5. fashion-CLIP  (synthetic seeded images + CLIP text prompts)
# ---------------------------------------------------------------------------
FASHIONPEDIA_CATEGORIES = [
    "shirt", "dress", "jacket", "pants", "skirt",
    "coat", "shoe", "bag", "hat", "sweater"
]

def prepare_fashionclip():
    print("Preparing fashionclip inputs (synthetic seeded images)...")
    tok = CLIPTokenizer.from_pretrained(
        "patrickjohncyh/fashion-clip", cache_dir=HF_CACHE)
    rng = np.random.default_rng(42)  # fixed seed for reproducibility
    for i, cat in enumerate(FASHIONPEDIA_CATEGORIES):
        # Synthetic 224x224 RGB image with seed — sufficient for benchmarking
        img_arr = rng.random((1, 3, 224, 224), dtype=np.float32).astype(np.float32)
        np.save(os.path.join(INPUTS_DIR, "fashionclip", f"image_{i}.npy"), img_arr)

        prompt = f"a photo of a {cat}"
        enc = tok(prompt, max_length=77, truncation=True, padding="max_length",
                  return_tensors="np")
        np.save(os.path.join(INPUTS_DIR, "fashionclip", f"text_{i}.npy"),
                enc["input_ids"].astype(np.int64))
    print(f"  Saved 10 fashionclip image+text tensor pairs (synthetic seed=42)")


# ---------------------------------------------------------------------------
# Run all
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    prepare_minilm()
    prepare_gpt2()
    prepare_modernbert()
    prepare_vit()
    prepare_fashionclip()

    print("\nAll inputs saved to", INPUTS_DIR)
    for sub in ("minilm", "modernbert", "gpt2", "vit", "fashionclip"):
        d = os.path.join(INPUTS_DIR, sub)
        files = sorted(os.listdir(d))
        print(f"\n  {sub}/  ({len(files)} files)")
        if files:
            sample = np.load(os.path.join(d, files[0]))
            print(f"    {files[0]}: shape={sample.shape} dtype={sample.dtype}")
