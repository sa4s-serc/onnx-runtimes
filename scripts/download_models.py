#!/usr/bin/env python3
"""Download ONNX models from Hugging Face Hub."""

import os
import shutil
from huggingface_hub import hf_hub_download

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

MODELS = [
    {
        "repo_id":   "sentence-transformers/all-MiniLM-L6-v2",
        "filename":  "onnx/model.onnx",
        "out_name":  "all-MiniLM-L6-v2.onnx",
    },
    {
        "repo_id":   "answerdotai/ModernBERT-base",
        "filename":  "onnx/model.onnx",
        "out_name":  "ModernBERT-base.onnx",
    },
    {
        "repo_id":   "onnx-community/gpt2",
        "filename":  "onnx/decoder_model.onnx",
        "out_name":  "gpt2.onnx",
    },
    {
        "repo_id":   "trpakov/vit-face-expression",
        "filename":  "onnx/model.onnx",
        "out_name":  "vit-face-expression.onnx",
    },
    {
        "repo_id":   "patrickjohncyh/fashion-clip",
        "filename":  "onnx/model.onnx",
        "out_name":  "fashion-clip.onnx",
    },
]

for m in MODELS:
    out_path = os.path.join(MODELS_DIR, m["out_name"])
    if os.path.exists(out_path):
        print(f"Already exists: {m['out_name']}")
        continue
    print(f"Downloading {m['repo_id']} / {m['filename']} ...")
    try:
        cached = hf_hub_download(
            repo_id=m["repo_id"],
            filename=m["filename"],
            cache_dir=os.path.join(REPO_ROOT, ".hf_cache"),
        )
        shutil.copy2(cached, out_path)
        size_mb = os.path.getsize(out_path) / (1024**2)
        print(f"  -> {out_path} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"  ERROR: {e}")
        # Try alternate paths
        alternates = [
            "model.onnx",
            "onnx/model_quantized.onnx",
            "model/model.onnx",
        ]
        success = False
        for alt in alternates:
            try:
                cached = hf_hub_download(
                    repo_id=m["repo_id"],
                    filename=alt,
                    cache_dir=os.path.join(REPO_ROOT, ".hf_cache"),
                )
                shutil.copy2(cached, out_path)
                size_mb = os.path.getsize(out_path) / (1024**2)
                print(f"  -> (alt: {alt}) {out_path} ({size_mb:.1f} MB)")
                success = True
                break
            except Exception:
                continue
        if not success:
            print(f"  FAILED to download {m['repo_id']}")

print("\nModel files in", MODELS_DIR)
for f in sorted(os.listdir(MODELS_DIR)):
    p = os.path.join(MODELS_DIR, f)
    if os.path.isfile(p):
        print(f"  {f}: {os.path.getsize(p)/(1024**2):.1f} MB")
