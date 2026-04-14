#!/usr/bin/env python3
"""
Generate a randomised run order for the full factorial experiment.

Output: data/run_order.json — array of treatment descriptors, one per run.
Each descriptor contains:
  run_order, run_id, language, model_short, model_id, volume, phase, repetition

Parameters (all constant for the paper):
  7 languages × 5 models × 4 volumes × 2 phases × 5 repetitions = 1400 runs
  RNG seed = 42 (reported in paper)
"""

import json
import os
import random
import uuid

SEED = 17

LANGUAGES = ["python", "c", "cpp", "java", "csharp", "rust", "ruby"]

MODELS = {
    "minilm":      "sentence-transformers/all-MiniLM-L6-v2",
    "modernbert":  "answerdotai/ModernBERT-base",
    "gpt2":        "openai-community/gpt2",
    "vit":         "trpakov/vit-face-expression",
    "fashionclip": "patrickjohncyh/fashion-clip",
}

VOLUMES     = [1, 10, 100, 1000]
PHASES      = ["cold", "steady"]
REPETITIONS = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
os.makedirs(DATA_DIR, exist_ok=True)

rng = random.Random(SEED)

treatments = []
for rep in range(1, REPETITIONS + 1):
    block = []
    for lang in LANGUAGES:
        for model_short, model_id in MODELS.items():
            for volume in VOLUMES:
                for phase in PHASES:
                    block.append({
                        "language":    lang,
                        "model_short": model_short,
                        "model_id":    model_id,
                        "volume":      volume,
                        "phase":       phase,
                        "repetition":  rep,
                    })
    rng.shuffle(block)
    treatments.extend(block)

for i, t in enumerate(treatments, 1):
    t["run_order"] = i
    t["run_id"]    = str(uuid.uuid4())

out_path = os.path.join(DATA_DIR, "run_order.json")
with open(out_path, "w") as f:
    json.dump(treatments, f, indent=2)

print(f"[gen_run_order] Seed={SEED} | Generated {len(treatments)} runs -> {out_path}")
