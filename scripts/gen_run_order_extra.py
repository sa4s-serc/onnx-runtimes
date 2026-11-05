#!/usr/bin/env python3
"""
Generate a STANDALONE run order for the revision's extra experiments.

Completes the full 9-language x 6-model factorial by generating exactly the
cells NOT covered by the original 1,400-run campaign:

  * ruby34, ruby40 (Ruby 3.4.9 / 4.0.5, YJIT enabled) x all 6 models
  * the 7 original languages x pythia (2nd autoregressive decoder)

  = 19 (language, model) cells x 4 volumes x 2 phases x 5 reps = 760 runs

run_order indices continue after the original campaign (1401..2160) so the
combined dataset keeps a monotonic, globally-unique run_order. Writes
data/run_order.json with fresh run_ids.

Usage:
    python scripts/gen_run_order_extra.py
"""

import json
import os
import random
import uuid

SEED = 99
ORIGINAL_RUN_COUNT = 1400

ALL_LANGUAGES = ["c", "cpp", "python", "java", "csharp", "rust",
                 "ruby", "ruby34", "ruby40"]
ORIGINAL_LANGUAGES = {"c", "cpp", "python", "java", "csharp", "rust", "ruby"}

MODELS = {
    "minilm":      "sentence-transformers/all-MiniLM-L6-v2",
    "modernbert":  "answerdotai/ModernBERT-base",
    "gpt2":        "openai-community/gpt2",
    "vit":         "trpakov/vit-face-expression",
    "fashionclip": "patrickjohncyh/fashion-clip",
    "pythia":      "EleutherAI/pythia-160m",
}
ORIGINAL_MODELS = {"minilm", "modernbert", "gpt2", "vit", "fashionclip"}

VOLUMES     = [1, 10, 100, 1000]
PHASES      = ["cold", "steady"]
REPETITIONS = 5

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR  = os.path.join(REPO_ROOT, "data")
out_path  = os.path.join(DATA_DIR, "run_order.json")


def is_new_cell(lang, model_short):
    """A cell is new iff it is NOT in the original 7-lang x 5-model design."""
    return not (lang in ORIGINAL_LANGUAGES and model_short in ORIGINAL_MODELS)


rng = random.Random(SEED)

treatments = []
for rep in range(1, REPETITIONS + 1):
    block = []
    for lang in ALL_LANGUAGES:
        for model_short, model_id in MODELS.items():
            if not is_new_cell(lang, model_short):
                continue
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
    rng.shuffle(block)   # randomise execution order within each replicate block
    treatments.extend(block)

for i, t in enumerate(treatments, ORIGINAL_RUN_COUNT + 1):
    t["run_order"] = i
    t["run_id"]    = str(uuid.uuid4())

with open(out_path, "w") as f:
    json.dump(treatments, f, indent=2)

print(f"[gen_run_order_extra] Wrote {len(treatments)} new runs -> {out_path} "
      f"(run_order {ORIGINAL_RUN_COUNT + 1}..{ORIGINAL_RUN_COUNT + len(treatments)})")
