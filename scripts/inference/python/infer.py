#!/usr/bin/env python3
"""
Python ONNX Runtime inference runner for the energy measurement study.

Energy/RAPL measurement is handled externally by scripts/measure.py.
This script only performs inference and signals the measurement boundary.

Usage (called by measure.py):
    python infer.py <model_path> <inputs_dir> <volume> <phase> <model_short>

READY protocol:
  Cold  — prints "READY" after session load, waits for 1 stdin byte, runs inference
  Steady— prints "READY" after warmup,        waits for 1 stdin byte, runs R×K inference

Stderr output (JSON):
  {"init_ms": <float|null>, "inference_ms": <float>, "total_ms": <float|null>}
"""

import sys
import os
import json
import time

import numpy as np
import onnxruntime as ort

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_EST_MS = {
    "minilm": 15, "modernbert": 80, "gpt2": 70, "vit": 80, "fashionclip": 100,
}

WARMUP_COUNT       = 300
GPT2_GENERATE_TOKENS = 20



def load_inputs(inputs_dir: str, model_short: str) -> list[dict]:
    sub = os.path.join(inputs_dir, model_short)
    inputs_list = []
    if model_short == "fashionclip":
        for i in range(10):
            img = np.load(os.path.join(sub, f"image_{i}.npy"))
            txt = np.load(os.path.join(sub, f"text_{i}.npy"))
            inputs_list.append({"pixel_values": img, "input_ids": txt})
    else:
        for i in range(10):
            arr = np.load(os.path.join(sub, f"{i}.npy"))
            if model_short == "gpt2":
                inputs_list.append({"input_ids": arr,
                                    "attention_mask": np.ones_like(arr)})
            elif model_short in ("minilm", "modernbert"):
                inputs_list.append({"input_ids": arr,
                                    "attention_mask": np.ones_like(arr),
                                    "token_type_ids": np.zeros_like(arr)})
            elif model_short == "vit":
                inputs_list.append({"pixel_values": arr})
    return inputs_list


def run_inference(session, inputs_list, K, input_names, model_short):
    if model_short == "gpt2":
        id_name   = next((n for n in input_names if "input_id" in n.lower()), input_names[0])
        mask_name = next((n for n in input_names if "attention" in n.lower() or "mask" in n.lower()), None)
        for j in range(K):
            inp = inputs_list[j % 10]
            current_ids = inp["input_ids"].copy()
            for _ in range(GPT2_GENERATE_TOKENS):
                feed = {id_name: current_ids}
                if mask_name:
                    feed[mask_name] = np.ones_like(current_ids)
                out = session.run(None, feed)
                logits = out[0][0, -1, :]
                next_tok = int(np.argmax(logits))
                current_ids = np.concatenate(
                    [current_ids, np.array([[next_tok]], dtype=np.int64)], axis=1
                )
    else:
        for j in range(K):
            inp = inputs_list[j % 10]
            feed = {}
            for name in input_names:
                lname = name.lower()
                if "pixel" in lname:
                    feed[name] = inp.get("pixel_values", list(inp.values())[0])
                elif "input_id" in lname:
                    feed[name] = inp.get("input_ids", list(inp.values())[0])
                elif "attention" in lname or "mask" in lname:
                    ref = inp.get("input_ids", list(inp.values())[0])
                    feed[name] = inp.get("attention_mask", np.ones_like(ref))
                elif "token_type" in lname:
                    ref = inp.get("input_ids", list(inp.values())[0])
                    feed[name] = inp.get("token_type_ids", np.zeros_like(ref))
                else:
                    feed[name] = list(inp.values())[0]
            session.run(None, feed)


def make_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(model_path, sess_options=opts,
                                providers=["CPUExecutionProvider"])


def signal_ready():
    """Print READY and flush stdout, then wait for 1 byte on stdin."""
    print("READY", flush=True)
    sys.stdin.buffer.read(1)


def report(init_ms, inference_ms, total_ms):
    print(json.dumps({
        "init_ms":      init_ms,
        "inference_ms": inference_ms,
        "total_ms":     total_ms,
    }), file=sys.stderr)


def main():
    if len(sys.argv) != 6:
        print(f"Usage: {sys.argv[0]} <model_path> <inputs_dir> <volume> <phase> <model_short>",
              file=sys.stderr)
        sys.exit(1)

    model_path  = sys.argv[1]
    inputs_dir  = sys.argv[2]
    K           = int(sys.argv[3])
    phase       = sys.argv[4]
    model_short = sys.argv[5]

    est_ms = MODEL_EST_MS[model_short]
    R = max(1, -(-(100) // (K * est_ms))) if phase == "steady" else 1

    inputs_list = load_inputs(inputs_dir, model_short)

    if phase == "cold":
        t_start = time.perf_counter()
        session = make_session(model_path)
        input_names = [inp.name for inp in session.get_inputs()]
        t_session_ready = time.perf_counter()

        signal_ready()  # measure.py reads RAPL_mid here

        t_infer_start = time.perf_counter()
        run_inference(session, inputs_list, K, input_names, model_short)
        t_infer_end = time.perf_counter()

        init_ms      = (t_session_ready - t_start) * 1000.0
        inference_ms = (t_infer_end - t_infer_start) * 1000.0
        total_ms     = (t_infer_end - t_start) * 1000.0

    else:  # steady
        session = make_session(model_path)
        input_names = [inp.name for inp in session.get_inputs()]

        run_inference(session, inputs_list, WARMUP_COUNT, input_names, model_short)

        signal_ready()  # measure.py reads RAPL_0 here

        t0 = time.perf_counter()
        for _ in range(R):
            run_inference(session, inputs_list, K, input_names, model_short)
        t1 = time.perf_counter()

        init_ms      = None
        inference_ms = (t1 - t0) * 1000.0 / R
        total_ms     = None

    report(init_ms, inference_ms, total_ms)


if __name__ == "__main__":
    main()
