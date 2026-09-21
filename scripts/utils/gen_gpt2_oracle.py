#!/usr/bin/env python
"""Self-contained decode-oracle generator for a RAW HF GPT-2 (any size).

The FHE decode gate (src/app/pipeline.cu: read_teacher_forced_inputs / read_lm_head_steps)
compares the encrypted model's per-position argmax against a plaintext reference. That
reference is two JSON files per horizon T, under ALL_BLOCKS_IO_DIR:

  all_blocks_L00_T{T}.json          -> {"inp": [[..n_embd..] x T]}   # block-0 input = embedding output
  all_blocks_lm_head_steps_T{T}.json-> {"steps": [{"logits": [..vocab..]}, x T]}

Because the deployed model is the raw HF checkpoint (no HE-aware fine-tuning), the reference
is just the raw model's own forward: we feed a fixed real token sequence (a slice of the
calibration token pool), record the embedding output (== input to block 0) and the final
logits at every position. The gate then checks FHE-argmax == plaintext-argmax (self-consistency),
so ANY in-distribution input sequence is a valid test.

Usage (offline, from cache):
  HF_HOME=$SCRATCH/.cache HF_HUB_OFFLINE=1 PYTHONPATH=$REPO \
    python scripts/utils/gen_gpt2_oracle.py \
      --model openai-community/gpt2-medium \
      --pool  $SCRATCH/.cache/perseus/pools/openwebtext_gpt2_2000000.npy \
      --out   /leonardo/pub/.../all_blocks_io_gpt2-medium \
      --T 16 32 64 128 --pool-offset 4096
"""
import argparse
import json
import os

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="Generate the FHE decode oracle for a raw HF GPT-2.")
    ap.add_argument("--model", default="openai-community/gpt2-medium")
    ap.add_argument("--pool", required=True, help="uint16 token pool .npy (same tokenizer as the model)")
    ap.add_argument("--out", required=True, help="output ALL_BLOCKS_IO dir")
    ap.add_argument("--T", type=int, nargs="+", default=[16, 32, 64, 128],
                    help="horizons to emit; a single forward over max(T) serves all")
    ap.add_argument("--pool-offset", type=int, default=4096,
                    help="start token index into the pool (skip the leading doc boundary)")
    args = ap.parse_args()

    from perseus.hub import load_model

    os.makedirs(args.out, exist_ok=True)
    Tmax = max(args.T)

    pool = np.load(args.pool, mmap_mode="r")
    toks = np.asarray(pool[args.pool_offset:args.pool_offset + Tmax], dtype=np.int64)
    if toks.shape[0] < Tmax:
        raise SystemExit(f"pool too small: need {Tmax} tokens at offset {args.pool_offset}, "
                         f"have {toks.shape[0]}")
    ids = torch.tensor(toks, dtype=torch.long).unsqueeze(0)  # (1, Tmax)

    model = load_model(args.model, device="cpu")  # raw HF, fp32, eval
    model.eval()

    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    emb = out.hidden_states[0][0]      # (Tmax, n_embd) == input to block 0
    logits = out.logits[0]             # (Tmax, vocab)
    n_embd, vocab = emb.shape[1], logits.shape[1]
    print(f"[oracle] {args.model}: Tmax={Tmax} n_embd={n_embd} vocab={vocab} "
          f"tokens[{args.pool_offset}:{args.pool_offset+Tmax}]")

    emb_l = emb.to(torch.float64).tolist()
    log_l = logits.to(torch.float64).tolist()

    for T in sorted(set(args.T)):
        if T > Tmax:
            print(f"[skip] T={T} > Tmax={Tmax}")
            continue
        inp_path = os.path.join(args.out, f"all_blocks_L00_T{T}.json")
        with open(inp_path, "w") as f:
            json.dump({"inp": emb_l[:T]}, f)
        lm_path = os.path.join(args.out, f"all_blocks_lm_head_steps_T{T}.json")
        with open(lm_path, "w") as f:
            json.dump({"steps": [{"logits": log_l[i]} for i in range(T)]}, f)
        am = [int(np.argmax(log_l[i])) for i in range(T)]
        print(f"[save] T={T:>4}  inp {T}x{n_embd} -> {inp_path}")
        print(f"[save] T={T:>4}  lm  {T}x{vocab}  argmax[:8]={am[:8]} -> {lm_path}")

    print("[oracle] done")


if __name__ == "__main__":
    main()
