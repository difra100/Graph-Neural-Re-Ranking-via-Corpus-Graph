"""Extract a model .pt from the best Lightning .ckpt saved in logs/.

Usage (run immediately after training finishes):
  python scripts/recover_checkpoint.py \
      --log_dir logs/hadamard_2_789_0.01_0.0_128_0.1_transformer_single_1_tctcolbert \
      --out     models/msmarco_data/hadamard_2_789_0.01_0.0_128_0.1_transformer_single_1_tctcolbert.pt

Or let it auto-detect from a glob:
  python scripts/recover_checkpoint.py --exp_name hadamard_2_789_0.01_0.0_128_0.1_transformer_single_1_tctcolbert
"""
import argparse
import glob
import os
import sys
from pathlib import Path

import torch


def best_ckpt(log_dir: Path):
    ckpts = sorted(log_dir.glob("**/*.ckpt"))
    if not ckpts:
        return None
    # prefer the one with the highest monitored score in the filename, else last modified
    def score(p):
        # Lightning names checkpoints like epoch=9-step=100-v_num=0.ckpt or
        # epoch=9-nDCG@10 on test=0.1197.ckpt; try to parse the metric
        try:
            part = p.stem.split("nDCG@10 on test=")[-1].split("-")[0]
            return float(part)
        except Exception:
            return p.stat().st_mtime
    return max(ckpts, key=score)


def extract(ckpt_path: Path, out_path: Path):
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state = ckpt.get("state_dict", ckpt)
    # Lightning wraps model as "model.*" keys
    model_state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
    if not model_state:
        # fallback: save the whole state dict
        model_state = state

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model_state, out_path)
    print(f"Saved model state dict → {out_path}  ({len(model_state)} keys)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_dir", type=Path, default=None,
                    help="Path to the Lightning log directory (contains *.ckpt)")
    ap.add_argument("--exp_name", default=None,
                    help="exp_name slug; will look in logs/<exp_name>/")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .pt path (default: models/msmarco_data/<exp_name>.pt)")
    args = ap.parse_args()

    if args.log_dir is None and args.exp_name is None:
        ap.error("Provide --log_dir or --exp_name")

    log_dir = args.log_dir or Path("logs") / args.exp_name
    exp_name = args.exp_name or log_dir.name
    out = args.out or Path("models/msmarco_data") / f"{exp_name}.pt"

    if not log_dir.exists():
        print(f"ERROR: log dir not found: {log_dir}", file=sys.stderr)
        sys.exit(1)

    ckpt = best_ckpt(log_dir)
    if ckpt is None:
        print(f"ERROR: no .ckpt files found under {log_dir}", file=sys.stderr)
        sys.exit(1)

    extract(ckpt, out)


if __name__ == "__main__":
    main()
