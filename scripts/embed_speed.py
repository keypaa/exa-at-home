#!/usr/bin/env python3
"""Measure embedding throughput vs max_seq_length and batch size.

Usage (on the box, with the current embed paused or after it finishes):
    python scripts/embed_speed.py --docs docs_130wet.jsonl --n 20000
    python scripts/embed_speed.py --docs docs_130wet.jsonl --n 20000 \
        --seq-lengths 128 256 512 --batch-sizes 1024 2048

This is a diagnostic, not a gate: it writes nothing. It reports docs/s so
the max-seq-length / batch choice is a measured decision (M10), not a guess.
"""

from __future__ import annotations

import argparse
import json
import time


def _read_docs(path: str, n: int) -> list[str]:
    out: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append((d.get("text") or "")[:12000])
            if len(out) >= n:
                break
    if not out:
        raise SystemExit(f"{path}: no documents")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", default="docs_130wet.jsonl")
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--seq-lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=[1024])
    ap.add_argument("--warmup", type=int, default=1024)
    args = ap.parse_args(argv)

    import torch

    from exa_home.embed import Embedder

    texts = _read_docs(args.docs, args.n)
    print(f"docs={len(texts)} from {args.docs}")
    if not torch.cuda.is_available():
        print("WARNING: no CUDA; CPU numbers are not the box numbers")

    for ms in args.seq_lengths:
        for bs in args.batch_sizes:
            emb = Embedder(max_seq_length=ms, batch_size=bs)
            if args.warmup:
                emb.encode_docs(texts[: min(args.warmup, len(texts))], batch_size=bs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            vecs = emb.encode_docs(texts, batch_size=bs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rate = len(texts) / max(1e-9, dt)
            print(
                f"seq={ms:<4} batch={bs:<5} {rate:7.1f} docs/s  "
                f"({dt:6.1f}s, shape={tuple(vecs.shape)})",
                flush=True,
            )
            del emb, vecs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
