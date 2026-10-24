"""Item 1b diagnostic — pre-tokenize offline throughput measurement.

Measures how much of bulk embed wall time is the tokenizer, so the
pre-tokenize lever can be sized before it is implemented.

Reports (one line):
  N docs, batch, max_seq_len, device, prec, tokenize_only_s,
  tokenize_ms_per_doc, encode_total_s, docs_per_s, tokenize_share_pct
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from exa_home.config import EMBEDDER_MODEL, EMBED_BATCH
from exa_home.embed import Embedder
from exa_home.ingest import read_wet


def load_corpus(corpus: str | None, n: int) -> list[str]:
    """Return up to n document texts.

    None -> synthetic (matches Embedder/HashEmbedder tests).
    .jsonl -> read the 'text' field from each row.
    anything else -> parse as WET (possibly .gz) via read_wet.
    """
    if corpus is None:
        return [
            f"This is synthetic document {i} about topic {i % 7}, "
            f"with additional text so the tokenizer has work to do, "
            f"and length varies slightly {i % 13}."
            for i in range(n)
        ]
    p = Path(corpus)
    texts: list[str] = []
    if p.suffix == ".jsonl":
        import json as _json

        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = _json.loads(line)
                t = (d.get("text") or "").strip()
                if t:
                    texts.append(t)
                    if len(texts) >= n:
                        break
        return texts
    for doc in read_wet(str(p)):
        if doc.error or not doc.text:
            continue
        texts.append(doc.text)
        if len(texts) >= n:
            break
    return texts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--corpus", default=None,
                    help="WET file (.gz ok) or docs jsonl; default = synth")
    ap.add_argument("--batch-size", type=int, default=EMBED_BATCH)
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    emb = Embedder(
        model=EMBEDDER_MODEL, dim=256, max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
    )
    texts = load_corpus(args.corpus, args.n)
    if not texts:
        raise SystemExit("no documents loaded from corpus")
    args.n = len(texts)
    bs = args.batch_size

    if args.warmup:
        emb.encode_docs(texts[: min(args.warmup, args.n)], batch_size=bs,
                        sort_by_length=False)

    t0 = time.perf_counter()
    emb.model.tokenizer(texts, padding=True, truncation=True,
                        max_length=args.max_seq_length)
    tokenize_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = emb.encode_docs(texts, batch_size=bs, sort_by_length=False)
    encode_s = time.perf_counter() - t0

    share = tokenize_s / encode_s if encode_s > 0 else float("nan")
    out = {
        "n_docs": args.n,
        "corpus": args.corpus,
        "batch_size": bs,
        "max_seq_length": args.max_seq_length,
        "device": emb.device,
        "precision": emb.precision,
        "tokenize_only_s": round(tokenize_s, 4),
        "tokenize_ms_per_doc": round(tokenize_s / args.n * 1000, 4),
        "encode_total_s": round(encode_s, 4),
        "docs_per_s": round(args.n / encode_s, 2),
        "tokenize_share_pct": round(share * 100, 2),
    }
    print(json.dumps(out, indent=2))
    print(
        f"\nn={args.n} bs={bs} msl={args.max_seq_length} {emb.device} "
        f"tok={tokenize_s:.3f}s encode={encode_s:.3f}s "
        f"share={share*100:.1f}% docs/s={args.n/encode_s:.1f}"
    )


if __name__ == "__main__":
    main()