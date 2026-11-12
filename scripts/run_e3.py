"""Run the E3 question-grounded retrieval eval on the local index.

Wraps eval/E3_PROTOCOL.md steps 1 + 3 into one reproducible entry point so
the G2 answer-coverage gate can be re-run after any index/corpus change.
Writes eval/e3_result.json.
"""

from __future__ import annotations

import json
import os
import random
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

QUESTIONS = ROOT / "eval" / "questions_toy_20261021.json"
TARGETS = ROOT / "eval" / "sampled_targets.json"


def targets_path_for(docs: Path, fingerprint: str = "") -> Path:
    """One target set per CORPUS, not per file name.

    The old name was derived from docs.stem only, so re-ingesting into the
    same path silently reused target ids that no longer existed in the new
    corpus (finding C10). The fingerprint is the docs content hash.
    """
    suffix = f".{fingerprint[:12]}" if fingerprint else ""
    return ROOT / "eval" / f"sampled_targets.{docs.stem}{suffix}.json"


def infer_docs_path(root: Path, index_dir: Path) -> Path:
    """Pick the docs file belonging to `index_dir`.

    Corpus-named docs (index_1wet -> docs_1wet.jsonl) win over the root
    docs.jsonl: preferring the root file meant `--index index_1wet` was
    evaluated against the TOY corpus whenever both existed (finding C10).
    The result is still validated against the index's recorded source.
    """
    stem = index_dir.name
    if stem.startswith("index"):
        candidate = root / ("docs" + stem[len("index") :] + ".jsonl")
        if candidate.is_file():
            return candidate
    return root / "docs.jsonl"


def docs_fingerprint(docs: Path) -> str:
    from exa_home.index_format import sha256

    return sha256(str(docs))


def validate_docs_for_index(docs: Path, index_dir: Path) -> str:
    """Return the docs content hash, or raise when it is not this index's corpus.

    The index records its source identity at build time. Without this check
    a stale docs.jsonl (or an explicit --docs pointing at another corpus)
    produced target ids from a corpus the index never saw.
    """
    from scripts.build_index import read_source

    src = read_source(str(index_dir))
    got = docs_fingerprint(docs)
    if not src:
        print(
            f"warning: {index_dir} has no source provenance; "
            "cannot validate --docs against it",
            file=sys.stderr,
        )
        return got
    want = src.get("docs_sha256")
    if want:
        if got != want:
            raise ValueError(
                f"{docs} does not match the corpus {index_dir} was built from "
                f"(docs sha256 {got[:12]} != {want[:12]})"
            )
        return got
    # Legacy index written before docs hashing: only size is available.
    want_size = src.get("docs_size")
    if want_size is not None and os.path.getsize(docs) != want_size:
        raise ValueError(
            f"{docs} does not match the corpus {index_dir} was built from "
            f"(size {os.path.getsize(docs)} != {want_size})"
        )
    print(
        "warning: index provenance predates docs hashing; only size checked",
        file=sys.stderr,
    )
    return got


def find_repo_root() -> Path:
    if os.path.isdir(ROOT / "exa_home"):
        return ROOT
    raise SystemExit(f"repo not found; cwd={os.getcwd()}")


def sample_targets(
    root: Path,
    n: int = 50,
    docs_path: Path | None = None,
    target_file: Path | None = None,
) -> list[dict]:

    def ascii_heavy(s):
        return sum(ch.isascii() for ch in s) / max(1, len(s)) > 0.9

    docs = docs_path if docs_path is not None else (root / "docs.jsonl")
    tf = target_file if target_file is not None else TARGETS
    pool = []
    with open(docs, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            body = " ".join((d.get("text") or "").split())
            if len(body) < 800 or not ascii_heavy(body[:400]):
                continue
            pool.append((d["id"], d.get("url", ""), body[:320]))
    print(f"pool: {len(pool)}")
    rng = random.Random(20261021)
    picked = rng.sample(pool, n)
    out = [{"n": i, "id": d, "url": u, "text": t} for i, (d, u, t) in enumerate(picked)]
    tf.parent.mkdir(parents=True, exist_ok=True)
    tf.write_text(json.dumps(out, indent=1))
    return out


def ensure_questions(root: Path) -> dict:
    if QUESTIONS.exists():
        return json.loads(QUESTIONS.read_text())
    url = "https://raw.githubusercontent.com/keypaa/exa-at-home/main/eval/questions_toy_20261021.json"
    print("downloading questions from", url)
    QUESTIONS.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, QUESTIONS)
    return json.loads(QUESTIONS.read_text())


def main(argv: list[str] | None = None) -> int:
    import argparse

    import numpy as np
    from ann_core import IvfIndex

    from exa_home.embed import Embedder
    from exa_home.rerank import Reranker
    from exa_home.store import ContentStore

    root = find_repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--index", default="index")
    ap.add_argument(
        "--docs", default=None, help="docs.jsonl (defaults via infer_docs_path)"
    )
    ap.add_argument("--out", default="eval/e3_result.json")
    args = ap.parse_args(argv)
    index_arg = args.index
    index_dir = Path(index_arg) if Path(index_arg).is_absolute() else (root / index_arg)
    docs_path = Path(args.docs) if args.docs else infer_docs_path(root, index_dir)
    if not docs_path.is_absolute():
        docs_path = root / docs_path
    try:
        docs_fp = validate_docs_for_index(docs_path, index_dir)
    except ValueError as e:
        raise SystemExit(f"run_e3: {e}") from e
    store = ContentStore(str(index_dir / "store"))
    ann = IvfIndex.load(str(index_dir))

    target_file = targets_path_for(docs_path, docs_fp)
    targets = (
        json.loads(target_file.read_text())
        if target_file.exists()
        else sample_targets(root, n=50, docs_path=docs_path, target_file=target_file)
    )
    Qdoc = ensure_questions(root)
    QQ = Qdoc["questions"]
    Q = [{"question": q["question"], "target_id": targets[q["n"]]["id"]} for q in QQ]
    print(f"questions: {len(Q)} (targets by index from sampled_targets.json)")

    emb = Embedder()
    rr = Reranker(max_pair_tokens=64, batch_size=32, max_query_tokens=32)
    qv = np.atleast_2d(emb.encode_queries([q["question"] for q in Q]))

    rows = []
    for i, q in enumerate(Q):
        vec, tgt = qv[i], q["target_id"]
        c50, _ = ann.search(vec, nprobe=8, n_coarse=50)
        texts = [(store.get(c) or {}).get("text", "")[:2000] for c in c50]
        cands = [{"id": c, "text": t} for c, t in zip(c50, texts)]
        ranked, deg = rr.rerank(q["question"], cands)
        r50 = [r["id"] for r in ranked]
        c_pos = c50.index(tgt) + 1 if tgt in c50 else 0
        r_pos = r50.index(tgt) + 1 if tgt in r50 else 0
        rows.append(
            {
                "n": i,
                "target": tgt,
                "coarse_pos": c_pos,
                "reranked_pos": r_pos,
                "degraded": bool(deg),
                "question": q["question"],
            }
        )
        print(f"Q{i:<3} coarse_pos={c_pos:<3} reranked_pos={r_pos:<3} deg={deg}")

    n = len(rows)
    if n == 0:
        c_hit = r_hit = 0
        c_mrr = r_mrr = 0.0
        wins = losses = 0
    else:

        def hit(p):
            return 1 if 1 <= p <= 10 else 0

        def mrr(p):
            return (1.0 / p) if 1 <= p <= 10 else 0.0

        c_hit = sum(hit(r["coarse_pos"]) for r in rows)
        r_hit = sum(hit(r["reranked_pos"]) for r in rows)
        c_mrr = sum(mrr(r["coarse_pos"]) for r in rows) / n
        r_mrr = sum(mrr(r["reranked_pos"]) for r in rows) / n
        wins = sum(
            1 for r in rows if hit(r["reranked_pos"]) and not hit(r["coarse_pos"])
        )
        losses = sum(
            1 for r in rows if hit(r["coarse_pos"]) and not hit(r["reranked_pos"])
        )

    print(f"\n=== E3 RESULT (n={n}) ===")
    print(
        f"target recall@10   coarse={c_hit / n if n else 0.0:.3f}   reranked={r_hit / n if n else 0.0:.3f}"
    )
    print(f"target MRR@10       coarse={c_mrr:.3f}   reranked={r_mrr:.3f}")
    print(f"paired: reranker wins {wins} / loses {losses} / ties {n - wins - losses}")

    result = {
        "rows": rows,
        "coarse_recall10": c_hit / n,
        "reranked_recall10": r_hit / n,
        "coarse_mrr10": c_mrr,
        "reranked_mrr10": r_mrr,
        "wins": wins,
        "losses": losses,
        "ties": n - wins - losses,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(json.dumps(result, indent=1))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
