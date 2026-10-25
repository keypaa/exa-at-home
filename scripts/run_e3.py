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


def targets_path_for(docs: Path) -> Path:
    """One target set per corpus, so stale 4WET targets never get reused."""
    return ROOT / "eval" / f"sampled_targets.{docs.stem}.json"


def infer_docs_path(root: Path, index_dir: Path) -> Path:
    if (root / "docs.jsonl").is_file():
        return root / "docs.jsonl"
    stem = index_dir.name
    if stem.startswith("index") and (root / ("docs" + stem[len("index"): ] + ".jsonl")).is_file():
        return root / ("docs" + stem[len("index"): ] + ".jsonl")
    return root / "docs.jsonl"


def find_repo_root() -> Path:
    if os.path.isdir(ROOT / "exa_home"):
        return ROOT
    raise SystemExit(f"repo not found; cwd={os.getcwd()}")


def sample_targets(root: Path, n: int = 50, docs_path: Path | None = None,
                    target_file: Path | None = None) -> list[dict]:

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
    ap.add_argument("--out", default="eval/e3_result.json")
    args = ap.parse_args(argv)
    index_arg = args.index
    index_dir = Path(index_arg) if Path(index_arg).is_absolute() else (root / index_arg)
    store = ContentStore(str(index_dir / "store"))
    ann = IvfIndex.load(str(index_dir))

    docs_path = infer_docs_path(root, index_dir)
    if args.docs is not None:
        docs_path = Path(args.docs)
        if not docs_path.is_absolute():
            docs_path = root / docs_path
    target_file = targets_path_for(docs_path)
    targets = json.loads(target_file.read_text()) if target_file.exists() else sample_targets(root, n=50, docs_path=docs_path, target_file=target_file)
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
        rows.append({"n": i, "target": tgt, "coarse_pos": c_pos,
                     "reranked_pos": r_pos, "degraded": bool(deg),
                     "question": q["question"]})
        print(f"Q{i:<3} coarse_pos={c_pos:<3} reranked_pos={r_pos:<3} deg={deg}")

    n = len(rows)

    def hit(p):
        return 1 if 1 <= p <= 10 else 0

    def mrr(p):
        return (1.0 / p) if 1 <= p <= 10 else 0.0

    c_hit = sum(hit(r["coarse_pos"]) for r in rows)
    r_hit = sum(hit(r["reranked_pos"]) for r in rows)
    c_mrr = sum(mrr(r["coarse_pos"]) for r in rows) / n
    r_mrr = sum(mrr(r["reranked_pos"]) for r in rows) / n
    wins = sum(1 for r in rows if hit(r["reranked_pos"]) and not hit(r["coarse_pos"]))
    losses = sum(1 for r in rows if hit(r["coarse_pos"]) and not hit(r["reranked_pos"]))

    print(f"\n=== E3 RESULT (n={n}) ===")
    print(f"target recall@10   coarse={c_hit/n:.3f}   reranked={r_hit/n:.3f}")
    print(f"target MRR@10       coarse={c_mrr:.3f}   reranked={r_mrr:.3f}")
    print(f"paired: reranker wins {wins} / loses {losses} / ties {n-wins-losses}")

    result = {
        "rows": rows, "coarse_recall10": c_hit / n, "reranked_recall10": r_hit / n,
        "coarse_mrr10": c_mrr, "reranked_mrr10": r_mrr,
        "wins": wins, "losses": losses, "ties": n - wins - losses,
    }
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    out_path.write_text(json.dumps(result, indent=1))
    print(f"\nsaved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
