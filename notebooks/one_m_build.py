"""One million (1.0 M–2.6 M) build — literal 130-WET pipeline on marimo/molab.

Open with:  `marimo edit notebooks/one_m_build.py`
Mode: attended, ~2.5 h on the RTX Pro 6000 box.
Each cell is a single runnable stage; kill any one cell and resume from the
artifact it committed ($HOME at /marimo/exa-at-home, nothing else survives
the wipe).
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import os
    import sys

    REPO_URL = "https://github.com/keypaa/exa-at-home.git"

    def _has_markers(d):
        return os.path.isdir(os.path.join(d, "exa_home")) and os.path.isdir(
            os.path.join(d, "scripts")
        )

    def _find_root():
        cands = [os.getcwd()]
        seen = set()
        queue = list(cands)
        try:
            while queue and len(seen) < 200:
                base = queue.pop(0)
                if base in seen:
                    continue
                seen.add(base)
                if _has_markers(base):
                    return base
                try:
                    kids = sorted(os.listdir(base))
                except Exception:  # noqa: BLE001, S110
                    continue
                for child in kids[:50]:
                    full = os.path.join(base, child)
                    if os.path.isdir(full) and full not in seen:
                        if _has_markers(full):
                            return full
                        queue.append(full)
        except Exception:  # noqa: BLE001, S110
            pass
        for base in cands:
            d = base
            for _ in range(5):
                if _has_markers(d):
                    return d
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        return cands[0]

    ROOT = _find_root()
    if not _has_markers(ROOT):
        print("Repo not found; clone it in a terminal, then re-open:", REPO_URL)
    for p in (ROOT, os.path.join(ROOT, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    print(f"repo root: {ROOT}")
    return ROOT, os


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_download, stage_fetch

    stage_fetch(
        "CC-MAIN-2026-34", 130, os.path.join(ROOT, "wet_urls.txt"),
        force=True, verbose=True,
    )
    stage_download(
        os.path.join(ROOT, "wet_urls.txt"), os.path.join(ROOT, "data/wet_130wet"),
        workers=16, force=True, verbose=True,
    )
    print("DONE: fetch + download")


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_ingest

    src = os.path.join(ROOT, "data/wet_130wet")
    dst = os.path.join(ROOT, "docs_130wet.jsonl")
    res = stage_ingest(src, dst, force=True, verbose=True)
    print("INGEST DONE:", res)


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_embed

    docs = os.path.join(ROOT, "docs_130wet.jsonl")
    vecs = os.path.join(ROOT, "vecs_130wet.npy")
    ids = os.path.join(ROOT, "ids_130wet.json")
    shape, dtype = stage_embed(
        docs, vecs, ids, batch=1024, chunk=8192,
        force=True, verbose=True, show_progress=True, resume=True,
    )
    print("EMBED DONE:", shape, dtype)


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_index

    shape = stage_index(
        os.path.join(ROOT, "vecs_130wet.npy"),
        os.path.join(ROOT, "ids_130wet.json"),
        os.path.join(ROOT, "docs_130wet.jsonl"),
        os.path.join(ROOT, "index_130wet"),
        k=10000, use_filter=True, force=True, verbose=True,
    )
    print("INDEX DONE:", shape)


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_queries

    res = stage_queries(
        os.path.join(ROOT, "docs_130wet.jsonl"),
        os.path.join(ROOT, "vecs_130wet.npy"),
        os.path.join(ROOT, "ids_130wet.json"),
        200,
        queries_path=os.path.join(ROOT, "q_130wet.jsonl"),
        gt_path=os.path.join(ROOT, "gt_130wet.jsonl"),
        force=True, verbose=True,
    )
    print("QUERIES/GT DONE:", res)


@app.cell
def _(ROOT):
    import os
    import sys
    sys.path.insert(0, ROOT)
    from scripts.e2e_latency import main as e2e_main
    from scripts.gate import main as gate_main
    from scripts.run_e3 import main as run_e3_main
    from scripts.sweep import main as sweep_main

    base = [
        "--ann", "ivf", "--index", os.path.join(ROOT, "index_130wet"),
        "--queries", os.path.join(ROOT, "q_130wet.jsonl"),
        "--k", "10", "--budget-ms", "100", "--embedder", "real",
        "--reranker", "real", "--top-coarse", "50",
        "--max-pair-tokens", "64", "--batch-size", "32",
        "--max-query-tokens", "32", "--ledger", os.path.join(ROOT, "runs.jsonl"),
    ]
    assert e2e_main(base) == 0
    import glob
    sweep_main([
        "--index", os.path.join(ROOT, "index_130wet"),
        "--queries", os.path.join(ROOT, "q_130wet.jsonl"),
        "--gt", os.path.join(ROOT, "gt_130wet.jsonl"),
        "--nprobe", "8", "16", "--top-coarse", "50", "100",
        "--max-pair-tokens", "64", "--batch-size", "32",
        "--ledger", os.path.join(ROOT, "runs.jsonl"),
    ])
    sweeps = sorted(glob.glob(os.path.join(ROOT, "sweeps") + "/*"))
    te = sweeps[-1]
    rc = gate_main([
        "--index", os.path.join(ROOT, "index_130wet"),
        "--queries", os.path.join(ROOT, "q_130wet.jsonl"),
        "--gt", os.path.join(ROOT, "gt_130wet.jsonl"),
        "--pred", os.path.join(te, "pred_np8_tc50_pt64_b32.jsonl"),
        "--e3-result", os.path.join(ROOT, "eval/e3_result.json"),
        "--budget-ms", "100", "--json",
        "--ledger", os.path.join(ROOT, "runs.jsonl"),
    ])
    print("GATE EXIT:", rc)
    assert rc == 0
    run_e3_main()
    print("DONE gate+sweep+E3")

    import tarfile
    out = os.path.join(ROOT, "exa-at-home-handoff.tgz")
    with tarfile.open(out, "w:gz") as t:
        for p in ["runs.jsonl", "index_130wet", "q_130wet.jsonl", "gt_130wet.jsonl",
                  "vecs_130wet.npy", "ids_130wet.json",
                  "eval/e3_result.json", "eval/e3_result_toy_20261021.json"]:
            full = os.path.join(ROOT, p)
            if os.path.exists(full):
                t.add(full, arcname=p)
    print("SNAPSHOT:", out)


if __name__ == "__main__":
    pass
