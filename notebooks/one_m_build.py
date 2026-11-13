"""One million (1.0 M–2.6 M) build — literal 130-WET pipeline on marimo/molab.

Open with:  `marimo edit notebooks/one_m_build.py`
Mode: attended, ~2.5 h on the RTX Pro 6000 box.
Each cell is a single runnable stage. Stages run with force=False, so a
killed cell reuses the artifact it committed (or resumes a partial embed);
$HOME at /marimo/exa-at-home is all that survives the wipe.
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
                except Exception:  # noqa: BLE001, S112
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
    import shutil
    import sys

    sys.path.insert(0, ROOT)
    import ann_core  # noqa: F401  (fails early if box_setup.sh was skipped)
    import torch

    assert torch.cuda.is_available(), "CUDA is not available on this box"
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(
        f"ann_core OK; GPU: {torch.cuda.get_device_name(0)}; "
        f"free disk: {free_gb:.0f} GB"
    )
    assert free_gb > 100, f"only {free_gb:.0f} GB free; need >100 GB for the 1M run"


@app.cell
def _(ROOT):
    import os
    import sys

    sys.path.insert(0, ROOT)
    from scripts.run_pipeline import stage_download, stage_fetch

    urls = stage_fetch(
        "CC-MAIN-2026-34",
        130,
        os.path.join(ROOT, "wet_urls.txt"),
        force=False,
        verbose=True,
    )
    assert len(urls) == 130, f"expected 130 WET URLs, got {len(urls)}"
    stage_download(
        os.path.join(ROOT, "wet_urls.txt"),
        os.path.join(ROOT, "data/wet_130wet"),
        workers=16,
        force=False,
        verbose=True,
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
    res = stage_ingest(src, dst, force=False, verbose=True)
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
        docs,
        vecs,
        ids,
        batch=1024,
        chunk=8192,
        force=False,
        verbose=True,
        show_progress=True,
        resume=True,
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
        k=10000,
        use_filter=True,
        force=False,
        verbose=True,
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
        force=False,
        verbose=True,
    )
    print("QUERIES/GT DONE:", res)


@app.cell
def _(ROOT):
    """Anchor the golden 50 questions to the SAME pages in the 1M haystack.

    E3's questions are tied to target pages by index. Re-sampling targets
    from a new corpus would pair each question with a different page, so
    the G2 number would be meaningless. The toy E3 result records the 50
    target ids; map those ids into the 1M docs and pre-write the target
    file run_e3 expects (targets_path_for uses the docs hash).
    """
    import json
    import sys
    from pathlib import Path

    sys.path.insert(0, ROOT)
    from scripts.run_e3 import docs_fingerprint, targets_path_for

    docs = Path(ROOT) / "docs_130wet.jsonl"
    toy = json.loads((Path(ROOT) / "eval/e3_result_toy_20261021.json").read_text())
    want = {r["target"] for r in toy["rows"]}
    found = {}
    with open(docs, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in want:
                found[d["id"]] = d

    rows, missing = [], []
    for r in toy["rows"]:
        d = found.get(r["target"])
        if d is None:
            missing.append(r["target"])
            continue
        body = " ".join((d.get("text") or "").split())
        rows.append(
            {
                "n": r["n"],
                "id": d["id"],
                "url": d.get("url", ""),
                "text": body[:320],
            }
        )
    assert not missing, (
        f"{len(missing)} toy target ids are not in docs_130wet.jsonl; "
        "the 1M slice must include the toy pages for a valid G2 re-anchor"
    )
    fp = docs_fingerprint(docs)
    out = targets_path_for(docs, fp)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"anchored {len(rows)} golden targets -> {out}")
    return out


@app.cell
def _(ROOT, anchored):
    import os
    import sys

    assert anchored.exists(), "target anchor cell did not run"
    sys.path.insert(0, ROOT)
    from scripts.e2e_latency import main as e2e_main
    from scripts.gate import main as gate_main
    from scripts.run_e3 import main as run_e3_main
    from scripts.sweep import main as sweep_main

    base = [
        "--ann",
        "ivf",
        "--index",
        os.path.join(ROOT, "index_130wet"),
        "--queries",
        os.path.join(ROOT, "q_130wet.jsonl"),
        "--k",
        "10",
        "--budget-ms",
        "100",
        "--embedder",
        "real",
        "--reranker",
        "real",
        "--top-coarse",
        "50",
        "--max-pair-tokens",
        "64",
        "--batch-size",
        "32",
        "--max-query-tokens",
        "32",
        "--ledger",
        os.path.join(ROOT, "runs.jsonl"),
    ]
    assert e2e_main(base) == 0
    import glob

    assert (
        sweep_main(
            [
                "--index",
                os.path.join(ROOT, "index_130wet"),
                "--queries",
                os.path.join(ROOT, "q_130wet.jsonl"),
                "--gt",
                os.path.join(ROOT, "gt_130wet.jsonl"),
                "--nprobe",
                "8",
                "16",
                "--top-coarse",
                "50",
                "100",
                "--max-pair-tokens",
                "64",
                "--batch-size",
                "32",
                "--out",
                os.path.join(ROOT, "sweeps"),
                "--ledger",
                os.path.join(ROOT, "runs.jsonl"),
            ]
        )
        == 0
    )
    sweeps = sorted(glob.glob(os.path.join(ROOT, "sweeps") + "/*"))
    te = sweeps[-1]

    # G2 first: the gate reads eval/e3_result.json, so E3 must run before it.
    # Pass the 1M index/docs explicitly — run_e3 defaults to the toy index/.
    e3_index = os.path.join(ROOT, "index_130wet")
    e3_docs = os.path.join(ROOT, "docs_130wet.jsonl")
    e3_out = os.path.join(ROOT, "eval/e3_result.json")
    assert (
        run_e3_main(
            [
                "--index",
                e3_index,
                "--docs",
                e3_docs,
                "--out",
                e3_out,
            ]
        )
        == 0
    )

    rc = gate_main(
        [
            "--index",
            os.path.join(ROOT, "index_130wet"),
            "--queries",
            os.path.join(ROOT, "q_130wet.jsonl"),
            "--gt",
            os.path.join(ROOT, "gt_130wet.jsonl"),
            "--pred",
            os.path.join(te, "pred_np8_tc50_pt64_b32.jsonl"),
            "--e3-result",
            e3_out,
            "--budget-ms",
            "100",
            "--json",
            "--ledger",
            os.path.join(ROOT, "runs.jsonl"),
        ]
    )
    print("GATE EXIT:", rc)
    assert rc == 0
    print("DONE gate+sweep+E3")

    import tarfile

    out = os.path.join(ROOT, "exa-at-home-handoff.tgz")
    with tarfile.open(out, "w:gz") as t:
        for p in [
            "runs.jsonl",
            "wet_urls.txt",
            "index_130wet/manifest.json",
            "q_130wet.jsonl",
            "gt_130wet.jsonl",
            "eval/e3_result.json",
            "eval/e3_result_toy_20261021.json",
            "eval/questions_toy_20261021.json",
        ]:
            full = os.path.join(ROOT, p)
            if os.path.exists(full):
                t.add(full, arcname=p)
        # The sweep directory is the evidence for the 1M recall/latency
        # cells: grid.json + one pred file per combo + sweep.log.
        t.add(te, arcname=os.path.relpath(te, ROOT))
        # Corpus-specific E3 target sample (name embeds the docs hash).
        for full in glob.glob(
            os.path.join(ROOT, "eval/sampled_targets.docs_130wet.*.json")
        ):
            t.add(full, arcname=os.path.relpath(full, ROOT))
        # The full index_130wet/ (codes + lists + store) and vecs_130wet.npy
        # are large and derived, so they are NOT in the evidence tar. Rebuild
        # from wet_urls.txt + this repo, or archive the index separately if a
        # serving copy is needed.
    print("SNAPSHOT:", out)


if __name__ == "__main__":
    pass
