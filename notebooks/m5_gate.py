"""M5 latency gate — literate notebook (synth <-> toy rebuild).

Run natively: `marimo edit notebooks/m5_gate.py` (needs `.[dev]` extras).
On molab: the workspace is wiped between sessions, so this notebook assumes
NOTHING persists. Open it from a fresh clone, pick a mode, run top to
bottom — every stage rebuilds from the repo, every number comes from
`import exa_home` library calls. No shell, no subprocess.

Modes:
- `synth`: in-memory fakes (HashEmbedder + MockReranker). Seconds. Proves
  the notebook, the graphs, the ledger plumbing. Runs anywhere.
- `toy`: full rebuild — 4 WET → ingest → mxbai embed → K=1024 index →
  gate + recall on the 81k toy (box, ~30 min attended, progress ticks).
"""

import marimo

__generated_with = "0.25.0"
app = marimo.App()


@app.cell
def _():
    import os
    import sys

    def _find_root():
        """Locate the repo root without trusting __file__.

        Mirrored/hosted marimo serves the notebook from a virtual path
        (e.g. marimo://notebook.py), so __file__-based anchoring resolves
        to garbage and `import scripts` fails. Instead, walk upward from
        the CWD (and __file__, when it is a real path) looking for the
        repo markers (exa_home/ + scripts/).
        """
        cands = [os.getcwd()]
        try:
            f = __file__  # noqa: F821 — absent on some hosts
            if isinstance(f, str) and os.path.sep in f:
                cands.append(os.path.dirname(
                    os.path.dirname(os.path.abspath(f))))
        except Exception:
            pass
        for base in cands:
            d = base
            for _ in range(5):
                if os.path.isdir(os.path.join(d, "exa_home")) and \
                   os.path.isdir(os.path.join(d, "scripts")):
                    return d
                parent = os.path.dirname(d)
                if parent == d:
                    break
                d = parent
        return cands[0]

    ROOT = _find_root()
    for p in (ROOT, os.path.join(ROOT, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    print(f"repo root: {ROOT}")
    return ROOT, os


@app.cell
def _(ROOT, os):
    import json
    import time

    import marimo as mo
    import plotly.graph_objects as go
    from scripts import e2e_latency as e2e
    from scripts import run_pipeline as pipe

    def R(*parts):
        """Repo-anchored path (the box is wiped; never rely on CWD)."""
        return os.path.join(ROOT, *parts)

    return R, e2e, go, json, mo, pipe, time


@app.cell
def _(mo):
    mo.md(
        r"""
        # M5 latency gate: the funnel, measured

        Exa's funnel principle — expensive full-precision work touches only
        ~200 docs, never 5M. This notebook proves our toy obeys it:

        | Stage | Budget (p50) | What it is |
        |---|---|---|
        | embed | 5–10ms | query text → 256-dim vector (GPU on box, hash stub in synth) |
        | ann | 10–20ms | binary IVF scan, Rust LUT scoring (CPU) |
        | rerank fetch | — | `store.get` × top_coarse (the M5b lesson: this hid 1400ms) |
        | rerank model | 20–40ms | cross-encoder over candidates (GPU on box) |
        | snippet | <10ms | first 300 chars per hit |
        | **total** | **<100ms** | PASS/FAIL below |

        The stacked bar splits fetch vs compute (observability pass), so a
        store regression can never masquerade as GPU time again.
        """
    )
    return


@app.cell
def _(mo):
    # List options (not a dict): dropdown .value is then identical in
    # interactive and headless runs — dict label/value resolution differs
    # per context and once sent this notebook down the box path in export.
    mode = mo.ui.dropdown(
        ["synth", "toy"],
        value="synth",
        label="Mode (synth = fakes, seconds / toy = 4-WET rebuild, box, ~30 min)",
    )
    top_coarse = mo.ui.slider(10, 200, value=50, step=10, label="top_coarse")
    nprobe = mo.ui.slider(1, 16, value=8, step=1, label="nprobe")
    mo.hstack([mode, top_coarse, nprobe])
    return mode, nprobe, top_coarse


@app.cell
def _(R, mode, pipe):
    if mode.value == "toy":
        slice_urls = pipe.stage_fetch("CC-MAIN-2026-34", 4, R("wet_urls.txt"),
                                      verbose=True)
        print(f"slice: {len(slice_urls)} URLs -> wet_urls.txt")
    else:
        print("synth: fetch skipped (no corpus needed)")
    return


@app.cell
def _(R, mode, pipe):
    if mode.value == "toy":
        pipe.stage_download(R("wet_urls.txt"), R("data", "wet"),
                            workers=16, verbose=True)
    print("download: done (or synth-skipped)")
    return


@app.cell
def _(R, mode, pipe):
    if mode.value == "toy":
        pipe.stage_ingest(R("data", "wet"), R("docs.jsonl"), verbose=True)
    print("ingest: done (or synth-skipped)")
    return


@app.cell
def _(R, mode, pipe):
    if mode.value == "toy":
        pipe.stage_embed(R("docs.jsonl"), R("vecs.npy"), R("ids.json"),
                         batch=256, verbose=True, show_progress=True)
    print("embed: done (or synth-skipped)")
    return


@app.cell
def _(R, e2e, mode, nprobe, pipe, top_coarse):
    if mode.value == "toy":
        pipe.stage_index(R("vecs.npy"), R("ids.json"), R("docs.jsonl"),
                         R("index"), k=1024, verbose=True)
        pipe.stage_queries(R("docs.jsonl"), R("vecs.npy"), R("ids.json"),
                           200, queries_path=R("q.jsonl"),
                           gt_path=R("gt.jsonl"))
        dag = e2e.build_real_dag(R("index"), None, "real", "real",
                                 top_coarse=top_coarse.value,
                                 nprobe=nprobe.value,
                                 max_pair_tokens=64, batch_size=32,
                                 max_query_tokens=32)
        queries = e2e.load_queries(R("q.jsonl"))
        print(f"toy ready: {len(queries)} queries, "
              f"top_coarse={top_coarse.value}, nprobe={nprobe.value}")
    else:
        dag = e2e.build_synth_dag(n_docs=200, seed=0,
                                  top_coarse=top_coarse.value,
                                  nprobe=nprobe.value)
        queries = [{"query": f"synthetic topic {i} about neural search",
                    "filters": {}} for i in range(10)]
        print(f"synth ready: {len(queries)} queries")
    return dag, queries


@app.cell
def _(dag, e2e, queries, time):
    gate_t0 = time.perf_counter()
    stats = e2e.run_latency(dag, queries, k=10, verbose=True)
    print(f"gate: {stats['n']} queries in "
          f"{time.perf_counter() - gate_t0:.1f}s, "
          f"{len(stats['violations'])} waterfall violations")
    return (stats,)


@app.cell
def _(e2e, go, mo, stats):
    table = e2e.report(stats, 100.0)
    stages = list(e2e.STAGE_ORDER)
    p50s = [e2e.pct(stats["stages"][s], 0.5) for s in stages]
    bd = stats.get("breakdown") or {}

    def _p50(key):
        return e2e.pct(bd.get(key, [0.0]), 0.5)

    fetch = [_p50("rerank_fetch_ms") if s == "rerank_ms"
             else _p50("snippet_fetch_ms") if s == "snippet_ms"
             else 0.0 for s in stages]
    compute = [p - f for p, f in zip(p50s, fetch)]
    p99s = [e2e.pct(stats["stages"][s], 0.99) for s in stages]

    fig_water = go.Figure()
    fig_water.add_bar(x=stages, y=compute, name="compute",
                      marker_color="#636EFA")
    fig_water.add_bar(x=stages, y=fetch, name="store fetch",
                      marker_color="#EF553B")
    fig_water.update_layout(barmode="stack", template="plotly_white",
                            title="M5 waterfall — p50 per stage (fetch vs compute)",
                            xaxis_title="stage", yaxis_title="p50 ms")
    fig_p99 = go.Figure()
    fig_p99.add_bar(x=stages, y=p50s, name="p50", marker_color="#636EFA")
    fig_p99.add_bar(x=stages, y=p99s, name="p99", marker_color="#AB63FA")
    fig_p99.update_layout(template="plotly_white",
                          title="p50 vs p99 per stage",
                          xaxis_title="stage", yaxis_title="ms")
    mo.vstack([mo.md(f"```\n{table}\n```"), fig_water, fig_p99])
    return


@app.cell
def _(R, json, mo, mode, queries):
    if mode.value == "synth":
        mo.md("Synth mode skips predictions by design — they need the box "
              "corpus and GPU models (toy mode runs this cell).")
    else:
        # Imports live INSIDE the branch: synth never touches GPU libs.
        from ann_core import IvfIndex
        from exa_home.embed import Embedder
        from exa_home.orch import build_search_dag, run_search
        from exa_home.rerank import Reranker
        from exa_home.store import ContentStore

        dag_pred = build_search_dag(Embedder(), IvfIndex.load(R("index")),
                                    Reranker(max_pair_tokens=64, batch_size=32,
                                             max_query_tokens=32),
                                    ContentStore(R("index", "store")),
                                    top_coarse=50)
        with open(R("pred.jsonl"), "w") as fout:
            for k, q in enumerate(queries):
                res = run_search(dag_pred, q["query"], {}, top_k=10)
                fout.write(json.dumps({"query_id": k, "top10": [r["id"] for r in res["results"]]}) + "\n")
                if (k + 1) % 50 == 0:
                    print(f"pred: {k + 1}/{len(queries)}", flush=True)
        print(f"{len(queries)} predictions -> pred.jsonl")
    return


@app.cell
def _(R, go, json, mo, mode, os):
    if mode.value == "synth" or not (
            os.path.exists(R("gt.jsonl")) and os.path.exists(R("pred.jsonl"))):
        mo.md("Recall graph appears in toy mode once `gt.jsonl` + `pred.jsonl` "
              "exist (previous cells). Tier-0 stops the story at latency.")
    else:
        gt_rows = [json.loads(line) for line in open(R("gt.jsonl"), encoding="utf-8")]
        pr_by_id = {json.loads(line)["query_id"]: json.loads(line)["top10"]
                    for line in open(R("pred.jsonl"), encoding="utf-8")}
        recs = [len(set(g["top10"]) & set(pr_by_id[g["query_id"]])) / 10
                for g in gt_rows]
        recall = sum(recs) / len(recs)
        fig_rec = go.Figure()
        fig_rec.add_bar(x=["reranked top-10"], y=[recall], name="recall@10",
                        marker_color="#00CC96")
        fig_rec.add_hline(y=0.027, line_dash="dash", line_color="red",
                          annotation_text="coarse-only skew bound 0.027")
        fig_rec.update_layout(template="plotly_white",
                              title=f"Recall@10 = {recall:.4f} over {len(recs)} queries",
                              yaxis_title="recall@10",
                              yaxis_range=[0, max(0.5, recall * 1.2)])
        fig_rec
    return


@app.cell
def _(R, e2e, mo, stats):
    total_p50 = e2e.pct(stats["total"], 0.5)
    verdict = "PASS" if total_p50 < 100.0 else "OVER BUDGET"
    row = e2e.append_ledger(R("runs.jsonl"),
                            {"notebook": "m5_gate", "n": stats["n"]}, stats)
    mo.md(
        f"## Verdict: {verdict} — total p50 {total_p50:.2f}ms vs 100ms budget\n\n"
        f"Waterfall integrity: "
        f"{'OK' if not stats['violations'] else 'FAIL'} "
        f"({len(stats['violations'])} violations). "
        f"Run banked to `runs.jsonl` — **commit it + push before leaving the box** "
        f"(git {row['git']}, {row['ts']})."
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## How to read this (and stay welcome on molab)

        - **Fetch vs compute inside rerank** is the M5b lesson made visible:
          on the 81k toy the fetch bar hid ~1400ms of JSON shard scans behind
          a "GPU" span. `store/offsets.json` (built at index time) killed it —
          total p50 4025ms → 46ms.
        - **Recall pairs with latency**: the recall bar needs the serving
          stack (`pred.jsonl`), the waterfall needs the gate. Report both or
          report neither.
        - **This notebook IS the interface**: markdown reasoning, library
          imports, plots, sharing. No shell heredocs, no background jobs, no
          keep-alives — everything attended, everything committed.
        """
    )
    return


if __name__ == "__main__":
    app.run()
