"""M5 latency gate — literate notebook (synth <-> toy rebuild).

Self-contained: importing JUST this file is enough. The first cells
bootstrap the repo (clone-if-missing) and check the environment, so a
wiped box goes from empty workspace to running gate with no terminal.

Run natively: `marimo edit notebooks/m5_gate.py` (needs `.[dev]` extras).
On molab: open the file (mirror or import), run top to bottom — every
stage rebuilds from the repo, every number comes from `import exa_home`
library calls. Shell/subprocess appear ONLY in the bootstrap cell (setup,
not compute).

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
    import subprocess
    import sys

    REPO_URL = "https://github.com/keypaa/exa-at-home.git"

    def _has_markers(d):
        return os.path.isdir(os.path.join(d, "exa_home")) and \
            os.path.isdir(os.path.join(d, "scripts"))

    def _find_root():
        """Locate the repo root without trusting __file__.

        Mirrored/hosted marimo serves the notebook from a virtual path
        (e.g. marimo://notebook.py), so __file__-based anchoring resolves
        to garbage and `import scripts` fails. Instead, look for the repo
        markers (exa_home/ + scripts/): upward from the CWD and from
        __file__ (when real), plus one level DOWN from the CWD (mirrored
        workspaces often nest the repo, e.g. <workspace>/exa-at-home/).
        """
        # NOTE: mirroring the notebook file alone is NOT enough — the
        # library lives in the repo. Clone it once per session (terminal):
        #   git clone https://github.com/keypaa/exa-at-home.git
        # then this cell finds it below.
        cands = [os.getcwd()]
        try:
            f = __file__  # noqa: F821 — absent on some hosts
            if isinstance(f, str) and os.path.sep in f:
                cands.append(os.path.dirname(
                    os.path.dirname(os.path.abspath(f))))
        except Exception:
            pass
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
                except Exception:
                    continue
                cands.append(base)
                for child in kids[:50]:
                    full = os.path.join(base, child)
                    if os.path.isdir(full) and full not in seen:
                        if _has_markers(full):
                            return full
                        queue.append(full)
        except Exception:
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
        # Only the notebook file came over (mirror/import a single file):
        # fetch the library. Attended one-time setup — the only
        # subprocess in the notebook; all compute stays in library calls.
        dest = os.path.join(os.getcwd(), "exa-at-home")
        print(f"repo not on box — cloning {REPO_URL}\n  -> {dest} ...")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, dest],
                       check=True)
        ROOT = dest
    for p in (ROOT, os.path.join(ROOT, "tests")):
        if p not in sys.path:
            sys.path.insert(0, p)
    try:
        kids = sorted(os.listdir(os.getcwd()))[:20]
    except Exception as e:
        kids = [f"<unlistable: {e}>"]
    print(f"repo root: {ROOT} (markers: "
          f"{os.path.isdir(os.path.join(ROOT, 'exa_home'))} / "
          f"{os.path.isdir(os.path.join(ROOT, 'scripts'))})")
    print(f"cwd: {os.getcwd()}")
    print(f"cwd contents: {kids}")
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
def _(R, mo):
    import importlib.util

    rows = []
    for mod in ("torch", "sentence_transformers", "duckdb", "sklearn",
                "pyroaring", "warcio", "plotly", "ann_core"):
        rows.append((mod, "yes" if importlib.util.find_spec(mod) is not None else "MISSING"))
    cuda = "n/a (no torch)"
    try:
        import torch

        cuda = f"cuda={torch.cuda.is_available()}"
    except Exception:
        pass
    status = "\n".join(f"| {m} | {s} |" for m, s in rows)
    missing = [m for m, s in rows if s == "MISSING"]
    advice = ""
    if missing:
        advice = ("\n**Missing**: " + ", ".join(missing) +
                  " — in the box terminal: `pip install -e .[dev]` from the repo root" +
                  ("; then `cd crates/ann-core && maturin develop`" if "ann_core" in missing else "") +
                  " (RUNBOOK §0). Re-run this cell after.")
    setup_btn = mo.ui.run_button(label="Run box setup (rust + deps + ann_core build)")
    mo.vstack([mo.md(f"## 0. Environment\n| module | importable |\n|---|---|\n{status}\n\ntorch: {cuda}"
          f"\n\nRepo: `{R()}`" + advice), setup_btn])
    return (setup_btn,)


@app.cell
def _(R, mo, setup_btn):
    def _do_setup():
        """One-shot box setup: rust toolchain + repo deps + ann_core build.

        Idempotent (every step checks first) and attended (runs only when
        the button above is pressed). All imports stay inside the function
        so the notebook namespace is untouched.
        """
        import os
        import shutil
        import subprocess
        import sys

        repo = R()
        env = dict(os.environ)
        if shutil.which("cargo") is None:
            print("rust: no cargo — installing stable toolchain via rustup "
                  "(minutes, attended) ...", flush=True)
            subprocess.run(
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs"
                " | sh -s -- -y --default-toolchain stable --profile minimal",
                shell=True, check=True)
            cargo_bin = os.path.expanduser("~/.cargo/bin")
            env["PATH"] = cargo_bin + os.pathsep + env.get("PATH", "")
            os.environ["PATH"] = env["PATH"]
        else:
            print(f"rust: {shutil.which('cargo')}", flush=True)
        if sys.version_info >= (3, 14):
            env["PYO3_USE_ABI3_FORWARD_COMPATIBILITY"] = "1"
            print("python >= 3.14: PyO3 forward-compat flag set", flush=True)
        print("pip: installing repo + dev extras ...", flush=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev]"],
            cwd=repo, env=env, check=True)
        # Belt-and-braces: extras don't always land (observed on a fresh
        # uv-venv: maturin missing despite exit 0), so verify the load-
        # bearing modules and fill gaps individually.
        import importlib.util

        for mod, pkg in (("pyroaring", "pyroaring>=0.4"),
                         ("warcio", "warcio>=1.7"),
                         ("maturin", "maturin>=1.0")):
            if importlib.util.find_spec(mod) is None:
                print(f"pip: {mod} missing after extras — installing {pkg} ...",
                      flush=True)
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", pkg],
                    env=env, check=True)
        try:
            import ann_core  # noqa: F401
            print("ann_core: already importable", flush=True)
        except ImportError:
            print("rust: building ann_core (maturin develop, minutes) ...",
                  flush=True)
            mat_bin = shutil.which("maturin")
            cmd = [mat_bin, "develop"] if mat_bin else \
                [sys.executable, "-m", "maturin", "develop"]
            subprocess.run(
                cmd,
                cwd=os.path.join(repo, "crates", "ann-core"),
                env=env, check=True)
            print("ann_core: built", flush=True)
        print("setup complete — re-run the §0 Environment cell, then continue below.")

    if not setup_btn.value:
        mo.md("Press **Run box setup** in the §0 cell above to install "
              "missing pieces (rust + deps + `ann_core` build — one click, "
              "attended, minutes). Already green? Just keep scrolling.")
    else:
        _do_setup()
        mo.md("Setup ran — re-run the **§0 Environment** cell to confirm "
              "everything reads `yes`, then continue below.")
    return


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
def _(R, json, mo, mode, nprobe, queries, top_coarse):
    if mode.value == "synth":
        mo.md("Synth mode skips predictions by design — they need the box "
              "corpus and GPU models (toy mode runs this cell).")
    else:
        # Imports live INSIDE the branch: synth never touches GPU libs.
        # Knobs follow the sliders (a hardcoded nprobe=8 here once
        # measured one pred five times — M5c lesson, never again).
        from ann_core import IvfIndex
        from exa_home.embed import Embedder
        from exa_home.orch import build_search_dag, run_search
        from exa_home.rerank import Reranker
        from exa_home.store import ContentStore

        dag_pred = build_search_dag(Embedder(), IvfIndex.load(R("index")),
                                    Reranker(max_pair_tokens=64, batch_size=32,
                                             max_query_tokens=32),
                                    ContentStore(R("index", "store")),
                                    top_coarse=top_coarse.value,
                                    nprobe=nprobe.value)
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
        # Print first: stdout always renders even if the figure below fails.
        print(f"recall@10 = {recall:.4f} over {len(recs)} queries "
              f"({recall / 0.027:.1f}x the coarse skew bound)")
        fig_rec = go.Figure()
        fig_rec.add_bar(x=["reranked top-10"], y=[recall], name="recall@10",
                        marker_color="#00CC96")
        fig_rec.add_hline(y=0.027, line_dash="dash", line_color="red",
                          annotation_text="coarse-only skew bound 0.027")
        fig_rec.update_layout(template="plotly_white",
                              title=f"Recall@10 = {recall:.4f} over {len(recs)} queries",
                              yaxis_title="recall@10",
                              yaxis_range=[0, max(0.5, recall * 1.2)])
        mo.vstack([mo.md(f"## Recall@10 = {recall:.4f}"), fig_rec])
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
