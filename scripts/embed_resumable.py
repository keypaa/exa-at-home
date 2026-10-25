"""Resumable, chunked embedder driver (Item 2).

This module owns stage_embed's new run/resume lifecycle so the runner can
survive SIGKILL between chunks. It deliberately mirrors the constraints of
the old stage_embed (streaming 8192-doc chunks, memmap output) and adds:

  * a `.state.json` written atomically after every complete chunk
  * a `.part` memmap that grows in place
  * ids as a `.part.ids` JSONL (one id per line) assembled into the final
    `ids.json` JSON array only on completion
  * a `.done` marker + `.provenance.json` so a completed build is reused only
    when source sha and count match
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time


def _count_lines(path: str) -> int:
    n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_replace_json(dst: str, obj) -> None:
    tmp = dst + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)


def _cleanup(parts) -> None:
    for p in parts:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass


def run_embed(
    docs_path: str,
    vecs_path: str,
    ids_path: str,
    batch: int | None = None,
    chunk: int | None = None,
    force: bool = False,
    verbose: bool = False,
    show_progress: bool = False,
    resume: bool = True,
    progress_every: int = 100_000,
) -> tuple:
    import numpy as np

    from exa_home.config import EMBED_BATCH, EMBED_CHUNK
    from exa_home.embed import Embedder
    from scripts.jsonl import read_jsonl

    if batch is None:
        batch = EMBED_BATCH
    if chunk is None:
        chunk = EMBED_CHUNK

    t0 = time.perf_counter()

    n_docs = _count_lines(docs_path)
    if n_docs == 0:
        raise ValueError(f"{docs_path}: no documents to embed")
    src_sha = _sha256_file(docs_path)

    done_marker = vecs_path + ".done"
    prov_path = vecs_path + ".provenance.json"
    vecs_part = vecs_path + ".part"
    ids_part = ids_path + ".part.ids"
    state_path = vecs_path + ".state.json"

    # --- already complete? -------------------------------------------
    if not force and os.path.exists(vecs_path) and os.path.exists(ids_path) and os.path.exists(done_marker) and os.path.exists(prov_path):
        try:
            with open(prov_path, encoding="utf-8") as _pf:
                prov = json.load(_pf)
            if prov.get("n_docs") == n_docs and prov.get("source_sha256") == src_sha:
                import numpy as _np

                v = _np.load(vecs_path, mmap_mode="r")
                if tuple(v.shape) == (n_docs, 256):
                    if verbose:
                        print(f"[embed] reusing {vecs_path} {v.shape}", flush=True)
                    return v.shape, str(v.dtype)
        except (OSError, ValueError, KeyError, TypeError):
            pass  # fall through to partial/rebuild

    # --- force wipe ---------------------------------------------------
    if force:
        _cleanup([
            vecs_path, ids_path, vecs_part, ids_part, state_path, done_marker,
            prov_path, ids_path + ".tmp", vecs_path + ".tmp.npy",
        ])

    # --- resume state handling ----------------------------------------
    state = None
    if os.path.exists(state_path):
        try:
            with open(state_path, encoding="utf-8") as _sf:
                state = json.load(_sf)
        except (OSError, ValueError):
            raise ValueError(f"{state_path}: corrupted resume state; run with --force")
        if (
            not isinstance(state, dict)
            or state.get("n_docs") != n_docs
            or state.get("source_sha256") != src_sha
        ):
            print("[embed] stale state: restarting", file=sys.stderr, flush=True)
            _cleanup([state_path, vecs_part, ids_part])
            state = None
        else:
            rows = int(state.get("rows", 0))
            if rows % chunk != 0 or rows > n_docs:
                raise ValueError(f"{state_path}: rows={rows} not on chunk boundary; run --force")
            # ids progress must match rows
            if os.path.exists(ids_part):
                with open(ids_part, encoding="utf-8") as _if:
                    n_ids = sum(1 for _ in _if)
                if n_ids != rows:
                    raise ValueError(f"{ids_part}: {n_ids} ids != {rows} rows; run --force")
            # memmap shape sanity
            if os.path.exists(vecs_part):
                mm_test = np.load(vecs_part, mmap_mode="r")
                if mm_test.shape != (n_docs, 256):
                    raise ValueError(f"{vecs_part}: shape {mm_test.shape} != ({n_docs}, 256); run --force")
            # fall through to resume with start_rows=rows

    rows_start = int(state.get("rows", 0)) if state else 0
    is_fresh = state is None

    if is_fresh:
        # make sure no stale part
        _cleanup([vecs_part, ids_part])
    else:
        if not os.path.exists(vecs_part):
            raise ValueError(f"{vecs_part}: missing while state says rows={rows_start}; run --force")

    emb = Embedder()

    mm = None
    rows_done = rows_start
    ids_fp = open(ids_part, "a", encoding="utf-8")  # noqa: SIM115

    # memmap handle (r+ if resuming, w+ lazily on first chunk otherwise)
    if not is_fresh:
        mm = np.lib.format.open_memmap(vecs_part, mode="r+")
        if mm.shape != (n_docs, 256):
            raise ValueError(f"{vecs_part}: bad shape {mm.shape}; run --force")

    def _commit_state(r: int) -> None:
        _atomic_replace_json(state_path, {
            "n_docs": n_docs, "chunk": chunk, "batch": batch,
            "source_sha256": src_sha, "rows": r,
        })

    # skip already committed rows
    read_iter = read_jsonl(docs_path)
    for _ in range(rows_start):
        try:
            next(read_iter)
        except StopIteration:
            break

    buf_texts: list[str] = []
    buf_ids: list[str] = []

    def _flush(buf_t, buf_i) -> None:
        nonlocal mm, rows_done
        import numpy as _np

        vecs = emb.encode_docs(buf_t, batch_size=batch)
        if mm is None:
            mm = _np.lib.format.open_memmap(
                vecs_part, mode="w+", dtype=vecs.dtype, shape=(n_docs, vecs.shape[1]),
            )
        mm[rows_done: rows_done + len(vecs)] = vecs
        mm.flush()
        # ids
        ids_fp.writelines(json.dumps(doc_id) + "\n" for doc_id in buf_i)
        ids_fp.flush()
        os.fsync(ids_fp.fileno())
        rows_done += len(vecs)
        if show_progress or verbose:
            print(f"[embed] {rows_done}/{n_docs} docs", flush=True)
        _commit_state(rows_done)

    try:
        for d in read_iter:
            buf_texts.append(d["text"][:12000])
            buf_ids.append(d["id"])
            if len(buf_texts) < chunk:
                continue
            _flush(buf_texts, buf_ids)
            buf_texts, buf_ids = [], []
        if buf_texts:
            _flush(buf_texts, buf_ids)
    except Exception:
        # leave state at last committed chunk; partials stay for resume
        try:
            ids_fp.close()
        except Exception:  # noqa: BLE001, S110
            pass
        raise
    finally:
        try:
            ids_fp.close()
        except Exception:  # noqa: BLE001, S110
            pass

    if rows_done != n_docs:
        raise ValueError(f"{docs_path}: embedded {rows_done} docs, expected {n_docs}")

    # finalized dimensions
    dim = int(mm.shape[1])
    dtype = mm.dtype
    mm.flush()
    del mm

    # assemble ids.json from the JSONL part (streaming)
    ids_tmp = ids_path + ".tmp"
    with open(ids_part, encoding="utf-8") as src, open(ids_tmp, "w", encoding="utf-8") as dst:
        dst.write("[")
        first = True
        for line in src:
            line = line.strip()
            if not line:
                continue
            if not first:
                dst.write(",")
            dst.write(line)
            first = False
        dst.write("]")
        dst.flush()
        os.fsync(dst.fileno())
    os.replace(vecs_part, vecs_path)
    os.replace(ids_tmp, ids_path)

    # provenance + done marker
    _atomic_replace_json(prov_path, {"n_docs": n_docs, "source_sha256": src_sha})
    try:
        os.remove(state_path)
    except FileNotFoundError:
        pass
    with open(done_marker, "w", encoding="utf-8") as f:
        f.write("done\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.remove(ids_part)
    except FileNotFoundError:
        pass

    shape = (n_docs, dim)
    _log_embed(t0, shape, dtype, verbose)
    return shape, str(dtype)


def _log_embed(t0, shape, dtype, verbose):
    import time as _t

    dt = (_t.perf_counter() - t0) * 1000
    # minimal local log to mirror run_pipeline._log shape
    print(f"[embed] {dt:.0f} ms ({shape} {dtype})", flush=True)
