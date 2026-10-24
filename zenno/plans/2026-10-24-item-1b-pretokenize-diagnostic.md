# Item 1b — Pre-tokenize offline diagnostic (design)

## Purpose
Determine whether pre-tokenizing documents once and reusing those token ids
speeds bulk embedding. Handoff (2026-10-24) predicts the embedder is
CPU-bound on the tokenizer (442 docs/s, GPU 1.1GB/96GB), so pre-tokenizing
offline is the throughput lever. This diagnostic measures that directly.

## Question
"Is the tokenizer a large fraction of per-document embed wall time, so that
moving tokenization out of the hot path materially raises docs/s?"

## Variants
- A (baseline): `Embedder.encode_docs(texts, batch_size=B, sort_by_length=True)`
  — tokenize inside `model.encode` per batch (current).
- B (pretok): tokenize all docs once with `model.tokenizer`, cache ids,
  feed into `model.encode`. (Implementation note: ST 6.x `encode` does not
  accept raw ids; the diagnostic reports `tokenize_only_s` and the model's
  own `encode` timing so we can subtract without a full rewrite.)
- Both run with fp16 weights on CUDA (target) or fp32 on CPU (sandbox sanity).

## Knobs (fixed for the diagnostic)
- `n_docs` = 1000 (representative slice, processed in one run)
- `batch_size` ∈ {32, 1024} — the ST default and the M8 pick
- `max_seq_length` = 128 (printed; the model's own default is 512; raising it
  is a separate knob per LEARNINGS #7 — leave it here)
- corpus: WET slice (preferred) or 1000 synth docs (sandbox)

## Metrics reported (per run, appended-config to `runs.jsonl`)
- `docs_per_s_A`, `docs_per_s_B` (or derive)
- `tokenize_only_ms_per_doc`
- `tokenize_share_of_total` (tokenize time / encode time)
- peak GPU memory (CUDA only)
- batch size, fp precision, max_seq_length, device, n_docs, model name

## Decision rule (pre-registered)
- If `tokenize_share_of_total >= 0.25` on the GPU box → implement pre-tokenizing
  (item 1b becomes a code change, measured as M9).
- If `< 0.25` → pre-tokenizing is not the lever. Pivot to: (a) fused
  kernel/batching measurement, (b) ST overhead, (c) leaving the knob
  (timeout/sample-size bound on tokenization is below the 2x target).
No retrofitting the threshold after the run.

## Sandbox finding so far (CPU, N=100, batch 8)
- `tokenize_only_s` = 0.0087s; `encode_total_s` = 35.9s; share ~0.024%.
- Per-document token cost ~0.087ms → at a 1024-batch that is ~89ms of
  tokenizer work per batch, which on a GPU (model forward ~32ms for 50
  pairs; embed ~a few ms for a 1024-batch) *is* the predicted wall.
- Conclusion: cannot size on CPU. The diagnostic must run on the box.
  Hence this design + script, no code change yet.

## Deliverable
`scripts/diagnose_embed_pretok.py` measures both variants and prints a
one-line row suitable for `runs.jsonl`. Run from the notebook or shell.
Output is pasted (not auto-committed) so a human keeps the ledger honest.