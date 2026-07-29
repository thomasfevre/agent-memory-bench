# Repository instructions

- Keep raw source data immutable and reconstruct derived indexes from it.
- Separate ingestion, retrieval, context construction, reading and judging.
- Do not publish private data, authenticated session content, provider secrets or downloaded dataset payloads.
- Every public metric must name its dataset, task, budget, repetition count, reader and evidence file.
- A top-k comparison is invalid when the retrieved units have materially different token or word budgets.
- Keep long-context and BM25 baselines.
- Do not call a proxy metric an end-to-end score.
- Add or update tests with every new adapter or public result schema.
- Update `results/published/registry.json`, run `python tools/build_dashboard_data.py`, then run the full test suite.
- Never publish or schedule social content from this repository.

