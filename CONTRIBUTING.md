# Contributing

This project compares complete memory configurations, not product names in the
abstract. A contribution should be reproducible on public or synthetic data and
must preserve the boundary between:

1. ingestion and extraction;
2. representation and storage;
3. retrieval and context construction;
4. reading or agent execution;
5. judging and statistical analysis.

## Add a result

1. Add or extend a runner under `src/`.
2. Add deterministic tests under `tests/`.
3. Keep downloaded datasets and raw provider traces outside Git.
4. Add a compact record to `results/published/registry.json`.
5. Include hashes and relative filenames for every local raw artifact.
6. Run:

```bash
python tools/build_dashboard_data.py
python tools/validate_public_registry.py
python -m pytest -q
```

## Evidence levels

- `controlled`: repeated under an aligned local protocol.
- `official-data`: uses an official public dataset, but may evaluate only one layer.
- `smoke`: proves that a path executes, not that it is good.
- `timeout`: records an operational failure within a declared budget.
- `not-reproduced`: researched but not executed.

Do not promote a result to a stronger level without adding the missing
evidence. In particular, retrieval recall is not answer accuracy, a successful
installation is not memory quality, and a single run is not stability.

