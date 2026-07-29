# Methodology

## The unit of comparison

The benchmark compares a complete configuration:

```text
source data
  -> ingestion
  -> representation
  -> storage
  -> retrieval
  -> context construction
  -> reader or agent
  -> judge
```

A result is meaningful only when the dataset, unit of retrieval, context budget,
reader, prompt, judge and repetition count are recorded.

## Canonical data and rebuildable views

Raw records are append-only and immutable. Chunks, facts, embeddings, graphs,
snapshots and temporal indexes are derived views. They may be deleted and
rebuilt when an implementation changes.

The repository commits the synthetic corpus under `data/`. Official datasets
must be downloaded from their primary source at run time and are not vendored.
Raw run artifacts stay local by default. The public registry stores compact
metrics, provenance and SHA-256 hashes.

## Required baselines

Every retrieval comparison should retain:

- full or bounded long context;
- BM25;
- a dense retriever;
- a fixed hybrid retriever when embeddings are in scope.

Structured or graph approaches are additions to these baselines, not
replacements.

## Budgets

Use a common word or token budget when retrieved units differ in size. Top-k
alone is not a common budget: five sessions can contain far more context than
five dialogue turns.

Record separately:

- index build time;
- ingestion model calls and tokens;
- retrieval latency;
- context words or tokens;
- reader tokens and latency;
- judge cost.

## Metrics by boundary

### Ingestion

- source retention;
- fact fidelity;
- temporal-window fidelity;
- provenance accuracy;
- duplication and unwanted merging;
- stability across repetitions.

### Retrieval and context

- evidence recall;
- context precision;
- MRR or hit rate;
- temporal correctness;
- correct abstention;
- context words and tokens.

### Reading and agent execution

- exact, normalized and token F1;
- calibrated semantic judgment where required;
- citation validity and support;
- task reward;
- tool calls, messages, tokens and latency.

### Durability

- replay equivalence;
- crash-before-commit rollback;
- crash-after-commit preservation;
- hash-chain and database integrity;
- derived-index reconstruction.

## Statistical discipline

Questions from the same conversation, book or paired near-miss are correlated.
Bootstrap and cross-validation at the group level. Use paired tests such as
McNemar when two configurations answer the same questions. Report uncertainty
instead of ranking systems on tiny, unstable differences.

## Claims policy

The dashboard distinguishes:

- what was measured;
- what the result supports;
- what it does not support;
- what remains open.

No aggregate leaderboard is published because the protocols currently cover
different layers and budgets.

