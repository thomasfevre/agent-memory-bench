# Roadmap

## Priority 1: comparable and publishable

- [x] Immutable synthetic corpus and expected facts.
- [x] Long-context, BM25, dense, hybrid and temporal baselines.
- [x] Boundary-level metrics for ingestion, retrieval, context and generation.
- [x] Repeated topology and product-adapter runs.
- [x] Compact public result registry with raw-artifact hashes.
- [x] Static dashboard generated from the registry.
- [x] CI validation and GitHub Pages deployment workflow.
- [ ] Interleave configuration order in every model-backed campaign.

## Priority 2: stronger evidence

- [ ] Extend LongMemEval generation from the diagnostic pairs to a calibrated
  full protocol.
- [ ] Extend MemoryAgentBench generation beyond the two complete 6k variants.
- [ ] Complete MemGym-DR with a reader and calibrated judge.
- [ ] Finish at least two real GraphRAG engines under one aligned budget.
- [ ] Add real human review measurements for Context Shards.
- [ ] Calibrate semantic judging with blinded double annotation.

## Priority 3: realistic and adversarial

- [ ] Long-running tasks on a realistic public repository.
- [ ] Same tasks, tools and model across jcode, Letta Code, Codex and Claude Code.
- [ ] Late, out-of-order, expired and retracted memories under model extraction.
- [ ] Persistent derived-index recovery after crash.
- [ ] Verifiable deletion, compaction and retention policies.
- [ ] Cost per correct answer and quality under fixed time and token budgets.

Every item needs a preregistered protocol, raw evidence, a compact public result
and an explicit statement of what remains unproven.

