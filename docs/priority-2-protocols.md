# Priority 2 protocols

This document separates completed evidence from campaigns that are still
running or require human input.

## LongMemEval role-aware chunking

- Dataset: pinned LongMemEval-S cleaned artifact.
- Sample: all 29 matched answerable/near-miss pairs available to the paired
  premise protocol.
- Reader: isolated `gpt-5.6-sol` Codex subscription sessions.
- Contexts: hybrid retrieval over all turns versus hybrid retrieval over user
  turns only.
- Budget: 4,000 context words for both conditions.
- Primary metric: answer-versus-abstain decision accuracy.
- Secondary metrics: deterministic answer lower bound, citation validity,
  token F1, latency and tokens.
- Pairing: architecture comparisons keep each answerable/near-miss pair
  together in the bootstrap.
- Status: complete, 116/116 successful calls.

The run does not include a no-context memorization control. Deterministic answer
aliases were preregistered for the original pilot, so answer accuracy on the
larger set remains a lower bound.

## MemoryAgentBench scale sweep

- Dataset: pinned official Conflict Resolution parquet.
- Sample: the same 15 SHA-256-selected question indices used in the repetition
  study.
- Sources: multi-hop and single-hop variants at 32k, 64k and 262k.
- Reader: isolated `gpt-5.6-sol` Codex subscription sessions.
- Architectures: long context, BM25 top-20 and hybrid top-20.
- Primary metric: official substring match.
- Secondary metrics: strict exact match, token F1, provider success, context
  words, latency and tokens.
- Expected calls: 270.
- Status: running.

Input-length failures are retained as capacity results. They are not silently
removed from the provider success denominator.

## MemGym-DR reader and judge

- Dataset: pinned official 3-hop, 4-hop and 5-6-hop JSONL artifacts.
- Sample: a SHA-256-selected ten ids per stratum from the already seeded
  30-per-stratum retrieval sample.
- Reader contexts: currently visible documents, then official BM25 chunking at
  top-1, top-2 and top-5.
- Reader: isolated `gpt-5.6-sol` Codex subscription sessions.
- Judge: the official MemGym semantic prompt through an isolated
  `gpt-5.6-luna` session.
- Deterministic checks: exact match, substring match and token F1 remain
  alongside the judge.
- Status: runner implemented and tested; model campaign waits for the
  MemoryAgentBench scale sweep to finish.

The semantic judge cannot be called calibrated against humans until two blinded
annotations exist. The annotation pack and agreement report are a separate
deliverable and must not be replaced by model self-agreement.

## Human review gates

Two Priority 2 claims require real human observations:

1. time and decision quality for Context Shard approval;
2. semantic-judge agreement against two blinded annotators.

Code may prepare randomized review packs and compute agreement, but the
measurements remain pending until two humans submit independent labels.
