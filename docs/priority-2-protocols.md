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
- Status: complete, 270/270 configurations attempted, 240 successful reader
  calls and 30 retained capacity failures.

All 30 failures are the two raw 262k long-context variants exceeding Codex's
1,048,576-character input ceiling. BM25 and hybrid remained executable at
262k. On the 15-question slice, single-hop substring accuracy was 86.7% for
both BM25 and hybrid at 32k, 80.0% for both at 64k, and 53.3% versus 66.7% at
262k. Multi-hop accuracy stayed between 0% and 13.3% for every architecture and
scale. These failures are retained as capacity results and are not silently
removed from the provider-success denominator.

The paired single-hop difference between compressed retrieval and long context
was 40 points at 64k (exact McNemar p=0.03125). At 32k the same 33.3-point
difference had p=0.0625. The sample is intentionally small, so conclusions
remain specific to this fixed slice.

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
- Tool policy: browser, apps, shell, computer-use, multi-agent and workspace
  tools are disabled at the Codex command line. Native web search is disabled
  independently through `web_search="disabled"` and
  `tools.web_search=false`. Any remaining web-search or MCP trace fails the
  call closed.
- Status: complete, with 120/120 readers and 120/120 judges successful and no
  forbidden tool trace in the retained artifact.

The provisional semantic-judge macro scores were 0.635 for visible documents,
0.650 for BM25 top-1, 0.743 for top-2 and 0.713 for top-5. Paired by the same
30 questions, top-2 exceeded visible-only by 0.108 on average, with 15 wins,
9 ties and 6 losses. Top-5 was not monotonically better: it exceeded
visible-only by 0.078, but had 13 wins and 12 losses. These results support a
bounded-context trade-off, not a universal top-k optimum.

The semantic judge cannot be called broadly calibrated against humans from
model self-agreement. The current first gate is a blinded single-owner review.
It can measure alignment with the intended owner but not inter-annotator
agreement or population-level preference. A future independent second reviewer
would be a separate extension.

Two earlier artifacts were quarantined after raw stderr showed web search. They
are excluded from every summary and public result. The first established that
prompt-only restrictions are insufficient. The second showed that disabling
feature flags alone is also insufficient in the tested Codex CLI version,
which is why the runner now applies root tool configuration and a fail-closed
trace gate.

## Aligned real GraphRAG engines

- Dataset: a fixed 20-document, 10-question slice from the pinned
  GraphRAG-Benchmark Novel-30752 corpus.
- Questions: five complex-reasoning and five fact-retrieval questions.
- Shared settings: `qwen2.5:14b`, `nomic-embed-text`, top-5, 30-minute
  ingestion ceiling and 60-second query ceiling.
- Cognee: version 1.4.0 with embedded Kuzu and LanceDB.
- Graphiti: version 0.29.3 with FalkorDB Lite.
- Status: complete as a capacity comparison, one repetition.

Cognee did not complete ingestion within 1,800 seconds, so no retrieval score
was generated. A separately identified single-item concurrency ablation also
timed out at 1,800 seconds. Graphiti indexed 13 of 20 documents, timed out on
three, left four unattempted after exhausting the shared budget, and answered all
10 questions from the partial index. That partial index reached 0.083 mean
source recall, 0.033 context precision and 0.000 temporal correctness after
124,735 local model tokens.

The Graphiti quality numbers must not be compared as if both engines had built
complete indexes. The supported conclusion is operational: neither engine
completed the 20-document temporal graph workload inside this local budget,
and Cognee did not reach retrieval at all.

## Human review gates

Two Priority 2 questions require real owner observations:

1. time and decision quality for Context Shard approval;
2. semantic-judge agreement against one blinded owner-reviewer.

Code may prepare randomized review packs and compute agreement, but the
measurements remain pending until the owner submits the labels.

The local review interface reads randomized public packs, keeps source keys and
reference labels in a separate mapping, autosaves scorer-compatible CSV files
and resumes interrupted work. The `score-single` command computes coverage,
reference agreement, median review time and model-to-owner judge error. The
current Context Shard fixture has only three items, while the dedicated review
pack has 24 balanced candidates with synthetic reference statuses. The latter
is large enough to exercise the workflow, but no owner-quality estimate exists
until the review is complete.

To keep the manual burden practical, the current owner campaign derives a
frozen ten-item exploratory subset: five MemGym answers and five Context
Shards. Selection maximizes qualitative coverage across model-score bands,
architectures, reasoning depths, hidden shard decisions, scopes and evidence
counts, then minimizes reading length. This improves diversity per reviewed
item but is not statistically representative. The original 40 plus 24-item
packs remain unchanged and available for a future larger study.

The frozen rubrics, local pack hashes, blindness rules, interface command,
claim boundaries and scoring commands are documented in
[`human-review-runbook.md`](human-review-runbook.md).
