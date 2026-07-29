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
- Status: the configuration-isolated reader and judge smoke passed with three
  readers and three judges. The full campaign is running from a fresh artifact.

The semantic judge cannot be called calibrated against humans until two blinded
annotations exist. The annotation pack and agreement report are a separate
deliverable and must not be replaced by model self-agreement.

Two earlier artifacts were quarantined after raw stderr showed web search. They
are excluded from every summary and public result. The first established that
prompt-only restrictions are insufficient. The second showed that disabling
feature flags alone is also insufficient in the tested Codex CLI version,
which is why the runner now applies root tool configuration and a fail-closed
trace gate.

## Human review gates

Two Priority 2 claims require real human observations:

1. time and decision quality for Context Shard approval;
2. semantic-judge agreement against two blinded annotators.

Code may prepare randomized review packs and compute agreement, but the
measurements remain pending until two humans submit independent labels.

The `human_calibration.py` CLI now prepares two independent CSV templates,
keeps source keys and reference labels in a separate mapping, and computes
coverage, Cohen's kappa, reference agreement, median review time and
model-to-human judge error. The current Context Shard fixture has only three
items, while the dedicated review pack has 24 balanced candidates with
synthetic reference statuses. The latter is large enough to exercise the
workflow, but no human-quality estimate exists until two independent reviewers
complete it.
