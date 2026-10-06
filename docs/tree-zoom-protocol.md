# LoCoMo tree + zoom protocol (pre-registered)

Status: frozen before any model call on 2026-10-06.

## Question

When an agent sees its conversation memory only as a recency-weighted binary
tree of one-line summaries (the OptChat / OptMem "view"), does a `zoom` tool
that re-opens summaries down to the verbatim messages recover long-range
recall? Does it beat plain lexical retrieval at an equal context budget?

## Corpus and sample

- Dataset: LoCoMo `locomo10.json` from `snap-research/locomo` (main),
  sha256 `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
- Each conversation is its own memory (11k-22k tokens).
- Category mapping (upstream evaluation code): 1 multi-hop, 2 temporal,
  3 open-domain, 4 single-hop, 5 adversarial.
- Long-range question: every evidence turn lies in the first half of the
  conversation's sessions.
- Sample (seed 20261006): 40 long-range questions each for categories 4, 1
  and 2 (4 per conversation when available, topped up from other
  conversations), plus 20 adversarial (category 5) questions, 2 per
  conversation. 140 questions in total.

## Memory representation

- Leaves: verbatim turns with dialogue id, session date and speaker. Shared
  images are rendered with their BLIP caption.
- Internal nodes: balanced binary tree over the turn sequence. Each internal
  node gets a one-line summary (at most 30 words) written by the summarizer
  from its two children (raw turns at the first level, child summaries above).
  Each displayed line also shows the node id, date range and turn count,
  computed deterministically.
- View budget: 10% of the conversation's estimated tokens (chars / 4), so the
  compression ratio is 10:1 regardless of the conversation length. The view
  is filled greedily from the root by expanding the frontier node with the
  largest `span / (turns_after_node + 1)`, which yields a log-structured,
  recency-weighted view (recent turns verbatim, old turns folded).

## Strategies (same reader, same answer instructions)

| ID | Strategy | Context |
|---|---|---|
| A | Full conversation | All turns verbatim (ceiling) |
| B | Tree only | The view, no tool |
| C | Tree + zoom | The view plus `zoom(node)`, which reveals up to 16 descendants (verbatim turns once reached) |
| D | BM25 at equal budget | Four-turn windows (stride 2) ranked by BM25, filled to the median context tokens consumed by C |
| E | Tree + zoom + search | C plus `search(query)`, which returns the top 5 BM25 turns |

Agent caps for C and E: at most 8 agent steps; each step may open up to 3
nodes or run 1 search; at most 12 zoom/search operations in total. When a cap
is reached the agent must answer.

## Models

- Summarizer: `claude-sonnet-5-5`.
- Reader / agent: `claude-sonnet-5-5`.
- Judge: `claude-sonnet-5-5`, blind to the strategy, with the gold answer.
  Lenient on format (dates, paraphrase), strict on facts. For category 5 a
  response is correct only if it says the information is not in the
  conversation.
- All calls go through the Claude Code CLI (`claude -p`) with a replaced
  system prompt, no tools and no MCP servers. Every call is cached on disk by
  content hash so the run is resumable.

## Metrics

- Judged answer accuracy, overall and per category; accuracy on the 120
  answerable long-range questions is the primary metric.
- Correct-abstention rate on category 5.
- Evidence reach: fraction of answerable questions where at least one
  annotated evidence turn was shown verbatim to the reader (A trivially 1).
- Zoom/search operations, agent steps, reader context tokens, wall time.
- Paired comparisons use exact McNemar tests on the same questions.

## Hypotheses

- H1: B is clearly below A on long-range answerable questions.
- H2: C recovers at least half of the A - B gap.
- H3: At an equal context budget, C does not beat D.

If H3 is rejected this is reported as evidence for the tree + zoom design.

## Known limitations

- LoCoMo gold answers contain some annotation errors; all strategies share
  them.
- Single run, single reader model, 140 questions: differences under about
  10 points are not conclusive.
- LoCoMo conversations are short; the 10:1 ratio emulates, but does not
  reproduce, a multi-million-token memory.

## Amendments (after the first full run, before looking at final results)

1. Agent actions are also accepted in the reader's native tool-call markup
   (`<invoke name="zoom|search|answer">`). In the first full run the reader
   used that markup in 125 agent steps, which the JSON-only parser rejected
   as format errors, costing C and E steps on about 45 questions each.
   Prompts, caps, sample and scoring are unchanged; the first run is kept as
   `results/locomo-tree-zoom-20261006-v1.json` and both are reported.
2. CLI calls time out after 180 s and are retried (one call stalled for
   387 s in the smoke test). Latency is reported as medians.
