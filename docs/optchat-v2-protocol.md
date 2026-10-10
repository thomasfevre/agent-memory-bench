# OptChat v2 protocol (pre-registered)

Status: frozen before any model call of the v2 run on 2026-10-06. Changes
after this point go to the Amendments section, with the reason, and both
versions are reported.

v1 ([tree-zoom-protocol.md](tree-zoom-protocol.md),
[tree-zoom-results.md](tree-zoom-results.md)) re-implemented the OptChat
idea from a description. v2 follows the published OptChat specification
(gist `VictorTaelin/91837951a5ce5b38f341ec1ba1df6449`, read on 2026-10-06),
moves to a memory that is 4-5 times larger (LongMemEval-S), adds two
strategies that isolate the questions v1 left open, and fixes the analysis
plan before any call.

## Questions

1. With a faithful OptChat memory (byte-budgeted binary tree, incremental
   view, binary `zoom(id, n)`), how close does the agent get to full
   context, and does it beat lexical retrieval at the same budget?
2. Does forcing verification (quote a message read in full before
   answering) fix the under-verification failure seen in v1 (21 of C's 34
   errors)? (OQ-002)
3. Does the tree add anything over a plain search agent with no tree?

## Corpora and samples (seed 20261006)

- **LongMemEval-S** `longmemeval_s_cleaned.json` (Hugging Face
  `xiaowu0162/longmemeval-cleaned`), sha256
  `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442`.
  Each question has its own haystack (median 491 messages, 494 KB,
  about 124k tokens), so one tree per question. Sample: 15 questions for
  each of the 6 question types among non-abstention questions, plus 10
  abstention (`_abs`) questions, 100 in total. Message kinds: `user` for the
  user, `talk` for the assistant. The question is asked with its
  `question_date`.
- **LoCoMo** `locomo10.json` (`snap-research/locomo`), sha256
  `79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4`.
  One tree per conversation. Long-range answerable questions (every
  evidence turn in the first half of the sessions, as in v1): all 84
  multi-hop, 96 temporal, 96 single-hop; plus 24 adversarial. 300 in
  total. Each turn is one message of kind `note` whose text starts with the
  speaker's name (the spec's kind for imported history; image captions
  appended as in v1).

## Memory (OptChat spec, sections 3-7)

- **Tree.** Node `(l, i)` covers messages `[i*2^l, (i+1)*2^l)`. Level 0
  compresses one message; level `l > 0` merges its two children. Purely
  binary. Free nodes: a message whose `kind: text` fits in `NODE` = 512
  bytes is its own level-0 line verbatim; two children whose lines joined
  by a newline fit in 512 bytes are their own parent. Every full node is
  built (zoom needs them all).
- **Compactor.** The `COMPACT` system prompt verbatim (agent name kept as
  OptChat). User message: the `<chat>` context (the current view lines up
  to the node, bare text, no ids), the 512-byte `SCALE` line, then the
  step(s) with the message whole or the two lines to merge written out.
  No ids anywhere. Size enforcement: lines over 512 bytes are sent back
  with the spec's cut-at-limit feedback, up to `TRIES` = 5, keeping the
  shortest try. Summarizer model `claude-sonnet-5-5`.
- **Batching (the one deviation, for cost).** The spec makes one call per
  node. Here one call does up to 32 independent steps (same level, or
  consecutive level-0 messages), returned as a JSON array of lines in
  order. Construction goes chronologically in windows of 256 messages:
  inside a window, level 0 first, then each merge level in turn; nodes
  that span windows are built as soon as their children exist. The
  `<chat>` context of a batch is the view folded over everything built so
  far up to the batch's last message (it can be temporarily over budget
  inside a window, as in the spec when parents are not built yet). A
  level-0 step therefore sees the earlier messages of its own batch whole,
  and a merge may see a few lines after its own end within the batch.
  Expected cost: about 35 calls per LongMemEval question instead of about
  670.
- **View.** Built with the spec's fold: append one level-0 part per
  message, and while the size (sum of line bytes) exceeds the budget,
  replace the adjacent same-level sibling pair with the largest
  `due = (T - start) / 2^(l+2)` by its parent. Never split. Rendered as
  `id+n|text` lines inside `<chat>`, newlines as spaces, no dates. The view
  never holds a whole message. Budget: 10% of the history's bytes (v1
  ratio; about 49 KB on LongMemEval-S, about 10 KB on LoCoMo).
- **Agent tools.** `zoom(id, n)`: `n` a power of 2, `id % n == 0`; `n = 1`
  returns the message whole (`id+0|kind: text`), otherwise the two child
  lines. `date(id)`: the date of message `id`. Tool results are capped at
  30,000 characters (head and tail kept).
- **Agent prompt.** The spec's `VIEW_DOC` verbatim, preceded by a short
  benchmark adaptation of `MASTER` (the agent answers one question about
  the chat; it keeps no memory between turns) and followed by the answer
  rules and the action format. Actions are JSON objects (or the reader's
  native `<invoke>` markup, v1 amendment 1).

## Strategies (reader `claude-sonnet-5-5`, same answer rules)

| ID | Strategy | Context |
|---|---|---|
| A | Full history | Every message verbatim with its date (ceiling and reference cost) |
| B | View only | The view, no tool |
| C | View + zoom | The view, `zoom`, `date` |
| C-verify | View + zoom, verified | C, but an answer must quote a message the agent read in full (`zoom(id, 1)` or search hit) and name its id; the harness rejects quotes not found verbatim (whitespace and case normalized). Abstaining requires having read at least one message in full. On the last step any answer is accepted and marked unverified |
| D | BM25 at C's budget | Windows of 4 messages (stride 2) ranked by BM25 on the question, filled to the median context bytes of C on that dataset, shown in order with dates |
| E | View + zoom + search | C plus `search(query)`: top 5 messages by BM25, whole, with id and date |
| F | Search agent, no tree | Only `search` (same as E) and the answer action; the prompt says how many messages the history holds and its date range |

Agent caps (C, C-verify, E, F): at most 10 steps; each step makes up to 4
operations (zooms, dates, or 1 search); at most 30 operations. When a cap
is reached the agent must answer.

## Judging

- LongMemEval: the official `evaluate_qa.py` prompts per question type
  (abstention prompt for `_abs`), judge `claude-sonnet-5-5`, blind to the
  strategy.
- LoCoMo: the v1 judge prompt, judge `claude-sonnet-5-5`, blind.
- Second judge: `claude-opus-5-5` on 200 answers drawn at random (seed)
  from all strategies and both datasets; agreement and Cohen's kappa are
  reported.
- Human check: 30 answers drawn at random (seed) are exported for review
  by the owner with the repo's local review tool; the result is reported
  when available and does not block the run.

## Metrics and analysis

- Primary: judged accuracy per strategy, on LongMemEval-S (100) and on
  LoCoMo answerable (276); LoCoMo adversarial abstention reported
  separately (24).
- 95% confidence intervals by paired bootstrap over questions (10,000
  resamples, seed), for each accuracy and for each paired difference.
- Pre-specified comparisons, per dataset, exact McNemar with Holm
  correction over these 8 tests (4 per dataset):
  1. C-verify vs C (does forced verification help?)
  2. E vs F (does the tree add to a search agent?)
  3. C vs D (tree + zoom vs retrieval at equal budget)
  4. C vs A (gap to full context)
- Secondary (descriptive, no correction): B vs A, E vs C, F vs D, per
  question type / category, verified-answer rate of C-verify, evidence
  reach (LoCoMo evidence turns, LongMemEval `has_answer` turns, shown in
  full), operations, steps, input tokens, latency, and build cost of the
  trees.

## Hypotheses

- H1: C-verify beats C on at least one dataset (corrected p < 0.05).
- H2: E does not beat F on either dataset (the tree adds little to search).
- H3: C does not beat D at equal budget on either dataset.
- H4: On LongMemEval-S, C is closer to A than on LoCoMo (A degrades on long
  histories).

Any rejected hypothesis is reported as such.

## Robustness and limits declared in advance

- One run per strategy; v1 showed about 2-3 points of run-to-run noise.
- Batched compactor calls (above) and the absence of medium effort control
  in the CLI are deviations from the spec.
- LongMemEval haystacks are mostly unrelated chats; a real OptChat history
  is one continuous project, where the compactor's context matters more.
- Raw results contain dataset text and stay local; only aggregates and
  hashes are published.

## Amendments

1. (After the smoke test, before the main run.) The action parser takes the
   first *usable* action: a bare `<invoke name="zoom">` with no parameters
   followed by the JSON action no longer counts as an empty zoom. Prompts,
   caps and scoring are unchanged.
2. (Operational, 2026-10-07.) The run stops at once, instead of waiting,
   on any usage-limit error, on overage or fallback credit, or when any
   rate-limit window reaches 98.5% utilization. LoCoMo runs first (cheap
   trees), then the full run. Methodology unchanged.
3. (2026-10-10, owner's request, before any answer was scored.) The
   compactor, the reader and the first judge run on `claude-haiku-5-5`
   instead of `claude-sonnet-5-5`, to fit the subscription quota. The
   second judge stays `claude-opus-5-5` on 200 answers, which checks the
   Haiku judge. Trees are rebuilt with Haiku (one model for the whole
   memory). The partial Sonnet run of 2026-10-07 (LoCoMo trees and about
   60 answers, stopped by the quota guard) is not reported. Everything
   else is unchanged.
4. (Operational, 2026-10-10.) Reaching 98.5% of the five-hour window now
   pauses the run until that window resets instead of stopping it; the
   seven-day window, overage or credit still stop it.

## Status (2026-10-07)

Paused by the owner before the main run. Smoke test (1 LongMemEval-S
question, all strategies) ran end to end: tree 473 messages, 44 compactor
calls (19 size retries), 1.24M input tokens, about 7 USD at list API prices;
about 1.6M input tokens per LongMemEval question with the readers. Projected
full targeted run: about 195M input tokens (about 950 USD at list prices),
twice the handoff estimate. Options left open: run as is, LongMemEval 60
questions, or TRIES = 2.
