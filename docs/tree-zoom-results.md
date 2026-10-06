# LoCoMo tree + zoom: results

Run: `locomo-tree-zoom-20261006` (registry). Protocol and amendments:
[tree-zoom-protocol.md](tree-zoom-protocol.md). Raw evidence stays local
(`results/locomo-tree-zoom-20261006*.json`, contains dataset text).

Summarizer, reader and blinded judge: `claude-sonnet-5-5` through the Claude
Code CLI. 120 long-range answerable questions (evidence in the first half of
the sessions) and 20 adversarial questions from the 10 LoCoMo conversations.
View budget: 10% of each conversation (median 2.7k tokens out of 27k).

## Headline

| Strategy | Answerable (120) | Single-hop | Multi-hop | Temporal | Adversarial abstention (20) | Verbatim evidence reached | Median reader context | Median input tokens / question | Median latency |
|---|---|---|---|---|---|---|---|---|---|
| A Full conversation | **0.892** | 0.95 | 0.80 | 0.925 | 0.40 | 1.00 | 27.1k | 43.1k | 3.9 s |
| B Tree view only | 0.333 | 0.375 | 0.475 | 0.15 | 0.75 | 0.00 | 2.7k | 5.7k | 4.1 s |
| C Tree + zoom | 0.717 | 0.85 | 0.60 | 0.70 | 0.50 | 0.13 | 3.4k | 12.9k | 8.7 s |
| D BM25 at C budget | 0.725 | 0.90 | 0.50 | 0.775 | 0.60 | 0.88 | 3.4k | 6.6k | 3.9 s |
| E Tree + zoom + search | 0.725 | 0.825 | 0.625 | 0.725 | 0.50 | 0.53 | 3.0k | 12.5k | 7.9 s |

Paired exact McNemar on the 120 answerable questions:

| Pair | Left only correct | Right only correct | p |
|---|---|---|---|
| A vs B | 70 | 3 | < 0.0001 |
| C vs B | 47 | 1 | < 0.0001 |
| A vs C | 25 | 4 | 0.0001 |
| C vs D | 16 | 17 | 1.0 |
| E vs C | 13 | 12 | 1.0 |
| E vs D | 15 | 15 | 1.0 |

## Hypotheses

- **H1 (B clearly below A): confirmed.** 33% vs 89%. Temporal questions
  collapse to 15% because one-line summaries drop dates and relative times.
- **H2 (C recovers at least half of the A - B gap): confirmed.** C recovers
  69% of the gap (0.717 vs 0.333 and 0.892). A 17-point gap to full context
  remains (p = 0.0001).
- **H3 (C does not beat D at equal budget): confirmed.** 71.7% vs 72.5%,
  16 vs 17 discordant questions. Adding search to the tree (E) does not
  change this.

## How the zoom actually helps

- The agent zooms little: median 1 operation, 25 of 120 answers with no
  zoom at all, only 6 questions with more than 4 operations.
- It almost never reads the evidence verbatim (13% of questions). The gain
  over B comes from opening one node into 16 finer summaries, which is
  usually enough on LoCoMo because Sonnet summaries keep names and objects.
- Under-verification is the dominant failure: 21 of C's 34 wrong answers
  used 0 or 1 zoom, e.g. answering "Not mentioned in the conversation"
  without opening anything, or "prize money" when the turn says "money and a
  trophy". The agent trusts lossy summaries instead of checking.
- Tree + zoom does better than BM25 on multi-hop (0.60 vs 0.50) and worse on
  temporal (0.70 vs 0.775). Neither difference is significant at n = 40.

## Cost

At the same reader context, C and E need about twice the input tokens of D
(re-sending the view at every agent step) and about twice the latency, plus
the one-off tree construction (about 6k summaries). The whole campaign used
2,373 CLI calls (16.3M input tokens, 0.56M output tokens; 82 USD at list
API prices, run on a subscription).

## Robustness

- Before amendment 1 the parser rejected native tool-call markup in 125
  agent steps. The first run gave C 0.700, D 0.700, E 0.717; the amended run
  gives 0.717, 0.725, 0.725. D's change (new BM25 budget, fresh calls) is a
  useful indication of run-to-run noise: about 2-3 points.
- The judge is lenient on format but occasionally inconsistent on
  near-paraphrases; this affects all strategies equally.
- Adversarial abstention rests on 20 questions. The full-context reader is
  the most often fooled (40%), and the tree reader abstains more partly
  because it knows less (B also abstains on 35 answerable questions).

## Limits

- One run, one model family, 140 questions: differences under about 10
  points are not conclusive.
- LoCoMo conversations hold 15k-30k tokens. At this size full context is
  both the best and a cheap option; the 10:1 view emulates the compression
  of a much larger memory but not the navigation depth of a multi-million
  token tree.
- The tree used strong summaries. A weaker summarizer would likely lower B,
  C and E but not A or D.

## Implication

On this evidence a summary tree with zoom is a viable way to fold memory
into a small window, but it is not better than plain lexical retrieval at
the same budget, it costs more per question, and its main weakness is the
agent's reluctance to verify. Retrieval (or full context when it fits)
remains the stronger default; a tree is only worth building if a real
workload shows failures that retrieval misses (for example multi-hop
aggregation), and that would need a larger sample to confirm.
