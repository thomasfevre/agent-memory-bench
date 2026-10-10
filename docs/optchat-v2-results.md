# OptChat v2 on LoCoMo (Haiku 5.5): results

Run: `optchat-v2-haiku-locomo-20261010` (registry). Protocol and amendments:
[optchat-v2-protocol.md](optchat-v2-protocol.md). Raw evidence stays local
(`results/optchat-v2-haiku-locomo.json`, contains dataset text).

**Scope.** This file reports the LoCoMo half of the pre-registered run.
The LongMemEval-S half was stopped by the owner during tree construction
(about 32 of 100 trees, no answers) and is not reported. Following
amendment 3, the compactor, the reader and the first judge are all
`claude-haiku-5-5`. The second judge is `claude-opus-5-5` on 200 answers.

Memory: the OptChat specification (lines of at most 512 bytes, a
compactor that sees the view, purely binary merges, a byte-budgeted view
folded incrementally, `zoom(id, n)` and `date(id)`). The view is 10% of
each conversation's bytes: 19 to 30 lines for 369 to 689 messages.
Building the trees took 422 compactor calls (185 size retries, about
3 USD at list prices); 39 of 11,711 nodes end slightly over 512 bytes.

## Headline (276 long-range answerable questions, 24 adversarial)

| Strategy | Accuracy | 95% CI | Single-hop | Multi-hop | Temporal | Adversarial | Evidence read in full | Median input tokens | Median latency |
|---|---|---|---|---|---|---|---|---|---|
| A Full conversation | **0.725** | 0.67-0.78 | 0.85 | 0.63 | 0.68 | 0.79 | 1.00 | 42.0k | 6.0 s |
| B View only | 0.109 | 0.07-0.15 | 0.14 | 0.20 | 0.00 | 0.92 | 0.00 | 5.4k | 6.0 s |
| C View + zoom | 0.341 | 0.29-0.40 | 0.38 | 0.29 | 0.35 | 1.00 | 0.01 | 16.7k | 19.6 s |
| C-verify | 0.384 | 0.33-0.44 | 0.46 | 0.27 | 0.41 | 0.79 | 0.45 | 41.7k | 45.6 s |
| D BM25 at C budget (11.3 KB) | **0.587** | 0.53-0.64 | 0.82 | 0.37 | 0.54 | 0.83 | 0.84 | 5.6k | 9.3 s |
| E View + zoom + search | 0.540 | 0.48-0.60 | 0.69 | 0.32 | 0.58 | 0.88 | 0.60 | 12.0k | 13.2 s |
| F Search agent, no tree | 0.507 | 0.45-0.57 | 0.64 | 0.30 | 0.56 | 0.63 | 0.74 | 3.8k | 13.7 s |

C-verify answers carried a valid verbatim quote 96.7% of the time.

## Pre-registered comparisons (exact McNemar)

Holm correction here covers the 4 LoCoMo tests only, because the
LongMemEval half has no answers.

| Pair | Difference | 95% CI | Left only | Right only | p | p (Holm) |
|---|---|---|---|---|---|---|
| C-verify vs C | +0.043 | -0.007 to +0.094 | 31 | 19 | 0.119 | 0.238 |
| E vs F | +0.033 | -0.022 to +0.087 | 33 | 24 | 0.289 | 0.289 |
| C vs D | -0.246 | -0.315 to -0.178 | 21 | 89 | < 0.0001 | < 0.0001 |
| C vs A | -0.384 | -0.449 to -0.319 | 11 | 117 | < 0.0001 | < 0.0001 |

Secondary: B vs A -0.62, E vs C +0.20 (p < 0.0001), F vs D -0.08
(p = 0.014).

## Hypotheses (LoCoMo only)

- **H1 (C-verify beats C): not confirmed.** The difference is +4 points
  and not significant, even though forced quoting raises full reads of
  evidence from 1% to 45%.
- **H2 (E does not beat F): consistent.** The difference is +3 points and
  not significant. This is an absence of evidence, not proof that the
  tree adds nothing.
- **H3 (C does not beat D): confirmed, and more strongly than expected.**
  C is 25 points below D.
- **H4 (LongMemEval vs LoCoMo): not tested.**

## Why tree + zoom fails here

- Reaching a message takes 5 to 7 zooms from the view. Haiku stops early:
  most of its zooms open lines that cover 16 to 128 messages, and it read
  an evidence message in full on 0.7% of questions.
- C answered "Not mentioned" on 85 of 276 answerable questions. This
  explains its perfect adversarial abstention: it abstains whatever the
  question.
- View lines carry no dates (spec), so B scores 0 on temporal questions.
  The agent can call `date(id)`, but only after it has found the message.
- Search does most of the work. Adding it to the tree lifts C by 20
  points, to the level of a search agent without a tree.

## Judge

The Opus second judge agreed with the Haiku judge on 90.5% of 200 answers
(Cohen's kappa 0.81).

## Cost

The whole LoCoMo run used 7,557 CLI calls and 48.9M input tokens, about
19 USD at list API prices (run on a subscription).

## Limits

- One dataset with short memories (15k-30k tokens), where full context is
  both the best option and an affordable one.
- One small model in every role. With Sonnet and a simpler tree, v1 found
  tree + zoom tied with BM25 (72% vs 73%), so most of the gap here may
  come from the model. This result says a cheap model cannot navigate the
  tree; it does not say the design fails with strong models.
- Single run. Compactor calls are batched (declared deviation from the
  spec).
- The Holm correction covers 4 tests instead of the 8 pre-registered.

## Implication

With a small model, a spec-faithful OptChat memory is clearly worse than
plain BM25 at the same budget, and the zoom tool does not reach the
evidence. To keep the design, the reader needs either a stronger model or
help reaching messages (search, which closes most of the gap). The
question OptChat is built for, memories far larger than the context, still
needs the LongMemEval-S half.
