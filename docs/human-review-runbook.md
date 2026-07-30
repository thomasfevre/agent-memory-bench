# Priority 2 owner-review runbook

This runbook describes the first human calibration gate for the two Priority 2
questions that code cannot answer by itself:

1. how the project owner decides whether a Context Shard should be promoted;
2. how closely the semantic judge agrees with the owner's blinded assessment.

This is deliberately a **single-owner calibration**. It measures alignment with
one intended user of the system. It does not measure inter-annotator agreement,
team consensus or population-level human preference.

The packs and annotation CSV files remain local and are excluded from Git.

## Frozen local packs

| Review | Items | Local directory | Pack SHA-256 |
| --- | ---: | --- | --- |
| MemGym semantic answers | 40 | `.cache/human-calibration/memgym-judge` | `cd6b91058fed12d36d62d547c5fab32b7554c345cdc3742c173fe6f0c99af576` |
| Context Shards | 24 | `.cache/human-calibration/context-shards-24` | `ee23d85e9f045ed320ee0d8cb27ad92071e60498b658f89c37c40bbfb00ba3b6` |

Do not regenerate a pack after annotation starts. A changed hash defines a
different calibration campaign.

## Start the local interface

From the repository root:

```bash
.venv/bin/python tools/human_review_server.py
```

The command opens `http://127.0.0.1:8766`. The interface:

- shows only the public review packs, never the private mapping;
- saves after every change;
- resumes from the existing local CSV;
- records active review time;
- writes the exact columns expected by the scorer.

The two output files are:

```text
.cache/human-calibration/memgym-judge/annotator-owner.csv
.cache/human-calibration/context-shards-24/annotator-owner.csv
```

The server binds only to `127.0.0.1`. No external service, account or API is
used.

## Blindness

During review, do not open:

- `private-mapping.jsonl`;
- model-judge scores;
- synthetic Context Shard reference labels.

The interface never loads these files. Notes and labels are not added to the
public result.

## MemGym semantic-answer rubric

Read the question, gold answer and predicted answer for one item. Select exactly
one frozen score:

| Score | Meaning |
| ---: | --- |
| `1.0` | Semantically correct, complete enough and without a material contradiction |
| `0.7` | Mostly correct, with a minor omission or imprecision |
| `0.5` | Mixed or partially correct, with a meaningful missing element |
| `0.3` | Contains a small correct element but is materially incomplete or misleading |
| `0.0` | Wrong, unsupported, contradictory or non-responsive |

Also enter confidence from 0 to 1. A short note is optional but useful for
partial scores. Intermediate scores such as `0.8` are not accepted.

## Context Shard rubric

Read the candidate and every evidence item. Select exactly one decision:

| Decision | Meaning |
| --- | --- |
| `approved` | Repeated evidence supports the candidate and it is safe enough to promote |
| `rejected` | Contradicted, unsupported, too specific, redundant or unsafe to inject |
| `deferred` | Plausible, but evidence, wording, scope or durability is still unclear |

Then select:

- `scope`: `personal`, `team` or `task`;
- `injection`: `always_on`, `task_specific` or `never`;
- confidence from 0 to 1;
- an optional short reason.

The scorer rejects aliases such as `accept`.

## Verify the frozen packs

Run before and after annotation:

```bash
shasum -a 256 \
  .cache/human-calibration/memgym-judge/review-pack.jsonl \
  .cache/human-calibration/context-shards-24/review-pack.jsonl
```

The hashes must match the table above.

## Score the completed owner reviews

MemGym:

```bash
.venv/bin/python src/human_calibration.py score-single \
  --pack .cache/human-calibration/memgym-judge/review-pack.jsonl \
  --mapping .cache/human-calibration/memgym-judge/private-mapping.jsonl \
  --annotator .cache/human-calibration/memgym-judge/annotator-owner.csv \
  --output results/P2-MEMGYM-OWNER-CALIBRATION.json
```

Context Shards:

```bash
.venv/bin/python src/human_calibration.py score-single \
  --pack .cache/human-calibration/context-shards-24/review-pack.jsonl \
  --mapping .cache/human-calibration/context-shards-24/private-mapping.jsonl \
  --annotator .cache/human-calibration/context-shards-24/annotator-owner.csv \
  --output results/P2-CONTEXT-SHARDS-OWNER-CALIBRATION.json
```

The MemGym report includes coverage, model-judge mean absolute error relative
to the owner, Pearson correlation and median review time. The Context Shard
report includes coverage, decision distribution, agreement with the synthetic
reference and median review time.

## Publication gate

Do not publish an output automatically. First verify:

- coverage is reported and no missing rows are hidden;
- the pack hashes still match;
- Context Shard references are described as synthetic;
- MemGym error is described as alignment with one owner, not a universal judge
  score;
- Context Shard accuracy is described as one owner's promotion policy, not team
  consensus;
- no question text, answer text, notes or reviewer identity enters the compact
  public registry.

## Optional future extension

A second independent reviewer can later upgrade this diagnostic into a
double-annotation study. The existing `score` command remains available for
paired coverage, exact agreement, Cohen's kappa or Pearson agreement. It is not
required for the current owner-focused campaign.
