# Priority 2 human review runbook

This runbook is the final manual gate for the two Priority 2 claims that code
cannot honestly produce:

1. whether two people agree on Context Shard promotion decisions;
2. whether the semantic judge agrees with two blinded human reviewers.

The packs are prepared locally and intentionally excluded from Git.

## Frozen local packs

| Review | Items | Local directory | Pack SHA-256 |
| --- | ---: | --- | --- |
| MemGym semantic answers | 40 | `.cache/human-calibration/memgym-judge` | `cd6b91058fed12d36d62d547c5fab32b7554c345cdc3742c173fe6f0c99af576` |
| Context Shards | 24 | `.cache/human-calibration/context-shards-24` | `ee23d85e9f045ed320ee0d8cb27ad92071e60498b658f89c37c40bbfb00ba3b6` |

Do not regenerate a pack after either reviewer starts. A changed hash defines a
different calibration campaign.

## Independence and blindness

Assign one person as annotator A and another as annotator B.

Give each reviewer:

- the same `review-pack.jsonl`;
- only their own `annotator-a.csv` or `annotator-b.csv`;
- this rubric.

Never give a reviewer:

- `private-mapping.jsonl`;
- the other reviewer's CSV;
- model-judge scores;
- the synthetic Context Shard reference labels.

Reviewers must complete their files independently and should not discuss
individual items before both files are returned.

## MemGym semantic-answer rubric

Read the question, gold answer and predicted answer for one item. Enter exactly
one frozen score:

| Score | Meaning |
| ---: | --- |
| `1.0` | Semantically correct, complete enough and without a material contradiction |
| `0.7` | Mostly correct, with a minor omission or imprecision |
| `0.5` | Mixed or partially correct, with a meaningful missing element |
| `0.3` | Contains a small correct element but is materially incomplete or misleading |
| `0.0` | Wrong, unsupported, contradictory or non-responsive |

Also enter:

- `confidence`: reviewer confidence from 0 to 1;
- `time_seconds`: active review time for that item;
- `notes`: optional short reason, especially for partial credit.

Do not use intermediate values such as `0.8`. The scorer now rejects values
outside the frozen scale.

## Context Shard rubric

Read the candidate and every evidence item. Enter exactly one decision:

| Decision | Meaning |
| --- | --- |
| `approved` | Repeated evidence supports the candidate and it is safe enough to promote |
| `rejected` | Contradicted, unsupported, too specific, redundant or unsafe to inject |
| `deferred` | Plausible, but evidence, wording, scope or durability is still unclear |

Then select:

- `scope`: `personal`, `team` or `task`;
- `injection`: `always_on`, `task_specific` or `never`;
- `confidence`: reviewer confidence from 0 to 1;
- `time_seconds`: active review time for that item;
- `notes`: optional short reason.

The scorer rejects decision aliases such as `accept`.

## Verify the frozen packs

Run before and after annotation:

```bash
shasum -a 256 \
  .cache/human-calibration/memgym-judge/review-pack.jsonl \
  .cache/human-calibration/context-shards-24/review-pack.jsonl
```

The hashes must match the table above.

## Score both completed reviews

MemGym:

```bash
.venv/bin/python src/human_calibration.py score \
  --pack .cache/human-calibration/memgym-judge/review-pack.jsonl \
  --mapping .cache/human-calibration/memgym-judge/private-mapping.jsonl \
  --annotator-a .cache/human-calibration/memgym-judge/annotator-a.csv \
  --annotator-b .cache/human-calibration/memgym-judge/annotator-b.csv \
  --output results/P2-MEMGYM-HUMAN-CALIBRATION.json
```

Context Shards:

```bash
.venv/bin/python src/human_calibration.py score \
  --pack .cache/human-calibration/context-shards-24/review-pack.jsonl \
  --mapping .cache/human-calibration/context-shards-24/private-mapping.jsonl \
  --annotator-a .cache/human-calibration/context-shards-24/annotator-a.csv \
  --annotator-b .cache/human-calibration/context-shards-24/annotator-b.csv \
  --output results/P2-CONTEXT-SHARDS-HUMAN-CALIBRATION.json
```

The scorer reports paired coverage, exact agreement, Cohen's kappa or Pearson
agreement, median review time and comparison with the hidden model or
synthetic reference labels.

## Publication gate

Do not publish an output automatically. First verify:

- paired coverage is reported and no missing rows are hidden;
- both annotation files are independent and complete;
- the output describes synthetic Context Shard references as synthetic;
- MemGym model-to-human error is not presented as a universal judge score;
- no question text, answer text, notes or reviewer identity enters the compact
  public registry.

Only aggregate metrics and input hashes should be added to the public result.
