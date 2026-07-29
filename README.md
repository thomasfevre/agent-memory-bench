# Agent Memory Bench

[Dashboard](https://thomasfevre.github.io/agent-memory-bench/) ·
[Methodology](docs/methodology.md) ·
[Priority 2 protocols](docs/priority-2-protocols.md) ·
[Roadmap](docs/roadmap.md) ·
[Contributing](CONTRIBUTING.md)

An evidence-first, provider-neutral benchmark for AI agent memory systems.

The project asks a narrower and more useful question than “which memory product
is best?”:

> Under an explicit corpus, budget, reader and judge, what information survives
> ingestion, retrieval, context construction and final reading?

The raw corpus is immutable. Every strategy produces a reconstructible view
over the same records. The public dashboard is generated from
`results/published/registry.json`; raw dataset and provider outputs remain local
until their license and privacy bounds have been reviewed.

The current campaign covers long context, BM25, dense and hybrid RAG, temporal
facts and graphs, Context Shards, jcode, Mem0, Cognee, Graphiti, Letta Code,
MemGym, MemoryData, LoCoMo, LongMemEval, MemoryAgentBench and GraphRAG-Bench.
There is deliberately no aggregate leaderboard because many runs evaluate
different layers and budgets.

Public upstreams are pinned in `config/upstreams.lock.json`. Materialize one or
more isolated checkouts without mutating them:

```bash
.venv/bin/python tools/checkout_upstreams.py \
  --only MemGym GraphRAG-Benchmark Cognee Graphiti
```

Pinned public datasets can be downloaded and checksum-verified in the same
ignored cache:

```bash
.venv/bin/python tools/download_datasets.py \
  --only "LongMemEval-S cleaned" \
  "MemoryAgentBench Accurate Retrieval"
```

## One-command run

```bash
./run.sh
```

The default run is fully local and deterministic. It compares:

- long context;
- BM25;
- local MiniLM dense retrieval;
- hybrid BM25 + MiniLM;
- atomic facts with temporal filtering;
- a small relation graph;
- human-reviewed Context Shards;
- a routed strategy;
- a merged parallel retrieval strategy.

Results are written under `results/PROTOTYPE-*`.

The local regression suite is:

```bash
.venv/bin/python -m pytest -q
```

The current suite contains 122 tests.

## Optional local-model run

Ollama readers are deliberately a separate phase:

```bash
./run.sh --with-ollama qwen3:1.7b qwen3:8b
```

No remote API or authenticated account is required.

## Local topology run

The companion run compares a single reader, a retrieval pipeline, conditional
routing, an adversarial verification gate, a verify-and-retry loop, and a
three-worker supervisor topology:

```bash
./run-topologies.sh --model qwen3:1.7b
```

Results are written under `results/TOPOLOGY-*`.

The canonical node contracts and tested graph shapes are recorded in
`config/architectures.json`.

## Ingestion run

```bash
.venv/bin/python src/ingestion_benchmark.py --models qwen3:1.7b qwen3:8b
```

This measures atomic-fact extraction, temporal metadata preservation, and
repeated-pattern candidates before human review.

The Context Shard policy benchmark separates repetition, activation, decision
history, and delayed re-review:

```bash
.venv/bin/python src/context_shard_policy_benchmark.py \
  --shards data/shards.jsonl \
  --events data/shard-lifecycle-events.jsonl \
  --output results/CONTEXT-SHARD-POLICIES-20260729.json
```

The two repeated candidates and one deferred candidate come from the common
corpus. Review timestamps and the post-rejection recurrence are explicitly
synthetic policy events. This validates state-machine safety, not real reviewer
effort or organizational usefulness.

The incremental lifecycle runner then replays one ordered stream of 21
structured events and scores 25 checkpoints:

```bash
PYTHONPATH=src .venv/bin/python \
  src/incremental_memory_lifecycle.py \
  --output results/INCREMENTAL-MEMORY-LIFECYCLE-20260729.json
```

It isolates four deterministic policies:

- immutable event-log scan with reviewed state reconstructed at query time;
- latest-write snapshot with repeated-pattern auto-promotion;
- temporal version index with repeated-pattern auto-promotion;
- temporal version index with explicit human review.

| Policy | Exact state | Correct abstention | Unsafe answer | Mean events scanned | Mean records inspected |
|---|---:|---:|---:|---:|---:|
| Latest-write snapshot | 52% | 33.3% | 66.7% | 0 | 0.88 |
| Raw event-log scan | 100% | 100% | 0% | 13.40 | 21.32 |
| Reviewed temporal index | 100% | 100% | 0% | 0 | 6.00 |
| Temporal auto-promotion | 76% | 33.3% | 66.7% | 0 | 6.00 |

The workload includes current and historical corrections, a low-confidence
rumor, a competing rare exception, expiry, explicit retraction, a historical
record delivered late, answerable and unanswerable queries, and shard approval,
rejection, recurrence, and historical review state. The 100% scores are
expected consequences of preregistered policy contracts on a small synthetic
stream, not evidence that a product or LLM is universally superior. Expiry and
retraction prove non-retrieval only, not physical deletion or storage
compaction.

Crash-safe local persistence of the immutable event log can be tested
separately:

```bash
PYTHONPATH=src .venv/bin/python \
  src/incremental_memory_persistence.py \
  --output results/INCREMENTAL-MEMORY-PERSISTENCE-20260729.json \
  --crash-points 0 5 10 15 19 20
```

This creates isolated SQLite databases for each crash point and exits child
processes immediately before and immediately after commit. It then reopens the
databases, finishes the replay, tries a duplicate event, verifies the linked
hash chain and `PRAGMA integrity_check`, and compares an independently rebuilt
semantic signature with an uninterrupted run. All six boundary pairs pass.
Recovery is structurally exact after canonical JSON serialization, not
byte-exact to the source JSONL. This proves the tested transaction boundaries
and local replay path only, not arbitrary kernel failure, WAL-checkpoint
failure, persisted derived-index recovery, secure erasure, or distributed
durability.

## Official-dataset retrieval

After downloading the official datasets, run:

```bash
.venv/bin/python src/external_retrieval.py \
  --locomo /path/to/locomo10.json \
  --longmemeval /path/to/longmemeval_s_cleaned.json \
  --output results/EXTERNAL-OFFICIAL.json
```

LoCoMo granularity can be compared within the same 1,982 questions:

```bash
.venv/bin/python src/locomo_granularity.py \
  --dataset /path/to/locomo10.json \
  --output results/LOCOMO-GRANULARITY-20260729.json
```

The runner compares individual turns, overlapping four-turn windows, and full
sessions with BM25, MiniLM, and hybrid retrieval. It reports both top-5 items
and a common 500-word context budget, because five sessions contain far more
text than five turns. Results checkpoint after every conversation.

Window-size sensitivity uses the same 500-word protocol:

```bash
.venv/bin/python src/locomo_window_grid.py \
  --dataset /path/to/locomo10.json \
  --window-sizes 2 4 8 16 \
  --output results/LOCOMO-WINDOW-GRID-20260729.json
```

The output compares each retriever at every size and bootstraps the top two
configurations over conversations rather than treating 1,982 correlated
questions as independent samples.

Hybrid-fusion sensitivity can then be tested without tuning on the evaluated
conversation:

```bash
.venv/bin/python src/locomo_hybrid_fusion.py \
  --dataset /path/to/locomo10.json \
  --window-sizes 2 4 \
  --alphas 0 0.25 0.5 0.75 1 \
  --output results/LOCOMO-HYBRID-FUSION-20260729.json
```

`alpha` is the BM25 contribution to weighted RRF; `1 - alpha` is the dense
contribution. The runner selects a configuration on nine conversations and
evaluates it on the tenth, rotating through all ten groups. It reports
categories 1 to 4 separately from the adversarial category 5, and includes an
oracle category-router upper bound whose gold labels would not be available in
production. Query latency and index-build latency are kept separate.

Pay the prediction cost of that router with nested grouped validation:

```bash
.venv/bin/python src/locomo_predicted_router.py \
  --dataset /path/to/locomo10.json \
  --fusion-results results/LOCOMO-HYBRID-FUSION-20260729.json \
  --output results/LOCOMO-PREDICTED-ROUTER-20260729.json
```

The outer fold holds out the evaluated conversation. Inner conversation folds
select the BM25 nearest-question vote depth. The classifier then predicts a
category for each held-out question, and its measured latency is added to the
category-specific retriever selected on outer-training data.

LongMemEval-S also contains 30 deliberate near-miss questions that require
abstention. Calibrate retrieval-score thresholds out of fold:

```bash
.venv/bin/python src/longmemeval_abstention.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output results/LONGMEMEVAL-ABSTENTION-20260729.json
```

The runner checkpoints all 500 feature rows before fitting five-fold
thresholds. It compares raw score, margin, query-normalized score, and
within-history z-score for BM25 and MiniLM. This tests retrieval confidence, not
final answer generation.

Test premise-aware abstention without allowing a near-miss and its answerable
counterpart to land in different folds:

```bash
.venv/bin/python src/longmemeval_premise_verifier.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --retrieval-features results/LONGMEMEVAL-ABSTENTION-20260729.features.json \
  --output results/LONGMEMEVAL-PREMISE-VERIFIER-20260729.json
```

The runner groups all 29 matched positive/negative pairs, extracts deterministic
lexical premise-support features, selects logistic regularization and thresholds
with nested grouped validation, and bootstraps balanced-accuracy differences at
the pair-group level. It remains a local verifier ablation, not an LLM answer
judge.

Weighted hybrid retrieval on the 470 answerable questions uses both top-5
sessions and a shared context budget:

```bash
.venv/bin/python src/longmemeval_hybrid_fusion.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --alphas 0 0.25 0.5 0.75 1 \
  --word-budget 8000 \
  --output results/LONGMEMEVAL-HYBRID-FUSION-20260729.json
```

Five deterministic held-out folds select alpha before evaluation. The output
keeps query latency, BM25 index construction, dense index construction, session
count, and actual context words separate. The 30 `_abs` near-misses remain
outside evidence recall and are evaluated by the abstention runner above.

Run the bounded paired generation challenge through Codex subscription models:

```bash
.venv/bin/python src/longmemeval_codex_generation.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output results/LONGMEMEVAL-CODEX-GENERATION-8PAIRS-20260729.json \
  --repetitions 3 \
  --retry-errors
```

The default slice contains eight SHA-256 selected answerable/near-miss pairs,
two per represented question type. It compares no context, BM25 chunks, and
fixed 50/50 BM25-MiniLM RRF through `gpt-5.6-luna` and `gpt-5.6-sol`. Every
context uses the same 224-token chunks, a 4,000-word budget including headers,
and chronological presentation. The runner uses ephemeral read-only Codex
sessions in empty directories, checkpoints every call, caches public contexts,
verifies citations and the exact abstention sentinel, and reports paired
McNemar tests plus pair-group bootstrap intervals.

This is a subscription-agent diagnostic with deterministic answer aliases, not
the official LongMemEval LLM-judged score or a pinned raw model API comparison.

Filter assistant turns while keeping the same reader, prompts, budgets, and
retrievers:

```bash
.venv/bin/python src/longmemeval_codex_generation.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output results/LONGMEMEVAL-CODEX-ROLE-ABLATION-8PAIRS-3X-20260729.json \
  --seed-results results/LONGMEMEVAL-CODEX-GENERATION-8PAIRS-20260729.json \
  --models gpt-5.6-sol \
  --architectures bm25_chunks hybrid_chunks \
    bm25_user_chunks hybrid_user_chunks \
  --repetitions 3 \
  --allow-legacy-results \
  --retry-errors
```

The seed file reuses matching all-role calls. The user-only architectures index
only user messages but preserve the same chunking, context budget, temporal
ordering, and reader. This isolates role policy from model strength.

Confirm the role-policy decision on all 29 matched pairs with one frozen reader:

```bash
.venv/bin/python src/longmemeval_codex_generation.py \
  --dataset /path/to/longmemeval_s_cleaned.json \
  --output results/LONGMEMEVAL-CODEX-ROLE-ABLATION-29PAIRS-20260729.json \
  --seed-results \
    results/LONGMEMEVAL-CODEX-ROLE-ABLATION-8PAIRS-3X-20260729.json \
  --models gpt-5.6-sol \
  --architectures hybrid_chunks hybrid_user_chunks \
  --pair-ids <all-29-matched-base-ids> \
  --repetitions 1 \
  --allow-legacy-results \
  --retry-errors
```

For this larger confirmation, use `decision_accuracy` and
`architecture_decision_comparisons` as the primary outputs. The deterministic
answer aliases are preregistered only for the initial eight-pair slice, so
strict answer accuracy on the other pairs is a lower bound rather than the
official semantic score.

## MemoryAgentBench conflict-resolution slice

The bounded runner below uses the official Conflict Resolution parquet and
compares a full 6k-token context with BM25-selected facts through the same local
Ollama reader:

```bash
.venv/bin/python src/memoryagentbench_slice.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --model qwen2.5:14b \
  --questions 100 \
  --output results/MEMORYAGENTBENCH-CONFLICT-100Q.json \
  --resume
```

It reports the benchmark's official substring exact-match metric alongside
strict exact match, token F1, latency, and token counts. This is a bounded slice,
not a reproduction of the paper's complete result table.

Each strategy result is checkpointed immediately. `--resume` skips completed
question-strategy pairs and rejects a checkpoint if model, source, offset,
question count, top-k, or seed differ.

Long runs can be split without repeating prior questions:

```bash
.venv/bin/python src/memoryagentbench_slice.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --offset 10 \
  --questions 10 \
  --output results/MEMORYAGENTBENCH-CONFLICT-Q10-19.json

.venv/bin/python src/merge_memoryagentbench_slices.py \
  results/MEMORYAGENTBENCH-CONFLICT-Q0-9.json \
  results/MEMORYAGENTBENCH-CONFLICT-Q10-19.json \
  --output results/MEMORYAGENTBENCH-CONFLICT-20Q.json
```

After merging all 100 questions, generate paired outcomes, Wilson intervals,
per-decile scores, efficiency ratios, and the exact McNemar test:

```bash
.venv/bin/python src/analyze_memoryagentbench_full.py \
  results/MEMORYAGENTBENCH-CONFLICT-100Q.json \
  --output results/MEMORYAGENTBENCH-CONFLICT-100Q-ANALYSIS.json
```

On `factconsolidation_mh_6k`, BM25 scores 10/100 and long context 2/100.
All long-context successes are shared with BM25; BM25 has eight unique
successes. The exact paired McNemar p-value is 0.0078125. Long context uses
16.46 times more prompt tokens and is 9.47 times slower on average.

The provider-isolated Codex runner extends this comparison to both official 6k
variants and adds hybrid BM25 plus MiniLM retrieval:

```bash
PYTHONPATH=src .venv/bin/python \
  src/memoryagentbench_codex_generation.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-CODEX-CONFLICT-6K-200Q.json \
  --models gpt-5.6-sol \
  --sources factconsolidation_mh_6k factconsolidation_sh_6k \
  --architectures long_context bm25 hybrid \
  --question-start 0 \
  --questions 100 \
  --repetitions 1 \
  --retry-errors
```

Every Codex call runs in an empty temporary directory, read-only, with user
configuration and repository rules disabled. The model receives only the public
benchmark facts and cannot use tools, files, or web search. Results checkpoint
after each call and can be reused with `--seed-results`. Reuse now verifies the
dataset, schema, prompt version, embedding artifact, Codex version, retrieval
parameters, and reasoning effort before accepting prior rows.

This is a static final-context test. It evaluates retrieval and reading after
the memory has already been assembled. It does not evaluate incremental
ingestion, consolidation, updating, maintenance, forgetting, or human review.
The completed historical runs used a blocked architecture order. Current
runners now deterministically interleave architecture, repetition and question
configurations to reduce sensitivity to provider drift. Local Ollama models stay
grouped to avoid reload confounding; subscription models can be interleaved. The
historical result keeps its original limitation.

The complete 600-call run reports:

| Variant | Long context | BM25 top-20 | Hybrid top-20 |
|---|---:|---:|---:|
| Multi-hop 6k | 1% | 13% | 13% |
| Single-hop 6k | 47% | 69% | 65% |

On multi-hop, BM25 and hybrid each beat long context by 12 points
(`p = 0.001831`, question bootstrap 95% interval 5 to 19 points). On
single-hop, BM25 beats long context by 22 points (`p = 0.0000105`, interval 13
to 31 points), while hybrid gains 18 points (`p = 0.000277`, interval 9 to 27
points). BM25 and hybrid remain statistically indistinguishable on both
variants.

The compressed views inject about 189 to 200 words instead of 4,431 and consume
about 3,154 to 4,437 Codex tokens per call instead of 10,247 to 13,266. The
official substring metric is permissive: on single-hop, strict exact match is
25% for both compressed views even though substring match is 65% to 69%.
Expanded answers that include both a reference value and a conflicting value
can therefore count as official successes. Exact match, token F1, explicit
conflict cues, citations, and substring match are all retained separately.
Citation validity here is only formal: a cited id must exist in the supplied
context. It does not prove that the cited fact entails the answer or that its
original provenance is correct. Because every selected question has a gold
answer, the reported abstention rate is a refusal rate, not abstention accuracy.
Reported latency is reader latency and excludes retrieval and index
construction. Provider success rate is reported independently from quality.

A deterministic SHA-256 sample of 15 question indices was repeated three times
per architecture and variant, for 270 successful calls:

```bash
PYTHONPATH=src .venv/bin/python \
  src/memoryagentbench_codex_generation.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-CODEX-CONFLICT-15Q-3X.json \
  --models gpt-5.6-sol \
  --question-indices 3 7 21 30 32 60 71 85 86 88 90 93 94 95 99 \
  --repetitions 3 \
  --retry-errors
```

On the repeated single-hop slice, long context scores 53.3%, BM25 73.3%, and
hybrid 75.6%. The compressed views keep a 20 to 22.2 point call-level
advantage. The question-group bootstrap intervals exclude zero, but majority
vote McNemar tests on only 15 independent questions do not, with `p = 0.25`
for BM25 and `p = 0.125` for hybrid. Correctness is unanimous across
repetitions for only 73.3% of questions. Normalized answer text is unanimous
for 53.3% of BM25 questions and 60% of hybrid or long-context questions.

The same 15 questions were also paired across `gpt-5.6-luna` and
`gpt-5.6-sol`. No model difference is significant for any architecture. This
slice supports an architectural effect, not a provider-model leaderboard.

The deterministic retrieval matrix covers all eight Conflict Resolution
variants and all 800 questions before generation:

```bash
.venv/bin/python src/memoryagentbench_conflict_retrieval.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-CONFLICT-RETRIEVAL-ALL-20260729.json
```

It reports answer-string evidence hit rates at 1, 5, 10, 20, 50, and 100,
reciprocal rank, compression, and retrieval latency. It also compares direct
BM25 with a deterministic greedy expansion that adds each selected fact to the
next query. This is not the official end-to-end score. It isolates whether a
fact containing the reference answer survives retrieval before the reader is
involved.

Pair completed or partial generation files with that retrieval evidence:

```bash
.venv/bin/python src/summarize_conflict_generation_matrix.py \
  --retrieval results/MEMORYAGENTBENCH-CONFLICT-RETRIEVAL-ALL-20260729.json \
  --generation results/MEMORYAGENTBENCH-CONFLICT-100Q.json \
    results/MEMORYAGENTBENCH-CONFLICT-SH-6K-100Q.json \
  --output results/MEMORYAGENTBENCH-CONFLICT-GENERATION-MATRIX.json
```

The output separates evidence-plus-reader success, evidence-plus-reader
failure, and success without a literal answer match. Literal answer evidence is
not a complete multi-hop proof, so the decomposition remains a diagnostic rather
than an official metric.

Generation can be restricted to one strategy when full context exceeds the
local model window:

```bash
.venv/bin/python src/memoryagentbench_slice.py \
  --parquet /path/to/Conflict_Resolution-00000-of-00001.parquet \
  --source factconsolidation_mh_262k \
  --questions 100 \
  --strategies bm25 \
  --output results/MEMORYAGENTBENCH-CONFLICT-mh_262k-BM25-100Q.json \
  --resume
```

The Test-Time Learning retrieval baseline parses the five balanced
classification corpora into labeled examples and compares weighted BM25 votes:

```bash
.venv/bin/python src/memoryagentbench_ttl_retrieval.py \
  --parquet /path/to/Test_Time_Learning-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-TTL-RETRIEVAL-20260729.json
```

It evaluates all 500 classification questions at top-1, 3, 5, and 10.

ReDial uses its own evidence-retrieval adapter and the official movie entity
mapping:

```bash
.venv/bin/python src/memoryagentbench_redial_retrieval.py \
  --parquet /path/to/Test_Time_Learning-00000-of-00001.parquet \
  --entity-map /path/to/entity2id.json \
  --output results/MEMORYAGENTBENCH-REDIAL-RETRIEVAL-20260729.json
```

It measures whether the reference movies are mentioned in the top-1, 5, 10,
20, and 100 BM25-retrieved dialogues. This is an evidence upper bound before
recommendation generation, not the official twenty-item recommendation recall.

The RULER QA retrieval baseline preserves the official document boundaries:

```bash
.venv/bin/python src/memoryagentbench_ruler_retrieval.py \
  --parquet /path/to/Accurate_Retrieval-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-RULER-RETRIEVAL-20260729.json
```

It measures answer-string document evidence coverage on all 200 questions from
the 197K and 421K RULER contexts. EventQA requires a separate query-construction
protocol because each question contains all candidate answers.

The EventQA ablation compares full-question retrieval, previous-event retrieval,
sequential anchor neighborhoods, and an explicit oracle upper bound:

```bash
.venv/bin/python src/memoryagentbench_eventqa_query_ablation.py \
  --parquet /path/to/Accurate_Retrieval-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-EVENTQA-QUERY-ABLATION-20260729.json
```

It covers all 1,500 EventQA questions with fixed 120-word windows and a
deterministic lexical multiple-choice reader. Oracle-answer retrieval is
reported only as an upper bound and is not a valid production strategy.

The DetectiveQA ablation covers all 71 questions in Long-Range Understanding:

```bash
.venv/bin/python src/memoryagentbench_detectiveqa_ablation.py \
  --parquet /path/to/Long_Range_Understanding-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-DETECTIVEQA-QUERY-ABLATION-20260729.json
```

It compares the complete demonstrated prompt, the target question with options,
the target stem at top-5 and top-20, and an explicit oracle upper bound. The 100
long-document summarization tasks use a separate fixed-budget evidence protocol:

```bash
.venv/bin/python src/memoryagentbench_summarization_coverage.py \
  --parquet /path/to/Long_Range_Understanding-00000-of-00001.parquet \
  --output results/MEMORYAGENTBENCH-SUMMARIZATION-COVERAGE-20260729.json
```

It covers all 100 books and compares the first 20 chunks, 20 uniformly sampled
chunks, prompt-driven BM25, an oracle keypoint RRF selector, and the full
context. The metric is lexical keypoint coverage before summary generation, not
the official judged summary score. The oracle and full-context results are
upper bounds. Controlled generation still requires a calibrated local judge.

## GraphRAG-Bench retrieval slice

The official Novel corpus can be evaluated without a provider by comparing the
same fixed chunks under lexical, dense, hybrid, and corpus-only chunk-graph
retrieval:

```bash
PYTHONPATH=src .venv/bin/python src/graphrag_bench_chunk_graph.py \
  --corpus /path/to/GraphRAG-Benchmark/Datasets/Corpus/novel.json \
  --questions /path/to/GraphRAG-Benchmark/Datasets/Questions/novel_questions.json \
  --output results/GRAPHRAG-BENCH-CHUNK-GRAPH-FULL-20260729.json
```

The full run covers 20 books, 2,010 questions, 839,608 words, and 7,003
fixed chunks. It scores retrieval against all 4,689 official evidence
statements. A statement is considered lexically representable when one fixed
chunk contains a sentence or adjacent-sentence window with at least 85% of its
tokens and the same explicit polarity markers. The output separates pooled and
per-question recall over all official evidence, recall conditional on
representability, and full coverage on fully representable questions. It does
not run answer generation, semantic entailment, or the official LLM judge.

The result includes summaries by question type and book, question-paired
McNemar tests, and 10,000-repetition bootstrap intervals with the book as the
sampling cluster. This chunk graph is an intentionally simple deterministic
neighbor-replacement ablation. It is not LightRAG, HippoRAG, FastGraphRAG, or
an LLM-extracted knowledge graph.

Sensitivity runs at 80%, 85%, and 90% are stored separately. The paired
chunk-neighbor effect remains negative under all three lexical definitions, but
its magnitude shrinks as the criterion becomes stricter. This sensitivity is
part of the result, not a reason to select whichever threshold looks best.

A deterministic sample of 40 lexical positives was reviewed semantically by
one independent Codex subagent. It labeled 34 supported, five unclear, and one
contradicted. The audit is useful calibration evidence, but it is not a
substitute for the official judge: it has one auditor, no blinding, no
inter-annotator agreement, and no false-negative sample.

## Common temporal-graph slice

Cognee and Graphiti can be run against the same ordered corpus and questions.
Each adapter uses its project's isolated environment and writes the same result
envelope:

```bash
/tmp/agent-memory-systems.0ruwY3/Graphiti/.venv-codex/bin/python \
  src/graphiti_common_benchmark.py \
  --corpus data/corpus.jsonl \
  --questions data/questions.jsonl \
  --output results/GRAPHITI-COMMON.json \
  --limit-docs 8 \
  --limit-questions 8 \
  --document-timeout 90 \
  --query-timeout 30

LLM_MODEL=qwen2.5:14b \
  /tmp/agent-memory-systems.0ruwY3/Cognee/.venv-codex/bin/python \
  src/cognee_common_benchmark.py \
  --corpus data/corpus.jsonl \
  --questions data/questions.jsonl \
  --output results/COGNEE-COMMON.json \
  --limit-docs 8 \
  --limit-questions 8 \
  --ingestion-timeout 600 \
  --query-timeout 60
```

The common metrics distinguish evidence recall from temporal context precision.
This matters when a system returns both the valid and superseded versions of a
fact: recall can be perfect while temporal selection remains ambiguous.

Three Qwen 2.5 14B repetitions can be aggregated with:

```bash
.venv/bin/python src/summarize_graph_repetitions.py \
  --cognee results/COGNEE-COMMON-8DOCS-20260729.json \
    results/COGNEE-QWEN25-14B-8DOCS-REPEAT2-20260729.json \
    results/COGNEE-QWEN25-14B-8DOCS-REPEAT3-20260729.json \
  --graphiti results/GRAPHITI-COMBINED-8DOCS-20260729.json \
    results/GRAPHITI-QWEN25-14B-8DOCS-REPEAT2-20260729.json \
    results/GRAPHITI-QWEN25-14B-8DOCS-REPEAT3-20260729.json \
  --output results/GRAPH-REPETITIONS-QWEN25-20260729.json
```

The repeated runs expose a useful distinction: Cognee completed all documents
but varied substantially in recall, while Graphiti repeated the same quality
scores and the same document timeout but varied in ingestion duration.

These adapters do not vendor either project. They expect isolated, pinned
checkouts and locally served Ollama models.

## jcode common-corpus adapter

The jcode adapter runs the official CLI import, CLI keyword and semantic
searches, and the shipped `MemoryManager.find_similar_hybrid` retriever against
the same 28 documents and 20 questions:

```bash
.venv/bin/python src/jcode_common_benchmark.py \
  --corpus data/corpus.jsonl \
  --questions data/questions.jsonl \
  --output results/JCODE-COMMON-3X-20260729.json \
  --jcode /path/to/jcode \
  --native-bench /path/to/jcode_common_memory_bench \
  --model-dir /path/to/all-MiniLM-L6-v2 \
  --top-k 5 \
  --repetitions 3
```

Every repetition uses a fresh `JCODE_HOME`. The run compares the default CLI
import, which applies semantic storage deduplication, with the shipped upsert
API, which preserves stable document IDs. On this corpus the default import
retains 17 of 28 IDs and the hybrid retriever recalls 52.6% of gold sources.
Preserving all 28 IDs raises the same retriever to 89.5% recall.

Create the cross-strategy comparison with:

```bash
.venv/bin/python src/analyze_common_retrieval.py \
  --questions data/questions.jsonl \
  --prototype results/PROTOTYPE-latest.json \
  --jcode results/JCODE-COMMON-3X-20260729.json \
  --mem0 results/MEM0-COMMON-3X-20260729.json \
  --output results/COMMON-RETRIEVAL-COMPARISON-20260729.json
```

## Mem0 common-corpus adapter

The Mem0 adapter compares raw storage with local LLM extraction on the same 28
documents and 20 questions:

```bash
PYTHONPATH=src /path/to/Mem0/.venv/bin/python src/mem0_common_benchmark.py \
  --corpus data/corpus.jsonl \
  --questions data/questions.jsonl \
  --output results/MEM0-COMMON-3X-20260729.json \
  --model qwen3:1.7b \
  --top-k 5 \
  --repetitions 3
```

Each repetition uses fresh embedded Qdrant stores. `infer=False` keeps source
records unchanged, while `infer=True` exercises Mem0's extraction and update
path with a local Ollama model. The adapter requests more than Mem0's default
20-item `get_all` limit so retention counts cover the complete corpus.

On this bounded run, raw storage retained all 28 source records and reached
100% recall. Local inference produced 41 memories, retained identifiable
provenance for 26 sources, reached 92.1% recall, and made ingestion about 403
times slower. These numbers are configuration-specific and do not generalize to
larger extraction models.

## MemGym-DR provider-free retrieval

The public MemGym-DR corpus can be evaluated before answer generation with the
official MemGym BM25 and MiniLM memory managers:

```bash
/path/to/MemGym/.venv-codex/bin/python src/memgym_dr_retrieval.py \
  --dataset-dir /path/to/memgym-dr-instances-snapshot \
  --memgym-repo /path/to/MemGym \
  --sample-per-stratum 30 \
  --top-k 1 2 5 10 \
  --bm25-full \
  --output results/MEMGYM-DR-RETRIEVAL-20260729.json
```

BM25 was run on all 1,194 public instances. Its official lexical
memory-required-fact recall rises from 68.9% at top-1 to 91.8% at top-2,
99.7% at top-5, and 100% at top-10. The paired 90-instance comparison finds
MiniLM indistinguishable at top-1, 5.0 points below BM25 at top-2 with a 95%
bootstrap interval of -9.9 to -0.2 points, and saturated with BM25 by top-10.
At top-10 both methods inject about 3,800 words, so the perfect proxy score is
not a perfect-memory result. It mainly shows that the repository's 40%
substring fact-recall proxy saturates under a large context allowance.

The paired analysis is in
`results/MEMGYM-DR-RETRIEVAL-PAIRED-20260729.json`. This slice does not run
answer generation or an LLM judge.

The Priority 2 reader and judge campaign uses isolated Codex subscription
sessions with browser, apps, shell and agent tools disabled at the command
line, native web search disabled independently, and a fail-closed trace audit.
All 120 readers and 120 judges completed. The provisional semantic-judge macro
score was 0.635 for visible documents, 0.650 for BM25 top-1, 0.743 for top-2
and 0.713 for top-5. Top-2 improved the paired score over visible-only by
0.108 across the same 30 questions, but this is not a human-calibrated metric.

A bounded blinded calibration pack can be prepared after the run:

```bash
PYTHONPATH=src .venv/bin/python src/human_calibration.py prepare-judge \
  --result results/P2-MEMGYM-DR-CODEX-30X4-20260729.json \
  --dataset-dir /path/to/memgym-dr-instances-snapshot \
  --output-dir .cache/human-calibration/memgym-judge \
  --sample-size 40 \
  --seed 20260729
```

The generated 40-item pack resolves question text locally from the pinned
dataset hashes. It does not copy questions into the compact public result.

Context Shard review uses the same blinded workflow:

```bash
PYTHONPATH=src .venv/bin/python src/human_calibration.py prepare-shards \
  --shards data/context-shard-review-candidates.jsonl \
  --corpus data/corpus.jsonl \
  --output-dir .cache/human-calibration/context-shards \
  --seed 20260729
```

After two people complete the generated CSV files independently:

```bash
PYTHONPATH=src .venv/bin/python src/human_calibration.py score \
  --pack .cache/human-calibration/context-shards/review-pack.jsonl \
  --mapping .cache/human-calibration/context-shards/private-mapping.jsonl \
  --annotator-a .cache/human-calibration/context-shards/annotator-a.csv \
  --annotator-b .cache/human-calibration/context-shards/annotator-b.csv \
  --output results/P2-CONTEXT-SHARDS-HUMAN-CALIBRATION.json
```

The repository does not invent these labels. Until both humans submit them,
the related Priority 2 claims remain open.

## MemGym MemRM full CPU reproduction

The public 1.7B MemRM checkpoint was also rerun on all 6,209 rows of the
official `scenario-ood-tau2` split with the official CLI, CPU execution,
disabled NF4 quantization, and 1,000 bootstrap resamples. The rerun obtains
AUROC 0.507 with a 95% interval of 0.487 to 0.528. The released eight-GPU
artifact records AUROC 0.520.

The aggregate AUROCs are close, but the individual predictions are not a
numerical reproduction: class agreement is 69.9%, probability correlation is
0.472, and mean absolute `P(SAFE)` difference is 0.316. This is consistent with
a hardware and quantization sensitivity warning, not evidence that one artifact
is intrinsically better. The current official runner uses the trained `" Y"`
and `" N"` completion logits; its separate SAFE/HARMFUL tokenization warning
does not apply to this evaluation path.

Raw rerun predictions, the released comparison artifact, and the keyed
comparison are retained under:

- `results/MEMGYM-MEMRM-TAU2-CPU-FULL-20260729.json`
- `results/MEMGYM-MEMRM-TAU2-RELEASED-20260729.json`
- `results/MEMGYM-MEMRM-TAU2-REPRODUCTION-20260729.json`

This evaluates the reward model's classification behavior, not the end-to-end
quality of a memory architecture.

## MemoryData official-runner smoke

MemoryData was also executed through its official `main.py` runner against the
official MemoryAgentBench EventQA configuration. External agent configs keep
the local-model setup outside the upstream checkout:

```bash
OPENAI_API_KEY=ollama \
  /path/to/MemoryData/.venv-codex/bin/python /path/to/MemoryData/main.py \
  --agent_config config/memorydata-long-context-qwen25-ollama.yaml \
  --dataset_config benchmark/memoryagentbench/Accurate_Retrieval/config/EventQA/Eventqa_full.yaml \
  --max_test_queries_ablation 1 \
  --artifact_root /tmp/memorydata-smoke
```

The bounded result is summarized in
`results/MEMORYDATA-SMOKE-20260729.json`. It includes long-context, BM25 top-10,
BM25 top-2, and dense top-2 runs. All use the same official EventQA item and all
answer it incorrectly. The top-k change nevertheless exposes a large
cost-latency effect, while inspection shows that the official runner sends the
question and all answer choices to retrieval. This validates the runner and
several preset paths, but it does not rank MemoryData as a whole.

## Scope

The prototype measures retrieval before generation. This is intentional: a strong reader must not hide information lost during ingestion or retrieval.

The local reader phase remains separate, and the common graph adapters currently
cover Cognee, Graphiti, and jcode. Letta Code deliberately uses agent-driven
MemFS exploration rather than a top-k retrieval API, so it belongs in a
separate end-to-end agentic-memory protocol.
