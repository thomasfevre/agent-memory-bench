# Priority 3 preregistered protocols

This document fixes the comparison rules before the realistic and adversarial
campaigns begin. A failed or unavailable configuration remains in the result.
It is not replaced by an easier model, larger budget or different task.

## Shared invariants

- Public inputs only. No private repository, personal file, wallet or paid API.
- Every repository, model and dataset is pinned by commit or SHA-256.
- The canonical model is local `qwen2.5:14b` at temperature zero.
- The model receives no browser, network, user account or unrelated memory.
- Harnesses receive the same task text, starting checkout, tools and budgets.
- Run order is seeded and interleaved. Each completed condition has three
  repetitions unless the protocol explicitly retains an operational failure.
- Tool transcripts, patches, tests, model tokens, wall time and exit states are
  captured at the same boundaries.
- Hidden tests remain unavailable to the agent during execution and are
  published only after the campaign is frozen.

## A. Long-running tasks across four coding harnesses

### Question

Does persistent memory or harness orchestration improve completion of realistic
multi-step repository work when the model and tools are held constant?

### Starting repository

The public `thomasfevre/agent-memory-bench` repository is pinned to a campaign
start commit. Each task begins from a fresh worktree at that exact commit.
Using this repository keeps installation reproducible on macOS ARM and makes
the complete task and evaluator publishable.

The frozen fixture commit is
`cf0dfa68b028f8dab5b39d0a5dddd9e14f2298ea`. The task specification SHA-256
is `65f6b58fadc7354dbc1baddec6ff1e0704290b6ff8d3e283ab3abeb7982ea471`.
Hidden evaluator hashes, in task order, are:

1. resume runner:
   `877b723b0444dbfff51bd5456e759d0f81a545e90cf5e933362a155df22a83cb`;
2. provenance:
   `740c623d71fd0d501eb3442e9f0d294323300a8928a1b8c7078eb133e02aa113`;
3. process resume:
   `accbb557c1b0404300f9cbd2b88302ed2a4ace59e54f97a87c112581de63ed8b`;
4. requirement change:
   `e628023b34868ab5ed803b0fcfaae321045304d4dafe32067293a71aa7703d21`.

### Harnesses

1. jcode;
2. Letta Code;
3. Codex CLI in local-provider mode;
4. Claude Code through a local Anthropic-compatible proxy.

All four must route to the same Ollama model digest. If one harness cannot
complete a provider-free local-model handshake, its result is
`provider_incompatible`. It is not rerun with a subscription model.

### Tasks

Four held-out tasks cover different memory demands:

1. resume a partially implemented benchmark runner and satisfy hidden
   regression tests;
2. diagnose a multi-file provenance bug, implement the fix and preserve
   backward compatibility;
3. add a resumable campaign checkpoint, then continue after a forced process
   interruption;
4. reconcile a changed requirement introduced halfway through the task without
   reintroducing the superseded behavior.

The task authoring branch, golden patch and hidden tests remain outside every
agent worktree until scoring. Each task must modify at least two production
files and one test-facing contract.

### Budgets

- 20 minutes per task attempt;
- 100,000 total model tokens per attempt;
- 80 tool calls;
- four CPU cores;
- one clean worktree;
- no network after dependency preparation.

### Metrics

- hidden-test pass fraction;
- task completion;
- regression count;
- unsupported or invented claim count;
- resume fidelity after interruption;
- superseded-requirement reintroduction;
- input, output and cached tokens;
- wall time and tool calls;
- correct tasks per 100,000 tokens and per hour.

The primary comparison is paired by task and repetition. Aggregate harness
scores are secondary to task-level outcomes.

## B. Late and out-of-order memory under model extraction

### Question

Can a model-backed ingestion pipeline preserve temporal truth when updates
arrive late, out of order, duplicated, expired or retracted?

### Dataset

A public synthetic stream contains at least 60 natural-language observations
covering 20 entities. Every entity includes an initial fact and one or more of:

- correction with an earlier effective date;
- late arrival;
- duplicate observation;
- explicit expiration;
- explicit retraction;
- lower-confidence contradiction;
- reviewed Context Shard approval or rejection.

The canonical structured events are committed before any model run.

### Extraction

The model receives one natural-language observation at a time and returns:

- entity key;
- value;
- asserted timestamp;
- effective-from and effective-until;
- event type;
- superseded or retracted event identifier when present;
- confidence;
- source identifier.

### Arrival schedules

1. chronological;
2. 10% deterministic late arrivals;
3. 25% deterministic late arrivals;
4. reverse order within fixed five-event windows;
5. duplicate and retry schedule.

### Metrics

- field-level extraction F1;
- temporal-window exact match;
- event-type accuracy;
- provenance exact match;
- final-state exact match;
- historical query accuracy;
- correct abstention after expiration or retraction;
- duplicate amplification;
- stability across three repetitions.

## C. Persistent derived-index recovery

### Question

Can persisted retrieval views recover after a real process crash without
silently diverging from the canonical event journal?

### Views

- temporal current-state table;
- full-text index;
- persisted dense-vector table;
- graph edge table;
- generation manifest linking each view to the journal sequence and hash.

### Crash matrix

At eight deterministic event boundaries, a child process is terminated:

1. before derived transaction begin;
2. after temporal update;
3. after full-text update;
4. after vector update;
5. after graph update;
6. before manifest commit;
7. immediately after commit;
8. during a full rebuild.

Each boundary is tested before and after acknowledgement.

### Recovery gate

After restart, the implementation must either:

- expose a complete generation whose manifest matches every view; or
- reject the generation and rebuild it from the canonical journal.

Success requires identical canonical query results, source identifiers and
generation signatures to an uninterrupted run. File-byte identity is not
required.

## D. Deletion, compaction and retention

### Question

Can a sovereign memory system prove that logically deleted information is no
longer retrievable from active derived state?

### Operations

- tombstone one source;
- retract one fact but retain its audit event;
- expire one Context Shard;
- apply a 30-day retention policy;
- compact the journal into a new signed generation;
- rebuild every derived view from that generation.

### Verification

Deleted payload text and identifiers must be absent from:

- direct current-state lookup;
- BM25 and full-text search;
- dense-vector nearest neighbors;
- graph traversal;
- active export;
- compacted database pages after `VACUUM`;
- newly created backup artifacts.

The report separately lists old backups and immutable audit events that still
exist. This protocol demonstrates verifiable logical deletion and compaction,
not secure erasure from SSD flash cells.

## E. Fixed-budget quality and cost

Every Priority 3 result is rescored under common ceilings:

- 25,000, 50,000 and 100,000 model tokens;
- 5, 10 and 20 wall-clock minutes;
- 20, 40 and 80 tool calls.

For each ceiling the report includes:

- task success;
- correct answers or passed hidden tests;
- cost per correct result;
- time per correct result;
- proportion of budget spent on retrieval, reading, verification and retry;
- Pareto-efficient configurations.

No monetary API price is assigned to subscription or local runs. Tokens,
hardware time and wall time remain separate observable costs.

## Completion gates

Priority 3 is complete only when:

- all four harnesses have either three completed repetitions per task or a
  retained, evidenced incompatibility;
- the temporal extraction matrix has all five arrival schedules;
- every derived view passes the crash matrix or records a reproducible failure;
- deletion verification covers every listed surface;
- fixed-budget reports can be rebuilt from raw manifests;
- the public registry and dashboard expose both results and unresolved limits.

## Deviation log

### Post-hoc observable-field diagnostic

After the first 60 temporal extractions completed, an audit found that the
frozen text does not state the expected confidence for expiration and
retraction events, and does not state a separate `effective_from` for
expiration events. The preregistered strict field score remains unchanged and
primary. A secondary `text_observable_field_accuracy` excludes only those
three event-field combinations and is explicitly marked post hoc in every
result. It must not replace the strict score in rankings.

### Active-state set correction

The first-pass audit also showed that a correct selected answer can hide a
stale record that was never invalidated. The scorer therefore defines
`final_state_exact` over the complete active record set and reports
`historical_active_state_accuracy` plus `stale_record_leakage_rate`.
The earlier winner-only checks remain visible as `selected_final_value_exact`
and `historical_query_accuracy`. This correction was made before inspecting
the second and third repetitions.

### Coding-harness compatibility stop

The first two independent tasks produced no production-file change with
jcode, Letta Code or Codex CLI. All three harnesses reached the local
`qwen2.5:14b` model, but the model did not use each harness tool protocol
successfully. Claude Code was attempted twice and failed before task execution
because its local provider adapter sent an unsupported `thinking` parameter.

The campaign therefore retained:

- `tool_protocol_incompatible` for jcode, Letta Code and Codex CLI;
- `provider_incompatible` for Claude Code;
- zero completed tasks for every harness.

The remaining two tasks and third repetitions were not executed because the
completion gate explicitly allows a retained, evidenced incompatibility. More
attempts with the same pinned model and protocol would measure repeated
handshake failure, not task quality. This result is not a quality ranking of
the harnesses with their recommended models.

The frozen hidden evaluators were published only after these results were
sealed. Their committed SHA-256 values match the preregistered hashes above.
