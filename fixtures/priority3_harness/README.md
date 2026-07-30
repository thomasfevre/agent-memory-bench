# Priority 3 coding-harness fixture

This small public package is the shared starting checkout for the four
long-running harness tasks. Public tests check only API compatibility.
Task prompts, golden patches and hidden regression tests are frozen separately
and remain outside agent worktrees during scoring.

The modules intentionally contain incomplete behavior. They are benchmark
fixtures, not production components of Agent Memory Bench.

## Released evaluators

The hidden evaluators under `hidden_tests/` were released after the campaign
was frozen. They remain separate from the root test suite so the intentionally
incomplete fixture does not turn normal project CI red.

Their SHA-256 values are preregistered in
`docs/priority-3-protocols.md`. A public-test or hidden-test baseline pass does
not count as task completion unless the agent changed the required production
files and satisfied the full task contract.
