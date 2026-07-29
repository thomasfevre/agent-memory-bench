# Priority 3 coding-harness fixture

This small public package is the shared starting checkout for the four
long-running harness tasks. Public tests check only API compatibility.
Task prompts, golden patches and hidden regression tests are frozen separately
and remain outside agent worktrees during scoring.

The modules intentionally contain incomplete behavior. They are benchmark
fixtures, not production components of Agent Memory Bench.
