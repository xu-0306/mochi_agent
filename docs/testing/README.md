# Test Suite

Pytest uses strict marker validation. The default invocation still collects the
whole offline suite; markers are added at the narrowest package boundary as the
structural refactor moves tests.

## Commands

```powershell
# Quick development feedback
rtk pytest tests\unit tests\backends -q

# API/runtime integration
rtk pytest tests\integration -q

# Security and platform suites
rtk pytest tests\security -q
rtk pytest -m windows -q
rtk pytest -m posix -q

# Slow-test review lane
rtk pytest -m slow -q

# Full offline suite
rtk pytest -m "not network" -q

# Collect node IDs without running tests
rtk pytest --collect-only -q > docs/testing/test-inventory.txt
```

The quick command uses the current unit and backend contract packages. A
missing marker or misspelled marker fails collection because strict validation is
enabled.

## Marker Policy

- `integration`: crosses FastAPI, runtime, persistence, or store boundaries.
- `security`: checks an invariant, attack surface, authorization boundary, or
  fail-closed behavior.
- `slow`: one of the tests at or above the approximately 4.5-second measured
  baseline, or a later explicitly reviewed heavyweight test.
- `windows`: depends on Windows-native ACL, Win32, or reparse-point behavior.
- `posix`: depends on POSIX dir-fd, symlink, or native filesystem behavior.
- `network`: requires live external network access. Offline runs exclude it.

The collection hook in `tests/conftest.py` applies `integration` to tests below
`tests/integration/`, `security` to tests below `tests/security/`, and
`windows` or `posix` when the path names that platform. The measured slow
test list is explicit and is updated from the duration baseline. Network is not
inferred from a URL or a mock transport; mark a genuinely external test
explicitly with `pytest.mark.network`.

Do not add a `unit` marker. Tests under `tests/unit/` are already identified by
their directory, and the same test should not be classified twice by default.

Use `tmp_path` or an explicit workspace-local base temp for test artifacts. A
test must close the runtime, store, subprocess, and background tasks it creates.
Polling helpers must have a timeout and report the last observed state.

The measured collection and execution baseline is recorded in
`test-suite-baseline.md`; the node inventory is generated in
`test-inventory.txt`.
