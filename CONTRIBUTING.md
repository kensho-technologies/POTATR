# Contributing

Thank you for taking the time to contribute to this project!

## Code of Conduct

This project adheres to the Contributor Covenant [code of conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold this code.
Please report unacceptable behavior to opensource@kensho.com.

## Development setup

Install the package with its development tools from the lock file, then activate the virtual
environment. The scripts and `just` recipes below call `python` directly, so they need it active:

```bash
uv sync --extra dev
source .venv/bin/activate
```

## Style Guide

- **Lint and types:** `scripts/lint.sh` runs mypy (with strict settings: every function needs
  type annotations), the ruff linter and the ruff formatter check. Run `scripts/lint.sh -f`, or
  `just lint`, to apply ruff's fixes and formatting.
- **Tests:** `scripts/test.sh` runs the tests with coverage (`just unit-test` runs them without).
  Add a test with each bug fix or new feature.
- **Docstrings:** every public function, method and `__init__` needs a docstring, with a one-line
  summary first.
- **Logging:** use the module's `logger`, not `print`.
- **Copyright:** every Python file under `src/`, other than vendored code, starts with
  `# Copyright <year>-present Kensho Technologies, LLC.`, which
  `scripts/copyright_line_check.sh` checks.
- **Vendored code:** `src/potatr/detr/` (DETR) and `src/potatr/teds.py` (TEDS) are third-party
  code. They are excluded from linting and keep their upstream style; see
  [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Keep changes to them minimal, and keep the
  modification notes in their headers and in that file up to date.
- **Results:** if a change affects training or evaluation results, say so in the pull request,
  with the metrics before and after.
