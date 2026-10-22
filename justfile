set dotenv-load

# Just is a command runner. It can be used to provide a shorthand to run
# complex commands, for example `just lint` for `python -m ruff...`

# Here are some installation instructions:
# https://github.com/casey/just?tab=readme-ov-file#installation
# Once installed, you can run any of the commands in this file by
# prepending them with `just`, for example `just lint` to run the linter.
# Run `just` shows a list of all available commands

default:
    just --list

alias l := lint
# Lint the app directory (both lint and l work). For verbose, use `just lint --verbose`
lint *args:
    python -m mypy --config-file pyproject.toml src/potatr {{args}}
    # The ruff linters (check) and formatters (format) are separate.
    # See https://docs.astral.sh/ruff/formatter/#sorting-imports
    python -m ruff --config pyproject.toml check src/potatr --fix {{args}}
    python -m ruff --config pyproject.toml format src/potatr {{args}}

alias t := unit-test
# Run unit tests. Use args for optional settings
unit-test *args:
    python -m pytest {{args}}

# Build the sphinx documents locally
sphinx *args:
    sphinx-build docs docs/output
