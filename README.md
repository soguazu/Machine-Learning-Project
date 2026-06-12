# boilerplate-ds

A reusable, opinionated **data-science project starter** managed with
[uv](https://docs.astral.sh/uv/). Clone it, rename it, and start analyzing —
the structure, dependencies, and tooling are already wired up so you can focus
on the data instead of the plumbing.

---

## Table of contents

- [Features](#features)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Project structure](#project-structure)
- [Recommended workflow](#recommended-workflow)
- [Command reference](#command-reference)
- [Managing dependencies](#managing-dependencies)
- [Testing](#testing)
- [Linting & formatting](#linting--formatting)
- [Data & model handling](#data--model-handling)
- [Configuration & secrets](#configuration--secrets)
- [Using this as a template for a new project](#using-this-as-a-template-for-a-new-project)
- [Troubleshooting](#troubleshooting)

---

## Features

- **One-command environment** — `uv sync` builds a reproducible virtualenv from
  a locked dependency graph (`uv.lock`).
- **Sensible DS layout** — separate places for raw vs. processed data,
  exploratory notebooks, reusable source code, models, and reports.
- **Notebooks stay thin** — reusable logic lives in `src/`, imported into
  notebooks so analysis is testable and not trapped in cells.
- **Linting & formatting** — [Ruff](https://docs.astral.sh/ruff/) preconfigured
  for fast linting and formatting.
- **Testing** — [pytest](https://docs.pytest.org/) ready out of the box.
- **Safe defaults** — data, models, secrets, and generated figures are
  gitignored so you never accidentally commit large or sensitive files.

## Tech stack

| Purpose            | Tool                                              |
| ------------------ | ------------------------------------------------- |
| Package / env mgmt | [uv](https://docs.astral.sh/uv/)                  |
| Python             | >= 3.14.6                                         |
| Numerics           | NumPy                                             |
| Dataframes         | pandas                                            |
| Plotting           | Matplotlib, seaborn                               |
| Modeling           | scikit-learn                                      |
| Notebooks          | Jupyter, ipykernel                                |
| Lint / format      | Ruff                                              |
| Testing            | pytest                                            |

## Prerequisites

- **uv** installed ([install guide](https://docs.astral.sh/uv/getting-started/installation/)):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- A compatible Python (>= 3.14.6). uv can install it for you automatically;
  the pinned version lives in [`.python-version`](.python-version).

## Quick start

```bash
# 1. Install all runtime + dev dependencies into .venv
uv sync

# 2. (optional) activate the environment
source .venv/bin/activate
#    ...or just prefix every command with `uv run`

# 3. Launch Jupyter and start exploring
uv run jupyter lab
```

That's it — you now have a fully isolated, reproducible environment.

## Project structure

```
boilerplate-ds/
├── data/
│   ├── raw/         # Immutable source data — never edit by hand (gitignored)
│   ├── interim/     # Partially cleaned / transformed data (gitignored)
│   └── processed/   # Final, analysis/model-ready datasets (gitignored)
├── notebooks/       # Exploratory Jupyter notebooks
├── src/             # Reusable, importable Python package (data prep, features, models)
├── models/          # Serialized trained models & artifacts (gitignored)
├── reports/
│   └── figures/     # Exported plots & figures (gitignored)
├── tests/           # pytest test suite
├── pyproject.toml   # Project metadata, dependencies, tool config
├── uv.lock          # Fully resolved, pinned dependency graph (commit this)
├── .python-version  # Pinned Python version
└── .gitignore
```

**Why this split?**

- `data/raw` is treated as read-only — every transformation reads from `raw`
  and writes to `interim`/`processed`, so you can always reproduce results from
  scratch.
- `src/` holds the real logic. Import it into notebooks
  (`from src.features import build_features`) so code is reusable and testable.
- `notebooks/` is for exploration and storytelling, not for hoarding
  production logic.

## Recommended workflow

1. **Drop source data** into `data/raw/` (or write a loader in `src/` that
   fetches it).
2. **Explore** in a notebook under `notebooks/`.
3. **Promote** reusable steps (cleaning, feature engineering, model code) out of
   the notebook into modules under `src/`.
4. **Write tests** in `tests/` for the logic in `src/`.
5. **Persist** processed datasets to `data/processed/` and trained models to
   `models/`.
6. **Export** final figures to `reports/figures/`.

## Command reference

```bash
uv sync                       # Install/update env from pyproject + lock
uv run jupyter lab            # Launch JupyterLab
uv run python src/script.py   # Run a script inside the env
uv run pytest                 # Run the test suite
uv run ruff check .           # Lint
uv run ruff check --fix .     # Lint and auto-fix
uv run ruff format .          # Format code
uv add <pkg>                  # Add a runtime dependency
uv add --dev <pkg>            # Add a dev-only dependency
uv remove <pkg>               # Remove a dependency
uv lock --upgrade             # Upgrade locked versions
```

## Managing dependencies

Dependencies are declared in [`pyproject.toml`](pyproject.toml) and pinned in
`uv.lock`. **Always** use uv to change them so the lockfile stays in sync:

```bash
uv add pandas-stubs           # runtime dep
uv add --dev mypy             # dev/tooling dep
```

Commit both `pyproject.toml` and `uv.lock` after any change — the lockfile is
what makes the environment reproducible for collaborators and CI.

## Testing

Tests live in `tests/` and are run with pytest:

```bash
uv run pytest                 # all tests
uv run pytest tests/test_x.py # a single file
uv run pytest -k "feature"    # tests matching an expression
uv run pytest -q              # quiet output
```

Test discovery is configured in `[tool.pytest.ini_options]` in
`pyproject.toml`.

## Linting & formatting

[Ruff](https://docs.astral.sh/ruff/) handles both. Config lives under
`[tool.ruff]` in `pyproject.toml` (line length 100; rule sets `E, F, I, W, UP,
B`).

```bash
uv run ruff check .           # report issues
uv run ruff check --fix .     # auto-fix what it can
uv run ruff format .          # format the codebase
```

## Data & model handling

By default the following are **gitignored** so they never end up in version
control:

- everything under `data/raw`, `data/interim`, `data/processed`
- `models/`
- `reports/figures/`
- loose `*.csv`, `*.parquet`, `*.pkl`, `*.joblib`, `*.h5` files

For sharing or versioning large artifacts, use external storage
(e.g. [DVC](https://dvc.org/), S3, or a data registry) rather than committing
them to git.

## Configuration & secrets

Put secrets (API keys, DB URLs, etc.) in a `.env` file at the project root —
it's gitignored. Load it in code with something like:

```python
import os
# pip/uv add python-dotenv if you want automatic .env loading
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ["API_KEY"]
```

Never commit real credentials. Consider checking in a `.env.example` with blank
placeholders to document required variables.

## Using this as a template for a new project

1. Copy or clone this directory under a new name.
2. In [`pyproject.toml`](pyproject.toml), update `name`, `version`, and
   `description`.
3. Update this README's title and overview.
4. (Optional) reset git history:
   ```bash
   rm -rf .git && git init && git add -A && git commit -m "Initial commit"
   ```
5. Run `uv sync` and start working.

## Troubleshooting

- **`uv: command not found`** — install uv (see [Prerequisites](#prerequisites))
  and restart your shell.
- **Wrong Python / resolution errors** — make sure the version in
  `.python-version` is available. `uv python install 3.14.6` will fetch it.
- **`uv` operating on the wrong folder** — confirm your terminal's working
  directory with `pwd`; a shell left inside a moved/deleted folder will run
  commands in the wrong place.
- **Notebook can't import from `src/`** — launch Jupyter from the project root
  (`uv run jupyter lab`) so the working directory is correct, or add the root to
  `sys.path`.
- **Stale environment after editing `pyproject.toml`** — re-run `uv sync`.
```
