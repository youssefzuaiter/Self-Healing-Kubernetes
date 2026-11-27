# Contributing

This project is built in audited stages. A change is welcome when it keeps the frozen
baseline intact, follows the specifications, and comes with honest evidence that it works.
Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ground rules

1. **The frozen baseline never changes.** The 14 files tagged `stage-1b-frozen` must stay
   byte-identical: `CLAUDE.md`, `docs/SPEC_01_SIMULATOR.md`,
   `docs/SPEC_02_BASELINES.md`, `docs/DECISIONS.md`, everything in `sim/` and `tests/`, and
   `tools/mutation_check.py`. The tag is never moved.
2. **Specifications come first.** The specs in `docs/` are the source of truth. If a change
   needs a formula or behavior the specs don't define, or two requirements conflict, stop
   and raise it before writing code.
3. **One stage at a time.** Passing tests does not authorize the next stage. Stage 2 is
   split into gates 2A, 2B and 2C ([SPEC_03 Section 6.5](docs/SPEC_03_RL_TRAINING.md)), and
   each needs explicit approval.
4. **Never weaken a test.** If a test fails, fix the code, not the assertion.
5. **Never fabricate results.** Don't claim a test passed unless `pytest` exited with code
   0, and don't report a number the code in the repository can't reproduce.
6. **Don't hide bugs.** Internal checks raise `SimulationInvariantError`. Clipping is only
   allowed for normalized observation features and the documented action conversion.

## Setting up

```bash
python3 -m venv ~/.venvs/autoscaler
source ~/.venvs/autoscaler/bin/activate
pip install -r requirements.txt
```

`requirements.txt` works on Python 3.11, 3.12 and 3.13. A `.venv/` inside the repository
is also fine, because `.gitignore` excludes it.

## Checks to run before every pull request

Run the test suite. Expected: `1070 passed`, with no failures or skips.

```bash
python -B -m pytest -v -p no:cacheprovider
```

Confirm the frozen files are untouched. Exit code 0 means they match the tag. Stage or
commit new files first, because `git diff` doesn't see untracked files.

```bash
git diff --exit-code --stat stage-1b-frozen -- CLAUDE.md docs/SPEC_01_SIMULATOR.md docs/SPEC_02_BASELINES.md docs/DECISIONS.md sim tests tools/mutation_check.py
```

If you touched anything the simulator's behavior depends on, also run the mutation check.
It takes about 9 minutes on an Apple M2, and exit code 0 means every mutant was caught.

```bash
python tools/mutation_check.py
```

## Where changes go

| Change | Location |
|---|---|
| Stage 2 RL code | `rl/` (layout in SPEC_03 Section 6.1) |
| Stage 2 tests | `rl/tests/check_*.py`, run as a separate pytest process |
| Stage 2 dependencies | `requirements-rl.txt`, never `requirements.txt` |
| Stage 2 decisions | SPEC_03 Section 7.1, because `docs/DECISIONS.md` is frozen |

Stage 2 tests must not go in `tests/`. A frozen Stage 1A test checks that `torch` is never
imported in its process, and the mutation check copies only `sim/` and `tests/`. Run the
Stage 2 tests with:

```bash
python -B -m pytest -v -p no:cacheprovider -o python_files='check_*.py' rl/tests
```

`sim/` must never import `rl/` or `torch`, and `rl/` uses only the public frozen API.

## Reporting problems

Open an issue with one of the [issue forms](.github/ISSUE_TEMPLATE): a bug report, a spec
problem, or a Stage 2 task. Report a spec contradiction or missing decision before writing
any code that depends on it. Report security problems privately, as described in
[SECURITY.md](SECURITY.md), never in a public issue.

## Workflow

1. Branch from `main`. Stage 2 work happens on `stage-2-rl`.
2. Make small commits. The subject line says what changed, in the imperative mood.
3. Add a line for each notable change under `Unreleased` in [CHANGELOG.md](CHANGELOG.md).
4. Open a pull request into `main`. The [Tests workflow](.github/workflows/tests.yml) must
   pass on Python 3.11, 3.12 and 3.13.
5. Never force-push `main` or `stage-2-rl`.

## What to include in a pull request

The [pull request template](.github/pull_request_template.md) fills in these sections
for you.

- The exact commands you ran, with exit codes and test counts
- Which files changed, and confirmation that the frozen-file check passed
- Every decision the specs left open, recorded where the table above says
- Anything you couldn't verify, stated plainly

## Licence

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
