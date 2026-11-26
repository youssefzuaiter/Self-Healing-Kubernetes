## What this changes

<!-- One or two sentences. Link the spec section this implements, if any. -->

## Stage

<!-- Housekeeping, or a Stage 2 gate (2A, 2B, 2C). Passing tests does not authorize the next gate. -->

## Checks

<!-- Tick what you ran and paste the commands, exit codes and counts below. See CONTRIBUTING.md. -->

- [ ] Test suite exited 0 with `1070 passed`: `python -B -m pytest -v -p no:cacheprovider`
- [ ] Frozen-file check exited 0: `git diff --exit-code --stat stage-1b-frozen -- CLAUDE.md docs/SPEC_01_SIMULATOR.md docs/SPEC_02_BASELINES.md docs/DECISIONS.md sim tests tools/mutation_check.py`
- [ ] If `rl/` changed, Stage 2 tests exited 0: `python -B -m pytest -v -p no:cacheprovider -o python_files='check_*.py' rl/tests`
- [ ] If dependencies or anything the simulator relies on changed, the mutation check exited 0: `python tools/mutation_check.py`
- [ ] The Tests workflow passes on Python 3.11, 3.12 and 3.13

```text
Paste command output here.
```

## Decisions

<!-- Every choice the specs left open, and where it is recorded (SPEC_03 Section 7.1 for Stage 2). Write "None" if there are none. -->

## Not verified

<!-- Anything you could not check, stated plainly. Write "None" if everything was verified. -->
