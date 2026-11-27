# Changelog

Notable changes to this project. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The project has no numbered
releases; milestones are git tags.

## [Unreleased]

### Added

- Stage 2 specification `docs/SPEC_03_RL_TRAINING.md`, version `1.0.0-rc4`, awaiting audit
  sign-off. It defines the RL ablation ladder; no Stage 2 code exists yet (`2b1b937`,
  pull request #1).
- `README.md` explaining the project, results and how to run it (`d9bbec0`).
- MIT `LICENSE` and `.gitignore` (`e8555c0`).
- `requirements.txt` pinning NumPy, Gymnasium and pytest (`33a4afc`).
- GitHub Actions workflow running the test suite on Python 3.11, 3.12 and 3.13 (`7237b2b`).
- `CONTRIBUTING.md` with the project's rules and checks (`1217888`).
- Pull request template (`6507ed1`).
- Issue forms for bugs, spec problems and Stage 2 tasks (`4b09d42`).
- `CODE_OF_CONDUCT.md`, Contributor Covenant 2.0 (`a4cf19e`).
- `SECURITY.md` for private vulnerability reports (`696392c`).
- This changelog, and a step in `CONTRIBUTING.md` to keep it updated.
- `CITATION.cff` with citation metadata.

### Changed

- `requirements.txt` pins NumPy 2.4.6 on Python 3.11, because NumPy 2.5.5 needs Python
  3.12 or later. CI now installs from it on every Python version (`50b4153`).

## [stage-1b-frozen] - 2026-11-25

The audited Stage 1A and Stage 1B baseline, committed as `be74326`. These 14 files never
change afterwards. Their development history predates the git repository; the choices made
along the way are recorded in `docs/DECISIONS.md` as IMPL-01 to IMPL-48.

### Added

- Stage 1A: deterministic five-service simulator, safety shield and `ApprovedAction`
  tokens (`sim/config.py`, `sim/env.py`, `sim/shield.py`), specified in
  `docs/SPEC_01_SIMULATOR.md`, with 500 tests and the mutation check
  `tools/mutation_check.py`.
- Stage 1B: seven workloads, three non-RL baselines and a runner that writes JSON
  manifests (`sim/workloads.py`, `sim/baselines.py`, `sim/runner.py`), specified in
  `docs/SPEC_02_BASELINES.md`, with 570 tests.

[Unreleased]: https://github.com/youssefzuaiter/Self-Healing-Kubernetes/compare/stage-1b-frozen...main
[stage-1b-frozen]: https://github.com/youssefzuaiter/Self-Healing-Kubernetes/tree/stage-1b-frozen
