# CLAUDE.md — Engineering Laws & Execution Contract

## 1. Project Identity & Scope
Repository for **Topology-Aware Spatio-Temporal Graph RL Autoscaler for Kubernetes Microservices**.
- **Stage 0:** Specification (`CLAUDE.md`, `docs/DECISIONS.md`, `docs/SPEC_01_SIMULATOR.md`).
- **Stage 1A:** Validated 5-Node Discrete-Time Fluid Simulator + Deterministic Safety Shield + Unit Tests.
- **Stage 1B:** Workloads + Non-RL Baselines + Reproducibility Manifests.
- **Stage 2:** Progressive RL Ablation Ladder.

**Current stage: 1A.** Passing tests do not authorize Stage 1B. Wait for explicit
authorization from the principal engineer before advancing.

## 2. Non-Negotiable Engineering Rules
1. **Stage Isolation:** Do NOT import `torch`, `stable_baselines3`, or `kubernetes` in Stage 1A/1B.
2. **Zero Fabricated Results:** Never mock benchmark outputs or claim a test passed unless `pytest` returned `0`.
3. **Never Weaken Tests:** Fix the physics, do not relax assertions.
4. **Action-Safety Invariant:** `env.step()` must verify a one-use, step-bound `ApprovedAction` token and assert all invariants.

## 3. Mandatory Reporting Format
Use this structure for Stage 1A reports. It covers the files, commands, test
status, invariants and raw code audit this project requires.

- **A. Files Created or Modified:** Every exact path.
- **B. Architecture Summary:** Simulator state, queue processing, replica lifecycle, shield arbitration, token validation, metric calculations.
- **C. Test Execution:** Exact command, actual exit code, tests collected, passed, failed, skipped, xfailed, warnings.
- **D. Invariant Verification:** Evidence for flow conservation, error-accounting bounds, queue capacity, replica lifecycle, cluster quota, cooldown, token enforcement, observation and reward validity, reproducibility.
- **E. Known Limitations:** Approximations, incomplete behavior, risks.
- **F. Scope Confirmation:** Confirm no Stage 1B or later work was added.
- **G. Raw Source Code:** Full contents of `sim/env.py`, `sim/shield.py`, `tests/test_stage1a_physics.py`. If output limits prevent this, state exactly which file was not fully included.

## 4. Sources of Truth
- `docs/SPEC_01_SIMULATOR.md` is authoritative. Do not alter physics or formulas silently.
- `docs/DECISIONS.md` records every choice the spec leaves open. Add to it rather than deciding silently.
- If requirements contradict each other, or a necessary decision is missing, stop and report before coding.

## 5. Layout
| Path | Purpose |
|---|---|
| `sim/config.py` | Frozen `SimulatorConfig`; defaults reproduce the spec tables |
| `sim/shield.py` | Action conversion, arbitration, `ApprovedAction`, `SafetyShield` |
| `sim/env.py` | `MicroserviceMeshEnv`: physics, lifecycle, metrics, token validation |
| `tests/test_stage1a_physics.py` | Stage 1A acceptance tests |
| `tools/mutation_check.py` | Reproducible mutation check; run with `python tools/mutation_check.py` |

Controllers call `env.step(env.shield.approve(raw_proposal))`. Raw proposals never reach `step()`.

## 6. Working Rules for Future Sessions
- Permitted dependencies: NumPy, Gymnasium, pytest.
- Run tests with `pytest -v` from the repository root.
- Keep diagnostics raising `SimulationInvariantError`. Never clip internal state to hide a bug. Clipping is only for normalized observation features and documented action conversion.
- Do not create `sim/baselines.py` or `sim/workloads.py` before Stage 1B is authorized.
