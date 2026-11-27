# Topology-Aware Spatio-Temporal Graph RL Autoscaler for Kubernetes Microservices

[![Tests](https://github.com/youssefzuaiter/Self-Healing-Kubernetes/actions/workflows/tests.yml/badge.svg)](https://github.com/youssefzuaiter/Self-Healing-Kubernetes/actions/workflows/tests.yml)

A deterministic simulator of a five-service microservice application, a safety shield,
and three non-RL autoscaling baselines. It is the test bed for a planned graph-based
reinforcement learning autoscaler. **No RL code exists yet, and nothing here runs on a
real Kubernetes cluster.**

## Status

| Stage | Content | State |
|---|---|---|
| 1A | Simulator physics, safety shield, 500 tests | Done, audited, frozen |
| 1B | Workloads, baselines, reproducible runner, 570 tests | Done, audited, frozen |
| 2 | RL ablation ladder: [specification](docs/SPEC_03_RL_TRAINING.md) | Spec `1.0.0-rc4`, awaiting audit sign-off. No code |

The frozen Stage 1A/1B code is tagged `stage-1b-frozen`. Stage 2 work happens on the
`stage-2-rl` branch.

## Problem

Scaling one microservice changes the load on the services it calls. Here a Gateway calls
Auth, Catalog and Order, and all three call a Database that cannot scale. New pods take
20 s to boot, scale-downs are blocked for 30 s after a scale-up, and the whole
application shares a 28-pod quota.

The research question: can a policy that sees the call graph and the recent past scale
upstream services while accounting for the downstream bottleneck, the boot delay and the
cost of capacity left over after a burst?

## Approach

### Stage 1A: simulator and safety shield

- **Time:** 1-second substeps, 5 per control step. An episode is 360 control steps
  (30 simulated minutes).
- **Traffic flow:** each completed request calls downstream services with fixed
  multiplicities (Gateway → Auth 1.0, Catalog 0.8, Order 0.2; each → Database). Calls
  arrive one substep later.
- **Queues:** FIFO with 5 age buckets. Work older than 5 s times out, and half of it
  retries. Arrivals beyond queue capacity overflow.
- **Database contention:** Auth, Catalog and Order slow down once the database queue is
  more than 50% full.
- **Replicas:** separate ready, pending and terminating pods. Boot takes 20 s,
  termination 5 s. The quota of 28 pods includes the database pod.
- **Observation:** 9 features per node, a `(5, 9)` array clipped to `[0, 1]`.
- **Action:** a replica change of −1, 0, +1 or +2 for each of the four scalable services.
- **Reward:** the negative of a weighted cost (SLO violation 2.0, failure 5.0,
  resources 0.5, scaling 0.2).
- **Safety shield:** every proposal goes through `env.shield.approve(raw)`. It enforces
  replica bounds, cooldown and the quota, then issues a one-use token bound to that step.
  `env.step()` accepts only that token and re-checks the action itself.

Full formulas: [docs/SPEC_01_SIMULATOR.md](docs/SPEC_01_SIMULATOR.md).

### Stage 1B: workloads, baselines and runner

- **7 workloads:** `constant`, `step_jump`, `gradual_ramp`, `flash_crowd`,
  `burst_and_drop`, `db_bottleneck`, `seeded_bursty_poisson`. Traces are immutable and
  identified by SHA-256.
- **3 baselines:** `static` (never scales), `sim_hpa` (a proportional utilization rule
  inspired by the Kubernetes HPA, not the real HPA) and `tuned_reactive_threshold`.
- **Runner:** runs every controller on every workload and writes one strictly
  JSON-native manifest per run. Each manifest records the config, seed, git commit,
  workload hash and metrics. All inputs are checked before anything runs or is written.

Details: [docs/SPEC_02_BASELINES.md](docs/SPEC_02_BASELINES.md).

## Results

Total episode reward for each baseline on each workload. Higher is better; every value is
negative because it is a cost. Rewards come from `python -m sim.runner` at commit
`be74326` with default settings. The best value in each row is in bold.

| Workload | `static` | `sim_hpa` | `tuned_reactive_threshold` |
|---|---:|---:|---:|
| constant | −57.86 | **−52.52** | **−52.52** |
| step_jump | −4099.87 | −4069.89 | **−4007.37** |
| gradual_ramp | −3861.33 | **−2702.69** | −2720.83 |
| flash_crowd | **−415.10** | −433.86 | −431.70 |
| burst_and_drop | **−700.74** | −715.84 | −710.98 |
| db_bottleneck | −1653.44 | −1543.36 | **−1542.73** |
| seeded_bursty_poisson | −2407.53 | **−2275.33** | −2287.59 |

- On `flash_crowd` and `burst_and_drop` the scaling baselines cost more than never
  scaling. A phase analysis showed that both reach the 28-pod quota before the first
  extra pod they requested is ready.
- The database limits the whole application to about 333 requests/s when every call is
  made (0.9 database calls per request, 300 calls/s capacity). Extra upstream capacity
  beyond that changes where requests queue and fail, not how many are served.
- Latency and failure are modeled proxies from the simulator, not measurements from a
  real system.

## How to run

The tests pass on Python 3.11, 3.12 and 3.13. `requirements.txt` pins Gymnasium 1.4.0,
pytest 9.2.0 and NumPy 2.5.5, or NumPy 2.4.6 on Python 3.11, because 2.5.5 needs 3.12 or
later. Create the environment outside the repository or in `.venv/`, which `.gitignore`
excludes. Any other untracked file makes the runner record the working tree as `dirty`.

```bash
python3 -m venv ~/.venvs/autoscaler
source ~/.venvs/autoscaler/bin/activate
pip install -r requirements.txt
```

Run the test suite. Expected: `1070 passed`.

```bash
python -B -m pytest -p no:cacheprovider
```

GitHub Actions runs the same command on Ubuntu with Python 3.11, 3.12 and 3.13 for every
push to `main` or `stage-2-rl` and for every pull request
([`.github/workflows/tests.yml`](.github/workflows/tests.yml)).

Print the benchmark table above:

```bash
python -B -m sim.runner
```

Write the 21 JSON manifests (3 controllers × 7 workloads) to a directory outside the
repository. Existing manifest files are never overwritten.

```bash
python -B -m sim.runner --output-dir ~/autoscaler-manifests
```

Run the mutation check, which edits a temporary copy of the code 156 ways and confirms
the tests catch every change. Exit code 0 means every mutant was caught. It takes about
9 minutes on an Apple M2.

```bash
python tools/mutation_check.py
```

## Project structure

```
CLAUDE.md                         rules for the AI coding assistant
docs/SPEC_01_SIMULATOR.md         Stage 1A simulator specification
docs/SPEC_02_BASELINES.md         Stage 1B workloads, baselines and runner
docs/DECISIONS.md                 48 implementation decisions (IMPL-01 to IMPL-48)
docs/SPEC_03_RL_TRAINING.md       Stage 2 specification
sim/config.py                     frozen, validated SimulatorConfig
sim/env.py                        MicroserviceMeshEnv: physics, lifecycle, reward
sim/shield.py                     action conversion, arbitration, ApprovedAction tokens
sim/workloads.py                  the 7 workload generators
sim/baselines.py                  BaseController and the 3 baselines
sim/runner.py                     benchmark runner and JSON manifests
tests/test_stage1a_physics.py     Stage 1A tests
tests/test_stage1b_baselines.py   Stage 1B tests
tools/mutation_check.py           reproducible mutation check
requirements.txt                  pinned direct dependencies
.github/workflows/tests.yml       CI: the test suite on Python 3.11 to 3.13
.github/pull_request_template.md  checklist pre-filled into new pull requests
.github/ISSUE_TEMPLATE/           forms for bugs, spec problems and Stage 2 tasks
LICENSE                           MIT licence
CONTRIBUTING.md                   rules and checks for contributors
CODE_OF_CONDUCT.md                Contributor Covenant 2.0
SECURITY.md                       how to report a vulnerability privately
```

## How the work was done

The project was built in stages. Each stage started from a written specification, and an
independent audit had to approve it before the next one began. Every choice the
specifications left open is recorded in [docs/DECISIONS.md](docs/DECISIONS.md).

1. **Stage 1A:** built the simulator, shield and tests from the specification
   (IMPL-01 to IMPL-29).
2. **Corrective audit:** floored failure ratios at a denominator of 1, added validation for
   every config field, and added the mutation check (IMPL-30 to IMPL-32).
3. **Stage 1B:** added workloads, baselines and the runner (IMPL-33 to IMPL-41).
4. **Stage 1B audits:** made workload traces truly immutable, made manifest writing safe
   against overwrites and partial output, read the git state once per run, rejected
   numbers too large for a float and anything that is not JSON-native, and checked every
   input before running (IMPL-42 to IMPL-48).
5. **Freeze:** committed the 14 audited files and tagged them `stage-1b-frozen`.
6. **Stage 2 specification:** drafted
   [SPEC_03](docs/SPEC_03_RL_TRAINING.md)
   and revised it through four release candidates after a dual audit.

## Stage 2 plan

The specification defines five primary PPO models trained on the frozen simulator with
identical data, parameter budgets (about 53,000) and tuning effort:

| Model | Inputs | Purpose |
|---|---|---|
| M1 flat MLP | current observation | feedforward floor |
| M2 flat MLP | last 8 observations | history without a graph |
| M3 GATv2 | last 8 observations, call graph | graph message passing |
| M3-self-loop | same as M3, self-loops only | M3 without message passing |
| M4 GATv2 + GRU | current observation, recurrent state | recurrent memory |

The two primary questions are M3 against M3-self-loop (does message passing help?) and
M4 against M3 (does recurrence beat a fixed window?). The shield stays in the loop during
training. The campaign is labelled exploratory, because no power calculation exists.

## Limitations

- **Simulation only:** there is no Kubernetes integration, and latency and failure are
  modeled proxies.
- **Default config:** results and the Stage 2 analysis assume the default configuration.
- **Untested platforms:** only macOS was tested. Python 3.10 was not tested.
- **Partial pinning:** `requirements.txt` pins only the three direct dependencies, not
  the packages they pull in.
- **Stale `CLAUDE.md`:** it still says the current stage is 1A, but it is a frozen file.
- **Diagnostics script:** the phase analysis behind the quota timing in Results came from
  a script that is not in the repository.

## License

MIT. See [LICENSE](LICENSE).

## Credits

Implementation written with Claude Code, an AI coding assistant, working from stage-gated
specifications and independent audits.
