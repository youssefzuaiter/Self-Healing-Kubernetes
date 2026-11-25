# SPEC 02 — Stage 1B Workloads, Baselines and Runner

Topology-Aware Spatio-Temporal Graph RL Autoscaler for Kubernetes Microservices.

Version 1.0.0. Stage 1B builds on the frozen Stage 1A simulator
(`docs/SPEC_01_SIMULATOR.md`, version 1.0.0) and does not modify it. Choices this
document leaves open are recorded in `docs/DECISIONS.md` as IMPL-33 onward.

Files: `sim/workloads.py`, `sim/baselines.py`, `sim/runner.py`,
`tests/test_stage1b_baselines.py`. Forbidden imports: `torch`, `torch_geometric`,
`stable_baselines3`, `kubernetes`, `prometheus`, `prometheus_client`, `locust`.

---

## 0. Stage 1A interfaces verified before implementation

Checked against the frozen source. No discrepancy was found.

| Assumption | Frozen source |
|---|---|
| `MicroserviceMeshEnv(traffic_trace=None, config=None)` | `sim/env.py` `__init__` |
| `env.reset(seed=...)` returns `(observation, info)` | `sim/env.py` `reset` |
| `env.step(token)` returns `(observation, reward, terminated, truncated, info)` | `sim/env.py` `step` |
| `env.config`, `env.step_index` | public attribute and property |
| `env.shield.approve(raw)` returns `ApprovedAction` with `.deltas` | `sim/shield.py` |
| `convert_proposal(raw, config)` returns a tuple or `None` | `sim/shield.py` |
| Feature 2 queue pressure, 3 interval utilization, 4 `ready / r_max`, 5 `pending / r_max` | `sim/env.py` `_build_features` |
| `info["reward_components"]` keys `slo_violation`, `failure_cost`, `resource_cost`, `scaling_cost`, `total_cost` | `sim/env.py` `_reward` |
| Weights `weight_slo`, `weight_failure`, `weight_resources`, `weight_scaling` | `sim/config.py` |
| `terminated` is `True` after `control_horizon` steps; `truncated` is always `False` | `sim/env.py` `step` |

### 0.1 Shield fallback counter semantics

`SafetyShield._fallback_count` is incremented once per rejected proposal inside
`approve()` and reset only by `shield.reset()`, which `env.reset()` calls. So
`info["fallback_count"]` is **cumulative within an episode**: at step k it is the
number of fallbacks in steps 0 to k. The runner reports the value from the last
step and never sums the per-step values.

### 0.2 Reward accounting

`_reward()` returns `r_k = -J_k` with
`J_k = weight_slo·V_k + weight_failure·E_k + weight_resources·C_res,k + weight_scaling·C_scale,k`
and reports the unweighted `V_k`, `E_k`, `C_res,k`, `C_scale,k` and `J_k` in
`info["reward_components"]`. So, up to float rounding, the sum of the weighted
components equals the summed `total_cost`, and the summed `total_cost` equals
`-total_reward`.

---

## 1. Workload suite (`sim/workloads.py`)

### 1.1 `WorkloadTrace`

A frozen dataclass `WorkloadTrace(name, params, values, sha256="")`.

- `values` is copied into a C-contiguous little-endian `float64` (`"<f8"`) 1-D
  array of shape `(horizon,)`, with `horizon >= 1`. Every entry must be finite and
  non-negative. Booleans, complex numbers, strings and objects are rejected. The
  validated array is converted to an immutable `bytes` object. Every access to
  `values` returns a new `np.frombuffer` array over those bytes, so writing can
  never be re-enabled, and changing the `dtype`, `shape` or `strides` of a returned
  array does not affect the trace.
- `params` is deep-copied and frozen. Mappings become `MappingProxyType`, while
  lists, tuples and sets become tuples or frozensets, recursively. Keys must be
  plain strings. Leaves must be exactly `None`, `bool`, `int`, `float` or `str`;
  NumPy scalars and arrays are converted to these. Anything else, including
  `bytearray`, `bytes`, subclasses of those types and custom objects, raises
  `TypeError`.
- `sha256 = hashlib.sha256(values.tobytes()).hexdigest()`, computed from the frozen
  array. A caller-supplied `sha256` must match it, or construction fails.
- Other invalid input raises `ValueError`.

### 1.2 Generators

Every generator validates its parameters. Rates must be finite, non-negative real
numbers, not booleans. Integer parameters must be integers, not booleans or
floats. `horizon >= 1`. Time windows must be non-empty and lie inside
`[0, horizon)`. Invalid input raises `ValueError`. `t` below is the substep index,
and `values[t]` is in requests per second.

| Name | Signature | Definition |
|---|---|---|
| `constant` | `(rate=100.0, horizon=1800)` | `rate` for every `t` |
| `step_jump` | `(base_rate=100.0, jump_rate=600.0, jump_time=300, horizon=1800)` | `base_rate` for `t < jump_time`, else `jump_rate`. `0 <= jump_time < horizon` |
| `gradual_ramp` | `(start_rate=100.0, end_rate=700.0, horizon=1800)` | `np.linspace(start_rate, end_rate, horizon)` |
| `flash_crowd` | `(base_rate=100.0, peak_rate=900.0, start_time=300, duration=120, horizon=1800)` | `peak_rate` on `[start_time, start_time + duration)`, else `base_rate` |
| `burst_and_drop` | `(base_rate=100.0, peak_rate=800.0, ramp_start=300, ramp_duration=60, hold_duration=180, horizon=1800)` | see 1.3 |
| `db_bottleneck` | `(base_rate=100.0, surge_rate=450.0, start_time=300, duration=600, horizon=1800)` | `surge_rate` on `[start_time, start_time + duration)`, else `base_rate` |
| `seeded_bursty_poisson` | `(base_rate=150.0, burst_rate=750.0, burst_prob=0.02, min_burst_len=20, max_burst_len=60, seed=42, horizon=1800)` | see 1.4 |

### 1.3 `burst_and_drop`

With `r = ramp_start`, `d = ramp_duration >= 1` and `h = hold_duration >= 1`,
requiring `r + d + h <= horizon`:

- `[0, r)`: `base_rate`.
- `[r, r + d)`: `base_rate + (peak_rate - base_rate) * (t - r) / d`, a linear ramp
  sampled at whole seconds, starting at `base_rate`.
- `[r + d, r + d + h)`: `peak_rate`.
- `[r + d + h, horizon)`: `base_rate`, an immediate drop.

### 1.4 `seeded_bursty_poisson`

`burst_prob` is in [0, 1], `1 <= min_burst_len <= max_burst_len`, and `seed` is an
integer `>= 0`. It uses an isolated `np.random.default_rng(seed)`; the global NumPy
random state is never read or written. For each `t` in order:

1. If no burst is active, draw `u = rng.random()`. If `u < burst_prob`, draw the
   burst length `rng.integers(min_burst_len, max_burst_len, endpoint=True)`. The
   burst covers `t` and the following substeps, and is cut off at the horizon.
2. Draw `rng.poisson(lam)`, with `lam = burst_rate` during a burst and `base_rate`
   otherwise, and store it as `float64`.

### 1.5 `db_bottleneck` and upstream capacity

Per external request, the database receives `1.0·0.3 + 0.8·0.5 + 0.2·1.0 = 0.9`
calls. A surge of 450 req/s therefore implies `405 > 300` req/s of database
demand, but **only if every upstream service passes the full surge**. That needs
at least 5 Gateway, 4 Auth, 5 Catalog and 2 Order ready pods. With the default 2
Gateway pods (200 req/s), the database sees at most 180 req/s. How much this trace
stresses the database therefore depends on how far the controller scales upstream.

### 1.6 Benchmark suite

`benchmark_suite(seed=42)` returns the six deterministic traces with default
parameters, followed by `seeded_bursty_poisson(seed=seed)`.

---

## 2. Controllers (`sim/baselines.py`)

### 2.1 Interface

`BaseController` is an abstract base class with:

- `name: str`, a property.
- `params: dict[str, Any]`, a property returning a new copy on every access.
- `reset() -> None`.
- `propose(observation, config, step_index) -> tuple[int, int, int, int]`.

A controller reads only `observation["node_features"]` (shape `(5, 9)`), `config`
and `step_index`. It never receives or stores the environment and never calls the
shield. `sim/baselines.py` does not import `sim.env` or `sim.shield`. The runner
passes every proposal through `env.shield.approve()`, which enforces bounds, quota
and cooldowns.

**Step-0 cold-telemetry rule:** when `step_index == 0`, every baseline returns
`(0, 0, 0, 0)`. The reset observation has no interval history.

Hyperparameters are validated at construction. They must be finite real numbers,
not booleans, within the stated ranges, or construction raises `ValueError`.

### 2.2 `StaticBaseline`

Always `(0, 0, 0, 0)`. Name `static`.

### 2.3 `SimHPABaseline(target_utilization=0.70, tolerance=0.10)`

A simulator-side proportional utilization approximation. **It is not the real
Kubernetes Horizontal Pod Autoscaler** and makes no claim to reproduce its
algorithm, metrics pipeline or timing. Name `sim_hpa`. Ranges:
`0 < target_utilization <= 1`, `0 <= tolerance < 1`.

For each scalable node `i` in `(0, 1, 2, 3)`:

1. `desired_i = int(np.rint((obs[i, 4] + obs[i, 5]) * config.max_replicas[i]))`.
   Stage 1A guarantees `desired = ready + pending`, so this recovers the desired
   count, including pending pods.
2. `u_i = float(obs[i, 3])` and `ratio_i = u_i / target_utilization`.
3. If `abs(ratio_i - 1.0) <= tolerance`, propose `0`.
4. Otherwise `raw_target = max(config.min_replicas[i], ceil(desired_i * ratio_i))`,
   `raw_delta = raw_target - desired_i`, clipped to
   `[config.action_min, config.action_max]`.

### 2.4 `TunedReactiveThresholdController`

Signature `(util_high=0.85, util_med=0.70, util_low=0.35, queue_high=0.25, queue_med=0.10)`.
Name `tuned_reactive_threshold`. Requires
`0 <= util_low < util_med < util_high <= 1` and
`0 < queue_med < queue_high <= 1`. For each scalable node `i`, with
`u = obs[i, 3]` and `q = obs[i, 2]`:

- `u > util_high` or `q > queue_high`: `+2`
- else `u > util_med` or `q > queue_med`: `+1`
- else `u < util_low` and `q == 0.0`: `-1`
- else `0`

---

## 3. Runner (`sim/runner.py`)

### 3.1 `run_episode(controller, trace, config=None, seed=0)`

1. `cfg = config if config is not None else SimulatorConfig()`.
2. Check the arguments before running anything: the types of the controller and
   trace, `trace.values.shape == (cfg.internal_horizon,)`, the seed, and that the
   names, params and config are JSON-native (3.3). Wrong types raise `TypeError`,
   and wrong values raise `ValueError`. The runner contains no literal horizon.
   Every length comes from `cfg`.
3. `controller.reset()`. Then build `MicroserviceMeshEnv(traffic_trace=trace.values, config=cfg)`
   and call `env.reset(seed=seed)`.
4. Until `terminated or truncated`, each step does:
   - `raw = controller.propose(obs, env.config, env.step_index)`
   - `converted = convert_proposal(raw, env.config)`
   - `token = env.shield.approve(raw)`
   - `obs, reward, terminated, truncated, info = env.step(token)`
5. Require `control_steps_executed == cfg.control_horizon`, or raise `RuntimeError`.
6. Return the manifest in 3.3.

`shield_arbitration_override_count` counts the steps where `converted is not None`
and `token.deltas != converted`: the proposal was valid but the shield changed it.
`shield_fallback_count` is the last `info["fallback_count"]` (see 0.1).

### 3.2 `run_benchmark_suite(controllers, traces, config=None, seed=0, output_dir=None)`

Runs `run_episode` for every controller and trace, controllers in the outer loop,
and returns the manifests in that order. With `output_dir`, each manifest is
written as
`<controller.name>__<trace.name>__<sha256[:12]>.json`, sorted keys, 2-space
indent, with a trailing newline.

Every controller, every trace, the config and the seed are checked as in 3.1 before
any episode runs, whether or not `output_dir` is given.

With `output_dir`, every output file is also checked before any episode runs, any
directory is created or any file is written. If any check fails, an existing
directory is left unchanged and a missing one is not created:

- Controller names, workload names and file names must be non-empty strings with
  no path separator (`/`, `\`), `..` or null byte. A file name must be UTF-8
  encodable and at most 255 bytes. Otherwise `ValueError`.
- The resolved destination must lie directly in the resolved output directory,
  otherwise `ValueError`.
- Two episodes must not share a file name, ignoring case and Unicode
  normalization, otherwise `ValueError`.
- No destination may already exist, otherwise `FileExistsError` naming it.

Each manifest is serialized and checked to survive a JSON round trip before its
file is opened. The file is opened with mode `"x"`, so a file that appears during
the run raises `FileExistsError` and is not overwritten. If writing fails after
creation, only the file created by that write is removed, and the error is
re-raised. Manifests completed earlier in the run are kept.

### 3.3 Manifest (schema 1.0.0)

The manifest is strictly JSON-native. Mappings, lists, tuples, sets, NumPy arrays
and NumPy scalars are converted, and their contents are checked recursively. Keys
must be exactly `str`. Leaves must be exactly `None`, `bool`, `int`, finite `float`
or `str`. Any other key or leaf raises `TypeError`, and a non-finite float raises
`ValueError`. Nothing is converted to a string. This is enforced by `run_episode`,
with or without `output_dir`. The manifest contains no timestamps, host names or paths. Identical inputs and an identical
repository state give byte-identical JSON, except that set-valued workload
parameters are listed in an order that depends on Python's hash seed.

| Key | Content |
|---|---|
| `manifest_schema_version` | `"1.0.0"` |
| `spec_versions` | `{"simulator": "1.0.0", "baselines": "1.0.0"}` |
| `git` | `{"commit": <HEAD sha or "unavailable">, "dirty": <bool or null>}`. The commit is only reported when the project root is the repository top level. A missing repository never raises. `run_benchmark_suite` reads it once, before it creates or writes anything, so a run never reports its own output as a change. Untracked files count as dirty. |
| `seed` | int |
| `config` | `dataclasses.asdict(cfg)`, JSON-safe |
| `controller` | `{"name", "params"}` |
| `workload` | `{"name", "params", "sha256", "horizon", "min_rps", "max_rps", "mean_rps"}` |
| `metrics` | below |

`metrics`, all computed from arrays recorded during the episode:

| Key | Definition |
|---|---|
| `control_steps_executed` | `len(rewards)` |
| `substeps_executed` | number of recorded substep latencies |
| `terminated`, `truncated` | from the last step |
| `total_reward`, `mean_reward` | sum and mean of step rewards |
| `unweighted_cost_sums` | sums of `slo_violation`, `failure_cost`, `resource_cost`, `scaling_cost` |
| `weighted_cost_sums` | each unweighted sum times its `cfg.weight_*`, plus `total_cost` summed from `info` |
| `substep_latency_proxy_ms` | `mean`, `p50`, `p95`, `p99`, `max` of the per-substep end-to-end **modeled latency proxy** (SPEC_01 Section 24). Percentiles use NumPy's default linear interpolation over substeps. These are not empirical request percentiles. |
| `mean_e2e_failure_proxy`, `max_e2e_failure_proxy` | over substeps |
| `slo_violation_substeps` | substeps with `e2e_latency_ms > cfg.slo_ms` |
| `total_pod_seconds` | int sum of `occupancy` over substeps |
| `total_scaling_events` | int sum of `abs(approved_delta)` over steps |
| `shield_arbitration_override_count`, `shield_fallback_count` | see 3.1 |

---

## 4. Acceptance tests (`tests/test_stage1b_baselines.py`)

The tests check mechanics, invariants, immutability, horizons and
reproducibility. **They never assert that one controller performs better than
another.**

1. **Workloads:** shape, dtype, read-only values, immutable params, hand-checked
   boundary values, SHA-256 determinism, seeded Poisson reproducibility and
   isolation, and invalid parameters.
2. **Controllers:** the step-0 rule, `SimHPABaseline` deadband boundaries, ceiling
   scaling and clipping, desired-replica reconstruction with pending pods checked
   against the real `env.desired`, all four threshold branches with strict
   boundaries, invalid hyperparameters, and defensive `params` copies.
3. **Runner:** default and shortened horizons, mismatched traces, `reset()` before
   every episode, the fallback count not double-counted, the override count, the
   accounting identities, JSON round trips, byte-identical repeat runs, and basic
   behavior (a static run has zero scaling events, and dynamic baselines scale up
   under sustained high load).
4. **Isolation:** AST inspection of imports in the three Stage 1B modules, plus a
   clean subprocess import check.
