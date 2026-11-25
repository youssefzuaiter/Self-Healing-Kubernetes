# Architectural Decisions Log

## Frozen decisions

Supplied by the principal engineer. ADR numbers 04, 06, 08, 10 and 11 were not
provided and are deliberately left unused, so nothing below claims those IDs.

| ID | Decision Area | Frozen Choice |
| :--- | :--- | :--- |
| **ADR-01** | Simulator Engine | Vectorized NumPy, fixed `1.0s` internal sub-steps. |
| **ADR-02** | Control Interval | `5.0 seconds` (`5` internal sub-steps per `env.step()`). |
| **ADR-03** | Downstream Delay | Explicitly delayed by `1` internal sub-step (`t -> t+1`). |
| **ADR-05** | Queue Timeouts | Exact discrete FIFO age buckets `Q[node, age]`. |
| **ADR-07** | Replica Lifecycle | Separate `desired`, `ready`, `pending`, and `terminating` states. |
| **ADR-09** | Safety Token | API-level one-use step-bound capability token. |
| **ADR-12** | Quota Arbitration | Deterministic round-robin (`0 -> 1 -> 2 -> 3`). |
| **ADR-13** | DB Contention | Piecewise linear above threshold $\theta_{\text{db}} = 0.50$. |

## Implementation decisions

These record choices the spec left open. None changes a spec formula. Each is
open to audit.

### Timing

- **IMPL-01 — Meaning of Section 9 step 1.** Step 16 graduates and exits pods at
  the end of t, so they take effect at the start of t+1. The only transition left
  for step 1 is applying the approved action at the first substep of a control
  interval. `step()` applies it immediately before the first substep. The other
  four substeps have no step-1 work.
- **IMPL-02 — When metrics see lifecycle changes.** Per-substep records are taken
  in step 15, before step 16, so they show the replicas in effect during t. A pod
  graduating at the end of t+19 is recorded as pending at t+19 and ready at t+20.
- **IMPL-03 — Observation timing.** Observations are built after step 16 of the
  last substep, so replica features describe the start of the next interval.
- **IMPL-04 — The substep length is not a config field.** Rates are per second and
  the substep is one second, so per-substep quantities equal rates numerically.
  A configurable length would need scaling the spec does not define.
- **IMPL-05 — Cooldown "30 seconds".** Under Section 29 a scale-up at interval k
  holds cooldown at 6 through k, then counts down at the end of k+1 to k+6. The
  first permitted scale-down is at interval k+7: 30 seconds after the scale-up
  interval ends, and 35 seconds after the scale-up was applied.
- **IMPL-06 — Only scale-ups start cooldown.** Section 29 sets cooldown on an
  approved scale-up and nowhere else. Consecutive scale-downs are therefore
  allowed. A partially approved scale-up (`+2` proposed, `+1` approved) sets it. A
  scale-up denied entirely does not. Rejected scale-downs do not change it.

### Horizon and traffic

- **IMPL-07 — Exhaustion applies to the constant trace too.** `traffic_trace=None`
  has no array to run out of, but Section 32.1 forbids continuing past 1,800
  seconds, so a request for index 1,800 raises `ValueError` in both modes.
- **IMPL-08 — A post-horizon step raises before mutating state.** `step()` reads
  the first traffic value of the interval right after consuming the token and
  before applying the action. The read does not depend on replica state, so this
  moves nothing observable in a valid step. The token stays consumed (Section
  30.1) and replicas stay untouched.
- **IMPL-09 — Trace inputs.** Only `numpy.ndarray` is accepted, so a Python list
  is rejected. Integer and floating dtypes are accepted. Boolean, complex, string
  and object dtypes raise `ValueError`. The trace is copied at construction and
  stored read-only, so later edits to the caller's array cannot change a run.

### Queues and numerics

- **IMPL-10 — Queue age buckets come from the timeouts.** The bucket count is
  `max(timeout_seconds)`, which is 5 by default. A node's work expires from bucket
  `W_i - 1`, so per-node timeouts stay configurable. With the default config this
  is exactly bucket 4.
- **IMPL-11 — Register lengths come from the delays.** Lengths are `max(boot)` and
  `max(termination)`, 20 and 5 by default. A pod enters slot `delay_i - 1`.
- **IMPL-12 — Float tolerance.** Conservation uses the spec's absolute `1e-9`. The
  queue-capacity diagnostic uses the same tolerance because `Q + (Qmax - Q)` can
  exceed `Qmax` by rounding. Neither one corrects state.
- **IMPL-13 — Diagnostics are explicit raises.** They raise
  `SimulationInvariantError` rather than using `assert`, so `python -O` cannot
  disable them.
- **IMPL-14 — Proxy formula constants.** Sections 22 and 24 are written with
  literal coefficients and are implemented that way for direct audit. If
  `call_multiplicity` is changed in config, those two proxies do not follow it.

### Replicas

- **IMPL-15 — Database in the quota.** The quota sum and resource cost run over all
  nodes, including the database's one replica, because Section 31.4 explicitly
  restricts its own sum to `i=0..3` and these do not. So 27 pods remain for the
  four scalable services, and default occupancy starts at 9.
- **IMPL-16 — Newest-first cancellation order.** The boot register is the order. A
  higher slot is a newer pod because every pod enters at the same slot and shifts
  left once per substep. Cancellation walks from the highest slot down. Pods in
  the same slot were created together and are interchangeable.
- **IMPL-17 — Only cancellation frees quota during arbitration.** A scale-down that
  terminates a ready pod keeps it in occupancy (Sections 16, 19), so it cannot fund
  a scale-up in the same step. A cancellation can.
- **IMPL-18 — Invariant `desired == ready + pending`.** It holds at all times and
  is checked every substep.

### Actions and tokens

- **IMPL-19 — Booleans are non-numeric.** Boolean arrays, and lists or tuples with
  a boolean element, are rejected. Complex and object dtypes are also rejected.
  Without this, `[True, 0, 0, 0]` would silently become `[1, 0, 0, 0]`.
- **IMPL-20 — Shield API.** Each environment owns one shield:
  `env.step(env.shield.approve(raw))`. Arbitration reads simulator state and never
  writes it. State changes happen only in `step()`, so an unused token has no
  effect. `approve()` changes only the fallback counter and the token ledger.
- **IMPL-21 — The ledger is the authority.** `env._tokens` maps each nonce to a
  record holding the exact issued object, its step, its deltas and `consumed`.
  `step()` applies the deltas stored in the ledger, not those on the object. Copies
  and edited tokens are rejected by an identity and field check that runs between
  Section 30.1 steps 4 and 5.
- **IMPL-22 — Several tokens per step.** `approve()` may be called more than once
  in a step. All those tokens are valid until one is consumed. The step index then
  advances and the others become stale, so only one action can apply per step.
- **IMPL-23 — Reuse reports as stale.** Section 30.1 checks the step index before
  consumption, so a token reused after a normal step fails as stale. It fails as
  consumed only if the step index did not advance, such as after a post-horizon
  `ValueError`. Both raise `PermissionError`.
- **IMPL-24 — Independent environment validation.** After consuming the token,
  `step()` re-checks the deltas against the action set, replica bounds, cooldown
  and quota without calling the shield. A violation raises `PermissionError`
  before any state changes, and the token stays consumed. A zero delta cannot break
  a bound, so corrupted state surfaces as `SimulationInvariantError`, not as a
  rejected hold action.

### Observations and API

- **IMPL-25 — Reset observation.** No previous interval exists at reset, so
  interval features (0, 1, 3, 6, 7) are zero. Features 2, 4, 5 and 8 use real state.
- **IMPL-26 — Cooldown feature with a zero config.** Feature 8 divides by
  `max(1, cooldown_control_steps)`, so a cooldown of 0 cannot divide by zero. The
  default is exactly `cooldown / 6`.
- **IMPL-27 — Spaces.** `action_space` describes the raw proposal given to the
  shield, as `Box(-1, 2, (4,))`, not the `ApprovedAction` type `step()` accepts.
  Observations stay `float64` so they match the simulator's values exactly.
- **IMPL-28 — Use before reset.** The constructor initializes full state, so the
  environment is usable before `reset()`.
- **IMPL-29 — Test import path.** `tests/test_stage1a_physics.py` inserts the
  repository root into `sys.path`. Bare `pytest -v` needs this, and it avoids
  adding a `conftest.py` or `pyproject.toml` outside the permitted file list.

### Corrective directives

- **IMPL-30 — Failure ratios use a `max(1.0, denominator)` floor.** The original
  master prompt defined plain `F/N`. The principal engineer's corrective directive
  required the floor, with `completed = 0.2, failed = 0.3` giving `0.3`, and then
  approved updating SPEC_01 Sections 21 and 21.1 to match. The floor applies to the
  substep ratio, the interval ratio and observation feature 7. At interval level it
  applies once, to the summed resolved work. So `p_succ`, the end-to-end failure
  proxy and the reward all change whenever a substep resolves less than 1.0 request.
  Results for `N >= 1` are unchanged. `F <= N` still holds, so ratios stay in [0,1].
  The Section 23.1 served-age ratio is not a failure ratio and is not floored.
- **IMPL-31 — `tools/mutation_check.py` is outside the Stage 1A module list.** It
  was added at the principal engineer's request as the reproducible mutation check.
  It is a development tool, not a simulator module, and `sim/` does not import it.
- **IMPL-32 — `SimulatorConfig` validates every field.** The principal engineer
  approved closing the gaps the configuration audit found. The spec sets default
  values, not legal ranges, so these rules are mine, and no default changes.
  Invalid input raises `ValueError` at construction, and types are checked first.
  - Integer fields reject floats and booleans, and number fields reject booleans,
    NaN and infinities. NumPy scalars are accepted. Per-node fields and
    `call_multiplicity` must be tuples, which keeps the frozen config immutable.
  - `db_contention_threshold` is limited to [0, 1], the range of database pressure.
  - `retry_coefficient` is limited to [0, 1], because above 1 retries would create
    more work than timed out.
  - `action_min <= 0 <= action_max`, because the `[0, 0, 0, 0]` fallback must be legal.
  - `capacity_multiplier`, `conservation_tolerance`, `slo_ms` and `scaling_cost_norm`
    must be positive. `conservation_tolerance` cannot be 0, because float rounding
    would then trip the diagnostic.
  - `default_traffic_rps`, `base_latency_ms`, `call_multiplicity` and
    `slo_violation_cap` must be >= 0, and so must the four reward weights. Zero stays
    allowed, so a cost term can be switched off.

## Known modeling approximations

- Fluid model: quantities are fractional expected values, not individual requests.
- The failure and latency proxies are heuristics, not measured probabilities or
  percentiles (Sections 22, 23).
- A child failure does not propagate back to its parent: a parent's completion
  counts as a success even when the child call fails later. The cost appears only
  through the end-to-end proxy, and backpressure appears only through database
  contention.
- Retries are limited only by the geometric decay of `rho = 0.5`. There is no
  per-request retry budget.
- Database contention slows upstream services, but the database's own `κ = 0`
  means it does not slow itself.

## Known behavior under the default configuration

- **Pending-pod cancellation cannot occur at default settings.** Every pending pod
  was created by a scale-up less than 20 seconds earlier, and that scale-up blocks
  scale-downs for at least 35 seconds (IMPL-05). Cancellation is implemented and
  tested with a reduced `cooldown_control_steps`. A test asserts that it is
  unreachable with defaults. It becomes reachable if cooldown is shortened.
- **Zero ready replicas cannot occur at default settings.** `min_replicas = 1`, and
  the last ready pod cannot be terminated. Tests reach this state by configuring
  `min_replicas` and `initial_replicas` to 0.
- **The `1e-9` absolute tolerance held at every traffic level measured.** Full
  episodes ran with constant traces up to `1e10` req/s and with irregular traces,
  and with single spikes onto a fractional queue, up to `1e9` req/s. The worst
  residual measured in the spike runs was `1.82e-12`. This is evidence, not a proof
  for all inputs. The spec sets no upper bound on trace values, and the diagnostic
  raises if a future input breaches the tolerance.

## Stage 1B decisions

Recorded for `docs/SPEC_02_BASELINES.md`. Stage 1A files are frozen and unchanged.

- **IMPL-33 — `WorkloadTrace` identity.** The dataclass uses `eq=False`, because
  comparing NumPy arrays field by field is ambiguous. Compare traces by `sha256`.
  `sha256` may be omitted. If it is given, it must equal the computed digest.
- **IMPL-34 — Time windows.** Every window must be non-empty and lie inside
  `[0, horizon)`. `step_jump` requires `0 <= jump_time < horizon`, so the jump
  always happens. `burst_and_drop` requires both `ramp_duration` and
  `hold_duration` to be at least 1.
- **IMPL-35 — `burst_and_drop` ramp sampling.** The ramp samples the continuous
  line from `base_rate` toward `peak_rate` at whole seconds:
  `base + (peak - base)(t - ramp_start) / ramp_duration`. It starts at `base_rate`
  and reaches `peak_rate` at the first hold substep. This differs from
  `gradual_ramp`, whose `np.linspace` includes both endpoints, as SPEC_02 requires.
- **IMPL-36 — Poisson draw order.** At each idle substep, one `rng.random()` draw
  decides whether a burst starts. A started burst draws its length with
  `rng.integers(..., endpoint=True)`. Every substep then draws one `rng.poisson`.
  Bursts longer than the remaining horizon are cut off, so
  `max_burst_len > horizon` is allowed.
- **IMPL-37 — Controller hyperparameter ranges.** `SimHPABaseline` requires
  `0 < target_utilization <= 1` and `0 <= tolerance < 1`. Utilization is clipped to
  [0, 1] in the observation, so a target above 1 could never be met.
- **IMPL-38 — Controllers cannot reach the environment.** `sim/baselines.py`
  imports neither `sim.env` nor `sim.shield`. It keeps its own copy of the feature
  count, 9, and a test checks that it matches `sim.env.NUM_FEATURES`.
- **IMPL-39 — Runner reproducibility.** Manifests contain no timestamps, host
  names or paths, so identical inputs give byte-identical JSON. The git commit is
  reported only when the project root is the repository top level, which avoids
  attributing a parent repository's commit. Episode-count mismatches raise
  `RuntimeError` explicitly rather than using `assert`.
- **IMPL-40 — Manifest file names.** Names include the first 12 hex digits of the
  trace digest, so two traces with the same name but different parameters cannot
  overwrite each other. A repeated name raises `ValueError`.
- **IMPL-41 — `CLAUDE.md` is stale but frozen.** It still says "Current stage: 1A"
  and forbids `sim/baselines.py` and `sim/workloads.py`. Stage 1B was authorized
  by the principal engineer with `CLAUDE.md` frozen, so it is left byte-identical.
- **IMPL-42 — Bytes-backed traces and parameter leaves.** An independent audit found
  that `setflags(write=True)` re-enabled writing and that mutable leaves stayed
  mutable. `values` is now `np.frombuffer` over immutable `bytes`. Parameter leaves
  are an exact-type allow-list, because subclass instances can carry mutable state.
  `_freeze` rebuilds every container, which is the deep copy the spec requires, so
  `copy.deepcopy` was removed: it ran custom `__deepcopy__` hooks before validation
  and failed on already frozen params.
- **IMPL-43 — Manifest output.** The audit found that existing manifests were
  silently overwritten. Beyond the directive, the pre-flight check also rejects
  names that are too long or not UTF-8 encodable, and collisions that differ only
  in case or Unicode normalization, because each would otherwise fail mid-run and
  leave a partial directory. It also stops at once if a destination already exists;
  mode `"x"` remains the guard against a file appearing later. Cleanup compares
  device and inode, so a file that replaced the incomplete one is not removed.
  `run_benchmark_suite` copies its arguments to lists, so iterators are not
  exhausted by the checks.
- **IMPL-44 — `values` is rebuilt on every access.** NumPy allows `dtype`, `shape`
  and `strides` to be reassigned in place on a read-only array. A byte-order change
  made the runner simulate different traffic under an unchanged digest. `values`
  stays a required dataclass field, backed by a descriptor that stores the bytes and
  returns a new array each time. It can be set only once, even through
  `object.__setattr__`. So `trace.values is trace.values` is false. As with tokens,
  code that edits private state (`trace.__dict__`) is out of scope.
- **IMPL-45 — Git state is read once per run (corrects IMPL-39).** IMPL-39 claimed
  that identical inputs give byte-identical manifests. That was wrong in a git
  repository: the state was read per episode, so writing the first manifest inside
  the repository marked the other 20 as dirty. `run_benchmark_suite` now reads it
  once, before creating or writing anything, and gives every manifest its own copy.
  Untracked files still count as dirty, so output left by an earlier run makes the
  next run dirty unless it is ignored or written outside the repository.
- **IMPL-46 — Numbers too large for a float.** `math.isfinite` raises
  `OverflowError` for such integers and fractions. Workload rates and controller
  hyperparameters now reject them with `ValueError`. The frozen `SimulatorConfig`
  has the same behavior and is unchanged.
- **IMPL-47 — Strict JSON-native manifests.** `json_safe` let unsupported objects
  through and turned keys into strings, so keys `1` and `"1"` collided and one value
  was lost. Keys must now be exactly `str`, and leaves exactly `None`, `bool`, `int`,
  finite `float` or `str`. Subclasses are rejected, as in workload params (IMPL-42).
  NumPy values without a Python equivalent, such as `longdouble` on this platform,
  are rejected.
- **IMPL-48 — Complete pre-flight.** Types, horizons, the seed, the JSON form of
  names, params and config, and all file names are checked before the first episode
  and before the file system is touched. The JSON checks were added to the list in
  the directive because they are equally predictable and also left partial output.
  `run_episode` makes the same checks before running, and checks the finished
  manifest again, because parameters can change during an episode. Failures that
  cannot be predicted, such as a full disk, can still leave earlier manifests.
