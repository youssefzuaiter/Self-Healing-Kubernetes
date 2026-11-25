# SPEC 01 — Stage 1A Simulator (Authoritative)

Topology-Aware Spatio-Temporal Graph RL Autoscaler for Kubernetes Microservices.

This document records the Stage 1A master prompt. It is the single source of
truth for the simulator. Section numbers match the master prompt and are cited
from code comments. Implementation choices the spec leaves open are recorded in
`docs/DECISIONS.md`, never here.

Revision: Sections 21 and 21.1 add the `max(1, ·)` failure ratio floor, approved by
the principal engineer after the original master prompt. See IMPL-30.

---

## 1. Objective

A deterministic, reproducible, discrete-time fluid simulator of a five-node
microservice application. It models external traffic, downstream call
propagation, queue admission and overflow, FIFO service, timeouts, retries,
replica boot and termination delays, database contention, failure accounting,
modeled end-to-end failure, latency proxies, observations, reward, deterministic
action arbitration, and safety-token validation.

This stage is not a real Kubernetes deployment.

## 2. Technical constraints

Python >= 3.10, NumPy, Gymnasium, pytest. Frozen dataclasses for configuration.
Type hints on public interfaces. Deterministic physics. No hidden random
variation in service capacity. Gymnasium environment interface.

## 3. Repository files

`CLAUDE.md`, `docs/DECISIONS.md`, `docs/SPEC_01_SIMULATOR.md`, `sim/__init__.py`,
`sim/config.py`, `sim/shield.py`, `sim/env.py`, `tests/test_stage1a_physics.py`.

---

## 4. Time discretization

| Parameter | Value |
|---|---:|
| Internal substep | 1 second |
| Control interval | 5 seconds |
| Substeps per control step | 5 |
| Control horizon | 360 steps |
| Internal horizon | 1,800 seconds |
| Queue age buckets | 5 |
| Traffic trace length | 1,800 |

Each `env.step()` advances exactly five internal substeps. Actions are approved
and applied at the beginning of the control interval and remain in effect during
its five substeps. Simulation time is integer-based; wall-clock time is not used.

## 5. Microservice topology

| ID | Name | Scalable |
|---:|---|---|
| 0 | Gateway | Yes |
| 1 | Auth | Yes |
| 2 | Catalog | Yes |
| 3 | Order | Yes |
| 4 | Database | No |

Node 4 always has exactly one replica.

### 5.1 Service parameters

| Node | μ req/s/pod | Qmax | Base latency ms | Min replicas | Max replicas | Initial replicas | Boot s | Termination s | Timeout s | κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Gateway | 100 | 2000 | 5 | 1 | 10 | 2 | 20 | 5 | 5 | 0 |
| Auth | 120 | 2000 | 8 | 1 | 10 | 2 | 20 | 5 | 5 | 1.0 |
| Catalog | 80 | 1500 | 15 | 1 | 10 | 2 | 20 | 5 | 5 | 1.5 |
| Order | 50 | 1000 | 25 | 1 | 10 | 2 | 20 | 5 | 5 | 2.5 |
| Database | 300 | 3000 | 10 | 1 | 1 | 1 | 20 | 5 | 5 | 0 |

All values are configurable through `SimulatorConfig`. The default configuration
must reproduce this table exactly.

## 6. Downstream call multiplicity

$$
M =
\begin{bmatrix}
0 & 1.0 & 0.8 & 0.2 & 0\\
0 & 0 & 0 & 0 & 0.3\\
0 & 0 & 0 & 0 & 0.5\\
0 & 0 & 0 & 0 & 1.0\\
0 & 0 & 0 & 0 & 0
\end{bmatrix}
$$

Gateway produces 1.0 Auth, 0.8 Catalog and 0.2 Order calls per completed request.
Auth produces 0.3, Catalog 0.5 and Order 1.0 Database calls. Fractional fluid
quantities are valid.

### 6.1 Propagation timing

$$A^{new}_{v,t+1} = \sum_u M_{u,v}C_{u,t}$$

Gateway additionally receives external traffic. A parent completion during
substep t produces child arrivals during substep t+1. Downstream arrivals are
never processed in the same substep as the parent completion. Current arrivals
and next-substep arrivals use separate buffers.

## 7. External traffic interface

Constructor: `MicroserviceMeshEnv(traffic_trace=None, config=None)`.

1. `None`: constant external traffic of 100 requests/second.
2. A NumPy array of exactly shape `(1800,)`, with finite, non-negative values.

Invalid inputs raise `ValueError` during initialization.

On reset: simulation time and the traffic pointer return to zero, and the same
configuration, trace, seed and actions reproduce the same trajectory.

When an unavailable traffic index is requested, raise `ValueError` at the point
of the request. Never repeat the final value, wrap around, pad the array, or
fabricate traffic. Initialization validation errors are distinct from runtime
horizon-exhaustion errors.

Stage 1A does not implement callable traces or `sim/workloads.py`.

---

## 8. Queue representation

$$q_{i,t}=[q_{i,0,t},q_{i,1,t},q_{i,2,t},q_{i,3,t},q_{i,4,t}]$$

Bucket a holds requests with a completed prior seconds of waiting. Bucket 0 is
the newest.

$$Q_{i,t}=\sum_{a=0}^{4}q_{i,a,t}$$

All queue quantities are finite and non-negative floating-point values.

## 9. Exact per-substep processing order

1. Process replica lifecycle transitions due at the start of t.
2. Read external traffic for t.
3. Read downstream arrivals generated during t-1.
4. Read retries generated during t-1.
5. Snapshot the database queue pressure at the start of the substep.
6. Calculate database contention.
7. Calculate service capacities.
8. Calculate total arrivals.
9. Perform queue admission and overflow accounting.
10. Serve queues using FIFO ordering.
11. Expire unserved bucket-4 requests.
12. Age surviving queue contents.
13. Generate retries for substep t+1.
14. Generate downstream arrivals for substep t+1.
15. Record metrics and conservation invariants.
16. Process replica lifecycle register transitions at the end of t.

All node calculations use the same start-of-substep database queue snapshot.

## 10. Database contention

$$P_{DB,t}=\frac{Q_{DB,t}}{Q^{max}_{DB}}$$

$Q_{DB,t}$ is the total database backlog at the start of substep t.

$$\phi_{i,t}=\frac{1}{1+\kappa_i\max(0,P_{DB,t}-0.50)}$$

Contention begins only when database queue pressure exceeds 50%.

### 10.1 Effective service capacity

$$K_{i,t}=R^{ready}_{i,t}\mu_i\phi_{i,t}$$

Deterministic capacity multiplier $\xi=1.0$. Pending and terminating replicas
provide zero capacity. If ready replicas are zero, capacity is zero.

## 11. Arrivals and admission

$$A^{total}_{i,t}=A^{new}_{i,t}+A^{retry}_{i,t}$$

$A^{new}$ includes external traffic or downstream calls. $A^{retry}$ includes
retries from the previous substep. Free space uses the queue at the beginning of
the substep:

$$S^{free}_{i,t}=\max(0,Q^{max}_i-Q_{i,t})$$

$$A^{accepted}_{i,t}=\min(A^{total}_{i,t},S^{free}_{i,t})$$

$$D^{overflow}_{i,t}=A^{total}_{i,t}-A^{accepted}_{i,t}$$

Accepted arrivals join bucket 0. Overflow is never silently discarded. Newly
arriving work may not exceed queue capacity on the grounds that service happens
later in the same substep.

## 12. FIFO service

Serve the oldest bucket first, $a=4,3,2,1,0$, starting from
$K^{remaining}=K_{i,t}$:

$$c_{i,a,t}=\min(q_{i,a,t},K^{remaining})$$

$$K^{remaining}\leftarrow K^{remaining}-c_{i,a,t}$$

$$C_{i,t}=\sum_{a=0}^{4}c_{i,a,t}$$

Ensure $C_{i,t}\leq K_{i,t}$ and $C_{i,t}\leq Q_{i,t}+A^{accepted}_{i,t}$.

## 13. Timeouts and queue aging

Timeout window $W_i=5$. At the end of substep t, unserved work in bucket 4
expires:

$$D^{timeout}_{i,t}=q_{i,4,t}-c_{i,4,t}$$

This represents requests that have reached five completed seconds of waiting.
Surviving buckets shift:

$$q'_{i,0,t+1}=0$$

$$q'_{i,a+1,t+1}=q_{i,a,t}-c_{i,a,t}\quad a=0,1,2,3$$

Bucket-4 residual work is expired, not shifted. Arrivals and retries for t+1
enter bucket 0 at the beginning of the next substep.

## 14. Retry behavior

Retry coefficient $\rho=0.5$.

$$A^{retry}_{i,t+1}=0.5D^{timeout}_{i,t}$$

Retries return to the same service, arrive at the beginning of the next substep,
enter bucket 0, are subject to normal admission, and can overflow or time out
again. Overflowed and completed requests are not retried.

## 15. Flow conservation

For every node and every substep:

$$Q_{i,t}+A^{new}_{i,t}+A^{retry}_{i,t}-C_{i,t}-D^{overflow}_{i,t}-D^{timeout}_{i,t}-Q_{i,t+1}=0$$

Absolute tolerance $10^{-9}$. A violation fails loudly; counters are never
silently corrected.

---

## 16. Cluster quota

$R^{cluster}_{max}=28$. Count ready, pending and terminating replicas. The
database remains fixed at one replica.

$$R^{cluster}=\sum_i(R^{ready}_i+R^{pending}_i+R^{terminating}_i)$$

Cancelled pending replicas leave occupancy immediately. Terminating replicas stay
in occupancy until their lifecycle timer completes.

## 17. Desired replicas

Desired replicas are tracked independently from ready, pending and terminating
counts. Approved scale-up increases desired and creates pending replicas.
Approved scale-down decreases desired.

$$R^{min}_i\leq R^{desired}_i\leq R^{max}_i$$

Database desired replicas remain one.

## 18. Boot timing

$\tau_{boot}=20$, length-20 register. A pod created at the beginning of substep t
enters slot 19. At the end of each substep: check slot 0, graduate those pods to
ready, then shift the register left. A pod created at t remains pending through
t+19 and becomes ready at the beginning of t+20.

## 19. Termination timing

$\tau_{term}=5$, length-5 register. A pod marked for termination at the beginning
of t enters slot 4. At the end of each substep: check slot 0, remove those pods
from occupancy, then shift the register left. A pod terminating at t consumes
quota through t+4 and exits at the beginning of t+5.

## 20. Pending-pod cancellation

When scaling down:

1. Cancel pending pods before terminating ready pods.
2. Cancel the newest pending pods first.
3. Remove cancelled pods immediately from the register and from occupancy.
4. Decrease desired replicas by the approved scale-down amount.
5. If the requested scale-down exceeds cancellable pending pods, terminate the
   remaining required number of ready replicas.
6. Never terminate below minimum replicas.
7. Never cancel or restore pods already in termination.

Cancellation is part of scale-down and is subject to cooldown.

---

## 21. Node failure accounting

$$F_{i,t}=D^{overflow}_{i,t}+D^{timeout}_{i,t}$$

$$N_{i,t}=C_{i,t}+D^{overflow}_{i,t}+D^{timeout}_{i,t}$$

$$f_{i,t}=\frac{F_{i,t}}{\max(1,N_{i,t})}$$

The floor of 1 keeps fractional volumes from producing large ratios: `completed = 0.2`
and `failed = 0.3` give `0.3`, not `0.6`. With zero resolved work the ratio is zero.

$$p^{succ}_{i,t}=1-f_{i,t}$$

All failure ratios remain in [0,1].

### 21.1 Control-interval failure ratio

Do not average substep ratios. Volume-weighted accounting:

$$f_{i,k}^{interval}=\frac{\sum_{t\in k}F_{i,t}}{\max\left(1,\sum_{t\in k}N_{i,t}\right)}$$

The floor applies once, to the summed resolved work, not to each substep. With zero
resolved work the ratio is zero.

## 22. End-to-end failure proxy

$$\hat S_{proxy,t}=p^{succ}_{0,t}p^{succ}_{1,t}(p^{succ}_{4,t})^{0.3}\left[0.8p^{succ}_{2,t}(p^{succ}_{4,t})^{0.5}+0.2p^{succ}_{3,t}(p^{succ}_{4,t})^{1.0}\right]$$

$$\hat E_{proxy,t}=1-\hat S_{proxy,t}$$

A bounded continuous survival heuristic using fractional expected call
multiplicities, not a literal probability for independent integer requests.
$0\leq\hat E_{proxy,t}\leq1$.

## 23. Node latency proxy

Latency is in milliseconds.

$$d^{served}_{i,t}=1000\frac{\sum_{a=0}^{4}a c_{i,a,t}}{C_{i,t}}$$

If completed work is zero, $d^{served}_{i,t}=0$.

$$d^{backlog}_{i,t}=1000\min\left(W_i,\frac{Q_{i,t+1}}{\max(1,K_{i,t})}\right)$$

$$l_{i,t}=l^{base}_i+\max(d^{served}_{i,t},d^{backlog}_{i,t})$$

A modeled proxy, not an empirical p95 or p99 measurement.

## 24. End-to-end latency proxy

$$L_{proxy,t}=l_{0,t}+l_{1,t}+0.3l_{4,t}+0.8(l_{2,t}+0.5l_{4,t})+0.2(l_{3,t}+1.0l_{4,t})$$

Follows the topology and the 80% Catalog / 20% Order branch weighting.

---

## 25. Observation space

Gymnasium Dict with `node_features` of shape `(5, 9)` and `call_multiplicity` of
shape `(5, 5)`. All values are finite. Normalized features are clipped to [0,1].
The topology matrix is returned unchanged.

Feature order: `[arrival_rate, retry_rate, queue_pressure, service_utilization,
ready_replicas, pending_replicas, latency_proxy, node_failure_ratio, cooldown]`.

| # | Feature | Definition |
|---:|---|---|
| 0 | Arrival rate | $clip(\overline{A^{new}_i}/(\mu_iR^{max}_i),0,1)$, mean over the previous control interval |
| 1 | Retry rate | $clip(\overline{A^{retry}_i}/(\mu_iR^{max}_i),0,1)$, mean over the previous control interval |
| 2 | Queue pressure | $clip(Q_i/Q^{max}_i,0,1)$, queue at the end of the control interval |
| 3 | Service utilization | $U_i=\sum_{t\in k}C_{i,t}/\sum_{t\in k}K_{i,t}$. If total capacity is zero: 0 when completed work is zero, 1 when positive. Clipped to [0,1] |
| 4 | Ready replicas | $clip(R^{ready}_i/R^{max}_i,0,1)$ |
| 5 | Pending replicas | $clip(R^{pending}_i/R^{max}_i,0,1)$ |
| 6 | Latency proxy | $clip(\overline{l_i}/1000,0,1)$, mean over the previous control interval |
| 7 | Node failure ratio | Interval volume-weighted ratio of Section 21.1 |
| 8 | Cooldown | $clip(cooldown_i/6,0,1)$ |

The database uses the same schema but remains non-scalable.

---

## 26. Raw action format

Raw proposal shape `(4,)`: 0 Gateway, 1 Auth, 2 Catalog, 3 Order. Values are
proposed replica deltas from $\{-1,0,+1,+2\}$.

### 26.1 Action conversion

For valid finite numeric components, $a_i^{int}=\operatorname{int}(\operatorname{np.round}(a_i))$
with NumPy half-to-even rounding:

| Raw input | Result before clipping |
|---:|---:|
| 0.5 | 0 |
| 1.5 | 2 |
| 2.5 | 2 |
| -0.5 | 0 |
| 3.0 | 3 |

Then clip to $\{-1,0,+1,+2\}$.

If the shape is not exactly `(4,)`, or any component is non-numeric, NaN or
infinite: reject the entire proposal, increment the fallback counter, map to
`[0,0,0,0]`, and do not partially accept valid components.

## 27. Replica constraints

$1\leq R^{desired}_i\leq10$ for scalable services, $R^{desired}_{DB}=1$, enforced
independently of the raw proposal.

## 28. Cluster quota arbitration

Maximum total occupancy 28, counting ready, pending and terminating pods.

1. Process scale-downs first.
2. Apply pending-pod cancellation before ready-pod termination.
3. Calculate remaining cluster capacity.
4. Process scale-up proposals in deterministic round-robin order:
   Gateway → Auth → Catalog → Order.
5. Allocate one pod per service per round.
6. Repeat until proposals are satisfied or quota is exhausted.

Never allocate beyond the cluster limit. All arbitration is deterministic.

## 29. Cooldown

$M_{down}=6$ control intervals, equivalent to 30 seconds. Per scalable service.

1. Set cooldown to six immediately after an approved scale-up.
2. Do not decrement cooldown during the scale-up interval.
3. Decrement once at the end of each subsequent control interval.
4. Block scale-down whenever cooldown is greater than zero.
5. Scale-up does not bypass replica maximums or cluster quota.

## 30. ApprovedAction token lifecycle

A shield-issued capability token binding the issuing environment identity, the
current control-step index, an internally generated nonce, the approved action
vector, and unconsumed status. Validation uses environment-owned state. A UUID or
strong random nonce is an identifier only: no claim of absolute unforgeability
against arbitrary code in the same Python process is made.

### 30.1 Validation sequence

When `env.step(action)` is called:

1. Reject raw arrays and non-token inputs.
2. Verify token type.
3. Verify issuing environment identity.
4. Verify nonce exists in environment-owned state.
5. Verify step index matches current environment step.
6. Verify token is unconsumed.
7. Atomically mark the token consumed.
8. Only then proceed with simulation.

Invalid token rejection raises `PermissionError` without modifying simulator
state. Once successfully consumed, a token stays consumed even if later action
processing raises. `env.step()` rejects raw actions, forged tokens, unknown
nonces, cross-environment tokens, stale tokens and reused tokens, and never
bypasses shield validation.

---

## 31. Reward calculation

$L_{SLO}=250$ ms; $\alpha=2.0,\ \beta=5.0,\ \gamma=0.5,\ \delta=0.2$.

### 31.1 SLO violation

$$V_t=\min\left(5,\max\left(0,\frac{L_{proxy,t}-250}{250}\right)\right)$$

$$V_k=\frac{1}{5}\sum_{t\in k}V_t$$

### 31.2 Failure cost

$$E_k=\frac{1}{5}\sum_{t\in k}\hat E_{proxy,t}$$

### 31.3 Resource cost

$$C_{resources,t}=\frac{\sum_i(R^{ready}_{i,t}+R^{pending}_{i,t}+R^{terminating}_{i,t})}{28}$$

$$C_{resources,k}=\frac{1}{5}\sum_{t\in k}C_{resources,t}$$

### 31.4 Scaling cost

Actual approved replica changes after shield arbitration, applied once per
control interval:

$$C_{scaling,k}=\frac{\sum_{i=0}^{3}|\Delta_i^{approved}|}{8}$$

### 31.5 Total cost and reward

$$J_k=2.0V_k+5.0E_k+0.5C_{resources,k}+0.2C_{scaling,k}$$

$$r_k=-J_k$$

Reward is always finite.

---

## 32. Environment behavior

`MicroserviceMeshEnv(gym.Env)` exposes `__init__`, `reset(seed=None, options=None)`
and `step(action)`, observation and action spaces, internal substep processing,
lifecycle processing, metric accounting and shield integration. `step()` accepts
`ApprovedAction`, so raw controller proposals must first pass through the shield.
The environment validates actions independently.

### 32.1 Gymnasium return format

`reset()` returns `(observation, info)`. `step()` returns
`(observation, reward, terminated, truncated, info)`.

`terminated=True` at the end of the 360th control step. The simulator does not
silently continue beyond the 1,800-second horizon. `truncated=False` for normal
horizon completion. No unspecified termination conditions.

## 33. Reproducibility

Reset clears simulation time, the traffic pointer, ready state, pending and
termination registers, desired state, queue buckets, retry and downstream
buffers, counters, interval accumulators, cooldowns, shield token state, fallback
counters and the step index. Gymnasium seeded reset conventions apply. Identical
configuration, trace, seed and actions produce identical trajectories.

---

## 34. Test acceptance criteria

`tests/test_stage1a_physics.py`, parameterized, asserting real boundary values.

**Physics:** (1) flow conservation across every node; (2) queue capacity and
overflow accounting; (3) FIFO service ordering; (4) downstream multiplicity;
(5) one-substep propagation delay; (6) timeout timing; (7) retry timing; (8) no
retries for overflow; (9) volume-weighted failure ratios; (10) failure ratio
bounds; (11) database contention threshold; (12) database contention
coefficients; (13) zero-ready-replica behavior; (14) latency calculations and
units.

**Replica lifecycle:** (15) pod boot pending through t+19; (16) graduation at the
start of t+20; (17) terminating pod consumes quota through t+4; (18) pod exits at
the start of t+5; (19) newest-first pending cancellation; (20) immediate quota
release on cancellation; (21) terminating pods continue consuming quota.

**Safety shield:** (22) action vector shape validation; (23) invalid action
fallback; (24) NaN and Inf handling; (25) half-to-even rounding; (26) replica
min/max constraints; (27) scale-down-before-scale-up arbitration; (28) round-robin
quota arbitration; (29) cooldown behavior; (30) database cannot scale; (31) raw
action rejection by `env.step()`; (32) forged token rejection; (33) stale token
rejection; (34) reused token rejection; (35) cross-environment token rejection;
(36) invalid token does not change environment state.

**Environment:** (37) trace shape validation; (38) trace finite/non-negative
validation; (39) trace reset; (40) trace exhaustion; (41) observation shape;
(42) observation bounds; (43) observation finiteness; (44) reward finiteness;
(45) seed reproducibility; (46) correct horizon termination.

## 35. Diagnostic assertions

Internal validation for non-negative queues, queue capacity, finite service
capacities, non-negative replica counts, cluster quota, failure ratio bounds,
finite observations, finite rewards and the conservation residual. Clipping is
permitted only for normalized observation features and documented action
conversion, never to hide invalid internal state.

## 37. Scope restrictions

Stage 1A excludes `sim/baselines.py`, `sim/workloads.py`, PPO, GATv2, GRU,
PyTorch, Stable-Baselines3, Kubernetes, kind, Prometheus, Locust, dashboards, UI,
and all Stage 1B or later functionality.
