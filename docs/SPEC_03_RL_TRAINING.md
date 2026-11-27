# SPEC 03 — Stage 2 Progressive RL Autoscaling Ablation Ladder

Topology-Aware Spatio-Temporal Graph RL Autoscaler for Kubernetes Microservices.

Version 1.0.0. **Specification only: no Stage 2 code exists and none is
authorized.** Stage 2 builds on Git tag `stage-1b-frozen`, commit
`be743262fe2d4372ee3abcef5f95512ca998d200`, and modifies none of its 14 files.
Section 7 is this document's decision register, because `docs/DECISIONS.md` is frozen.

Notation: control step `k` in `0..359`, substep `t = 5k`, nodes `0 Gateway, 1 Auth,
2 Catalog, 3 Order, 4 Database`, `B` batch size, `o_k = observation["node_features"]`
at decision `k`, `h_k` the observation history `o_0..o_k`.

Claims are marked by how they were established:
**[code]** read from frozen source; **[test]** also covered by a frozen test that
passes; **[derived]** proved here from [code] facts, not executed, and checked in
Stage 2 by the test named; **[measured]** from a Stage 1B run.

---

## 0. Frozen interfaces this specification depends on

| Contract | Source |
|---|---|
| `MicroserviceMeshEnv(traffic_trace=None, config=None)`; the trace is a constructor argument, so each training episode needs a new environment object | `sim/env.py:120-147` [code] |
| `env.reset(seed=...) -> (observation, info)`. Physics uses no randomness, so the reset seed does not change a trajectory | `sim/env.py:212-216` [code] |
| `env.step(token) -> (observation, reward, terminated, truncated, info)` | `sim/env.py:218-248` [test] |
| `terminated = (step_index >= control_horizon)`, first `True` on the 360th step; `truncated` is the literal `False` | `sim/env.py:236, 248` [test] |
| `observation["node_features"]`: shape `(5, 9)`, `float64`, clipped to `[0, 1]`, a fresh copy | `sim/env.py:138-143, 480-501` [test] |
| `observation["call_multiplicity"]`: shape `(5, 5)`, `float64`, constant within and across episodes | `sim/env.py:140-142, 501` [test] |
| `env.action_space = Box(-1.0, 2.0, (4,), float64)`: the raw proposal, not what `step` accepts | `sim/env.py:144` [code] |
| `env.shield.approve(raw) -> ApprovedAction(env_id, step_index, nonce, deltas)` | `sim/shield.py:25-36, 116-126` [test] |
| `convert_proposal(raw, config) -> tuple or None`: an integer vector of shape `(4,)` already inside `[-1, 2]` is returned unchanged; booleans in a list or tuple reject the whole proposal | `sim/shield.py:44-61` [test] |
| `arbitrate(proposal, desired, pending, cooldown, occupancy, config)`: deterministic, reads nothing else | `sim/shield.py:64-98` [code] |
| `info["approved_delta"]`: `int64` array `(4,)`, equal to `token.deltas` | `sim/env.py:224, 241` [code] |
| `info["fallback_count"]`: cumulative within the episode | `sim/env.py:242` [test] |
| `info["reward_components"]`, `info["substeps"]`: diagnostics, not observations | `sim/env.py:238-247` [code] |
| `BaseController`: abstract `name`, abstract `propose(observation, config, step_index)`, `reset()`, and `params` returning `dict(self._params)`, a **shallow** copy | `sim/baselines.py:69-91` [code] |
| `run_episode(controller, trace, config=None, seed=0)`: calls `controller.reset()`, then per step `propose`, `convert_proposal`, exactly one `approve`, and `step` | `sim/runner.py:123-156` [code] |
| Manifest file name `<controller.name>__<trace.name>__<sha256[:12]>.json`; collisions and existing files raise | `sim/runner.py:210-236` [test] |
| `json_safe`: keys exactly `str`; leaves exactly `None, bool, int, float, str`, finite | `sim/runner.py:52-74` [test] |

Reward per control step, with the default weights (`sim/env.py:460-478`):

$$r_k = -\left(2.0\,V_k + 5.0\,E_k + 0.5\,C^{res}_k + 0.2\,C^{scale}_k\right),\qquad V_k \in [0,5],\; E_k \in [0,1],\; C^{res}_k \in (0,1],\; C^{scale}_k = \tfrac{1}{8}\sum_{i=0}^{3}|\Delta^{approved}_i| \in [0,1]$$

So `-15.7 <= r_k < 0`. `C^{scale}` uses the **approved** deltas.

All Stage 2 work uses the default `SimulatorConfig()`. Code must refuse any other
config, because Section 2.2 depends on its values.

---

## 1. Research Questions and the Fair Ablation Ladder

### 1.1 Stage 1B findings that motivate the question

All are observations. None is a claim about cause.

1. **[measured]** Over the 7 canonical workloads, `static` has the lowest total cost on
   2 (`flash_crowd` 415.10 against 433.86 and 431.70; `burst_and_drop` 700.74 against
   715.84 and 710.98, in the order `sim_hpa`, `tuned_reactive_threshold`). A dynamic
   baseline is lowest on the other 5. Reproduce with `python -m sim.runner`.
2. **[measured]** On those two workloads both dynamic baselines reach the 28-pod quota
   (`t=320`, `t=325`) before the first replica they approved in response to the load is
   ready (`t=325`, `t=330`).
3. **[measured]** In the peak and hold phases the SLO cost sits at its cap for all three
   controllers, so it does not distinguish them there. Resource cost rises by more than
   failure cost falls.
4. **[measured]** Dynamic baselines hold queues at all five nodes, and total queue peaks
   near 6,900 (`flash_crowd`) and 5,800 (`burst_and_drop`) against 1,800 and 1,632 for
   `static`. Most of their extra post-drop cost is SLO cost.
5. **[derived]** If every call is made, one external request implies
   `1.0·0.3 + 0.8·0.5 + 0.2·1.0 = 0.9` database calls (SPEC_02 1.5). At 300 calls/s the
   database sustains at most 333.3 such requests/s. The quota leaves 27 pods for the four
   services. Ignoring retries and contention, the best integer split `(8, 7, 8, 4)` passes
   800 requests/s; more needs at least `9 + 7 + 9 + 4 = 29` pods. Upstream capacity beyond
   333 requests/s therefore cannot raise the rate of fully served requests. It changes
   where work queues and fails. `AT-ENV-5`.

Findings 2–4 come from the Stage 1B diagnostics report. Its script is not in the
repository. It was rerun unchanged while writing this document and reproduced the
reported values. Any publication must first make it reproducible from the repository.

### 1.2 Research question and the four separate claims

> Under a fixed cluster quota, 20 s boot delay, 30 s scale-down cooldown and a
> non-scalable database, can a topology-aware spatio-temporal graph RL policy coordinate
> upstream scaling while accounting for downstream bottlenecks, delayed replica readiness
> and post-burst costs, and does it do so better than non-graph or non-recurrent policies
> with the same information, parameters, data and tuning effort?

The study makes four kinds of claim. Each has its own hypotheses, results tables and
conclusions. A result for one is never reported as evidence for another, and inter-node
message passing is never conflated with optimality of the call topology.

1. **Primary graph contrast, `H-graph` (`M3` vs. `M3-self-loop`).** Does inter-node message
   passing improve performance under identical per-node weight sharing, heads, window and
   parameters? Tested by `D^msg` (Section 5.3).
2. **Primary temporal contrast, `H-rec` (`M4` vs. `M3`).** Does recurrent graph processing
   outperform fixed-window graph processing under approximately matched parameter budgets?
   Tested by `D^rec`. This is **not** a pure test of recurrence: fusion order and hidden
   width (64 vs. 48) also differ, and elapsed-time inference is a known potential confound
   (Section 1.10).
3. **Secondary topology and architecture contrasts** (descriptive):
   - `M3` vs. `M3-complete` (25 edges): does the specified call topology outperform dense
     all-to-all connectivity?
   - `M3-self-loop` vs. `M2`: the effect of shared per-node processing and graph heads,
     without message passing, against a windowed flat MLP.
   - `M2` vs. `M3`: the overall non-graph versus graph architecture comparison.
   - `M1` vs. `M2`: a fixed history window, without a graph.
4. **Baseline comparison** (descriptive): do learned controllers outperform the frozen
   deterministic Stage 1B controllers? Budget parity holds only **among learned models**.
   The frozen baselines were not tuned on the RL training distribution, and no statement
   may imply equal tuning effort between RL and the baselines.

Both primary tests are two-sided: the questions are phrased as "improve" and "outperform",
but no direction is assumed, and a null or negative result is a valid, reportable
outcome. **No test in this project may assert that one model beats another.**

### 1.3 Common rules for all models

- Separate actor and critic networks with the same trunk architecture and **no shared
  parameters**. The critic receives exactly the actor's inputs.
- Inputs come only from `observation["node_features"]`. `step_index`, `info`, queue ages,
  registers and any other simulator state are forbidden inputs for every model.
- Observations are cast `float64 -> float32`. No feature is added, dropped or rescaled,
  apart from the constant node identity of Section 1.5.
- ELU activations. No dropout and no normalization layers.
- Initialization: orthogonal for `Linear` and recurrent weights (gain `sqrt(2)` hidden,
  `0.01` actor output, `1.0` critic output), Xavier-uniform for GATv2 and GRU input
  weights, zero biases.
- Actor output: logits `[B, 4, 4]`, service by action index (Section 2.5).
- Critic output: `v̂ [B]`, with `V = c_V · v̂` and fixed `c_V = 100`. This is an output
  scale for conditioning only. It changes neither the reward nor the return.

### 1.4 History window

`W = 8` frames (40 s), fixed and **not tuned**, identical in Models 2, 3, 3-self-loop and
3-complete. Window
`o_{k-7}, ..., o_k`, oldest first. Frames before `o_0` are all-zero. An all-zero frame
cannot occur in an episode because the database's ready feature is always 1.0, so padding
is unambiguous without a mask input. The window is cleared at every episode start.

`W = 8` is justified by Section 2.2: 3 frames determine the boot state exactly, and 8
frames `k-7..k` reach back to a scale-up decided at `k-7`, the latest one that no longer
blocks scale-down at `k`.

### 1.5 The graph

Built once from `config.call_multiplicity = M`. For every `M[u][v] > 0` add a forward edge
`u -> v` and a reverse edge `v -> u`. Add one self-loop per node. Messages flow from
source to target. Forward edges carry load information downstream; reverse edges carry
backpressure information upstream.

| Type | Edges (source -> target) | `w = M[u][v] / max(M)` |
|---|---|---|
| Forward | 0->1, 0->2, 0->3, 1->4, 2->4, 3->4 | 1.0, 0.8, 0.2, 0.3, 0.5, 1.0 |
| Reverse | 1->0, 2->0, 3->0, 4->1, 4->2, 4->3 | 1.0, 0.8, 0.2, 0.3, 0.5, 1.0 |
| Self | 0->0, 1->1, 2->2, 3->3, 4->4 | 0.0 |

17 directed edges, in-degrees `4, 3, 3, 3, 4`. Edge attribute
`e = [w, is_forward, is_reverse, is_self]`, `d_e = 4`. The undirected call graph has
diameter 2, so two layers give every node a receptive field of all five nodes.

Topology is constant, so it is not extra state: as an input to an MLP it would be a
constant, equivalent to a bias. Models 1 and 2 know node identity by input position.
The graph models share weights across nodes, so each node also receives a constant one-hot
identity `[5]`.

Edge masks of the graph models: Models 3 and 4 use all 17 edges above. Model 3-self-loop
uses only the 5 self-loop rows. Model 3-complete adds the 8 ordered pairs of distinct nodes
that have no call edge (1↔2, 1↔3, 2↔3, 0↔4), with `e = [0, 0, 0, 0]`, for 25 edges.

### 1.6 GATv2 layer

Pure PyTorch, dense over the 5×5 adjacency mask, 4 heads of width `C = d/4`, for node
`i`, head `m`, and `N(i)` the sources of edges into `i`, including `i`:

$$s^m_{ij} = \mathbf a_m^\top \operatorname{LeakyReLU}_{0.2}\!\left(W^m_t x_i + W^m_s x_j + W^m_e e_{ji}\right),\qquad \alpha^m_{ij} = \frac{\exp s^m_{ij}}{\sum_{l \in N(i)} \exp s^m_{il}}$$

$$x'_i = \operatorname{ELU}\!\left(\Big\Vert_{m=1}^{4} \sum_{j \in N(i)} \alpha^m_{ij} W^m_s x_j + b\right) + x_i$$

`W_s, W_t, W_e` have no bias. Non-edges are set to `-inf` before the softmax. Edge
attributes affect attention only. Parameters per layer: `d(2d + d_e + 2)`.

### 1.7 The five primary models and one secondary model

| | Model 1 | Model 2 | Model 3 | Model 3-self-loop | Model 4 |
|---|---|---|---|---|---|
| ID | `m1_mlp_w1` | `m2_mlp_w8` | `m3_gatv2_w8` | `m3_gatv2_w8_selfloop` | `m4_gatv2_gru` |
| Graph | no | no | 17-edge call graph | 5 self-loops only | 17-edge call graph |
| Temporal input | current frame | window `W=8` | window `W=8` | window `W=8` | current frame, recurrent state |
| Trunk input | `[B, 45]` | `[B, 360]` | `[B, 5, 72+5]` | `[B, 5, 72+5]` | `[B, 5, 9+5]` per step |

Secondary: Model 3-complete, `m3_gatv2_w8_complete`, Model 3 with the 25-edge mask of
Section 1.5.

- **Model 1 — Single-Frame Flat MLP PPO.** `45 -> 102 -> 102 -> 102`, then actor
  `-> 16` reshaped to `[4, 4]`, critic `-> 1`. The feedforward RL floor.
- **Model 2 — Windowed Flat MLP PPO.** `360 -> 55 -> 55 -> 55`, same heads. The window is
  flattened `[W, 5, 9]` in row-major order.
- **Model 3 — Windowed GATv2 PPO.** Per node, the 8 frames' features are concatenated
  oldest first (`72`), plus identity (`5`). `Linear(77, 64)`, ELU, two GATv2 layers
  (`d = 64`, 4×16). Actor: a shared `Linear(64, 64)`, ELU, `Linear(64, 4)` applied to
  nodes 0–3. Critic: mean over 5 nodes, `Linear(64, 64)`, ELU, `Linear(64, 1)`.
- **Model 3-self-loop — Windowed GATv2 PPO without message passing (primary).**
  **Model 3 and Model 3-self-loop must use identical node encoders, shared weights, GATv2
  layers, attention heads, hidden dimensions, actor and critic heads, and parameter
  initialization protocol. The only architectural difference is the inter-node edge mask:
  Model 3 uses the specified 17-edge call graph, while Model 3-self-loop permits only the 5
  self-loops.** Consequences, all inherent to that single difference:
  - Each node attends only to itself, so `α_ii = 1` for every input and each layer reduces
    to `x'_i = ELU(W_s x_i + b) + x_i`. `W_t`, `W_e` and `a` receive exactly zero gradient
    and stay at initialization: 17,664 of the 52,933 parameters, leaving 35,269 that learn.
  - The actor head for service `i` sees only node `i`'s window and identity. Other services,
    the Database and quota headroom are invisible to it. The critic mean-pools all five
    nodes, exactly as in Model 3.
  - It is tuned on its own (Section 1.9) and never inherits Model 3's configuration.
- **Model 4 — GATv2 + GRU PPO.** Per step: `Linear(14, 48)`, ELU, two GATv2 layers
  (`d = 48`, 4×12), then a **per-node GRU with weights shared across nodes**, hidden state
  `[B, 5, 48]` for the actor and a second `[B, 5, 48]` for the critic. Heads as in Model 3
  with width 48, reading the GRU output. At step `k` each GRU consumes the encoded `o_k`, and
  the heads read the resulting state. A pooled graph GRU is rejected because actions
  are per service, and pooling would force one vector to carry all per-service memory.
  The hidden state is zero at the start of every episode and is never carried across
  `terminated` or `truncated`.
- **Model 3-complete (secondary).** Model 3 with the 25-edge mask. All 52,933 parameters can
  learn. It has no tuning search of its own and uses Model 3's selected configuration.

### 1.8 Parameter budget

Target 53,000 trainable parameters per model, actor plus critic, tolerance ±3%
(51,410–54,590). Counts below are exact for Section 1.7. `AT-MOD-2` asserts them.

| Model | Actor | Critic | Total | Versus 53,000 |
|---|---:|---:|---:|---:|
| 1 | 27,352 | 25,807 | 53,159 | +0.30% |
| 2 | 26,911 | 26,071 | 52,982 | −0.03% |
| 3 | 26,564 | 26,369 | 52,933 | −0.13% |
| 3-self-loop | 26,564 | 26,369 | 52,933 (35,269 can learn) | −0.13% |
| 4 | 27,172 | 27,025 | 54,197 | +2.26% |
| 3-complete (secondary) | 26,564 | 26,369 | 52,933 | −0.13% |

Formulas: MLP trunk `(in+1)h + 2(h²+h)`, heads `16(h+1)` and `h+1`; GATv2 layer
`d(2d+6)`; GRU `3d(2d+2)`; graph heads `d²+d+4d+4` and `d²+d+d+1`. In Model 3-self-loop
`W_t`, `W_e` and `a` of each layer, `d² + 5d = 4,416` parameters, never learn: four layers,
17,664 in total (Section 1.7).

The budget matches nominal parameter counts. Equal parameters do not mean equal compute,
capacity or learnable parameters. Wall-clock time and inference latency are reported per
model, and Model 3-self-loop's learnable count is reported with every `H-graph` result.

### 1.9 Training and tuning budgets, identical for all primary models

Primary models: Models 1, 2, 3, 3-self-loop and 4.

| Budget | Value |
|---|---|
| Rollout | 8 environments × 360 steps = 2,880 steps, always whole episodes |
| Final run | 360 PPO training iterations = **1,036,800 steps**, 2,880 episodes, 7,200 Adam updates |
| Tuning run | 120 PPO training iterations = 345,600 steps, 960 episodes, 2,400 Adam updates |
| Tuning search (Gate 2B) | per primary model, the same 8 configurations × the same 2 tuning seeds = 16 runs, 5,529,600 steps. In total 80 runs, 27,648,000 steps |
| Final runs (Gate 2C) | 5 primary models and Model 3-complete. N = 5: 30 runs, 31,104,000 steps. N = 10: 60 runs, 62,208,000 steps |
| Validation | 21 episodes (7,560 steps) per evaluation, at iteration 0 and after every 20th PPO training iteration: 7 per tuning run, 19 per final run, excluded from step counts |

**Iteration convention.**

- **Iteration 0** is the pre-training evaluation at initialization, before any optimizer
  update (0 Adam updates).
- Each **PPO training iteration** `1..K` is 1 rollout (2,880 steps) followed by
  `5 epochs × 4 minibatches = 20` Adam updates.
- A **tuning run** executes 120 PPO training iterations (2,400 Adam updates), with
  validation evaluations at iteration 0 and after iterations 20, 40, 60, 80, 100 and 120:
  7 evaluations, the last one after the final iteration.
- A **final run** executes 360 PPO training iterations (7,200 Adam updates), with
  validation evaluations at iteration 0 and after iterations 20, 40, ..., 360:
  19 evaluations, the last one after the final iteration.

| ID | learning rate | entropy `c_H` | clip `ε` | | ID | learning rate | entropy `c_H` | clip `ε` |
|---|---|---|---|---|---|---|---|---|
| H1 | 1e-4 | 0.001 | 0.1 | | H5 | 3e-4 | 0.001 | 0.1 |
| H2 | 1e-4 | 0.001 | 0.2 | | H6 | 3e-4 | 0.001 | 0.2 |
| H3 | 1e-4 | 0.01 | 0.1 | | H7 | 3e-4 | 0.01 | 0.1 |
| H4 | 1e-4 | 0.01 | 0.2 | | H8 | 3e-4 | 0.01 | 0.2 |

Per primary model, the configuration with the lowest mean validation score (Section 4.2)
over the two tuning seeds is used for every final seed. Ties go to the lowest ID. Model
3-self-loop is searched independently, like every primary model. Model 3-complete is
secondary and uses Model 3's selected configuration. No configuration may be added,
dropped or extended for one model only. All step budgets count step 0 and include every
environment step.

Planning figure **[measured]**: the frozen runner executed 7,560 control steps in
4.6–5.3 s on an Apple M2, about 1,400–1,600 steps/s. That is simulation only. Campaign
feasibility is **not assumed**: `AT-E2E-5` (Gate 2A) times the full training pipeline and
projects wall-clock hours for Gates 2B and 2C. Those projections are estimates, not
guarantees, and must be reviewed and approved before either gate is authorized. Secondary
analyses (`TRUNC-BOOT`, `GAMMA-1`, `*_ovp`, normalization ablations) are outside the totals
above and run only within an approved budget.

### 1.10 Scientific comparisons and their confounds

Reports state each confound next to its comparison, and keep the four claim groups of
Section 1.2 in separate tables.

**Primary family (Holm-corrected, Section 5.3):**

1. **`H-graph`: Model 3 vs. Model 3-self-loop (`D^msg`)** isolates inter-node message
   passing. Window, features, node encoder, weight sharing, layers, heads, widths, nominal
   parameters, initialization protocol, data, budgets, tuning effort and evaluation are
   identical; only the edge mask differs. Two consequences of that single difference are
   reported with every result: Model 3-self-loop's actor heads lose all inter-node
   information, so the contrast measures the mechanism and the information it carries
   together; and 17,664 of its parameters cannot learn (Section 1.8).
2. **`H-rec`: Model 4 vs. Model 3 (`D^rec`)** compares recurrent with fixed-window graph
   processing under approximately matched budgets (54,197 vs. 52,933 parameters). It does
   **not** isolate recurrence:
   - temporal fusion moves from before message passing (Model 3 concatenates 8 frames per
     node) to after it (Model 4 runs GATv2 per frame, then the GRU);
   - hidden width changes from 64 to 48 to match parameters;
   - **elapsed-time and horizon inference is a known potential confound for Model 4.** The
     episode start is visible through zero padding, the cold frame `o_0` (its latency
     feature is 0, which no later frame can have) and the GRU's zero state. Model 1 can tell
     only `k = 0`, the windowed models can tell the step up to `k = 7`, and Model 4 can in
     principle encode elapsed time, and so the remaining horizon, for the whole episode. A
     difference for Model 4 could come from horizon counting rather than load memory.
     `TRUNC-BOOT` (Section 3.3) is a sensitivity analysis only and does not remove this.
   - **Future control, not in this ladder:** per-frame GATv2 followed by a non-recurrent
     temporal aggregator over the same `W = 8` window would separate fusion order from
     recurrence.

**Secondary contrasts (descriptive, Section 5.3):**

3. **Model 3 vs. Model 3-complete:** the specified call topology against dense all-to-all
   connectivity. Model 3-complete uses Model 3's configuration, a tuning asymmetry stated
   with the result.
4. **Model 3-self-loop vs. Model 2:** shared per-node processing and graph heads without
   message passing, against a flat MLP. Model 2's trunk sees all nodes; Model
   3-self-loop's actor heads see one node each.
5. **Model 2 vs. Model 3:** overall non-graph versus graph architecture. Weight sharing,
   head structure and message passing change together, so it attributes nothing to any one
   of them.
6. **Model 1 vs. Model 2:** a fixed history window without a graph. Hidden width also
   changes (102 vs. 55).

**Baseline comparison (descriptive):** every learned model against `static`, `sim_hpa` and
`tuned_reactive_threshold` on the same traces. Budget parity applies only among learned
models; the baselines received no tuning on the training distribution.

---

## 2. Observation Space, Partial Observability Audit and Action Space

### 2.1 Frozen observation audit

`node_features[i, f]`, from `sim/env.py:480-498`. Interval means are over the 5 substeps
just completed. `rate_scale = service_rate × max_replicas = [1000, 1200, 800, 500, 300]`.
`q_max = [2000, 2000, 1500, 1000, 3000]`, `r_max = [10, 10, 10, 10, 1]`.

| f | Name | Exact definition before the clip | Clip active by default? |
|---:|---|---|---|
| 0 | Arrival rate | `mean(arrivals_new) / rate_scale` | Yes: Gateway above 1000 req/s, Database above 300 calls/s |
| 1 | Retry rate | `mean(arrivals_retry) / rate_scale` | Not proved impossible; treat as possible |
| 2 | Queue pressure | end-of-interval queue `/ q_max` | No, except rounding (queue capacity is enforced) |
| 3 | Utilization | `sum(completed) / sum(capacity)`; if capacity is 0: 1 when work completed, else 0 | No, except rounding (completed ≤ capacity) |
| 4 | Ready | `ready / r_max` | No |
| 5 | Pending | `pending / r_max` | No |
| 6 | Latency | `mean(node_latency_ms) / 1000` | **Yes: above 1000 ms. Node latency can reach base + 5000 ms** |
| 7 | Failure ratio | `F / max(1, F + completed)` over interval sums | No |
| 8 | Cooldown | `cooldown / max(1, 6)` | No |

At reset, features 0, 1, 3, 6, 7 are zero and 2, 4, 5, 8 use real state [test]. Features 5
and 8 of the Database are always 0 and its feature 4 is always 1.

### 2.2 Shield and lifecycle observability audit

Decision time is when `o_k` is available and `approve` has not been called. With the
default config (`boot 20 s`, `termination 5 s`, `substeps 5`, `cooldown 6`):

| Internal state | Size | Observable at decision `k`? | Argument |
|---|---|---|---|
| `ready`, `pending`, `desired` | 5 each | **Exactly**, one frame | `rint(f4·r_max)`, `rint(f5·r_max)`, and `desired = ready + pending` (invariant, `sim/env.py:452`) [test for desired] |
| Cooldown counters | 5 | **Exactly**, one frame | `rint(6·f8)`. Features are built after `_advance_cooldowns` (`sim/env.py:231-233`), so `f8` is the value the next arbitration reads [test] |
| Termination register | 5×5 | **Always empty** | Pods enter slot 4 only at the start of a step and are gone after the 5 advances of that step (`sim/env.py:302-324`) [derived, `AT-ENV-1`] |
| Cluster occupancy and quota headroom | 1 | **Exactly**, one frame | `occupancy = Σ ready + Σ pending + 0`; headroom `= 28 − Σ_i rint((f4+f5)·r_max)` including the Database [derived, `AT-ENV-2`] |
| **Shield output for any `a_raw`** | 4 | **Exactly**, one frame | `arbitrate` reads only proposal, desired, pending, cooldown, occupancy and config, all recoverable above [derived, `AT-ENV-3`] |
| Pending cancellation | – | **Never happens** | `pending > 0` implies `cooldown >= 4 > 0`, which blocks scale-down [test] |
| Boot register, per-slot ages | 5×20 | **Not from one frame. Exactly from 3 frames** | See below [derived, `AT-ENV-4`] |
| Queue age buckets | 5×5 | **No.** Totals only (f2) | Bucket 0 is empty at decision time (`sim/env.py:379-380`), leaving 4 buckets and one known sum per node: **15 hidden** continuous dimensions |
| Downstream buffer | 5 | **No** | A linear function of last-substep completions; under the default topology **2** independent dimensions (Gateway completions; the database-bound mixture) |
| Retry buffer | 5 | **No** | `0.5 × timeouts` of the last substep: **5** dimensions |
| Substep-level dynamics in the interval | – | **No** | Only interval means and sums are exposed |
| Values beyond active clips | – | **No** | f0, f1 and f6 (Section 2.1) |
| Time index, remaining horizon | 1 | **No** | Not in the observation |
| Future external traffic | – | **No** | Exogenous |

**Boot cohorts.** Approvals happen only at control-step boundaries and boot takes exactly
4 of them, so at decision time pending pods occupy only slots 14, 9 and 4: cohorts `c_1,
c_2, c_3` approved 1, 2 and 3 steps ago. No pod becomes ready inside an interval. Without
cancellation, `c_j(k) = max(Δapproved_{k-j}, 0)` and `Δapproved_{k-j} =
desired_{k-j+1} − desired_{k-j}`. So frames `k-2, k-1, k` give `c_1` and `c_2`, and `c_3 =
pending_k − c_1 − c_2`. From one frame, cooldown 4 implies `c_3 = pending`, but the split
is ambiguous when cooldown is 5 or 6 and `pending >= 2`. Near the episode start, missing
frames mean no approvals occurred.

**Aliasing example [derived, `AT-ENV-6`].** From reset, hold, with Gateway traffic
(a) `[900, 600, 200, 200, 200]` or (b) `[200, 200, 200, 600, 900]` req/s over the first 5
substeps. The 2 Gateway pods complete 200 in every substep of both, so all downstream
quantities are equal. Both give the Gateway f0 `0.42`, f1 `0`, f2 `1100/2000 = 0.55`, f3 `1`,
f4 `0.2`, f5 `0`, f7 `0`, f8 `0`. Mean backlog delay is 4,700 ms and 1,400 ms, so f6 clips to 1
in both. **The two observations are identical.** The queues are not: ages 1–4 s hold
`[200, 200, 200, 500]` in (a) and `[900, 200, 0, 0]` in (b). The next expiry is 300 requests at
`t = 5` in (a) and 300 at `t = 8` in (b), so the next interval's trajectories differ.

**Consequences:**

1. Cooldown, quota headroom and shield arbitration are **not** hidden. No model needs
   memory to predict the shield. The exception is Model 3-self-loop's actor: each head sees
   one node, so it cannot compute quota headroom.
2. Boot timing is hidden from Model 1 only. For the windowed models each node's boot state is
   inside its own window.
   **So recurrence has no lifecycle information that the window lacks.** A gain for Model 4
   would have to come from elsewhere: history older than 40 s, better inference of hidden
   queue and buffer state, horizon counting, optimization, or the confounds of Section 1.10.
3. Windowed and recurrent policies can also recover past approved actions, which equal
   the differences of `desired`. No model receives `a_raw`, `a_approved` or reward as input.

These results hold for the default config only. Other boot, termination, cooldown or
substep values break them.

### 2.3 Formal POMDP

$$\mathcal P = (\mathcal S, \mathcal A, P, R, \Omega, O, \gamma, H),\qquad H = 360,\; \gamma = 0.99$$

- **State** `s_k = (Q [5×5], ready, boot register, termination register, cooldown,
  downstream buffer, retry buffer, last-interval statistics, t, ξ)`, where `ξ =
  λ_{0:1799}` is the episode's external trace, drawn once per episode. Conditional on `ξ`
  the process is deterministic.
- **Action** `a_raw ∈ A = {-1, 0, 1, 2}^4`, 256 joint actions.
- **Transition** `s_{k+1} = F(s_k, Shield(s_k, a_raw))`. The shield is part of `P`.
- **Observation** `o_k = O(s_k)`, deterministic and many-to-one.
- **Reward** `R` as frozen (Section 0).

Because `ξ` is hidden and fixed per episode, this is a hidden-parameter POMDP. The
optimal policy depends on the full history `h_k`.

**Frozen boundary:** Stage 1A observations may not be modified, wrapped with extra
fields, or supplemented from `info`, `env` properties or `step_index`. No state may be
given to one architecture or to a critic only. Asymmetric ("privileged") critics are
forbidden in the primary ladder.

**Implications:**

- **Feedforward, `W=1` (Model 1):** a reactive policy. It cannot see load trend, nor the
  boot cohort split when ambiguous, and it must act on aliased states.
- **Feedforward, windowed (Models 2, 3, 3-self-loop, 3-complete):** lifecycle is fully
  resolved. Queue ages and buffers are only partly inferable, and history older than 40 s is
  unavailable. Model 3-self-loop's actor is further limited to its own node.
- **Recurrent (Model 4):** unbounded memory in principle, limited by BPTT length, stale
  stored states (Section 3.4) and optimization.
- **Value estimation:** the correct baseline is `V(h_k)`. Every critic here approximates
  it from its own input, so bootstrapped targets inherit aliasing bias. With `λ < 1` this
  bias reaches the advantages. The unobserved remaining horizon adds error near the end:
  under constant cost `c`, true value is `-c(1-γ^{360-k})/(1-γ)`, and `γ^{100} = 0.37`.
  A time-blind critic is therefore biased mainly in the last 100–200 steps.

### 2.4 Feature normalization policy

All 9 features are bounded in `[0, 1]`; none is unbounded. **The primary ladder uses
identity normalization**: no running statistics, no reward normalization, the same for all
models. Per-node running normalization is also ill-conditioned here: three Database columns
are constant, so their variance is zero.

If a secondary ablation uses running normalization, it must apply to every model;
statistics must be fitted on training rollouts only; they must be frozen and stored in the
checkpoint before any validation or test episode; variance must be floored at `1e-4`; and
evaluation code must be unable to update them.

### 2.5 Action space

- `MultiDiscrete([4, 4, 4, 4])` over `Gateway, Auth, Catalog, Order`, with
  `n = action_max − action_min + 1 = 4`. Index `j` means `delta = j + action_min`, so
  `0,1,2,3 -> -1,0,+1,+2`.
- Four independent categoricals given the trunk output:

$$\log\pi_\theta(a_{raw}\mid h_k) = \sum_{i=0}^{3}\log\operatorname{softmax}(z_i)[j_i],\qquad \mathcal H = \sum_{i=0}^{3}\mathcal H_i \le 4\ln 4 = 5.545$$

- **Training:** sample every head from its categorical with the run's action generator.
- **Evaluation:** per-head `argmax`, ties to the lowest index. Because the policy is
  factored, this is the joint mode.
- Proposals are passed to the shield as a tuple of 4 plain Python `int`. They are always
  valid, so `convert_proposal` returns them unchanged and `fallback_count` stays 0.
  Tensors, NumPy scalars and booleans are forbidden.
- No action masking in the primary ladder.

### 2.6 Step-0 cold telemetry

At `k = 0` there is no interval history, and every frozen baseline holds. For all models,
in training, validation and test:

- The action is the constant `(0, 0, 0, 0)`, approved by the shield once and stepped. The
  policy is not sampled.
- `o_0` still enters the window, and Model 4 still runs its encoder and GRU on it.
- The transition is stored with `train_mask = 0` and is excluded from the policy, value
  and entropy losses, from advantage normalization and from GAE targets. `r_0` still counts
  in every reported episode return. Trainable transitions: 359 per episode.
- Its training fields hold the finite sentinels of Section 3.2, and the NaN-safe masking
  rules there apply.

---

## 3. Shield-in-the-Loop PPO and Recurrent BPTT

### 3.1 Exact shield action semantics

Per environment step with `k >= 1`:

1. `z = actor(h_k)`; sample `j`; `a_raw = j + action_min`; `old_log_prob = log π_old(a_raw | h_k)`.
2. `token = env.shield.approve(tuple(int(x) for x in a_raw))`, **exactly once**.
3. `a_approved = token.deltas`, then `env.step(token)`. Tokens are never stored.

Rollout buffer, `T = 360`, `E = 8`:

| Field | Shape, dtype | Use |
|---|---|---|
| `obs` | `[T, E, 5, 9]` float32 | model input; windows are rebuilt from it per episode |
| `action_index` | `[T, E, 4]` int64 | sampled `j`; **the only action used by the loss** |
| `a_raw` | `[T, E, 4]` int64 | `action_index + action_min` |
| `old_log_prob` | `[T, E]` float32 | joint `log π_old(a_raw)` |
| `old_value` | `[T, E]` float32 | `V_old` in return units |
| `reward` | `[T, E]` float64 | the unmodified `env.step` reward |
| `terminated`, `truncated` | `[T, E]` bool | Section 3.3 |
| `train_mask` | `[T, E]` bool | `k >= 1` |
| `a_approved` | `[T, E, 4]` int64 | **diagnostic only** |
| `override_mask` | `[T, E, 4]` bool | `a_raw != a_approved`; **diagnostic only** |
| `fallback_count` | `[T, E]` int64 | must be 0; **diagnostic only** |
| `h_actor`, `h_critic` | `[T, E, 5, 48]` float32 | Model 4: state **before** processing step `k` |
| `episode_id`, `trace_sha256` | per environment | provenance |

Rules:

- $r_t(\theta) = \exp\left(\log\pi_\theta(a_{raw,t}\mid h_t) - \log\pi_{old}(a_{raw,t}\mid h_t)\right)$. The ratio uses `a_raw` only.
- The shield is deterministic transition dynamics. It is not a distribution over actions,
  and the executed action is not a sample from the policy. Computing `log_prob(a_approved)`
  when `a_approved != a_raw` is **forbidden**: the policy did not draw it, so the ratio
  would be wrong and the gradient biased. No code path may evaluate it, including logging.
- The estimator is valid because the behavior distribution is exactly `π_old(a_raw | h)`
  and the induced transition kernel is `P(s' | s, Shield(s, a_raw))`.
- `approve` is called exactly once per step: once in the collector, and once inside frozen
  `run_episode` during evaluation. Controller adapters never call it.
- `info["substeps"]` and `info["reward_components"]` may be logged and never enter a loss.

**Primary reward integrity:** primary training and evaluation use the frozen reward with
no shaping, scaling, clipping or normalization. `C^{scale}` uses approved deltas, so a fully
blocked proposal costs nothing and override rates may stay high. That is reported, not
corrected. An override penalty, `r' = r − 0.05 · |override_mask| / 4`, is allowed only as a
separate secondary ablation, in runs named `*_ovp`, excluded from every primary table and
statistic.

### 3.2 PPO objective

For one iteration, $\mathcal M$ is the set of slots with `train_mask = 1` and no padding.

$$\hat A_t = \frac{A_t - \operatorname{mean}_{\mathcal R}(A)}{\operatorname{std}_{\mathcal R}(A) + 10^{-8}}$$

normalized **once per rollout** over all 2,872 trainable slots $\mathcal R$, so feedforward
and recurrent models use the same statistics regardless of minibatch composition.

$$L_{clip} = -\frac{1}{|\mathcal M|}\sum_{t\in\mathcal M}\min\left(r_t\hat A_t,\ \operatorname{clip}(r_t, 1-\varepsilon, 1+\varepsilon)\hat A_t\right)$$

$$L_V = \frac{1}{2|\mathcal M|}\sum_{t\in\mathcal M}\left(\hat v_t - \frac{R_t}{c_V}\right)^2,\qquad L_H = -\frac{c_H}{|\mathcal M|}\sum_{t\in\mathcal M}\mathcal H_t,\qquad L = L_{clip} + L_V + L_H$$

| Fixed setting | Value |
|---|---|
| Optimizer | Adam, betas (0.9, 0.999), eps 1e-5, no weight decay, constant learning rate |
| Epochs, minibatches | 5 epochs × 4 minibatches of 720 slots |
| Gradient clipping | global norm 0.5, separately for actor and critic |
| `γ`, `λ`, `c_V` | 0.99, 0.95, 100 |
| Value clipping, KL early stopping, target networks | none, so every model does exactly 7,200 updates |

Feedforward minibatches shuffle all 2,880 slots. Masked slots keep their place and are
excluded from every loss by the NaN-safe rules below. Approximate KL, clip fraction,
explained variance, entropy, gradient norms and override rate are logged every iteration.

**Discount factor.** `γ = 0.99` is a **discounted optimization surrogate** for the
undiscounted 360-step evaluation cost. Policies are trained to optimize the surrogate, not
the evaluation metric itself. Secondary sensitivity check `GAMMA-1`: the same training with
`γ = 1.0`, or `γ = 0.995` if `γ = 1.0` is numerically unstable in Gate 2A, with `λ`, `c_V`
and the selected configurations unchanged. At `γ = 1` returns reach at most
`360 × 15.7 = 5,652` in magnitude, `56.5` after the output scale. Scope, fixed at the Gate
2C authorization from the `AT-E2E-5` projections: all five primary models at N = 5 if they
permit; otherwise Models 3, 3-self-loop and 4. If neither fits, `GAMMA-1` is not run and the
surrogate mismatch is stated as a methodological limitation in every results section.

**Masked and padded slots (NaN-safe).** A slot is masked if `train_mask = 0` (step 0) or if
it is padding (Section 3.4).

1. **Finite sentinels.** Masked slots store `action_index = 1` (HOLD), `old_log_prob = 0.0`,
   `old_value = 0.0`, `advantage = 0.0` and `return = 0.0`. Padded slots also store
   `reward = 0.0`, all-zero observations and zero hidden states. Step-0 rewards and
   observations are real and kept.
2. **Mask operands, not results.** Before any arithmetic with a gradient path, masked
   elements are either removed by indexing (`x[mask]`), or replaced with
   `torch.where(mask, x, 0.0)` where `x` is a stored field or a network input. Padded
   observations are replaced this way before the forward pass.
3. **Forbidden:** multiplying by a 0/1 mask (`m * x`) in any loss or logged mean, because
   `0.0 * NaN = NaN`. Also forbidden: applying `torch.where` only to a finished loss term.
   If the discarded branch is NaN, its local derivative is NaN, and the chain rule
   multiplies it by the zero upstream gradient, which again gives NaN.
4. Losses and logged means are sums over the selected elements divided by their count.
   `AT-PPO-6` and `AT-REC-4` inject NaN into masked and padded slots.

### 3.3 GAE and terminal versus truncation bootstrapping

With `d^{term}_t = terminated`, `d^{trunc}_t = truncated`, and `V(o'_t)` the value at the
observation returned by step `t`, with the recurrent state continued:

$$\delta_t = r_t + \gamma\,(1-d^{term}_t)\,V(o'_t) - V(o_t)$$

$$A_t = \delta_t + \gamma\lambda\,(1-d^{term}_t)(1-d^{trunc}_t)\,A_{t+1},\qquad R_t = A_t + V_{old}(o_t)$$

- **Terminal:** no bootstrap. **Truncation:** bootstrap from `V(o'_t)`, then stop the
  advantage recursion.
- Stage 1A sets `terminated=True` at `k == control_horizon` and never truncates.
  **Primary training stays faithful to the frozen environment: no bootstrap, `V = 0` after
  step 359.** It matches the evaluation metric, a 360-step sum.
- The task is finite-horizon by definition of the frozen environment, although a real
  autoscaler is not. With no time feature, the critic cannot see the horizon (Section 2.3).
- Rollouts contain whole episodes, so no rollout-boundary bootstrap exists. The collector
  raises `ValueError` if an episode does not end exactly at `T`.
- GAE runs in float64 for `k = 1..359`. Slot 0 gets no target.
- `TRUNC-BOOT` is a secondary **sensitivity analysis**, run for Models 3 and 4. It is the same
  training, except that GAE treats the horizon as truncation:

  $$\delta_{359} = r_{359} + \gamma V(o_{360}) - V(o_{359})$$

  `o_360` is the observation returned by the final `env.step`. `V_old(o_360)` is computed at
  collection with the collection parameters and stored, one value per episode:
  - windowed models slide the `W = 8` window forward by one step, to `o_353 .. o_360`;
  - Model 1 uses `o_360` alone;
  - Model 4 follows exactly four steps:
    1. take the critic GRU hidden state immediately after processing `o_359`, the state
       that produced `V(o_359)`;
    2. feed `o_360` exactly once, from that state;
    3. compute `V(o_360)` with the collection-time parameters;
    4. do not reset or replay the recurrent history.

  The advantage recursion still stops after step 359. `TRUNC-BOOT` removes the
  terminal-value discontinuity and with it part of the end-of-horizon incentive. **It does
  not eliminate horizon counting**, because a GRU can still encode elapsed time from
  `h_0 = 0` and `o_0`. Its results are never mixed into primary results, and never cited as
  having removed the confound.

### 3.4 Recurrent minibatch mechanics (Model 4)

1. **Collect** whole episodes with zero `h_actor` and `h_critic` at `k = 0`. Store the state
   held before each step, detached.
2. **Compute GAE, returns and normalized advantages on the full `[T, E]` rollout, before
   any splitting.**
3. **Split** each episode into consecutive chunks of `L = 36` steps. 360 = 10 × 36, so there
   are 80 chunks and no padding by default. Padding must still work: an episode whose
   length is not a multiple of `L` yields a final chunk padded at the end to `L`.
4. **Initial state:** each chunk starts from the stored `h_actor`, `h_critic` at its first
   step. The first chunk of an episode starts from zeros. Gradients do not flow into it.
5. **Never cross an episode.** A chunk holds steps of one episode only, in strict temporal
   order.
6. **Minibatch:** shuffle **chunks**, 20 per minibatch (720 slots), as `[L, 20, ...]`.
   **Timestep-level shuffling is forbidden for recurrent minibatches**, and the batching
   function must be unable to produce it.
7. **Forward** the encoder and GRU over all `L` steps. Step 0 runs through the GRU with zero
   loss weight.
8. **Masks:** `valid = not padding`, `m = valid AND train_mask`. Policy, value and entropy
   losses and all logged means are the mean over slots with `m = 1`, computed under the
   NaN-safe rules of Section 3.2: sentinels in masked slots, padded observations replaced
   before the forward pass, operands masked before arithmetic, never `m * x`. Padded slots
   are excluded from advantage normalization. A change to padded inputs, NaN included, must
   leave the loss and gradients unchanged.
9. **Staleness:** stored states come from the collection parameters and are not refreshed
   across epochs. Before the first gradient step they are exact, so the ratio is 1. After
   it they are approximate, which is standard and accepted. The fraction of slots clipped is
   logged per epoch as a monitor.

Truncated BPTT length is `L` = 36 steps, 180 s.

---

## 4. Strict Train / Validation / Held-Out Test Split

### 4.1 Training distribution

Episode index `e = 0..2879`, regime `e mod 7`. Each episode has one allocated
`workload_seed` (Section 4.4), which stays its identity across every redraw. Attempt
`a = 0, 1, 2, ...` derives two independent streams:

    param_ss, gen_ss = np.random.SeedSequence([workload_seed, a]).spawn(2)

- **Stream 0** drives the parameter draws: `rng = np.random.default_rng(param_ss)`, drawing in
  the listed order, with `rng.uniform(lo, hi)` for rates and probabilities and
  `rng.integers(lo, hi, endpoint=True)` for integers.
- **Stream 1** gives the Poisson generator seed, a plain non-negative `int`:
  `generator_seed = int(gen_ss.generate_state(1, dtype=np.uint32)[0])`, passed to frozen
  `seeded_bursty_poisson(seed=generator_seed)`. Each attempt therefore derives a distinct
  `SeedSequence` stream, while `workload_seed` never changes. Distinct traces are not
  guaranteed for every pair of attempts; actual duplicates and near-duplicates are caught
  by the leakage guard below (`AT-WRK-1`, `AT-WRK-3`).

The run log and provenance metadata record `workload_seed`, `attempt` and
`generator_seed` for every episode. Traces are built by the frozen `sim.workloads`
generators with `horizon = 1800`.

| # | Generator | Randomized parameters |
|---:|---|---|
| 0 | `constant` | `rate` U[50, 400] |
| 1 | `step_jump` | `base_rate` U[50, 200], `jump_rate` U[200, 800], `jump_time` I[120, 1500] |
| 2 | `gradual_ramp` | `start_rate` U[50, 200], `end_rate` U[300, 900] (upward only) |
| 3 | `flash_crowd` | `base_rate` U[50, 200], `peak_rate` U[400, 1000], `start_time` I[120, 1200], `duration` I[30, 300] |
| 4 | `burst_and_drop` | `base_rate` U[50, 200], `peak_rate` U[400, 900], `ramp_start` I[120, 1000], `ramp_duration` I[20, 180], `hold_duration` I[60, 360] |
| 5 | `db_bottleneck` | `base_rate` U[50, 200], `surge_rate` U[300, 600], `start_time` I[120, 900], `duration` I[180, 800] |
| 6 | `seeded_bursty_poisson` | `base_rate` U[80, 250], `burst_rate` U[400, 900], `burst_prob` U[0.005, 0.03], `min_burst_len` I[10, 30], `max_burst_len` I[min, 90], `seed = generator_seed` |

Every drawn window fits in 1,800 s. Training seed `s` determines the whole sequence, so
**all models see identical traces in identical order for the same `s`**.

**Leakage guard:** a draw is rejected and redrawn with `attempt + 1` (at most 100) if its
`sha256` equals any validation or test trace, or if its mean absolute difference from a
held-out trace of the same generator is below 10 req/s. Rejections are logged. Every
training trace's SHA-256 is written to the run log.

### 4.2 Validation suite

21 fixed traces: `workload_seed = 200_000_000 + i`, regime `i mod 7`, `i = 0..20`, drawn
like training traces, with the leakage guard applied against test traces only. Used **only**
for configuration and checkpoint selection, with deterministic actions.

With `C = −total_reward > 0`, the **validation score** is the mean log cost ratio to the
frozen static baseline on the same trace:

$$S_{val} = \frac{1}{21}\sum_{w}\ln\frac{C_{policy,w}}{C_{static,w}}\qquad\text{(lower is better)}$$

It weights traces equally, so high-cost regimes do not dominate. Mean `total_reward` is also
reported. The checkpoint with the lowest `S_val` is selected; ties go to the earliest. The
last-iteration checkpoint is reported as a secondary result.

### 4.3 Held-out test suite

**Canonical.** The 7 traces of `benchmark_suite(seed=42)`, head-to-head against `static`,
`sim_hpa` and `tuned_reactive_threshold` through frozen `sim/runner.py`, environment-reset
seed 0. These are unseen **traces** from trained **families**: in-distribution.

**Generalization.** 18 traces, also run for the three frozen baselines:

| ID | Trace | Shift from training |
|---|---|---|
| GEN-01..10 | `seeded_bursty_poisson(seed=300_000_000 + i)`, `i = 0..9` | unseen seeds |
| GEN-11 | `flash_crowd(start_time=30)` | earlier than any training start |
| GEN-12 | `flash_crowd(start_time=1500, duration=240)` | later than any training start |
| GEN-13 | `flash_crowd(peak_rate=1300.0)` | above training amplitude; Gateway f0 clips |
| GEN-14 | `step_jump(jump_rate=1000.0)` | above training amplitude |
| GEN-15 | `gradual_ramp(start_rate=700.0, end_rate=100.0)` | downward ramp |
| GEN-16 | `multi_burst`: base 100; 700 on `[300,360)`, `[600,660)`, `[900,960)` | unseen family |
| GEN-17 | `escalating_bursts`: base 100; 400 on `[240,330)`, 700 on `[660,750)`, 1000 on `[1080,1170)` | unseen family |
| GEN-18 | `triangle_wave`: `100 + 500·(1 − abs((t mod 300) − 150)/150)` | unseen family |

GEN-16..18 are built as `WorkloadTrace(name, params, values)` in `rl/`, without editing
`sim/workloads.py`.

**Access rule.** Test traces are never run by a learned policy before the **test-open
commit**, which lists every final checkpoint's SHA-256, all selected configurations and the
analysis code. After it, the suite runs once. A rerun needs written authorization and is
reported. Tuning is closed at that point.

**Known limits.** Stage 1B results on the canonical suite were known to the authors before
this specification, so canonical traces are unseen by the policies but not blind to the
designers. The frozen baselines were not tuned on the training distribution while RL is.

### 4.4 Disjoint seed ranges

| Group | Purpose | Range |
|---|---|---|
| (a) | RL training seeds, final | 10_000 – 10_999 (N = 5: 10_000..10_004; N = 10: 10_000..10_009) |
| (a) | RL training seeds, tuning | 11_000 – 11_999 (used: 11_000, 11_001) |
| (a) | RL training seeds, tests and smoke runs | 12_000 – 12_999 |
| (b) | Workload generation, training | 100_000_000 – 119_999_999 |
| (b) | Workload generation, validation | 200_000_000 – 200_000_999 |
| (c) | Environment reset, training | 400_000_000 – 419_999_999 |
| (c) | Environment reset, validation | 500_000_000 – 500_000_999 |
| (d) | Held-out workload generation | 300_000_000 – 300_000_999, plus 42 (frozen canonical) |
| (d) | Held-out environment reset | 0 only, matching the frozen baseline runs |
| – | Bootstrap analysis | 600_000_000 |
| – | Workloads and resets in tests and smoke runs | 700_000_000 – 700_999_999 |

Offsets: final `(s − 10_000)·10_000 + e`; tuning `10_000_000 + (s − 11_000)·10_000 + e`.
Workload seed = `100_000_000 + offset`, reset seed = `400_000_000 + offset`, `e < 10_000`.

`generator_seed` (Section 4.1) is derived, not allocated: it can numerically equal any
32-bit value, including 42. Disjointness of **traces** is enforced by the SHA-256 and
near-duplicate guard of Section 4.1, not by seed ranges.

A held-out evaluation episode is identified by `(controller name, trace sha256, reset seed
0)`. Reset seeds are recorded for provenance only, because physics ignores them.

---

## 5. Statistical Discipline and `sim/runner.py` Compatibility

### 5.1 Controller adapter

`rl/controllers.py` defines one `BaseController` subclass for all models:

- `name = "rl_<model id>_s<training seed>"`, for example `rl_m3_gatv2_w8_s10003`. The seed
  is required: two seeds with one name would collide on manifest file names.
- Reads only `observation["node_features"]`. Topology comes from the config at
  construction. Every `propose` call requires `config == SimulatorConfig()`, else
  `ValueError`.
- `reset()` clears the window, the hidden state and the expected step counter.
- `propose` requires `step_index` to equal the expected counter, else `ValueError`, uses it
  for nothing else except the step-0 hold, and returns 4 plain `int`. At step 0 it still
  updates its window or hidden state from `o_0`.
- Inference on CPU, float32, `torch.inference_mode()`, one thread, deterministic `argmax`.
- It never imports or receives the environment or shield.

`params` is **flat**, so the frozen shallow copy is a full copy. Keys are plain `str`:

`adapter_schema`, `spec_version`, `model_id`, `training_seed`, `checkpoint_sha256`,
`parameters_sha256`, `parameter_count`, `checkpoint_iteration`, `env_steps_trained`,
`config_id`, `learning_rate`, `entropy_coef`, `clip_epsilon`, `gamma`, `gae_lambda`,
`epochs`, `minibatches`, `n_envs`, `rollout_steps`, `chunk_length`, `window`,
`hidden_dim`, `gat_layers`, `gat_heads`, `gru_hidden`, `value_scale`, `max_grad_norm`,
`train_git_commit`, `torch_version`, `device`, `action_selection`, `step0_rule`.

Inapplicable values are `None`. All leaves must be exactly `None, bool, int, float, str`.
`torch.__version__` is a `str` **subclass**, which `json_safe` rejects, so it must be passed
through `str()`; NumPy and tensor scalars must be converted too. The manifest's
`spec_versions` is frozen and has no RL entry, so `spec_version` lives in `params`.

Evaluation is `run_benchmark_suite(controllers, traces, config=None, seed=0,
output_dir=...)`, emitting unmodified schema `1.0.0` manifests. `output_dir` must be outside
the repository and the tree must be clean, so manifests record `dirty: false`.

### 5.2 Seeds are different things

| Seed | Controls | Is it replication? |
|---|---|---|
| Training seed | initialization, action sampling, minibatch order, workload sequence | **Yes, the only one** |
| Workload-generation seed | one trace | No. Traces are fixed test conditions |
| Environment-reset seed | nothing in the physics | No |
| Evaluation episode | one deterministic run | No. Repeating it gives an identical manifest |

### 5.3 Statistical unit, primary family and exploratory status

The unit is the **training seed `s`**, paired across models through shared traces,
budgets and evaluation. `n` is the number of seeds, **never** seeds × workloads. Per model
`a` and seed `s`, the endpoint on the canonical suite is

$$S_{a,s} = \frac{1}{7}\sum_{w}\ln\frac{C_{a,s,w}}{C_{static,w}}$$

**Primary family: exactly two contrasts.**

$$D^{msg}_s = S_{M3,s} - S_{M3\_selfloop,s},\qquad D^{rec}_s = S_{M4,s} - S_{M3,s}$$

- Each is tested by a two-sided paired t-test, $t = \bar D / (s_D/\sqrt N)$ with `N − 1`
  degrees of freedom, giving an exact two-sided p-value from the t distribution.
- **Holm correction, family-wise α = 0.05, is applied to exactly these two p-values and to
  nothing else.** With $p_{(1)} \le p_{(2)}$: reject $H_{(1)}$ if $p_{(1)} \le 0.025$; only
  then, reject $H_{(2)}$ if $p_{(2)} \le 0.05$. Adjusted values:
  $\tilde p_{(1)} = \min(1, 2p_{(1)})$, $\tilde p_{(2)} = \max(\tilde p_{(1)}, p_{(2)})$.
- Each is reported with its unadjusted 95% paired t-interval,
  $\bar D \pm t_{0.975,N-1}\,s_D/\sqrt N$ (2.776 at N = 5, 2.262 at N = 10).

**Secondary and descriptive checks, not a second confirmatory family:**

- the exact two-sided sign-flip (permutation) test of each primary contrast;
- **95% paired percentile bootstrap intervals** over seeds, 10,000 resamples, seed
  600_000_000;
- every secondary contrast of claim 3 (Section 1.2), as unadjusted 95% paired t-intervals and
  bootstrap intervals, with no significance language;
- the baseline comparison of claim 4: per baseline, one-sample 95% t-intervals of the per-seed
  differences, because the baselines are deterministic. Reported in its own table, which
  states that the baselines were not tuned on the training distribution.

All per-seed values are tabulated and plotted as points next to every interval. Results
tables and conclusions are organized by the four claim groups of Section 1.2 and are never
merged.

**Exploratory status.** The Stage 2 campaign is **exploratory**. No formal power
calculation exists, so no result may be called confirmatory, and a non-rejection is not
evidence of no effect. It becomes confirmatory only if a pre-specified power calculation is
added to this document before the test-open commit: target effect size, a variance estimate
from data other than the test suite (for example Gate 2B validation scores), α = 0.05 with
Holm, and the resulting N. `N >= 5` seeds per model is the minimum exploratory baseline.
`N = 10` improves precision over `N = 5` and lets the two-sided sign-flip test go below 0.05
(minimum `2/2^10 ≈ 0.002`, against `2/2^5 = 0.0625` at N = 5), **but does not by itself
establish formal statistical power**. N is fixed before the test-open commit.

### 5.4 Mandatory metrics, per model, seed and workload

| Reported | Manifest path |
|---|---|
| Total reward | `metrics.total_reward` |
| Four weighted costs, total | `metrics.weighted_cost_sums.{slo_violation, failure_cost, resource_cost, scaling_cost, total_cost}` |
| Latency proxy p95, p99, max | `metrics.substep_latency_proxy_ms.{p95, p99, max}` |
| SLO-violating substeps | `metrics.slo_violation_substeps` |
| Mean failure proxy | `metrics.mean_e2e_failure_proxy` |
| Pod-seconds | `metrics.total_pod_seconds` |
| Scaling events | `metrics.total_scaling_events` |
| Shield overrides | `metrics.shield_arbitration_override_count` |
| Shield fallbacks | `metrics.shield_fallback_count`, **must equal 0**, else the evaluation is invalid |

Latency is the modeled proxy of SPEC_01 Section 24, not request percentiles.

### 5.5 No cherry-picking

- Every started seed is reported. Status: `completed`, `collapsed` (final-iteration `S_val`
  above `ln 1.5`, or mean entropy below 0.05 for 20 consecutive iterations), or
  `numerical_failure` (any non-finite loss, gradient or parameter).
- Labels never exclude a run. Collapsed runs enter every statistic. After a numerical
  failure, the best finite checkpoint is evaluated. The iteration-0 checkpoint always
  exists, so every seed always has a result.
- A seed may be rerun only after an infrastructure failure (power loss, out of memory),
  from scratch, with the incident logged.
- Seeds, configurations, endpoints and analysis are fixed before the test-open commit.

---

## 6. Software Layout, Determinism and Acceptance Criteria

### 6.1 Layout: isolated `rl/`, zero edits to the 14 frozen files

| Path | Purpose |
|---|---|
| `rl/__init__.py` | docstring only, imports nothing |
| `rl/settings.py` | frozen dataclasses for Sections 1.8–1.9 and 3.2 |
| `rl/seeds.py` | Section 4.4 ranges and offsets |
| `rl/graph.py` | edges and attributes from `call_multiplicity` |
| `rl/models.py` | GATv2 layer, four actors and critics |
| `rl/policy.py` | factored categorical, sampling, `argmax`, log-prob |
| `rl/workloads.py` | training, validation and generalization traces, leakage guard |
| `rl/collector.py` | environment stepping, single `approve`, rollout buffer |
| `rl/ppo.py` | GAE, losses, feedforward and recurrent minibatches |
| `rl/checkpoint.py` | save, load, hashes |
| `rl/controllers.py` | `BaseController` adapter |
| `rl/train.py`, `rl/evaluate.py`, `rl/stats.py` | entry points and analysis |
| `rl/tests/check_*.py` | Stage 2 tests |
| `requirements-rl.txt` | exact pins |

`sim/` never imports `rl/` or `torch`. `rl/` uses only the public frozen API of Section 0
and no attribute starting with `_`.

**Test isolation (verified constraint).** `tests/test_stage1a_physics.py:1347-1348` asserts
that `torch` is absent from `sys.modules` of its own pytest process. Pytest imports every
collected module before running any test, so one Stage 2 test module importing `torch` in
the same session would fail that frozen test. `tools/mutation_check.py:299` also copies
`sim/` and `tests/` without `rl/`, so a Stage 2 test placed in `tests/` would break it.
Therefore:

- Stage 2 tests live in `rl/tests/` and are named `check_*.py`, which default discovery
  ignores. There is no `conftest.py` under `rl/`.
- Two separate processes, both mandatory:
  `python -B -m pytest -v -p no:cacheprovider` (frozen: 1,070 tests) and
  `python -B -m pytest -v -p no:cacheprovider -o python_files='check_*.py' rl/tests`.
- No root `pytest.ini`, `conftest.py` or `pyproject.toml` is needed or allowed.

### 6.2 Dependencies

`numpy`, `gymnasium`, `pytest` and **`torch`** only. Forbidden: `torch_geometric`,
`torch_scatter`, `stable_baselines3`, `kubernetes`, `prometheus*`, `locust`. Pins are chosen
at implementation, not guessed here. The pinned environment must first pass the 1,070
frozen tests and reproduce all 21 benchmark manifests apart from `git`.

**Pure PyTorch GATv2 without `torch_geometric`:**

- The graph is static with 5 nodes and 17 edges. Dense masked attention is a few matrix
  products. Sparse message passing solves a scaling problem this graph does not have.
- Dense operations avoid scatter kernels, whose determinism and Apple MPS coverage vary
  with the version and would have to be verified for each release.
- The whole layer is short enough to audit line by line and to check by hand on small
  inputs (`AT-MOD-4`).
- Cost: no third-party reference. This is offset by hand-computed tests and, optionally, a
  one-off comparison outside the repository.

### 6.3 Device and determinism

- **CPU is the only device for reported training and evaluation.** The simulator is
  single-process NumPy, the networks have about 53,000 parameters, and CPU has the strongest
  reproducibility. `torch.set_num_threads(1)`, `torch.use_deterministic_algorithms(True)`.
- **MPS is exploratory only** and never enters a reported result. It lacks float64, and its
  bitwise determinism is not assumed. No speed claim is made, because nothing was measured.
  `torch` is not installed, so statements about it here are to be confirmed in Stage 2A.
- Seeding: from training seed `s`, `np.random.SeedSequence(s).spawn(3)` gives independent
  streams for initialization, action sampling and minibatch order. Workload seeds follow
  Section 4.4. No global generator is used. Environments are stepped sequentially in a
  fixed order.
- **Guarantee:** same machine, same pins, same seed, CPU, one thread gives an identical
  `parameters_sha256`. **No guarantee** across machines, operating systems, or `torch` or
  NumPy versions.
- Parallelism is across runs, one process per seed. `torch.compile` is forbidden.

### 6.4 Checkpoints

`torch.save` of a plain `state_dict` for actor and critic plus a JSON sidecar with
`params` (Section 5.1), log hashes and validation scores. Loading uses `weights_only=True`.
`checkpoint_sha256` is the SHA-256 of the file. `parameters_sha256` hashes, in sorted name
order, each tensor's name, dtype, shape and little-endian bytes, so it does not depend on
serialization details. Optimizer state is saved separately for resuming, never for evaluation.
Saved at iteration 0 and every 20 iterations, outside the repository, never overwritten.

### 6.5 Stage gates

| Gate | Content | Requires |
|---|---|---|
| 2A | `rl/` modules, tests, smoke training, and the `AT-E2E-5` throughput report | explicit authorization |
| 2B | tuning of the 5 primary models on training and validation traces | authorization after the 2A audit, including review and approval of the `AT-E2E-5` projection for Gate 2B |
| 2C | final runs (5 primary models and Model 3-complete), secondary analyses within the approved budget, test-open commit, held-out evaluation | authorization after the 2B audit, including approval of the `AT-E2E-5` projection for Gate 2C at the chosen N |

Passing a gate's tests does not authorize the next gate. `AT-E2E-5` projections are
estimates, not guarantees.

### 6.6 Acceptance tests

**Frozen boundary**

- `AT-FRZ-1` All 14 files match `stage-1b-frozen`; the frozen suite passes in its own process.
- `AT-FRZ-2` A bare root `pytest` collects 1,070 tests and never imports `torch`.
- `AT-FRZ-3` In the pinned environment, 21 benchmark manifests equal the frozen ones apart from `git`.
- `AT-FRZ-4` `sim/` has no import of `rl` or `torch`; `rl/` uses no private frozen attribute.

**Observability claims of Section 2.2** (random and adversarial proposals, full episodes)

- `AT-ENV-1` The termination register is all zero at every decision.
- `AT-ENV-2` Reconstructed ready, pending, desired, cooldown and occupancy equal the environment's.
- `AT-ENV-3` `arbitrate` on reconstructed inputs equals `token.deltas` at every step.
- `AT-ENV-4` Boot cohorts from 3 frames equal slots 14, 9, 4; all other slots are empty.
- `AT-ENV-5` The capacity arithmetic of Section 1.1 from config values.
- `AT-ENV-6` The aliasing example of Section 2.2: equal observations, different queue ages, different expiry times.
- `AT-ENV-7` Any non-default config is refused.

**Models**

- `AT-MOD-1` Exact input, logit, value and hidden shapes, `B = 1` and `B = 20`.
- `AT-MOD-2` Exact parameter counts of Section 1.8, including Model 3-self-loop and Model 3-complete.
- `AT-MOD-3` Exactly 17 edges, attributes and directions; 5 for Model 3-self-loop and 25 for Model 3-complete; non-edges get zero attention; attention sums to 1 per target and head.
- `AT-MOD-4` One GATv2 layer equals a hand-computed result.
- `AT-MOD-5` Message passing is real: changing Database features changes the Gateway logits after 2 layers and not after 1; in Model 3-self-loop, they never change.
- `AT-MOD-6` Every parameter receives a finite gradient, nonzero for at least one batch, and actor gradients do not reach the critic, nor the reverse. The one required exception, Model 3-self-loop, is checked in two parts. (a) After a backward pass of the full PPO loss, `W_t`, `W_e` and `a` of every GATv2 layer in its actor and critic, all 17,664 parameters, have gradients that are either `None` or all-zero tensors. (b) Starting from initialization, a fresh `torch.optim.Adam` with `weight_decay=0.0` takes one step on that loss, and those 17,664 parameters remain bitwise identical to their initial values. The test creates its own optimizer and never reuses one with accumulated momentum or other optimizer state.
- `AT-MOD-7` No NaN or Inf for all-zero, all-one and random inputs.
- `AT-MOD-8` Windows: ordering, zero padding, and clearing at reset. Models 2, 3, 3-self-loop and 3-complete get byte-identical windows.
- `AT-MOD-9` The forbidden inputs of Section 1.3 cannot reach a model.
- `AT-MOD-10` Models 3, 3-self-loop and 3-complete have identical parameter names, shapes and dtypes and, from one seed, identical initial parameters. Their modules differ only in the edge mask and edge attributes.

**PPO mathematics**

- `AT-PPO-1` Epoch 0, before any gradient step: `|ratio − 1| <= 1e-5` on every trainable slot, for every model.
- `AT-PPO-2` Joint log-prob equals the sum of four categorical log-probs; entropy likewise.
- `AT-PPO-3` GAE and returns against hand-computed values, for terminal, truncated and neither.
- `AT-PPO-4` `terminated` at step 359 gives no bootstrap; the truncation variant does.
- `AT-PPO-5` Clipped objective, value loss, entropy sign and gradient clipping against hand-computed values.
- `AT-PPO-6` Changing the stored step-0 action, log-prob, value or reward changes no loss, gradient, advantage statistic or target. Injecting NaN into every float training field of step-0 slots (`old_log_prob`, `old_value`, `advantage`, `return`) gives finite losses and gradients bitwise identical to the sentinel case. A static check finds no multiplication by a mask in any loss or logged mean.
- `AT-PPO-7` Buffer rewards equal `env.step` rewards exactly.
- `AT-PPO-8` Every model performs exactly the specified number of updates.

**Shield**

- `AT-SHD-1` Under forced overrides (bounds, cooldown, quota), stored log-probs equal `log π(a_raw)`, and `a_raw != a_approved` is recorded.
- `AT-SHD-2` Loss and gradients are unchanged when `a_approved`, `override_mask` and `fallback_count` are replaced by arbitrary values.
- `AT-SHD-3` A spy counts exactly one `approve` per `step`, in the collector and through `run_episode`.
- `AT-SHD-4` Proposals are 4 plain `int`; the fallback count is 0 over full episodes.
- `AT-SHD-5` The loss code never evaluates a log-prob of an approved action.

**Recurrence**

- `AT-REC-1` Hidden state is zero at each episode start and differs after one step.
- `AT-REC-2` Stored `h_0` equals the recomputed state; chunked forward equals full-episode forward under the same parameters.
- `AT-REC-3` No chunk crosses an episode; order inside a chunk is strictly temporal.
- `AT-REC-4` With a non-divisible length, changing padded inputs changes no loss or gradient, and means divide by unpadded counts. Injecting NaN into padded observations and into every float field of padded slots gives finite losses and gradients bitwise identical to the sentinel case.
- `AT-REC-5` GAE computed before splitting equals GAE on the full rollout; splitting first gives a different result on a constructed case.
- `AT-REC-6` The recurrent batcher cannot emit non-contiguous timesteps.
- `AT-REC-7` BPTT: the loss at the last step of a chunk has a nonzero gradient with respect to the input at its first step.

**Workloads and seeds**

- `AT-WRK-1` All drawn traces are valid and 1,800 long, and `generator_seed` is a plain `int` accepted by the frozen generator. (a) The same `(workload_seed, attempt)` pair reproduces the same `generator_seed` and a byte-identical trace. (b) Different `attempt` values derive distinct `SeedSequence` streams, and fixed test fixtures show distinct generated Poisson traces across attempts. The test does not claim that every possible pair of attempts gives distinct finite traces. (c) The recorded `workload_seed` stays equal to the allocated identity across retries, while actual duplicate or near-duplicate traces are rejected by the leakage guard (`AT-WRK-3`).
- `AT-WRK-2` Seed ranges are pairwise disjoint; offsets never leave their range.
- `AT-WRK-3` Training, validation and test hashes are disjoint. The leakage guard rejects a planted near-duplicate and redraws with `attempt + 1` under the same `workload_seed`, and the log records all three seed fields.
- `AT-WRK-4` The same training seed gives identical trace sequences for all models.

**Adapter and runner**

- `AT-ADP-1` `params` is JSON-native, flat, unchanged by callers, and passes `json_safe`.
- `AT-ADP-2` `run_episode` and `run_benchmark_suite` emit schema `1.0.0` manifests that survive a JSON round trip; a repeat run is byte-identical.
- `AT-ADP-3` Step-0 hold, step-counter enforcement, and no state leak between two consecutive episodes.
- `AT-ADP-4` Two training seeds of one model write different files.
- `AT-ADP-5` A loaded checkpoint reproduces the actions of the in-memory policy; a tampered file is refused.

**End to end**

- `AT-E2E-1` Smoke training: 2 iterations, 2 environments, every model, under 60 s in total, all losses finite, parameters changed. It shows that the pipeline runs and says nothing about campaign feasibility.
- `AT-E2E-2` Two runs with one seed give the same `parameters_sha256`; a different seed does not.
- `AT-E2E-3` PPO solves a trivial two-action bandit through the same loss code. This checks learning machinery, not model ranking.
- `AT-E2E-4` `numerical_failure` handling on an injected NaN keeps and evaluates the last finite checkpoint.
- `AT-E2E-5` **Full-pipeline CPU throughput benchmark (mandatory for Gate 2A).** For each primary model (Models 1, 2, 3, 3-self-loop and 4): 10 full training iterations, 28,800 environment steps, on one CPU thread, with the complete pipeline:
  1. 8-environment rollout collection;
  2. observation windowing or processing, and model inference;
  3. GAE and return calculation;
  4. all 5 PPO epochs × 4 minibatches;
  5. backpropagation and Adam updates;
  6. for Model 4, recurrent chunking and BPTT;
  7. checkpoint saving.

  One validation evaluation (21 episodes) is timed separately. Per model, the report gives **rollout collection time and optimization time separately** (plus checkpoint and validation time), total throughput in steps/s, and peak resident memory (RSS). It projects wall-clock hours, including validation evaluations (7 per tuning run, 19 per final run), always as a serial single-process estimate and, separately, for `P = min(physical cores, floor(available RAM / peak RSS))` parallel processes. If `P < 1`, parallel execution is infeasible: the report says so and gives no parallel projection, and the serial estimate stands alone. A parallel projection is an idealized, resource-bound estimate: it divides the runs among `P` processes as if each kept its measured single-process speed. Contention can make real runs slower, so it is a best case, not an assumed linear speedup. Both are given for:
  - **Gate 2B tuning:** 5 primary models × 8 configurations × 2 seeds, 80 runs of 120 iterations;
  - **Gate 2C at N = 5:** 5 primary models and Model 3-complete, 30 runs of 360 iterations;
  - **Gate 2C at N = 10:** the same 6 models, 60 runs of 360 iterations.

  Model 3-complete is projected with Model 3's timing, because dense masked attention costs the same for any mask. Pass criterion: every model completes with finite losses, and the report is produced. No speed threshold is asserted. **The projections are estimates, not guarantees,** and must be reviewed and approved before Gate 2B or Gate 2C is authorized.

Failure of any test blocks the gate. Tests are never weakened. Stage 2A also includes a
mutation check of the PPO, shield, masking and GAE code.

---

## 7. Decision Register and Open Trade-offs

### 7.1 Decisions made here

| ID | Decision |
|---|---|
| RL-01 | `W = 8`, fixed, not tuned |
| RL-02 | Separate actor and critic; critic sees only actor inputs |
| RL-03 | Per-node GRU with shared weights |
| RL-04 | Identity feature normalization; unmodified reward; fixed critic output scale 100 |
| RL-05 | Step 0 is a forced hold, masked from all losses |
| RL-06 | Horizon is terminal with no bootstrap, as frozen; `TRUNC-BOOT` is a sensitivity analysis only |
| RL-07 | `L = 36`, stored states, no refresh |
| RL-08 | Validation and primary endpoint: mean log cost ratio to `static` |
| RL-09 | CPU-only reported results; pure PyTorch |
| RL-10 | Stage 2 tests in `rl/tests/check_*.py`, in a separate process |
| RL-11 | `γ = 0.99` as a discounted surrogate, `λ = 0.95`, 5 epochs, an 8-configuration shared grid; `GAMMA-1` sensitivity if the budget allows |
| RL-12 | Model 3-self-loop is a primary model with its own tuning; the Holm family is exactly `{D^msg, D^rec}`; Stage 2 is exploratory |
| RL-13 | Masked and padded slots hold finite sentinels; masks apply to operands before arithmetic |
| RL-14 | `workload_seed` is the episode's identity; parameters and the Poisson seed come from `SeedSequence([workload_seed, attempt]).spawn(2)` |
| RL-15 | `AT-E2E-5` projections must be approved before Gates 2B and 2C |

### 7.2 Open trade-offs for independent audit

1. **Horizon handling.** Terminal plus a time-blind critic biases late-episode values, and
   Model 4 may count steps. `TRUNC-BOOT` is a sensitivity analysis and cannot remove horizon
   counting. Alternatives: truncation bootstrap as primary, or a time feature for all
   models. The latter adds input beyond the frozen observation.
2. **`γ = 0.99` is a discounted surrogate** for an undiscounted 360-step metric. `GAMMA-1`
   checks sensitivity if the budget allows; otherwise the mismatch is a stated limitation.
3. **Model 3-self-loop's learnable parameters and decentralized actor.** Attention over a
   single neighbor is constant, so 17,664 of its 52,933 parameters never learn, and each
   actor head sees one node. Both are inherent to removing message passing from the same
   architecture, and both are reported with `H-graph`.
4. **`H-rec` changes fusion order and width.** Per-frame GATv2 with a non-recurrent temporal
   aggregator would separate them; it is left as a future control.
5. **SLO saturation.** `V_k` is capped, and f6 clips at 1000 ms, so the SLO term and the
   latency feature carry no signal in deep overload. In Stage 1B, 892–1,500 of 1,800 substeps
   violate the SLO on `step_jump`, `gradual_ramp` and `seeded_bursty_poisson` under every
   baseline. The reward is frozen, so this is a limit, not something to fix.
6. **Canonical suite is not blind** to the designers.
7. **Baseline asymmetry.** Frozen baselines were not tuned on the training distribution.
   Tuning copies on validation traces would be fairer but is outside the frozen set.
8. **Budget sufficiency is unknown.** 1,036,800 steps may under-train any model. It may not be
   extended for one model only. Tuning at one third of the final length may favor fast
   starters.
9. **No power calculation exists, so Stage 2 is exploratory.** `N = 5` cannot reach
   `p < 0.05` in the sign-flip test; `N = 10` can and is more precise, but neither
   establishes power.
10. **Equal parameters are not equal capacity or compute.**
11. **Override incentive.** Blocked proposals are free, so policies may propose actions they
    know will be blocked. Action masking is possible because arbitration is observable, but
    it would change the action semantics for every model.
12. **Leakage guard threshold** (10 req/s) is a choice, not a derived value.
13. **Model 3-complete inherits Model 3's configuration**, so the topology contrast carries a
    tuning asymmetry.
14. **Only two p-values are corrected.** Secondary contrasts are many and uncorrected, which
    is why they are descriptive only.

### 7.3 Required before Stage 2 code

1. **`CLAUDE.md` conflicts with Stage 2 and is frozen.** It says the current stage is 1A,
   permits only NumPy, Gymnasium and pytest, and forbids Stage 1B or later work without
   authorization. The Stage 2A authorization must state that this document overrides those
   lines, or authorize a `CLAUDE.md` edit.
2. `docs/DECISIONS.md` is frozen for this task. Stage 2A must say whether Section 7.1 is
   copied there or stays here.
3. New files outside `rl/` are limited to `requirements-rl.txt`.
4. Installing `torch` needs explicit permission.
5. The Stage 1B diagnostics script (Section 1.1) is not in the repository.
