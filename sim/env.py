"""Deterministic five-node fluid simulator (docs/SPEC_01_SIMULATOR.md).

Section numbers in comments refer to that spec. ``step`` accepts only a token from
the environment's own shield, never a raw proposal::

    env = MicroserviceMeshEnv()
    observation, info = env.reset(seed=0)
    observation, reward, terminated, truncated, info = env.step(env.shield.approve(raw))
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.config import DATABASE, GATEWAY, NUM_NODES, NUM_SCALABLE, SimulatorConfig
from sim.shield import ApprovedAction, SafetyShield

NUM_FEATURES = 9


class SimulationInvariantError(RuntimeError):
    """An internal invariant failed. State is never silently corrected."""


def contention_multiplier(kappa: np.ndarray, db_queue: float, db_capacity: float, threshold: float) -> np.ndarray:
    """Section 10: phi for every node from one database queue snapshot."""
    pressure = db_queue / db_capacity
    return 1.0 / (1.0 + kappa * max(0.0, pressure - threshold))


def check_conservation(
    queue_start: np.ndarray,
    arrivals_new: np.ndarray,
    arrivals_retry: np.ndarray,
    completed: np.ndarray,
    overflow: np.ndarray,
    timeout: np.ndarray,
    queue_end: np.ndarray,
    tolerance: float,
) -> None:
    """Section 15: raise if any node's flow balance exceeds the tolerance."""
    residual = queue_start + arrivals_new + arrivals_retry - completed - overflow - timeout - queue_end
    if not np.all(np.abs(residual) <= tolerance):  # written so NaN also fails
        raise SimulationInvariantError(f"flow conservation violated: residual {residual}")


def safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    """Element-wise ratio that is zero where the denominator is zero (Section 23.1)."""
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def failure_ratio(failed: np.ndarray, resolved: np.ndarray) -> np.ndarray:
    """Sections 21 and 21.1: failed work over resolved work, floored at 1.0."""
    return failed / np.maximum(1.0, resolved)


def end_to_end_failure(success: np.ndarray) -> float:
    """Section 22: failure proxy from per-node success ratios."""
    p0, p1, p2, p3, p4 = (float(value) for value in success)
    survival = p0 * p1 * p4 ** 0.3 * (0.8 * p2 * p4 ** 0.5 + 0.2 * p3 * p4 ** 1.0)
    return 1.0 - survival


def end_to_end_latency(latency_ms: np.ndarray) -> float:
    """Section 24: latency proxy in milliseconds."""
    l0, l1, l2, l3, l4 = (float(value) for value in latency_ms)
    return l0 + l1 + 0.3 * l4 + 0.8 * (l2 + 0.5 * l4) + 0.2 * (l3 + 1.0 * l4)


def interval_utilization(completed: np.ndarray, capacity: np.ndarray) -> np.ndarray:
    """Section 25 feature 3, before clipping."""
    zero_capacity_value = np.where(completed > 0, 1.0, 0.0)
    ratio = np.divide(completed, capacity, out=np.zeros_like(completed), where=capacity > 0)
    return np.where(capacity > 0, ratio, zero_capacity_value)


def _validated_trace(trace: Any, length: int) -> Optional[np.ndarray]:
    """Section 7: return a private read-only float copy, or raise ValueError."""
    if trace is None:
        return None
    if not isinstance(trace, np.ndarray):
        raise ValueError("traffic_trace must be None or a NumPy array")
    if trace.shape != (length,):
        raise ValueError(f"traffic_trace must have shape ({length},), got {trace.shape}")
    if not (np.issubdtype(trace.dtype, np.integer) or np.issubdtype(trace.dtype, np.floating)):
        raise ValueError(f"traffic_trace must be numeric, got dtype {trace.dtype}")
    values = trace.astype(np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("traffic_trace values must be finite")
    if np.any(values < 0.0):
        raise ValueError("traffic_trace values must be non-negative")
    values.setflags(write=False)
    return values


@dataclass
class _TokenRecord:
    """Environment-owned truth about one issued token (Section 30)."""

    token: ApprovedAction
    step_index: int
    deltas: tuple[int, ...]
    consumed: bool = False


class MicroserviceMeshEnv(gym.Env):
    """Gymnasium environment for the five-node mesh.

    ``action_space`` describes the raw proposal given to ``env.shield.approve``.
    ``step`` itself accepts only the resulting ``ApprovedAction``.
    """

    metadata = {"render_modes": []}

    def __init__(self, traffic_trace: Optional[np.ndarray] = None, config: Optional[SimulatorConfig] = None) -> None:
        super().__init__()
        self.config = config if config is not None else SimulatorConfig()
        cfg = self.config
        self._trace = _validated_trace(traffic_trace, cfg.internal_horizon)
        self._env_id = uuid.uuid4().hex

        self._mu = np.array(cfg.service_rate, dtype=np.float64)
        self._q_max = np.array(cfg.queue_capacity, dtype=np.float64)
        self._base_latency = np.array(cfg.base_latency_ms, dtype=np.float64)
        self._kappa = np.array(cfg.contention_kappa, dtype=np.float64)
        self._window = np.array(cfg.timeout_seconds, dtype=np.int64)
        self._r_min = np.array(cfg.min_replicas, dtype=np.int64)
        self._r_max = np.array(cfg.max_replicas, dtype=np.int64)
        self._multiplicity = np.array(cfg.call_multiplicity, dtype=np.float64)
        self._nodes = np.arange(NUM_NODES)
        self._ages = np.arange(cfg.queue_age_buckets, dtype=np.float64)

        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(0.0, 1.0, (NUM_NODES, NUM_FEATURES), np.float64),
            "call_multiplicity": spaces.Box(
                0.0, max(1.0, float(self._multiplicity.max())), (NUM_NODES, NUM_NODES), np.float64
            ),
        })
        self.action_space = spaces.Box(float(cfg.action_min), float(cfg.action_max), (NUM_SCALABLE,), np.float64)

        self.shield = SafetyShield(self)
        self._initialize_state()

    # ------------------------------------------------------------ read-only state
    @property
    def env_id(self) -> str:
        return self._env_id

    @property
    def time(self) -> int:
        """The next substep to run."""
        return self._time

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def trace_pointer(self) -> int:
        return self._trace_pointer

    @property
    def queue(self) -> np.ndarray:
        return self._queue.copy()

    @property
    def ready(self) -> np.ndarray:
        return self._ready.copy()

    @property
    def desired(self) -> np.ndarray:
        return self._desired.copy()

    @property
    def pending(self) -> np.ndarray:
        return self._boot_register.sum(axis=1)

    @property
    def terminating(self) -> np.ndarray:
        return self._termination_register.sum(axis=1)

    @property
    def boot_register(self) -> np.ndarray:
        return self._boot_register.copy()

    @property
    def termination_register(self) -> np.ndarray:
        return self._termination_register.copy()

    @property
    def cooldown(self) -> np.ndarray:
        return self._cooldown.copy()

    @property
    def cluster_occupancy(self) -> int:
        return int(self._ready.sum() + self.pending.sum() + self.terminating.sum())

    @property
    def downstream_buffer(self) -> np.ndarray:
        return self._next_downstream.copy()

    @property
    def retry_buffer(self) -> np.ndarray:
        return self._next_retry.copy()

    # ------------------------------------------------------------------ gym API
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[dict, dict]:
        """Section 33. Physics uses no randomness; the seed follows Gymnasium convention."""
        super().reset(seed=seed)
        self._initialize_state()
        return self._observation(), {"time": self._time, "step_index": self._step_index}

    def step(self, action: ApprovedAction) -> tuple[dict, float, bool, bool, dict]:
        """Validate and consume a token, then run one control interval."""
        record = self._authorize(action)
        record.consumed = True  # Section 30.1 step 7: before anything can raise

        self._peek_traffic()  # a post-horizon step raises here, before any mutation
        approved = np.array(record.deltas, dtype=np.int64)
        self._validate_approved(approved)
        self._apply_approved(approved)

        substeps = [self._run_substep() for _ in range(self.config.substeps_per_control_step)]
        rows = {key: np.stack([substep[key] for substep in substeps]) for key in substeps[0]}

        self._advance_cooldowns(approved)
        self._step_index += 1
        self._features = self._build_features(rows)

        reward, components = self._reward(rows, approved)
        terminated = self._step_index >= self.config.control_horizon
        failed = rows["overflow"].sum(axis=0) + rows["timeout"].sum(axis=0)
        info = {
            "time": self._time,
            "step_index": self._step_index,
            "approved_delta": approved,
            "fallback_count": self.shield.fallback_count,
            "cluster_occupancy": self.cluster_occupancy,
            "reward_components": components,
            "interval_failure_ratio": failure_ratio(failed, failed + rows["completed"].sum(axis=0)),
            "substeps": rows,
        }
        return self._observation(), reward, terminated, False, info

    # --------------------------------------------------------------- token checks
    def _record_issued_token(self, token: ApprovedAction) -> None:
        """Called by the shield. The ledger, not the token, is the authority."""
        self._tokens[token.nonce] = _TokenRecord(token, token.step_index, token.deltas)

    def _authorize(self, action: Any) -> _TokenRecord:
        """Section 30.1 steps 1-6. Raises PermissionError and changes nothing."""
        if type(action) is not ApprovedAction:
            raise PermissionError(
                f"env.step() accepts only a shield-issued ApprovedAction, got {type(action).__name__}"
            )
        if action.env_id != self._env_id:
            raise PermissionError("token was issued by a different environment")
        record = self._tokens.get(action.nonce) if isinstance(action.nonce, str) else None
        if record is None:
            raise PermissionError("token has an unknown nonce")
        if record.token is not action or (action.step_index, action.deltas) != (record.step_index, record.deltas):
            raise PermissionError("token does not match the issued token")
        if record.step_index != self._step_index:
            raise PermissionError(f"stale token from step {record.step_index}, now at step {self._step_index}")
        if record.consumed:
            raise PermissionError("token was already consumed")
        return record

    def _validate_approved(self, approved: np.ndarray) -> None:
        """Re-check the approved vector without trusting the shield (Section 32)."""
        cfg = self.config
        if approved.shape != (NUM_SCALABLE,):
            raise PermissionError(f"approved vector has shape {approved.shape}")
        if approved.min() < cfg.action_min or approved.max() > cfg.action_max:
            raise PermissionError("approved vector is outside the action set")
        target = self._desired[:NUM_SCALABLE] + approved
        out_of_bounds = (target < self._r_min[:NUM_SCALABLE]) | (target > self._r_max[:NUM_SCALABLE])
        if np.any(out_of_bounds & (approved != 0)):  # a zero delta cannot break a bound
            raise PermissionError("approved vector breaks replica bounds")
        if np.any((approved < 0) & (self._cooldown[:NUM_SCALABLE] > 0)):
            raise PermissionError("approved vector scales down during cooldown")
        scale_ups = int(approved[approved > 0].sum())
        cancelled = int(np.minimum(self.pending[:NUM_SCALABLE], np.maximum(-approved, 0)).sum())
        if scale_ups > 0 and self.cluster_occupancy - cancelled + scale_ups > cfg.cluster_quota:
            raise PermissionError("approved vector exceeds the cluster quota")

    # ----------------------------------------------------------- replica lifecycle
    def _apply_approved(self, approved: np.ndarray) -> None:
        """Section 9 step 1: changes due at the start of the control interval."""
        for node, change in enumerate(approved.tolist()):
            if change < 0:
                self._scale_down(node, -change)
        for node, change in enumerate(approved.tolist()):
            if change > 0:
                self._scale_up(node, change)

    def _scale_down(self, node: int, count: int) -> None:
        """Section 20. Higher register slots hold newer pods."""
        remaining = count
        for slot in range(self._boot_register.shape[1] - 1, -1, -1):
            cancelled = min(remaining, int(self._boot_register[node, slot]))
            self._boot_register[node, slot] -= cancelled
            remaining -= cancelled
        self._ready[node] -= remaining
        self._termination_register[node, self.config.termination_seconds[node] - 1] += remaining
        self._desired[node] -= count

    def _scale_up(self, node: int, count: int) -> None:
        """Sections 17, 18 and 29 rule 1."""
        self._boot_register[node, self.config.boot_seconds[node] - 1] += count
        self._desired[node] += count
        self._cooldown[node] = self.config.cooldown_control_steps

    def _advance_lifecycle(self) -> None:
        """Sections 18-19: process slot 0, then shift the registers left."""
        self._ready += self._boot_register[:, 0]
        for register in (self._boot_register, self._termination_register):
            register[:, :-1] = register[:, 1:].copy()
            register[:, -1] = 0

    def _advance_cooldowns(self, approved: np.ndarray) -> None:
        """Section 29 rules 2-3: services that just scaled up do not decrement."""
        current = self._cooldown[:NUM_SCALABLE]
        self._cooldown[:NUM_SCALABLE] = np.where(approved > 0, current, np.maximum(current - 1, 0))

    # -------------------------------------------------------------------- physics
    def _peek_traffic(self) -> float:
        """Section 7: the value at the pointer, or ValueError if none is available."""
        index = self._trace_pointer
        if index >= self.config.internal_horizon:
            raise ValueError(f"traffic index {index} is beyond the {self.config.internal_horizon}-second horizon")
        return self.config.default_traffic_rps if self._trace is None else float(self._trace[index])

    def _run_substep(self) -> dict[str, Any]:
        """Section 9. Step numbers below match the spec's processing order."""
        cfg = self.config
        # Step 1 ran in _apply_approved; step 16 of t-1 already took effect.
        # Steps 2-4: read external traffic and the buffers filled during t-1.
        external = self._peek_traffic()
        self._trace_pointer += 1
        arrivals_new = self._next_downstream.copy()
        arrivals_new[GATEWAY] += external
        arrivals_retry = self._next_retry.copy()

        # Step 5: every node uses this one start-of-substep snapshot.
        queue_start = self._queue.sum(axis=1)

        # Steps 6-7: contention and capacity.
        contention = contention_multiplier(
            self._kappa, queue_start[DATABASE], self._q_max[DATABASE], cfg.db_contention_threshold
        )
        capacity = self._ready * self._mu * contention * cfg.capacity_multiplier

        # Steps 8-9: admission uses the queue at the start of the substep.
        arrivals_total = arrivals_new + arrivals_retry
        accepted = np.minimum(arrivals_total, np.maximum(0.0, self._q_max - queue_start))
        overflow = arrivals_total - accepted
        self._queue[:, 0] += accepted

        # Step 10: FIFO service, oldest bucket first.
        served = np.zeros_like(self._queue)
        remaining = capacity.copy()
        for age in range(self._queue.shape[1] - 1, -1, -1):
            served[:, age] = np.minimum(self._queue[:, age], remaining)
            remaining -= served[:, age]
        completed = served.sum(axis=1)

        # Step 11: unserved work in the last bucket expires.
        residual = self._queue - served
        timeout = residual[self._nodes, self._window - 1].copy()
        residual[self._nodes, self._window - 1] = 0.0

        # Step 12: survivors age by one second.
        self._queue[:, 1:] = residual[:, :-1]
        self._queue[:, 0] = 0.0
        queue_end = self._queue.sum(axis=1)

        # Steps 13-14: work for t+1 goes to separate buffers.
        self._next_retry = cfg.retry_coefficient * timeout
        self._next_downstream = completed @ self._multiplicity

        # Step 15: metrics and invariants, using replicas in effect during t.
        failed = overflow + timeout
        ratio = failure_ratio(failed, completed + failed)
        e2e_failure = end_to_end_failure(1.0 - ratio)
        served_delay = 1000.0 * safe_ratio(served @ self._ages, completed)
        backlog_delay = 1000.0 * np.minimum(self._window, queue_end / np.maximum(1.0, capacity))
        node_latency = self._base_latency + np.maximum(served_delay, backlog_delay)
        e2e_latency = end_to_end_latency(node_latency)
        occupancy = self.cluster_occupancy
        record = {
            "time": np.array(self._time),
            "external": np.array(external),
            "arrivals_new": arrivals_new,
            "arrivals_retry": arrivals_retry,
            "accepted": accepted,
            "overflow": overflow,
            "timeout": timeout,
            "completed": completed,
            "capacity": capacity,
            "contention": contention,
            "queue_start": queue_start,
            "queue_end": queue_end,
            "served_by_age": served,
            "ready": self._ready.copy(),
            "pending": self.pending,
            "terminating": self.terminating,
            "occupancy": np.array(occupancy),
            "failure_ratio": ratio,
            "node_latency_ms": node_latency,
            "e2e_latency_ms": np.array(e2e_latency),
            "e2e_failure": np.array(e2e_failure),
            "slo_violation": np.array(min(cfg.slo_violation_cap, max(0.0, (e2e_latency - cfg.slo_ms) / cfg.slo_ms))),
            "resource_cost": np.array(occupancy / cfg.cluster_quota),
        }
        self._check_substep(record)

        # Step 16: lifecycle registers, effective from the start of t+1.
        self._advance_lifecycle()
        self._time += 1
        return record

    def _check_substep(self, record: dict[str, Any]) -> None:
        """Section 35. Raises instead of correcting anything."""
        tolerance = self.config.conservation_tolerance
        check_conservation(
            record["queue_start"], record["arrivals_new"], record["arrivals_retry"], record["completed"],
            record["overflow"], record["timeout"], record["queue_end"], tolerance,
        )
        capacity, queue, served = record["capacity"], self._queue, record["served_by_age"]
        if not (np.all(np.isfinite(capacity)) and np.all(capacity >= 0.0)):
            raise SimulationInvariantError(f"invalid service capacity: {capacity}")
        # Serving a negative bucket would balance the books, so check served work too.
        if not (np.all(np.isfinite(queue)) and np.all(queue >= 0.0) and np.all(served >= 0.0)):
            raise SimulationInvariantError(f"invalid queue state: {queue}, served {served}")
        if not np.all(record["queue_end"] <= self._q_max + tolerance):
            raise SimulationInvariantError(f"queue over capacity: {record['queue_end']}")
        if not np.all(record["completed"] <= capacity + tolerance):
            raise SimulationInvariantError("completed work exceeds capacity")
        if not np.all(record["completed"] <= record["queue_start"] + record["accepted"] + tolerance):
            raise SimulationInvariantError("completed work exceeds available work")
        ratio, e2e_failure = record["failure_ratio"], float(record["e2e_failure"])
        if not (np.all((ratio >= 0.0) & (ratio <= 1.0)) and 0.0 <= e2e_failure <= 1.0):
            raise SimulationInvariantError(f"failure ratio out of bounds: {ratio}, {e2e_failure}")
        if min(self._ready.min(), self.pending.min(), self.terminating.min()) < 0:
            raise SimulationInvariantError("negative replica count")
        if not np.array_equal(self._desired, self._ready + self.pending):
            raise SimulationInvariantError("desired replicas differ from ready plus pending")
        if self.cluster_occupancy > self.config.cluster_quota:
            raise SimulationInvariantError(f"cluster quota exceeded: {self.cluster_occupancy}")
        if np.any(self._desired < self._r_min) or np.any(self._desired > self._r_max):
            raise SimulationInvariantError(f"desired replicas out of bounds: {self._desired}")

    # ---------------------------------------------------------- reward, observation
    def _reward(self, rows: dict[str, np.ndarray], approved: np.ndarray) -> tuple[float, dict[str, float]]:
        """Section 31. Returns r_k and its components."""
        cfg = self.config
        slo_violation = float(rows["slo_violation"].mean())
        failure_cost = float(rows["e2e_failure"].mean())
        resource_cost = float(rows["resource_cost"].mean())
        scaling_cost = float(np.abs(approved).sum()) / cfg.scaling_cost_norm  # once per interval
        total_cost = (cfg.weight_slo * slo_violation + cfg.weight_failure * failure_cost
                      + cfg.weight_resources * resource_cost + cfg.weight_scaling * scaling_cost)
        if not np.isfinite(total_cost):
            raise SimulationInvariantError(f"non-finite reward: {total_cost}")
        components = {
            "slo_violation": slo_violation,
            "failure_cost": failure_cost,
            "resource_cost": resource_cost,
            "scaling_cost": scaling_cost,
            "total_cost": total_cost,
        }
        return -total_cost, components

    def _build_features(self, rows: Optional[dict[str, np.ndarray]]) -> np.ndarray:
        """Section 25. ``rows`` is None at reset, when no interval exists yet."""
        features = np.zeros((NUM_NODES, NUM_FEATURES))
        if rows is not None:
            rate_scale = self._mu * self._r_max
            failed = rows["overflow"].sum(axis=0) + rows["timeout"].sum(axis=0)
            completed = rows["completed"].sum(axis=0)
            features[:, 0] = rows["arrivals_new"].mean(axis=0) / rate_scale
            features[:, 1] = rows["arrivals_retry"].mean(axis=0) / rate_scale
            features[:, 3] = interval_utilization(completed, rows["capacity"].sum(axis=0))
            features[:, 6] = rows["node_latency_ms"].mean(axis=0) / 1000.0
            features[:, 7] = failure_ratio(failed, completed + failed)
        features[:, 2] = self._queue.sum(axis=1) / self._q_max
        features[:, 4] = self._ready / self._r_max
        features[:, 5] = self.pending / self._r_max
        features[:, 8] = self._cooldown / max(1, self.config.cooldown_control_steps)
        if not np.all(np.isfinite(features)):
            raise SimulationInvariantError(f"non-finite observation: {features}")
        return np.clip(features, 0.0, 1.0)

    def _observation(self) -> dict[str, np.ndarray]:
        return {"node_features": self._features.copy(), "call_multiplicity": self._multiplicity.copy()}

    def _initialize_state(self) -> None:
        """Section 33: everything a reset must clear."""
        cfg = self.config
        self._time = 0
        self._step_index = 0
        self._trace_pointer = 0
        self._queue = np.zeros((NUM_NODES, cfg.queue_age_buckets))
        self._ready = np.array(cfg.initial_replicas, dtype=np.int64)
        self._desired = self._ready.copy()
        self._boot_register = np.zeros((NUM_NODES, cfg.boot_register_length), dtype=np.int64)
        self._termination_register = np.zeros((NUM_NODES, cfg.termination_register_length), dtype=np.int64)
        self._cooldown = np.zeros(NUM_NODES, dtype=np.int64)
        self._next_downstream = np.zeros(NUM_NODES)
        self._next_retry = np.zeros(NUM_NODES)
        self._tokens: dict[str, _TokenRecord] = {}
        self.shield.reset()
        self._features = self._build_features(None)
