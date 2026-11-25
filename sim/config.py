"""Immutable Stage 1A configuration. Defaults reproduce SPEC_01 Sections 4-6."""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any

NODE_NAMES = ("Gateway", "Auth", "Catalog", "Order", "Database")
NUM_NODES = 5
NUM_SCALABLE = 4
GATEWAY = 0
DATABASE = 4

_NODE_INTEGERS = (
    "min_replicas", "max_replicas", "initial_replicas", "boot_seconds", "termination_seconds", "timeout_seconds",
)
_NODE_NUMBERS = ("service_rate", "queue_capacity", "base_latency_ms", "contention_kappa")
_SCALAR_INTEGERS = (
    "substeps_per_control_step", "control_horizon", "cluster_quota", "cooldown_control_steps",
    "action_min", "action_max",
)
_SCALAR_NUMBERS = (
    "default_traffic_rps", "db_contention_threshold", "capacity_multiplier", "retry_coefficient",
    "conservation_tolerance", "slo_ms", "slo_violation_cap", "weight_slo", "weight_failure",
    "weight_resources", "weight_scaling", "scaling_cost_norm",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_integer(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


@dataclass(frozen=True)
class SimulatorConfig:
    """All simulator parameters. Per-node tuples follow ``NODE_NAMES`` order."""

    # Section 4: time discretization. The substep is fixed at one second.
    substeps_per_control_step: int = 5
    control_horizon: int = 360

    # Section 7: external traffic used when ``traffic_trace`` is None.
    default_traffic_rps: float = 100.0

    # Section 5.1: service parameters.
    service_rate: tuple[float, ...] = (100.0, 120.0, 80.0, 50.0, 300.0)
    queue_capacity: tuple[float, ...] = (2000.0, 2000.0, 1500.0, 1000.0, 3000.0)
    base_latency_ms: tuple[float, ...] = (5.0, 8.0, 15.0, 25.0, 10.0)
    min_replicas: tuple[int, ...] = (1, 1, 1, 1, 1)
    max_replicas: tuple[int, ...] = (10, 10, 10, 10, 1)
    initial_replicas: tuple[int, ...] = (2, 2, 2, 2, 1)
    boot_seconds: tuple[int, ...] = (20, 20, 20, 20, 20)
    termination_seconds: tuple[int, ...] = (5, 5, 5, 5, 5)
    timeout_seconds: tuple[int, ...] = (5, 5, 5, 5, 5)
    contention_kappa: tuple[float, ...] = (0.0, 1.0, 1.5, 2.5, 0.0)

    # Section 6: call_multiplicity[u][v] is calls to v per completion at u.
    call_multiplicity: tuple[tuple[float, ...], ...] = (
        (0.0, 1.0, 0.8, 0.2, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.3),
        (0.0, 0.0, 0.0, 0.0, 0.5),
        (0.0, 0.0, 0.0, 0.0, 1.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    )

    # Sections 10, 14, 15: physics constants.
    db_contention_threshold: float = 0.50
    capacity_multiplier: float = 1.0
    retry_coefficient: float = 0.5
    conservation_tolerance: float = 1e-9

    # Sections 16, 26, 29: shield limits.
    cluster_quota: int = 28
    cooldown_control_steps: int = 6
    action_min: int = -1
    action_max: int = 2

    # Section 31: reward.
    slo_ms: float = 250.0
    slo_violation_cap: float = 5.0
    weight_slo: float = 2.0
    weight_failure: float = 5.0
    weight_resources: float = 0.5
    weight_scaling: float = 0.2
    scaling_cost_norm: float = 8.0

    def __post_init__(self) -> None:
        self._validate_types()
        self._validate_ranges()

    def _validate_types(self) -> None:
        """Shapes and types first, so range checks only compare real numbers."""
        for name in _NODE_INTEGERS + _NODE_NUMBERS:
            values = getattr(self, name)
            _require(isinstance(values, tuple) and len(values) == NUM_NODES, f"{name} needs a tuple of {NUM_NODES} values")
        matrix = self.call_multiplicity
        _require(
            isinstance(matrix, tuple) and len(matrix) == NUM_NODES
            and all(isinstance(row, tuple) and len(row) == NUM_NODES for row in matrix),
            "call_multiplicity must be a 5x5 tuple of tuples",
        )
        for name in _NODE_INTEGERS:
            _require(all(_is_integer(value) for value in getattr(self, name)), f"{name} must hold integers")
        for name in _NODE_NUMBERS:
            _require(all(_is_number(value) for value in getattr(self, name)), f"{name} must hold finite numbers")
        _require(all(_is_number(value) for row in matrix for value in row), "call_multiplicity must hold finite numbers")
        for name in _SCALAR_INTEGERS:
            _require(_is_integer(getattr(self, name)), f"{name} must be an integer")
        for name in _SCALAR_NUMBERS:
            _require(_is_number(getattr(self, name)), f"{name} must be a finite number")

    def _validate_ranges(self) -> None:
        """Value ranges, recorded as IMPL-32 in docs/DECISIONS.md."""
        _require(self.substeps_per_control_step >= 1, "substeps_per_control_step must be >= 1")
        _require(self.control_horizon >= 1, "control_horizon must be >= 1")
        _require(self.default_traffic_rps >= 0, "default_traffic_rps must be >= 0")
        _require(all(rate > 0 for rate in self.service_rate), "service_rate must be positive")
        _require(all(size > 0 for size in self.queue_capacity), "queue_capacity must be positive")
        _require(all(latency >= 0 for latency in self.base_latency_ms), "base_latency_ms must be >= 0")
        _require(all(kappa >= 0 for kappa in self.contention_kappa), "contention_kappa must be >= 0")
        _require(all(value >= 0 for row in self.call_multiplicity for value in row), "call_multiplicity must be >= 0")
        for name in ("boot_seconds", "termination_seconds", "timeout_seconds"):
            _require(all(value >= 1 for value in getattr(self, name)), f"{name} must be >= 1")
        for low, start, high in zip(self.min_replicas, self.initial_replicas, self.max_replicas):
            _require(0 <= low <= start <= high and high >= 1, "need min <= initial <= max")
        _require(
            self.min_replicas[DATABASE] == self.initial_replicas[DATABASE] == self.max_replicas[DATABASE] == 1,
            "the database has exactly one replica",
        )
        _require(sum(self.initial_replicas) <= self.cluster_quota, "initial replicas exceed cluster_quota")
        _require(0 <= self.db_contention_threshold <= 1, "db_contention_threshold must be in [0, 1]")
        _require(self.capacity_multiplier > 0, "capacity_multiplier must be positive")
        _require(0 <= self.retry_coefficient <= 1, "retry_coefficient must be in [0, 1]")
        _require(self.conservation_tolerance > 0, "conservation_tolerance must be positive")
        _require(self.cooldown_control_steps >= 0, "cooldown_control_steps must be >= 0")
        _require(self.action_min <= 0 <= self.action_max, "the action set must contain 0")
        _require(self.slo_ms > 0, "slo_ms must be positive")
        _require(self.slo_violation_cap >= 0, "slo_violation_cap must be >= 0")
        for name in ("weight_slo", "weight_failure", "weight_resources", "weight_scaling"):
            _require(getattr(self, name) >= 0, f"{name} must be >= 0")
        _require(self.scaling_cost_norm > 0, "scaling_cost_norm must be positive")

    @property
    def internal_horizon(self) -> int:
        """Total one-second substeps, which is also the trace length."""
        return self.control_horizon * self.substeps_per_control_step

    @property
    def queue_age_buckets(self) -> int:
        return max(self.timeout_seconds)

    @property
    def boot_register_length(self) -> int:
        return max(self.boot_seconds)

    @property
    def termination_register_length(self) -> int:
        return max(self.termination_seconds)
