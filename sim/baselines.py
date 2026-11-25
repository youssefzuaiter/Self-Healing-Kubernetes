"""Stage 1B non-RL baseline controllers (docs/SPEC_02_BASELINES.md Section 2).

A controller sees only ``observation["node_features"]``, the config and the step
index. It never receives the environment, so it cannot read private state or
approve its own proposals: the runner passes every proposal through
``env.shield.approve()``. This module imports neither ``sim.env`` nor ``sim.shield``.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np

from sim.config import NUM_NODES, NUM_SCALABLE, SimulatorConfig

__all__ = [
    "BaseController", "StaticBaseline", "SimHPABaseline", "TunedReactiveThresholdController",
    "reconstruct_desired", "FEATURE_COUNT",
]

# Feature columns, as built by MicroserviceMeshEnv._build_features (SPEC_01 Section 25).
FEATURE_COUNT = 9  # equals sim.env.NUM_FEATURES; a test checks this
QUEUE_PRESSURE = 2
SERVICE_UTILIZATION = 3
READY_FRACTION = 4
PENDING_FRACTION = 5
HOLD = (0, 0, 0, 0)


def _node_features(observation: Mapping[str, Any]) -> np.ndarray:
    features = np.asarray(observation["node_features"], dtype=np.float64)
    if features.shape != (NUM_NODES, FEATURE_COUNT):
        raise ValueError(f"node_features must have shape ({NUM_NODES}, {FEATURE_COUNT}), got {features.shape}")
    return features


def _check_step(step_index: Any) -> int:
    if isinstance(step_index, bool) or not isinstance(step_index, Integral) or step_index < 0:
        raise ValueError(f"step_index must be an integer >= 0, got {step_index!r}")
    return int(step_index)


def _is_finite(value: Real) -> bool:
    """``math.isfinite``, with numbers too large for a float counted as not finite."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _number(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not _is_finite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return float(value)


def reconstruct_desired(node_features: np.ndarray, config: SimulatorConfig) -> tuple[int, ...]:
    """Desired replicas of the scalable nodes: (ready + pending) / r_max, times r_max."""
    return tuple(
        int(np.rint((node_features[node, READY_FRACTION] + node_features[node, PENDING_FRACTION])
                    * config.max_replicas[node]))
        for node in range(NUM_SCALABLE)
    )


class BaseController(ABC):
    """Section 2.1: a controller proposes raw replica deltas for the four scalable nodes."""

    def __init__(self, params: Mapping[str, Any]) -> None:
        self._params = dict(params)

    @property
    @abstractmethod
    def name(self) -> str:
        """A short identifier used in manifests and file names."""

    @property
    def params(self) -> dict[str, Any]:
        """A fresh copy on every access, so callers cannot change the controller."""
        return dict(self._params)

    def reset(self) -> None:
        """Clear per-episode state. The built-in baselines are stateless."""

    @abstractmethod
    def propose(self, observation: Mapping[str, Any], config: SimulatorConfig,
                step_index: int) -> tuple[int, int, int, int]:
        """Return a raw proposal; the shield decides what is approved."""


class StaticBaseline(BaseController):
    """Section 2.2: never proposes a change."""

    def __init__(self) -> None:
        super().__init__({})

    @property
    def name(self) -> str:
        return "static"

    def propose(self, observation: Mapping[str, Any], config: SimulatorConfig,
                step_index: int) -> tuple[int, int, int, int]:
        _check_step(step_index)
        return HOLD


class SimHPABaseline(BaseController):
    """Section 2.3: a simulator-side proportional utilization approximation.

    This is not the real Kubernetes Horizontal Pod Autoscaler and does not
    reproduce its algorithm, metrics pipeline or timing.
    """

    def __init__(self, target_utilization: float = 0.70, tolerance: float = 0.10) -> None:
        target = _number("target_utilization", target_utilization)
        tolerance = _number("tolerance", tolerance)
        if not 0.0 < target <= 1.0:
            raise ValueError(f"target_utilization must be in (0, 1], got {target}")
        if not 0.0 <= tolerance < 1.0:
            raise ValueError(f"tolerance must be in [0, 1), got {tolerance}")
        super().__init__({"target_utilization": target, "tolerance": tolerance})
        self._target, self._tolerance = target, tolerance

    @property
    def name(self) -> str:
        return "sim_hpa"

    def propose(self, observation: Mapping[str, Any], config: SimulatorConfig,
                step_index: int) -> tuple[int, int, int, int]:
        if _check_step(step_index) == 0:
            return HOLD  # cold telemetry: the reset observation has no interval history
        features = _node_features(observation)
        desired = reconstruct_desired(features, config)
        deltas = []
        for node in range(NUM_SCALABLE):
            ratio = float(features[node, SERVICE_UTILIZATION]) / self._target
            if abs(ratio - 1.0) <= self._tolerance:
                deltas.append(0)
                continue
            raw_target = max(config.min_replicas[node], int(math.ceil(desired[node] * ratio)))
            deltas.append(min(max(raw_target - desired[node], config.action_min), config.action_max))
        return tuple(deltas)


class TunedReactiveThresholdController(BaseController):
    """Section 2.4: fixed utilization and queue-pressure thresholds per node."""

    def __init__(self, util_high: float = 0.85, util_med: float = 0.70, util_low: float = 0.35,
                 queue_high: float = 0.25, queue_med: float = 0.10) -> None:
        values = {name: _number(name, value) for name, value in (
            ("util_high", util_high), ("util_med", util_med), ("util_low", util_low),
            ("queue_high", queue_high), ("queue_med", queue_med),
        )}
        if not 0.0 <= values["util_low"] < values["util_med"] < values["util_high"] <= 1.0:
            raise ValueError("need 0 <= util_low < util_med < util_high <= 1")
        if not 0.0 < values["queue_med"] < values["queue_high"] <= 1.0:
            raise ValueError("need 0 < queue_med < queue_high <= 1")
        super().__init__(values)
        self._thresholds = values

    @property
    def name(self) -> str:
        return "tuned_reactive_threshold"

    def propose(self, observation: Mapping[str, Any], config: SimulatorConfig,
                step_index: int) -> tuple[int, int, int, int]:
        if _check_step(step_index) == 0:
            return HOLD  # cold telemetry: the reset observation has no interval history
        features, limits = _node_features(observation), self._thresholds
        deltas = []
        for node in range(NUM_SCALABLE):
            utilization = float(features[node, SERVICE_UTILIZATION])
            queue = float(features[node, QUEUE_PRESSURE])
            if utilization > limits["util_high"] or queue > limits["queue_high"]:
                deltas.append(2)
            elif utilization > limits["util_med"] or queue > limits["queue_med"]:
                deltas.append(1)
            elif utilization < limits["util_low"] and queue == 0.0:
                deltas.append(-1)
            else:
                deltas.append(0)
        return tuple(deltas)
