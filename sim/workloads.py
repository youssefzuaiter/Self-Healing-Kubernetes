"""Stage 1B benchmark workloads (docs/SPEC_02_BASELINES.md Section 1).

Each generator returns an immutable ``WorkloadTrace`` of external request rates,
in requests per second, one value per one-second substep.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import FrozenInstanceError, dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

__all__ = [
    "WorkloadTrace", "constant", "step_jump", "gradual_ramp", "flash_crowd", "burst_and_drop",
    "db_bottleneck", "seeded_bursty_poisson", "benchmark_suite",
]


_IMMUTABLE_LEAVES = (type(None), bool, int, float, str)


def _freeze_key(key: Any) -> str:
    if type(key) is not str:
        raise TypeError(f"parameter names must be plain strings, got {type(key).__name__}")
    return key


def _freeze(value: Any) -> Any:
    """Rebuild every container as a read-only one and accept only immutable scalars.

    Leaves must be exactly None, bool, int, float or str: a subclass instance can
    carry mutable state. NumPy scalars and arrays are converted to these types.
    Nothing is shared with the caller, so no separate deep copy is needed.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({_freeze_key(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    if isinstance(value, np.ndarray):
        return _freeze(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if type(value) not in _IMMUTABLE_LEAVES:
        raise TypeError(f"unsupported parameter value of type {type(value).__name__}")
    return value


def _validated_bytes(values: Any) -> bytes:
    """Check trace values and return them as C-contiguous little-endian float64 bytes."""
    source = np.asarray(values)
    if source.dtype.kind not in "iuf":
        raise ValueError(f"values must be real numbers, got dtype {source.dtype}")
    if source.ndim != 1 or source.size < 1:
        raise ValueError(f"values must be a non-empty 1-D array, got shape {source.shape}")
    contiguous = np.array(source, dtype="<f8", order="C", copy=True)
    if not np.all(np.isfinite(contiguous)):
        raise ValueError("values must be finite")
    if np.any(contiguous < 0.0):
        raise ValueError("values must be non-negative")
    return contiguous.tobytes(order="C")


class _BytesBackedValues:
    """The ``values`` field: holds immutable bytes and returns a new array on every access.

    An array over immutable bytes can never be made writable, but NumPy still lets
    its ``dtype``, ``shape`` and ``strides`` be reassigned in place. Handing out one
    shared array would let a caller reinterpret the trace, so each access gets its own.
    """

    def __get__(self, instance: Any, owner: Any = None) -> np.ndarray:
        if instance is None:
            raise AttributeError("values")  # tells dataclass that the field has no default
        return np.frombuffer(instance.__dict__["_values_bytes"], dtype="<f8")

    def __set__(self, instance: Any, values: Any) -> None:
        if "_values_bytes" in instance.__dict__:
            raise FrozenInstanceError("cannot assign to field 'values'")
        instance.__dict__["_values_bytes"] = _validated_bytes(values)


@dataclass(frozen=True, eq=False)
class WorkloadTrace:
    """An immutable external-traffic trace (SPEC_02 Section 1.1). Compare by ``sha256``."""

    name: str
    params: Mapping[str, Any]
    values: np.ndarray = _BytesBackedValues()
    sha256: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")
        if not isinstance(self.params, Mapping):
            raise ValueError("params must be a mapping")
        digest = hashlib.sha256(self.__dict__["_values_bytes"]).hexdigest()
        if self.sha256 not in ("", digest):
            raise ValueError("sha256 does not match the values")
        object.__setattr__(self, "params", _freeze(self.params))
        object.__setattr__(self, "sha256", digest)


# ------------------------------------------------------------- validation
def _is_finite(value: Real) -> bool:
    """``math.isfinite``, with numbers too large for a float counted as not finite."""
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _rate(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not _is_finite(value) or value < 0:
        raise ValueError(f"{name} must be a finite number >= 0, got {value!r}")
    return float(value)


def _integer(name: str, value: Any, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _probability(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


def _window(name: str, start: int, length: int, horizon: int) -> None:
    """Section 1.2: a time window must fit inside [0, horizon)."""
    if start + length > horizon:
        raise ValueError(f"{name} window [{start}, {start + length}) does not fit in the horizon of {horizon}")


# ------------------------------------------------------------- generators
def constant(rate: float = 100.0, horizon: int = 1800) -> WorkloadTrace:
    rate, horizon = _rate("rate", rate), _integer("horizon", horizon, 1)
    return WorkloadTrace("constant", {"rate": rate, "horizon": horizon}, np.full(horizon, rate))


def step_jump(base_rate: float = 100.0, jump_rate: float = 600.0, jump_time: int = 300,
              horizon: int = 1800) -> WorkloadTrace:
    base_rate, jump_rate = _rate("base_rate", base_rate), _rate("jump_rate", jump_rate)
    jump_time, horizon = _integer("jump_time", jump_time, 0), _integer("horizon", horizon, 1)
    _window("jump_time", jump_time, 1, horizon)
    values = np.full(horizon, base_rate)
    values[jump_time:] = jump_rate
    params = {"base_rate": base_rate, "jump_rate": jump_rate, "jump_time": jump_time, "horizon": horizon}
    return WorkloadTrace("step_jump", params, values)


def gradual_ramp(start_rate: float = 100.0, end_rate: float = 700.0, horizon: int = 1800) -> WorkloadTrace:
    start_rate, end_rate = _rate("start_rate", start_rate), _rate("end_rate", end_rate)
    horizon = _integer("horizon", horizon, 1)
    params = {"start_rate": start_rate, "end_rate": end_rate, "horizon": horizon}
    return WorkloadTrace("gradual_ramp", params, np.linspace(start_rate, end_rate, horizon))


def flash_crowd(base_rate: float = 100.0, peak_rate: float = 900.0, start_time: int = 300, duration: int = 120,
                horizon: int = 1800) -> WorkloadTrace:
    base_rate, peak_rate = _rate("base_rate", base_rate), _rate("peak_rate", peak_rate)
    start_time, duration = _integer("start_time", start_time, 0), _integer("duration", duration, 1)
    horizon = _integer("horizon", horizon, 1)
    _window("flash crowd", start_time, duration, horizon)
    values = np.full(horizon, base_rate)
    values[start_time:start_time + duration] = peak_rate
    params = {"base_rate": base_rate, "peak_rate": peak_rate, "start_time": start_time, "duration": duration,
              "horizon": horizon}
    return WorkloadTrace("flash_crowd", params, values)


def burst_and_drop(base_rate: float = 100.0, peak_rate: float = 800.0, ramp_start: int = 300, ramp_duration: int = 60,
                   hold_duration: int = 180, horizon: int = 1800) -> WorkloadTrace:
    """Section 1.3: base, a linear ramp toward the peak, a hold, then an immediate drop."""
    base_rate, peak_rate = _rate("base_rate", base_rate), _rate("peak_rate", peak_rate)
    ramp_start = _integer("ramp_start", ramp_start, 0)
    ramp_duration, hold_duration = _integer("ramp_duration", ramp_duration, 1), _integer("hold_duration", hold_duration, 1)
    horizon = _integer("horizon", horizon, 1)
    _window("burst", ramp_start, ramp_duration + hold_duration, horizon)
    ramp_end, hold_end = ramp_start + ramp_duration, ramp_start + ramp_duration + hold_duration
    values = np.full(horizon, base_rate)
    offsets = np.arange(ramp_duration, dtype=np.float64)
    values[ramp_start:ramp_end] = base_rate + (peak_rate - base_rate) * offsets / ramp_duration
    values[ramp_end:hold_end] = peak_rate
    params = {"base_rate": base_rate, "peak_rate": peak_rate, "ramp_start": ramp_start,
              "ramp_duration": ramp_duration, "hold_duration": hold_duration, "horizon": horizon}
    return WorkloadTrace("burst_and_drop", params, values)


def db_bottleneck(base_rate: float = 100.0, surge_rate: float = 450.0, start_time: int = 300, duration: int = 600,
                  horizon: int = 1800) -> WorkloadTrace:
    """Section 1.5: the surge reaches the database only if upstream services pass it."""
    base_rate, surge_rate = _rate("base_rate", base_rate), _rate("surge_rate", surge_rate)
    start_time, duration = _integer("start_time", start_time, 0), _integer("duration", duration, 1)
    horizon = _integer("horizon", horizon, 1)
    _window("surge", start_time, duration, horizon)
    values = np.full(horizon, base_rate)
    values[start_time:start_time + duration] = surge_rate
    params = {"base_rate": base_rate, "surge_rate": surge_rate, "start_time": start_time, "duration": duration,
              "horizon": horizon}
    return WorkloadTrace("db_bottleneck", params, values)


def seeded_bursty_poisson(base_rate: float = 150.0, burst_rate: float = 750.0, burst_prob: float = 0.02,
                          min_burst_len: int = 20, max_burst_len: int = 60, seed: int = 42,
                          horizon: int = 1800) -> WorkloadTrace:
    """Section 1.4: Poisson arrivals with randomly started bursts, from an isolated generator."""
    base_rate, burst_rate = _rate("base_rate", base_rate), _rate("burst_rate", burst_rate)
    burst_prob = _probability("burst_prob", burst_prob)
    min_burst_len = _integer("min_burst_len", min_burst_len, 1)
    max_burst_len = _integer("max_burst_len", max_burst_len, min_burst_len)
    seed, horizon = _integer("seed", seed, 0), _integer("horizon", horizon, 1)
    rng = np.random.default_rng(seed)
    values = np.empty(horizon, dtype=np.float64)
    remaining = 0
    for t in range(horizon):
        if remaining == 0 and rng.random() < burst_prob:
            remaining = int(rng.integers(min_burst_len, max_burst_len, endpoint=True))
        values[t] = float(rng.poisson(burst_rate if remaining > 0 else base_rate))
        if remaining > 0:
            remaining -= 1  # a burst past the horizon is cut off when the loop ends
    params = {"base_rate": base_rate, "burst_rate": burst_rate, "burst_prob": burst_prob,
              "min_burst_len": min_burst_len, "max_burst_len": max_burst_len, "seed": seed, "horizon": horizon}
    return WorkloadTrace("seeded_bursty_poisson", params, values)


def benchmark_suite(seed: int = 42) -> tuple[WorkloadTrace, ...]:
    """Section 1.6: the six deterministic traces, then the seeded Poisson trace."""
    return (constant(), step_jump(), gradual_ramp(), flash_crowd(), burst_and_drop(), db_bottleneck(),
            seeded_bursty_poisson(seed=seed))
