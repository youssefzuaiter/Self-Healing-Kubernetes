"""Stage 1B acceptance tests. Section numbers refer to docs/SPEC_02_BASELINES.md.

These tests check mechanics, invariants, immutability, horizons and
reproducibility. They never assert that one controller performs better than another.
"""
from __future__ import annotations

import array
import ast
import contextlib
import copy
import dataclasses
import errno
import hashlib
import json
import os
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import sim.env as env_module  # noqa: E402
import sim.runner as runner_module  # noqa: E402
from sim import workloads  # noqa: E402
from sim.baselines import (  # noqa: E402
    FEATURE_COUNT,
    BaseController,
    SimHPABaseline,
    StaticBaseline,
    TunedReactiveThresholdController,
    reconstruct_desired,
)
from sim.config import SimulatorConfig  # noqa: E402
from sim.env import MicroserviceMeshEnv  # noqa: E402
from sim.runner import json_safe, run_benchmark_suite, run_episode  # noqa: E402
from sim.workloads import WorkloadTrace  # noqa: E402

GATEWAY = 0
HOLD = (0, 0, 0, 0)
SHORT = dataclasses.replace(SimulatorConfig(), control_horizon=20)  # 100 substeps
GENERATORS = (
    "constant", "step_jump", "gradual_ramp", "flash_crowd", "burst_and_drop", "db_bottleneck", "seeded_bursty_poisson",
)
CONTROLLERS = (StaticBaseline, SimHPABaseline, TunedReactiveThresholdController)
FORBIDDEN = {"torch", "torch_geometric", "stable_baselines3", "kubernetes", "prometheus", "prometheus_client", "locust"}
STAGE_1B_MODULES = ("sim/workloads.py", "sim/baselines.py", "sim/runner.py")
METRIC_KEYS = {
    "control_steps_executed", "substeps_executed", "terminated", "truncated", "total_reward", "mean_reward",
    "unweighted_cost_sums", "weighted_cost_sums", "substep_latency_proxy_ms", "mean_e2e_failure_proxy",
    "max_e2e_failure_proxy", "slo_violation_substeps", "total_pod_seconds", "total_scaling_events",
    "shield_arbitration_override_count", "shield_fallback_count",
}


# ------------------------------------------------------------------ helpers
def observation(utilization=0.0, queue=0.0, ready=0.2, pending=0.0) -> dict:
    """Node features with the same values on every node. ready 0.2 is 2 of 10 pods."""
    features = np.zeros((5, FEATURE_COUNT))
    features[:, 2], features[:, 3], features[:, 4], features[:, 5] = queue, utilization, ready, pending
    return {"node_features": features}


def burst_runs(mask: np.ndarray) -> list:
    """(start, length) of each run of True values."""
    runs, start = [], None
    for t, value in enumerate(mask.tolist() + [False]):
        if value and start is None:
            start = t
        elif not value and start is not None:
            runs.append((start, t - start))
            start = None
    return runs


def imported_modules(path: Path) -> set:
    names = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


class ScriptedController(BaseController):
    """Test double: logs every call and replays a fixed script of proposals."""

    def __init__(self, script=None, default=HOLD):
        super().__init__({})
        self.script, self.default, self.log = dict(script or {}), default, []

    @property
    def name(self) -> str:
        return "scripted"

    def reset(self) -> None:
        self.log.append("reset")

    def propose(self, observation, config, step_index):
        self.log.append(step_index)
        return self.script.get(step_index, self.default)


class NamedController(ScriptedController):
    """A scripted controller with an arbitrary, possibly unsafe, name."""

    def __init__(self, name):
        super().__init__()
        self._name = name

    @property
    def name(self):
        return self._name


def short_trace(name="constant", rate=100.0) -> WorkloadTrace:
    """A 100-substep trace. The name is set afterwards so that invalid ones can be tested."""
    trace = WorkloadTrace("placeholder", {"rate": rate}, np.full(100, rate))
    object.__setattr__(trace, "name", name)
    return trace


def manifest_name(controller_name: str, trace: WorkloadTrace) -> str:
    return f"{controller_name}__{trace.name}__{trace.sha256[:12]}.json"


def snapshot(directory: Path) -> dict:
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


# ------------------------------------------------------- 1. workload suite
@pytest.mark.parametrize("name", GENERATORS)
def test_generator_returns_a_frozen_little_endian_trace(name):
    trace = getattr(workloads, name)()
    assert trace.name == name
    assert trace.values.shape == (1800,)
    assert trace.values.dtype.str == "<f8"
    assert trace.values.flags.c_contiguous and not trace.values.flags.writeable
    assert np.all(np.isfinite(trace.values)) and trace.values.min() >= 0.0
    with pytest.raises(ValueError):
        trace.values[0] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        trace.values = np.zeros(1800)
    assert trace.sha256 == hashlib.sha256(trace.values.tobytes()).hexdigest()


@pytest.mark.parametrize("name", GENERATORS)
def test_trace_values_can_never_be_made_writable_again(name):
    trace = getattr(workloads, name)()
    original = trace.values.tobytes()
    for attempt in (
        lambda: trace.values.__setitem__(0, 1.0),
        lambda: trace.values.__setitem__(slice(None), 0.0),
        lambda: trace.values.setflags(write=True),
        lambda: setattr(trace.values.flags, "writeable", True),
        lambda: trace.values.view().setflags(write=True),
        lambda: trace.values.reshape(2, -1).setflags(write=True),
        lambda: trace.values.sort(),
        lambda: trace.values.fill(0.0),
        lambda: np.add(trace.values, 1.0, out=trace.values),
    ):
        with pytest.raises(ValueError):
            attempt()
    assert trace.values.tobytes() == original
    assert trace.sha256 == hashlib.sha256(original).hexdigest()


def test_trace_values_are_backed_by_immutable_bytes():
    values = WorkloadTrace("custom", {}, np.arange(5, dtype=">i4")).values
    base = values
    while isinstance(base, np.ndarray):
        base = base.base
    assert isinstance(base, bytes) and base == np.array([0, 1, 2, 3, 4], dtype="<f8").tobytes()
    assert not values.flags.owndata and not values.flags.writeable
    assert values.ndim == 1 and values.flags.c_contiguous and values.dtype.str == "<f8"


METADATA_CHANGES = [
    pytest.param("dtype", ">f8", id="byte-order"), pytest.param("dtype", "<i8", id="integers"),
    pytest.param("dtype", "<f4", id="narrower-floats"), pytest.param("shape", (2, 900), id="two-dimensional"),
    pytest.param("strides", (0,), id="repeat-first-element"),
]


# NumPy allows these on read-only arrays; newer versions warn and later ones may refuse.
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("attribute, value", METADATA_CHANGES)
def test_changing_array_metadata_in_place_never_affects_the_trace(attribute, value):
    trace, genuine = workloads.gradual_ramp(), workloads.gradual_ramp()
    handed_out = trace.values
    for target in (handed_out, trace.values, trace.values.view()):
        with contextlib.suppress(AttributeError):
            setattr(target, attribute, value)
    assert (trace.values.dtype.str, trace.values.shape, trace.values.strides) == ("<f8", (1800,), (8,))
    np.testing.assert_array_equal(trace.values, np.linspace(100.0, 700.0, 1800))
    assert trace.sha256 == hashlib.sha256(trace.values.tobytes()).hexdigest() == genuine.sha256
    assert run_episode(StaticBaseline(), trace) == run_episode(StaticBaseline(), genuine)


def test_every_access_returns_an_independent_read_only_array():
    trace = workloads.gradual_ramp()
    first, second = trace.values, trace.values
    assert first is not second and not first.flags.writeable and not second.flags.writeable
    np.testing.assert_array_equal(first, second)


def test_values_cannot_be_replaced_even_below_the_frozen_dataclass():
    trace = workloads.constant()
    with pytest.raises(dataclasses.FrozenInstanceError):
        object.__setattr__(trace, "values", np.zeros(1800))
    with pytest.raises(AttributeError):
        del trace.values
    with pytest.raises(AttributeError):
        object.__delattr__(trace, "values")
    assert trace.values.tolist() == [100.0] * 1800


def test_values_stays_a_required_dataclass_field():
    assert [field.name for field in dataclasses.fields(WorkloadTrace)] == ["name", "params", "values", "sha256"]
    with pytest.raises(TypeError):
        WorkloadTrace("custom", {})
    trace = workloads.flash_crowd()
    for duplicate in (dataclasses.replace(trace, name="renamed"), copy.copy(trace)):
        np.testing.assert_array_equal(duplicate.values, trace.values)
        assert duplicate.sha256 == trace.sha256 and not duplicate.values.flags.writeable


class MutableLeaf:
    """An arbitrary object whose state can change after it is stored."""

    def __init__(self):
        self.x = 1


class CopiesToAnInt:
    """Hostile to deep copying: its copy would look like a plain int."""

    def __deepcopy__(self, memo):
        return 7


class IntWithState(int):
    """An int subclass, so its instances carry a mutable __dict__."""


class StrWithState(str):
    """A str subclass, so its instances carry a mutable __dict__."""


UNSUPPORTED_LEAVES = [
    pytest.param(bytearray(b"ab"), id="bytearray"), pytest.param(memoryview(bytearray(b"ab")), id="memoryview"),
    pytest.param(array.array("d", [1.0]), id="array"), pytest.param(MutableLeaf(), id="custom-object"),
    pytest.param(CopiesToAnInt(), id="deepcopy-hook"), pytest.param(IntWithState(3), id="int-subclass"),
    pytest.param(StrWithState("x"), id="str-subclass"), pytest.param(lambda: None, id="function"),
    pytest.param(iter([1]), id="iterator"), pytest.param(b"ab", id="bytes"), pytest.param(1j, id="complex"),
    pytest.param(np.array([None, bytearray(b"ab")], dtype=object), id="object-array"),
    pytest.param(np.datetime64("2026-01-01"), id="datetime"),
]


@pytest.mark.parametrize("wrap", [
    pytest.param(lambda leaf: leaf, id="top"), pytest.param(lambda leaf: [leaf], id="list"),
    pytest.param(lambda leaf: (1, (leaf,)), id="tuple"), pytest.param(lambda leaf: {"inner": {"deep": leaf}}, id="dict"),
])
@pytest.mark.parametrize("leaf", UNSUPPORTED_LEAVES)
def test_unsupported_parameter_leaves_are_rejected(leaf, wrap):
    with pytest.raises((TypeError, ValueError)):
        WorkloadTrace("custom", {"value": wrap(leaf)}, np.ones(3))


@pytest.mark.parametrize("params", [
    pytest.param({"value": {MutableLeaf()}}, id="object-in-set"), pytest.param({MutableLeaf(): 1}, id="object-key"),
    pytest.param({"outer": {StrWithState("k"): 1}}, id="str-subclass-key"), pytest.param({1: "a"}, id="int-key"),
])
def test_mutable_set_members_and_keys_are_rejected(params):
    with pytest.raises((TypeError, ValueError)):
        WorkloadTrace("custom", params, np.ones(3))


def test_supported_parameter_leaves_become_plain_immutable_values():
    source_array = np.array([1, 2])
    trace = WorkloadTrace("custom", {
        "none": None, "flag": True, "count": 3, "rate": 1.5, "label": "x", "np_float": np.float64(2.5),
        "np_int": np.int64(4), "np_bool": np.bool_(False), "array": source_array, "scalar_array": np.array(3.0),
        "members": {"a"},
    }, np.ones(3))
    source_array[0] = 99
    assert dict(trace.params) == {
        "none": None, "flag": True, "count": 3, "rate": 1.5, "label": "x", "np_float": 2.5, "np_int": 4,
        "np_bool": False, "array": (1, 2), "scalar_array": 3.0, "members": frozenset({"a"}),
    }
    assert [type(trace.params[key]) for key in ("np_float", "np_int", "np_bool")] == [float, int, bool]


def test_every_container_in_params_is_read_only_at_any_depth():
    # Types are checked explicitly: a mutable set compares equal to a frozenset.
    trace = WorkloadTrace("custom", {"members": {"a"}, "outer": {"list": [1, {"deep": [2]}], "set": {3}}}, np.ones(3))
    outer = trace.params["outer"]
    assert [type(value) for value in (trace.params, outer, outer["list"][1])] == [MappingProxyType] * 3
    assert [type(value) for value in (outer["list"], outer["list"][1]["deep"])] == [tuple] * 2
    assert [type(value) for value in (trace.params["members"], outer["set"])] == [frozenset] * 2
    for attempt in (
        lambda: trace.params["members"].add("b"), lambda: outer["list"].append(4),
        lambda: outer["list"][1].update(deep=0), lambda: outer["set"].clear(),
    ):
        with pytest.raises(AttributeError):
            attempt()


def test_frozen_params_can_be_reused_for_a_new_trace():
    first = WorkloadTrace("custom", {"outer": {"inner": [1, 2]}}, np.ones(3))
    second = WorkloadTrace(first.name, first.params, first.values, first.sha256)
    assert second.params["outer"]["inner"] == (1, 2) and second.sha256 == first.sha256


@pytest.mark.parametrize("name", GENERATORS)
def test_generator_params_cannot_be_mutated(name):
    trace = getattr(workloads, name)()
    assert isinstance(trace.params, MappingProxyType)
    with pytest.raises(TypeError):
        trace.params["horizon"] = 5
    with pytest.raises(TypeError):
        trace.params["new_key"] = 1
    assert trace.params["horizon"] == 1800


def test_nested_params_are_deep_copied_and_frozen():
    source = {"outer": {"inner": [1, 2]}, "rates": [1.0, 2.0], "tags": {"a"}}
    trace = WorkloadTrace("custom", source, np.ones(3))
    source["outer"]["inner"].append(3)
    source["rates"][0] = 9.0
    assert trace.params["outer"]["inner"] == (1, 2)
    assert trace.params["rates"] == (1.0, 2.0)
    assert trace.params["tags"] == frozenset({"a"})
    with pytest.raises(TypeError):
        trace.params["outer"]["new"] = 0


def test_values_are_copied_into_little_endian_float64():
    source = np.arange(5, dtype=">i4")
    trace = WorkloadTrace("custom", {}, source)
    source[0] = 99
    assert trace.values.dtype.str == "<f8"
    np.testing.assert_array_equal(trace.values, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert trace.sha256 == hashlib.sha256(np.array([0, 1, 2, 3, 4], dtype="<f8").tobytes()).hexdigest()


@pytest.mark.parametrize("values", [
    pytest.param(np.zeros((2, 3)), id="2d"), pytest.param(np.zeros(0), id="empty"),
    pytest.param(np.array([1.0, np.nan]), id="nan"), pytest.param(np.array([1.0, np.inf]), id="inf"),
    pytest.param(np.array([1.0, -0.5]), id="negative"), pytest.param(np.array([True, False]), id="bool"),
    pytest.param(np.array([1 + 0j]), id="complex"), pytest.param(np.array(["1"]), id="string"),
    pytest.param(np.array([1.0], dtype=object), id="object"), pytest.param(5.0, id="scalar"),
])
def test_invalid_trace_values_are_rejected(values):
    with pytest.raises(ValueError):
        WorkloadTrace("custom", {}, values)


@pytest.mark.parametrize("name, params", [
    pytest.param("", {}, id="empty-name"), pytest.param(5, {}, id="non-string-name"),
    pytest.param("custom", [1], id="params-not-a-mapping"),
])
def test_invalid_trace_identity_is_rejected(name, params):
    with pytest.raises(ValueError):
        WorkloadTrace(name, params, np.ones(3))


def test_supplied_digest_must_match_the_values():
    good = hashlib.sha256(np.ones(3, dtype="<f8").tobytes()).hexdigest()
    assert WorkloadTrace("custom", {}, np.ones(3), good).sha256 == good
    with pytest.raises(ValueError):
        WorkloadTrace("custom", {}, np.ones(3), "0" * 64)


@pytest.mark.parametrize("name, t, expected", [
    ("constant", 0, 100.0), ("constant", 1799, 100.0),
    ("step_jump", 299, 100.0), ("step_jump", 300, 600.0), ("step_jump", 1799, 600.0),
    ("gradual_ramp", 0, 100.0), ("gradual_ramp", 1, 100.0 + 600.0 / 1799), ("gradual_ramp", 1799, 700.0),
    ("flash_crowd", 299, 100.0), ("flash_crowd", 300, 900.0), ("flash_crowd", 419, 900.0), ("flash_crowd", 420, 100.0),
    ("burst_and_drop", 299, 100.0), ("burst_and_drop", 300, 100.0), ("burst_and_drop", 301, 100.0 + 700.0 / 60),
    ("burst_and_drop", 359, 100.0 + 700.0 * 59 / 60), ("burst_and_drop", 360, 800.0),
    ("burst_and_drop", 539, 800.0), ("burst_and_drop", 540, 100.0), ("burst_and_drop", 1799, 100.0),
    ("db_bottleneck", 299, 100.0), ("db_bottleneck", 300, 450.0), ("db_bottleneck", 899, 450.0),
    ("db_bottleneck", 900, 100.0),
])
def test_default_trace_boundary_values(name, t, expected):
    assert getattr(workloads, name)().values[t] == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("name, value, count", [
    ("step_jump", 600.0, 1500), ("flash_crowd", 900.0, 120), ("burst_and_drop", 800.0, 180), ("db_bottleneck", 450.0, 600),
])
def test_default_trace_window_lengths(name, value, count):
    assert int(np.sum(getattr(workloads, name)().values == value)) == count


def test_burst_and_drop_ramp_rises_strictly_toward_the_peak():
    ramp = workloads.burst_and_drop().values[300:360]
    assert np.all(np.diff(ramp) > 0) and ramp.max() < 800.0


def test_windows_may_end_exactly_at_the_horizon():
    assert workloads.flash_crowd(start_time=95, duration=5, horizon=100).values[95:].tolist() == [900.0] * 5
    assert workloads.burst_and_drop(ramp_start=90, ramp_duration=5, hold_duration=5, horizon=100).values[-1] == 800.0
    assert workloads.step_jump(jump_time=0, horizon=10).values.tolist() == [600.0] * 10


def test_digest_is_deterministic_and_follows_the_values():
    assert workloads.flash_crowd().sha256 == workloads.flash_crowd().sha256
    assert workloads.flash_crowd().sha256 != workloads.flash_crowd(peak_rate=901.0).sha256
    assert workloads.constant().sha256 == hashlib.sha256(np.full(1800, 100.0, dtype="<f8").tobytes()).hexdigest()


def test_seeded_poisson_is_reproducible_and_seed_sensitive():
    first, again, other = (workloads.seeded_bursty_poisson(seed=seed) for seed in (42, 42, 43))
    np.testing.assert_array_equal(first.values, again.values)
    assert first.sha256 == again.sha256 != other.sha256
    assert np.all(first.values == np.round(first.values))  # Poisson counts are whole numbers


def test_seeded_poisson_never_touches_the_global_random_state():
    np.random.seed(123)
    before = np.random.get_state()
    trace = workloads.seeded_bursty_poisson(seed=7)
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    np.random.seed(999)
    np.testing.assert_array_equal(trace.values, workloads.seeded_bursty_poisson(seed=7).values)


@pytest.mark.parametrize("burst_prob, expected_mean", [(0.0, 150.0), (1.0, 750.0)])
def test_burst_probability_extremes_select_the_poisson_rate(burst_prob, expected_mean):
    # 1800 Poisson samples: the standard error of the mean is at most sqrt(750 / 1800), about 0.65.
    assert abs(workloads.seeded_bursty_poisson(burst_prob=burst_prob).values.mean() - expected_mean) < 5.0


def test_bursts_last_the_drawn_length():
    # Base rate 0 makes bursts visible. Back-to-back bursts merge into multiples of 30.
    trace = workloads.seeded_bursty_poisson(base_rate=0.0, burst_rate=1000.0, min_burst_len=30, max_burst_len=30)
    runs = burst_runs(trace.values > 0)
    assert len(runs) >= 2 and any(start + length < 1800 for start, length in runs)  # bursts end
    for start, length in runs:
        assert length % 30 == 0 or start + length == 1800


def test_bursts_longer_than_the_horizon_are_cut_off():
    trace = workloads.seeded_bursty_poisson(burst_prob=1.0, min_burst_len=5000, max_burst_len=5000, horizon=100)
    assert trace.values.shape == (100,)
    assert abs(trace.values.mean() - 750.0) < 15.0  # standard error about 2.7


INVALID_GENERATOR_ARGS = [
    ("constant", {"rate": -1.0}), ("constant", {"rate": float("nan")}), ("constant", {"rate": float("inf")}),
    ("constant", {"rate": True}), ("constant", {"rate": "100"}), ("constant", {"horizon": 0}),
    ("constant", {"horizon": 10.0}), ("constant", {"horizon": True}),
    ("step_jump", {"jump_time": 1800}), ("step_jump", {"jump_time": -1}), ("step_jump", {"jump_rate": -5.0}),
    ("step_jump", {"jump_time": 300.0}),
    ("gradual_ramp", {"end_rate": float("nan")}), ("gradual_ramp", {"start_rate": -1.0}), ("gradual_ramp", {"horizon": 0}),
    ("flash_crowd", {"duration": 0}), ("flash_crowd", {"start_time": 1700}), ("flash_crowd", {"start_time": -1}),
    ("flash_crowd", {"peak_rate": float("inf")}),
    ("burst_and_drop", {"ramp_duration": 0}), ("burst_and_drop", {"hold_duration": 0}),
    ("burst_and_drop", {"ramp_start": 1600}), ("burst_and_drop", {"base_rate": False}),
    ("db_bottleneck", {"duration": 1501}), ("db_bottleneck", {"surge_rate": -450.0}), ("db_bottleneck", {"start_time": True}),
    ("seeded_bursty_poisson", {"burst_prob": 1.5}), ("seeded_bursty_poisson", {"burst_prob": -0.1}),
    ("seeded_bursty_poisson", {"burst_prob": float("nan")}), ("seeded_bursty_poisson", {"min_burst_len": 0}),
    ("seeded_bursty_poisson", {"min_burst_len": 70}), ("seeded_bursty_poisson", {"seed": -1}),
    ("seeded_bursty_poisson", {"seed": True}), ("seeded_bursty_poisson", {"seed": 4.2}),
    ("seeded_bursty_poisson", {"base_rate": float("inf")}),
]


@pytest.mark.parametrize("name, kwargs", [
    pytest.param(name, kwargs, id=f"{name}-{key}={value!r}")
    for name, kwargs in INVALID_GENERATOR_ARGS for key, value in kwargs.items()
])
def test_invalid_generator_parameters_are_rejected(name, kwargs):
    with pytest.raises(ValueError):
        getattr(workloads, name)(**kwargs)


HUGE = 10 ** 400  # a valid integer that no float can hold
TOO_LARGE_FOR_A_FLOAT = [
    ("constant", {"rate": HUGE}), ("constant", {"rate": Fraction(HUGE)}), ("step_jump", {"base_rate": HUGE}),
    ("step_jump", {"jump_rate": HUGE}), ("gradual_ramp", {"start_rate": HUGE}), ("gradual_ramp", {"end_rate": HUGE}),
    ("flash_crowd", {"peak_rate": HUGE}), ("burst_and_drop", {"peak_rate": HUGE}), ("db_bottleneck", {"surge_rate": HUGE}),
    ("seeded_bursty_poisson", {"base_rate": HUGE}), ("seeded_bursty_poisson", {"burst_rate": HUGE}),
    ("seeded_bursty_poisson", {"burst_prob": HUGE}), ("constant", {"rate": -HUGE}), ("constant", {"horizon": HUGE}),
]


@pytest.mark.parametrize("name, kwargs", [
    pytest.param(name, kwargs, id=f"{name}-{index}") for index, (name, kwargs) in enumerate(TOO_LARGE_FOR_A_FLOAT)
])
def test_numbers_too_large_for_a_float_raise_value_error(name, kwargs):
    with pytest.raises(ValueError):
        getattr(workloads, name)(**kwargs)


@pytest.mark.parametrize("initial, database_arrivals", [
    ((5, 4, 5, 2, 1), 405.0),  # the smallest pod counts that pass the whole surge
    ((4, 4, 5, 2, 1), 360.0), ((5, 3, 5, 2, 1), 378.0), ((5, 4, 4, 2, 1), 385.0), ((5, 4, 5, 1, 1), 365.0),
    ((2, 2, 2, 2, 1), 180.0),  # the default pod counts
])
def test_db_bottleneck_needs_enough_pods_at_every_upstream_service(initial, database_arrivals):
    """SPEC_02 Section 1.5: dropping any one service by one pod keeps part of the surge from the database."""
    surge = workloads.db_bottleneck(start_time=0, duration=600).values
    env = MicroserviceMeshEnv(surge, dataclasses.replace(SimulatorConfig(), initial_replicas=initial))
    env.reset(seed=0)
    rows = env.step(env.shield.approve(HOLD))[4]["substeps"]
    assert rows["arrivals_new"][2, 4] == pytest.approx(database_arrivals)


def test_benchmark_suite_is_the_six_canonical_traces_then_poisson():
    names = [trace.name for trace in workloads.benchmark_suite()]
    assert names == list(GENERATORS[:6]) + ["seeded_bursty_poisson"]
    assert workloads.benchmark_suite(seed=5)[-1].sha256 == workloads.seeded_bursty_poisson(seed=5).sha256


# ---------------------------------------------------------- 2. controllers
def test_feature_count_matches_the_frozen_environment():
    assert FEATURE_COUNT == env_module.NUM_FEATURES


@pytest.mark.parametrize("make", CONTROLLERS)
def test_every_baseline_holds_at_step_zero(make):
    assert make().propose(observation(utilization=1.0, queue=1.0), SimulatorConfig(), 0) == HOLD


def test_static_baseline_never_scales():
    controller = StaticBaseline()
    for step, obs in enumerate([observation(1.0, 1.0), observation(0.0), observation(0.5, 0.2)], start=1):
        assert controller.propose(obs, SimulatorConfig(), step) == HOLD


@pytest.mark.parametrize("make", CONTROLLERS)
def test_proposals_are_four_python_ints_from_node_features_alone(make):
    proposal = make().propose({"node_features": observation(0.9)["node_features"]}, SimulatorConfig(), 3)
    assert isinstance(proposal, tuple) and len(proposal) == 4 and all(type(value) is int for value in proposal)


@pytest.mark.parametrize("make", [SimHPABaseline, TunedReactiveThresholdController])
def test_reactive_baselines_reject_malformed_features(make):
    with pytest.raises(ValueError):
        make().propose({"node_features": np.zeros((4, FEATURE_COUNT))}, SimulatorConfig(), 1)


@pytest.mark.parametrize("make", CONTROLLERS)
def test_params_are_defensive_copies(make):
    controller = make()
    snapshot = controller.params
    controller.params["injected"] = 1
    assert controller.params == snapshot and "injected" not in controller.params


def test_baselines_cannot_reach_the_environment_or_shield():
    assert not {"sim.env", "sim.shield"} & imported_modules(ROOT / "sim/baselines.py")


HPA = SimHPABaseline(target_utilization=0.5, tolerance=0.25)  # binary-exact ratios


@pytest.mark.parametrize("utilization, expected", [
    (0.5, 0),       # ratio 1.0
    (0.625, 0),     # ratio 1.25: on the upper edge of the deadband
    (0.375, 0),     # ratio 0.75: on the lower edge of the deadband
    (0.6875, 1),    # ratio 1.375: ceil(2 * 1.375) = 3
    (0.63, 1),      # ratio 1.26: just outside, ceil(2.52) = 3
    (0.37, 0),      # ratio 0.74: outside, but ceil(1.48) = 2 keeps 2 pods
    (0.25, -1),     # ratio 0.5: ceil(1.0) = 1
])
def test_sim_hpa_deadband_boundaries(utilization, expected):
    assert HPA.propose(observation(utilization), SimulatorConfig(), 1) == (expected,) * 4


@pytest.mark.parametrize("ready, utilization, expected", [
    (0.3, 0.75, 2),     # desired 3: ceil(3 * 1.5) = 5
    (0.4, 0.3125, -1),  # desired 4: ceil(4 * 0.625) = 3
    (0.2, 1.0, 2),      # desired 2: ceil(2 * 2.0) = 4
    (1.0, 1.0, 2),      # desired 10: target 20, clipped to +2
    (0.8, 0.0, -1),     # desired 8: target max(1, 0) = 1, clipped to -1
])
def test_sim_hpa_ceiling_target_and_clipping(ready, utilization, expected):
    assert HPA.propose(observation(utilization, ready=ready), SimulatorConfig(), 1) == (expected,) * 4


def test_sim_hpa_never_targets_below_min_replicas():
    config = dataclasses.replace(
        SimulatorConfig(), min_replicas=(5, 1, 1, 1, 1), initial_replicas=(5, 2, 2, 2, 1), action_min=-2,
    )
    # desired 6 and no utilization: Gateway's target is its minimum 5 (-1); the others reach -5, clipped to -2.
    assert HPA.propose(observation(0.0, ready=0.6), config, 1) == (-1, -2, -2, -2)


def test_desired_reconstruction_matches_the_environment_with_pending_pods():
    env = MicroserviceMeshEnv(workloads.constant(600.0).values)
    env.reset(seed=0)
    pattern = [(2, 2, 2, 2), (1, 0, 2, 1), HOLD, (-1, 2, -1, 0)]
    steps_with_pending = 0
    for step in range(40):
        obs = env.step(env.shield.approve(pattern[step % 4]))[0]
        assert reconstruct_desired(obs["node_features"], env.config) == tuple(int(value) for value in env.desired[:4])
        steps_with_pending += int(env.pending[:4].sum() > 0)
    assert steps_with_pending > 0


def test_sim_hpa_counts_pending_pods_as_desired():
    env = MicroserviceMeshEnv(workloads.constant(100.0).values)
    env.reset(seed=0)
    obs = env.step(env.shield.approve((2, 0, 0, 0)))[0]
    assert (env.ready[GATEWAY], env.pending[GATEWAY], env.desired[GATEWAY]) == (2, 2, 4)
    features = obs["node_features"].copy()
    features[:, 3] = 0.75  # ratio 1.5 at target 0.5
    proposal = HPA.propose({"node_features": features}, env.config, env.step_index)
    # desired 4 gives ceil(6.0) = 6, so +2. Counting only ready pods would give ceil(3.0) = 3, so +1.
    assert proposal[GATEWAY] == 2


TUNED = TunedReactiveThresholdController()


@pytest.mark.parametrize("utilization, queue, expected", [
    (0.86, 0.0, 2), (0.0, 0.26, 2),   # above a high threshold
    (0.85, 0.0, 1), (0.0, 0.25, 1),   # exactly at a high threshold is not above it
    (0.71, 0.0, 1), (0.0, 0.11, 1),   # above a medium threshold
    (0.70, 0.0, 0), (0.5, 0.10, 0),   # exactly at a medium threshold is not above it
    (0.34, 0.0, -1),                  # below util_low with an empty queue
    (0.35, 0.0, 0),                   # exactly at util_low is not below it
    (0.34, 1e-9, 0),                  # any queued work blocks a scale-down
    (0.5, 0.05, 0),                   # between thresholds
])
def test_tuned_threshold_branches_and_strict_boundaries(utilization, queue, expected):
    assert TUNED.propose(observation(utilization, queue), SimulatorConfig(), 1) == (expected,) * 4


def test_tuned_decides_each_node_independently():
    features = np.zeros((5, FEATURE_COUNT))
    features[:, 3] = (0.9, 0.75, 0.5, 0.1, 1.0)  # the database row is ignored
    assert TUNED.propose({"node_features": features}, SimulatorConfig(), 1) == (2, 1, 0, -1)


@pytest.mark.parametrize("kwargs", [
    {"target_utilization": 0.0}, {"target_utilization": -0.1}, {"target_utilization": 1.1},
    {"target_utilization": float("nan")}, {"target_utilization": True}, {"target_utilization": "0.7"},
    {"tolerance": -0.1}, {"tolerance": 1.0}, {"tolerance": float("inf")}, {"tolerance": False},
], ids=repr)
def test_sim_hpa_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        SimHPABaseline(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {"util_low": 0.7}, {"util_med": 0.85}, {"util_high": 1.01}, {"util_low": -0.01}, {"util_high": 0.6},
    {"queue_med": 0.0}, {"queue_med": 0.25}, {"queue_high": 1.5}, {"util_high": True},
    {"queue_high": float("nan")}, {"util_med": "0.7"},
], ids=repr)
def test_tuned_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        TunedReactiveThresholdController(**kwargs)


@pytest.mark.parametrize("make", [
    lambda value: SimHPABaseline(target_utilization=value), lambda value: SimHPABaseline(tolerance=value),
    lambda value: TunedReactiveThresholdController(util_high=value),
    lambda value: TunedReactiveThresholdController(queue_med=value),
], ids=["target", "tolerance", "util_high", "queue_med"])
@pytest.mark.parametrize("value", [HUGE, -HUGE, Fraction(HUGE)], ids=["positive", "negative", "fraction"])
def test_hyperparameters_too_large_for_a_float_raise_value_error(make, value):
    with pytest.raises(ValueError):
        make(value)


def test_boundary_hyperparameters_are_accepted():
    SimHPABaseline(target_utilization=1.0, tolerance=0.0)
    TunedReactiveThresholdController(util_low=0.0, util_high=1.0, queue_high=1.0)


# -------------------------------------------------------------- 3. runner
def test_default_horizon_episode_runs_to_termination():
    metrics = run_episode(StaticBaseline(), workloads.constant())["metrics"]
    assert (metrics["control_steps_executed"], metrics["substeps_executed"]) == (360, 1800)
    assert metrics["terminated"] is True and metrics["truncated"] is False


def test_short_horizon_comes_from_the_config():
    manifest = run_episode(TunedReactiveThresholdController(), workloads.constant(horizon=100), config=SHORT)
    assert (manifest["metrics"]["control_steps_executed"], manifest["metrics"]["substeps_executed"]) == (20, 100)
    assert manifest["config"]["control_horizon"] == 20 and manifest["workload"]["horizon"] == 100


@pytest.mark.parametrize("trace, config", [
    pytest.param(workloads.constant(), SHORT, id="long-trace-short-config"),
    pytest.param(workloads.constant(horizon=100), None, id="short-trace-default-config"),
    pytest.param(workloads.constant(horizon=1799), None, id="one-substep-short"),
])
def test_mismatched_trace_and_config_raise(trace, config):
    with pytest.raises(ValueError, match="horizon"):
        run_episode(StaticBaseline(), trace, config=config)


def test_runner_source_has_no_literal_horizon():
    tree = ast.parse((ROOT / "sim/runner.py").read_text())
    numbers = {
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    }
    assert not {360, 1800} & numbers


@pytest.mark.parametrize("controller, trace, seed", [
    pytest.param(object(), workloads.constant(), 0, id="not-a-controller"),
    pytest.param(StaticBaseline(), np.full(1800, 100.0), 0, id="raw-array-trace"),
    pytest.param(StaticBaseline(), workloads.constant(), True, id="bool-seed"),
    pytest.param(StaticBaseline(), workloads.constant(), -1, id="negative-seed"),
])
def test_runner_rejects_invalid_arguments(controller, trace, seed):
    with pytest.raises((TypeError, ValueError)):
        run_episode(controller, trace, seed=seed)


def test_reset_runs_before_every_episode():
    controller = ScriptedController()
    traces = [workloads.constant(horizon=100), workloads.constant(rate=50.0, horizon=100)]
    run_benchmark_suite([controller], traces, config=SHORT)
    assert controller.log == ["reset", *range(20), "reset", *range(20)]


def test_fallbacks_are_counted_once_and_overrides_only_for_valid_proposals():
    invalid = {5: (float("nan"), 0, 0, 0), 7: "junk", 9: (1, 2, 3)}
    controller = ScriptedController(script=invalid, default=(2, 2, 2, 2))
    metrics = run_episode(controller, workloads.constant(horizon=100), config=SHORT)["metrics"]
    # Summing the cumulative per-step counter would give 1*2 + 2*2 + 3*11 = 39.
    assert metrics["shield_fallback_count"] == 3
    # Steps 0 and 1 are approved in full. From step 2 the 28-pod quota trims every
    # valid +2, so 18 steps are overridden; the 3 invalid steps are fallbacks, not overrides.
    assert metrics["shield_arbitration_override_count"] == 15
    assert metrics["total_scaling_events"] == 8 + 8 + 3


@pytest.mark.parametrize("make", CONTROLLERS)
@pytest.mark.parametrize("trace_name", ["constant", "flash_crowd", "seeded_bursty_poisson"])
def test_cost_accounting_identities(make, trace_name):
    metrics = run_episode(make(), getattr(workloads, trace_name)())["metrics"]
    weighted, unweighted, config = metrics["weighted_cost_sums"], metrics["unweighted_cost_sums"], SimulatorConfig()
    weights = {"slo_violation": config.weight_slo, "failure_cost": config.weight_failure,
               "resource_cost": config.weight_resources, "scaling_cost": config.weight_scaling}
    for key, weight in weights.items():
        assert weighted[key] == pytest.approx(weight * unweighted[key], rel=1e-12)
    assert sum(weighted[key] for key in weights) == pytest.approx(weighted["total_cost"], rel=1e-9, abs=1e-9)
    assert weighted["total_cost"] == pytest.approx(-metrics["total_reward"], rel=1e-9, abs=1e-9)
    assert metrics["mean_reward"] == pytest.approx(metrics["total_reward"] / metrics["control_steps_executed"])


def test_static_run_matches_hand_computed_metrics():
    metrics = run_episode(StaticBaseline(), workloads.constant(horizon=100), config=SHORT)["metrics"]
    # 9 pods for 100 seconds, a 39 ms modeled latency at every substep, and no failures.
    assert metrics["total_pod_seconds"] == 900 and metrics["total_scaling_events"] == 0
    assert metrics["substep_latency_proxy_ms"] == pytest.approx(
        {"mean": 39.0, "p50": 39.0, "p95": 39.0, "p99": 39.0, "max": 39.0})
    assert metrics["slo_violation_substeps"] == 0 and metrics["max_e2e_failure_proxy"] == 0.0
    assert metrics["total_reward"] == pytest.approx(-20 * 0.5 * 9 / 28)
    assert metrics["unweighted_cost_sums"]["resource_cost"] == pytest.approx(20 * 9 / 28)
    assert (metrics["shield_fallback_count"], metrics["shield_arbitration_override_count"]) == (0, 0)


@pytest.mark.parametrize("make", [SimHPABaseline, TunedReactiveThresholdController])
def test_dynamic_baselines_scale_up_under_sustained_load(make):
    metrics = run_episode(make(), workloads.constant(rate=600.0, horizon=100), config=SHORT)["metrics"]
    # Occupancy can rise above the initial 9 pods only through approved scale-ups.
    assert metrics["total_scaling_events"] > 0
    assert metrics["total_pod_seconds"] > 9 * 100


def test_metrics_match_an_independent_replay():
    """Drive the environment by hand and recompute every metric from the raw arrays."""
    trace, controller = workloads.flash_crowd(), TunedReactiveThresholdController()
    metrics = run_episode(controller, trace)["metrics"]
    env = MicroserviceMeshEnv(trace.values)
    obs, _ = env.reset(seed=0)
    controller.reset()
    latencies, failures, occupancy, rewards, approved, overrides = [], [], [], [], [], 0
    terminated = truncated = False
    while not (terminated or truncated):
        raw = controller.propose(obs, env.config, env.step_index)
        token = env.shield.approve(raw)
        overrides += int(token.deltas != raw)  # this controller only makes valid proposals
        obs, reward, terminated, truncated, info = env.step(token)
        latencies.append(info["substeps"]["e2e_latency_ms"])
        failures.append(info["substeps"]["e2e_failure"])
        occupancy.append(info["substeps"]["occupancy"])
        rewards.append(reward)
        approved.extend(int(value) for value in info["approved_delta"])
    latency, failure = np.concatenate(latencies), np.concatenate(failures)
    p50, p95, p99 = np.percentile(latency, [50, 95, 99])
    assert metrics["substep_latency_proxy_ms"] == pytest.approx(
        {"mean": latency.mean(), "p50": p50, "p95": p95, "p99": p99, "max": latency.max()})
    assert metrics["slo_violation_substeps"] == int((latency > 250.0).sum())
    assert metrics["mean_e2e_failure_proxy"] == pytest.approx(failure.mean())
    assert metrics["max_e2e_failure_proxy"] == pytest.approx(failure.max())
    assert metrics["total_pod_seconds"] == int(np.concatenate(occupancy).sum())
    assert metrics["total_scaling_events"] == sum(abs(value) for value in approved)
    assert metrics["total_reward"] == pytest.approx(sum(rewards))
    assert metrics["shield_arbitration_override_count"] == overrides
    assert min(approved) < 0 < max(approved)  # the replay covers scale-ups and scale-downs


def test_traces_sharing_a_name_get_separate_files(tmp_path):
    traces = [workloads.constant(rate=100.0, horizon=100), workloads.constant(rate=200.0, horizon=100)]
    run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=tmp_path)
    assert len(list(tmp_path.iterdir())) == 2


def test_second_run_into_the_same_directory_never_overwrites(tmp_path):
    controller, traces = ScriptedController(), [short_trace(), short_trace(rate=200.0)]
    run_benchmark_suite([controller], traces, config=SHORT, output_dir=tmp_path)
    before, calls = snapshot(tmp_path), list(controller.log)
    assert len(before) == 2
    with pytest.raises(FileExistsError) as caught:
        run_benchmark_suite([controller], traces, config=SHORT, output_dir=tmp_path)
    assert str(tmp_path / manifest_name("scripted", traces[0])) in str(caught.value)
    assert snapshot(tmp_path) == before
    assert controller.log == calls  # refused before running any episode


def test_one_existing_destination_stops_the_run_before_anything_is_written(tmp_path):
    traces = [short_trace(), short_trace(rate=200.0)]
    existing = tmp_path / manifest_name("static", traces[1])
    existing.write_text("precious")
    with pytest.raises(FileExistsError, match=existing.name):
        run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=tmp_path)
    assert snapshot(tmp_path) == {existing.name: b"precious"}


def test_exclusive_creation_protects_a_file_that_appears_during_the_run(tmp_path):
    trace = short_trace()
    destination = tmp_path / manifest_name("scripted", trace)

    class CreatesTheFileMidRun(ScriptedController):
        def propose(self, observation, config, step_index):
            if step_index == 0:
                destination.write_text("from another process")
            return HOLD

    with pytest.raises(FileExistsError, match=destination.name):
        run_benchmark_suite([CreatesTheFileMidRun()], [trace], config=SHORT, output_dir=tmp_path)
    assert snapshot(tmp_path) == {destination.name: b"from another process"}


UNSAFE_NAMES = [
    pytest.param("", id="empty"), pytest.param("   ", id="spaces"), pytest.param("\t\n", id="whitespace"),
    pytest.param("../escape", id="traversal"), pytest.param("..", id="dots"), pytest.param("a..b", id="inner-dots"),
    pytest.param("a/b", id="slash"), pytest.param("/abs", id="absolute"), pytest.param("a\\b", id="backslash"),
    pytest.param("bad\x00name", id="null"), pytest.param("x" * 300, id="too-long"),
    pytest.param("\u00e9" * 150, id="too-long-in-bytes"), pytest.param("\ud800", id="lone-surrogate"), pytest.param(5, id="not-a-string"), pytest.param(None, id="none"),
]


@pytest.mark.parametrize("bad", UNSAFE_NAMES)
def test_unsafe_controller_name_late_in_the_suite_writes_nothing(tmp_path, bad):
    output, good = tmp_path / "new" / "nested", ScriptedController()
    with pytest.raises(ValueError):
        run_benchmark_suite([good, NamedController(bad)], [short_trace()], config=SHORT, output_dir=output)
    assert not (tmp_path / "new").exists() and good.log == []


@pytest.mark.parametrize("bad", UNSAFE_NAMES)
def test_unsafe_workload_name_late_in_the_suite_writes_nothing(tmp_path, bad):
    output, good = tmp_path / "new" / "nested", ScriptedController()
    with pytest.raises(ValueError):
        run_benchmark_suite([good], [short_trace(), short_trace(bad, 200.0)], config=SHORT, output_dir=output)
    assert not (tmp_path / "new").exists() and good.log == []


def test_tampered_digest_cannot_escape_the_output_directory(tmp_path):
    trace = short_trace()
    object.__setattr__(trace, "sha256", "/../../../x0")
    with pytest.raises(ValueError):
        run_benchmark_suite([StaticBaseline()], [trace], config=SHORT, output_dir=tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


def test_symlinked_destination_cannot_escape_the_output_directory(tmp_path):
    trace, output, outside = short_trace(), tmp_path / "out", tmp_path / "outside.json"
    output.mkdir()
    outside.write_text("precious")
    (output / manifest_name("static", trace)).symlink_to(outside)
    with pytest.raises(ValueError):
        run_benchmark_suite([StaticBaseline()], [trace], config=SHORT, output_dir=output)
    assert outside.read_text() == "precious"


@pytest.mark.parametrize("first, second", [
    pytest.param("same", "same", id="identical"), pytest.param("Static", "static", id="case-only"),
    pytest.param("caf\u00e9", "cafe\u0301", id="unicode-normalization"),
])
def test_colliding_file_names_are_rejected_before_writing(tmp_path, first, second):
    controllers = [NamedController(first), NamedController(second)]
    with pytest.raises(ValueError):
        run_benchmark_suite(controllers, [short_trace()], config=SHORT, output_dir=tmp_path / "out")
    assert list(tmp_path.iterdir()) == [] and controllers[0].log == []


def test_wrong_types_late_in_the_suite_write_nothing(tmp_path):
    good = ScriptedController()
    with pytest.raises(TypeError):
        run_benchmark_suite([good, object()], [short_trace()], config=SHORT, output_dir=tmp_path / "out")
    with pytest.raises(TypeError):
        run_benchmark_suite([good], [short_trace(), np.ones(100)], config=SHORT, output_dir=tmp_path / "out")
    assert list(tmp_path.iterdir()) == [] and good.log == []


def test_name_rules_apply_only_when_files_are_written():
    assert len(run_benchmark_suite([NamedController("a/b")], [short_trace("..")], config=SHORT)) == 1


def test_iterators_are_not_exhausted_by_the_checks(tmp_path):
    controllers = (make() for make in (StaticBaseline, SimHPABaseline))
    traces = (short_trace(rate=rate) for rate in (100.0, 200.0))
    assert len(run_benchmark_suite(controllers, traces, config=SHORT, output_dir=tmp_path)) == 4
    assert len(snapshot(tmp_path)) == 4


class BrokenHandle:
    """Wraps a real file opened with "x" and fails during write() or close()."""

    def __init__(self, handle, path, error, stage, before_failing=None):
        self.handle, self.path, self.error, self.stage = handle, path, error, stage
        self.before_failing, self.partial = before_failing, None

    def __enter__(self):
        return self

    def __exit__(self, *details):
        self.handle.close()
        if self.stage == "close":
            self.partial = os.stat(self.path).st_size
            raise self.error

    def write(self, text):
        self.handle.write(text[: len(text) // 2])
        self.handle.flush()
        if self.stage == "write":
            self.partial = os.stat(self.path).st_size
            if self.before_failing:
                self.before_failing(self.path)
            raise self.error


def break_nth_exclusive_open(monkeypatch, n, error, stage="write", before_failing=None) -> list:
    """Make the n-th Path.open(mode="x") return a BrokenHandle; returns every broken handle."""
    real_open, opened, broken = Path.open, [], []

    def fake_open(self, mode="r", *args, **kwargs):
        handle = real_open(self, mode, *args, **kwargs)
        if mode != "x":
            return handle
        opened.append(self)
        if len(opened) != n:
            return handle
        broken.append(BrokenHandle(handle, self, error, stage, before_failing))
        return broken[-1]

    monkeypatch.setattr(Path, "open", fake_open)
    return broken


@pytest.mark.parametrize("stage", ["write", "close"])
@pytest.mark.parametrize("error", [
    pytest.param(OSError(errno.ENOSPC, "disk full"), id="disk-full"), pytest.param(KeyboardInterrupt(), id="interrupt"),
])
def test_failed_write_removes_only_the_incomplete_file(tmp_path, monkeypatch, error, stage):
    (tmp_path / "unrelated.json").write_text("precious")
    traces = [short_trace(rate=rate) for rate in (100.0, 200.0, 300.0)]
    broken = break_nth_exclusive_open(monkeypatch, 2, error, stage)
    with pytest.raises(type(error)) as caught:
        run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=tmp_path)
    monkeypatch.undo()
    assert caught.value is error
    assert broken[0].partial > 0  # the incomplete file really existed before the cleanup
    first = manifest_name("static", traces[0])
    assert sorted(snapshot(tmp_path)) == sorted(["unrelated.json", first])
    assert (tmp_path / "unrelated.json").read_text() == "precious"
    assert json.loads((tmp_path / first).read_text())["workload"]["sha256"] == traces[0].sha256


def test_cleanup_leaves_a_file_that_replaced_the_incomplete_one(tmp_path, monkeypatch):
    def replace_with_foreign_file(path):
        os.unlink(path)
        with open(path, "w") as handle:
            handle.write("from another process")

    trace = short_trace()
    break_nth_exclusive_open(monkeypatch, 1, OSError(errno.EIO, "io error"), before_failing=replace_with_foreign_file)
    with pytest.raises(OSError, match="io error"):
        run_benchmark_suite([StaticBaseline()], [trace], config=SHORT, output_dir=tmp_path)
    monkeypatch.undo()
    assert snapshot(tmp_path) == {manifest_name("static", trace): b"from another process"}


@pytest.mark.parametrize("manifest", [
    pytest.param({"value": object()}, id="not-serializable"), pytest.param({"value": float("nan")}, id="nan"),
    pytest.param({"value": (1, 2)}, id="not-json-native"),
])
def test_manifest_is_serialized_and_checked_before_any_file_is_opened(tmp_path, monkeypatch, manifest):
    monkeypatch.setattr(runner_module, "run_episode", lambda *args, **kwargs: manifest)
    with pytest.raises((TypeError, ValueError)):
        run_benchmark_suite([StaticBaseline()], [short_trace()], config=SHORT, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


class ParamsController(ScriptedController):
    """A scripted controller with arbitrary parameters."""

    def __init__(self, params, name="params"):
        super().__init__()
        self._params, self._name = params, name

    @property
    def name(self):
        return self._name

    @property
    def params(self):
        return self._params


def directory_state(root: Path) -> list:
    """Every path below ``root``, and ``root`` itself, with contents and modification times."""
    return [(str(path), path.lstat().st_mtime_ns, path.read_bytes() if path.is_file() else None)
            for path in [root, *sorted(root.rglob("*"))]]


STRUCTURED = np.array([(1, b"x")], dtype=[("number", "i4"), ("text", "S1")])
NOT_JSON_VALUES = [
    pytest.param(b"ab", TypeError, id="bytes"), pytest.param(bytearray(b"ab"), TypeError, id="bytearray"),
    pytest.param(memoryview(b"ab"), TypeError, id="memoryview"), pytest.param(object(), TypeError, id="object"),
    pytest.param(MutableLeaf(), TypeError, id="custom-object"), pytest.param(Path("x"), TypeError, id="path"),
    pytest.param(Fraction(1, 3), TypeError, id="fraction"), pytest.param(1j, TypeError, id="complex"),
    pytest.param(range(2), TypeError, id="range"), pytest.param(lambda: None, TypeError, id="function"),
    pytest.param(IntWithState(3), TypeError, id="int-subclass"), pytest.param(StrWithState("x"), TypeError, id="str-subclass"),
    pytest.param(float("nan"), ValueError, id="nan"), pytest.param(float("inf"), ValueError, id="inf"),
    pytest.param(float("-inf"), ValueError, id="negative-inf"), pytest.param(np.float64("nan"), ValueError, id="numpy-nan"),
    pytest.param(np.array([1.0, np.inf]), ValueError, id="array-with-inf"), pytest.param(np.array(np.nan), ValueError, id="scalar-array-nan"),
    pytest.param(np.bytes_(b"ab"), TypeError, id="numpy-bytes"), pytest.param(np.complex128(1j), TypeError, id="numpy-complex"),
    pytest.param(np.datetime64("2026-01-01"), TypeError, id="numpy-datetime"), pytest.param(np.longdouble(1), TypeError, id="numpy-longdouble"),
    pytest.param(np.array([None, b"ab"], dtype=object), TypeError, id="object-array"),
    pytest.param(STRUCTURED, TypeError, id="structured-array"), pytest.param(STRUCTURED[0], TypeError, id="structured-scalar"),
    pytest.param({1: "a"}, TypeError, id="int-key"), pytest.param({None: "a"}, TypeError, id="none-key"),
    pytest.param({True: "a"}, TypeError, id="bool-key"), pytest.param({b"k": "a"}, TypeError, id="bytes-key"),
    pytest.param({StrWithState("k"): "a"}, TypeError, id="str-subclass-key"),
    pytest.param({np.str_("k"): "a"}, TypeError, id="numpy-str-key"), pytest.param({1: "a", "1": "b"}, TypeError, id="key-collision"),
]
NESTINGS = [
    pytest.param(lambda value: {"value": value}, id="top"), pytest.param(lambda value: {"outer": {"inner": value}}, id="dict"),
    pytest.param(lambda value: {"outer": [1, [value]]}, id="list"), pytest.param(lambda value: {"outer": (1, (value,))}, id="tuple"),
    pytest.param(lambda value: {"outer": MappingProxyType({"inner": value})}, id="proxy"),
]


@pytest.mark.parametrize("nest", NESTINGS)
@pytest.mark.parametrize("value, error", NOT_JSON_VALUES)
def test_values_that_are_not_json_native_never_reach_a_manifest(tmp_path, value, error, nest):
    with pytest.raises(error):
        json_safe(nest(value))
    alone = ParamsController(nest(value))
    with pytest.raises(error):
        run_episode(alone, short_trace(), config=SHORT)
    assert alone.log == []  # refused before the episode ran
    good, output = ScriptedController(), tmp_path / "out"
    with pytest.raises(error):
        run_benchmark_suite([good, ParamsController(nest(value))], [short_trace()], config=SHORT, output_dir=output)
    assert good.log == [] and not output.exists()


@pytest.mark.parametrize("value, error", [
    pytest.param({b"ab"}, TypeError, id="bytes-in-set"), pytest.param(frozenset({float("nan")}), ValueError, id="nan-in-frozenset"),
    pytest.param({MutableLeaf()}, TypeError, id="object-in-set"),
])
def test_set_members_are_validated_too(value, error):
    with pytest.raises(error):
        run_episode(ParamsController({"members": value}), short_trace(), config=SHORT)


@pytest.mark.parametrize("params", [
    pytest.param({1: "a", "1": "b"}, id="top-level"), pytest.param({"outer": {"1": "a", 1: "b"}}, id="nested"),
])
def test_keys_that_would_collide_as_strings_are_rejected_instead_of_losing_a_value(params):
    with pytest.raises(TypeError, match="key"):
        run_episode(ParamsController(params), short_trace(), config=SHORT)


def test_supported_containers_and_numpy_values_still_convert():
    converted = json_safe({
        "nested": (1, [2.5, {"deep": np.int64(3)}]), "proxy": MappingProxyType({"k": None}), "set": {5}, "frozen": frozenset({"x"}),
        "array": np.array([[1, 2], [3, 4]]), "scalar_array": np.array(2.5), "mixed": np.array([None, "x", 1.5], dtype=object),
        "np_float": np.float32(0.5), "np_bool": np.bool_(True), "np_str": np.str_("y"), "flag": False,
    })
    assert converted == {
        "nested": [1, [2.5, {"deep": 3}]], "proxy": {"k": None}, "set": [5], "frozen": ["x"],
        "array": [[1, 2], [3, 4]], "scalar_array": 2.5, "mixed": [None, "x", 1.5],
        "np_float": 0.5, "np_bool": True, "np_str": "y", "flag": False,
    }
    assert [type(converted[key]) for key in ("np_float", "np_bool", "np_str", "flag", "proxy", "set")] == [float, bool, str, bool, dict, list]
    assert type(converted["nested"][1][1]["deep"]) is int
    assert json.loads(json.dumps(converted)) == converted


def test_parameters_that_turn_invalid_during_the_episode_are_still_caught(tmp_path):
    class TurnsInvalid(ParamsController):
        def propose(self, observation, config, step_index):
            self._params = {"late": b"bytes"}
            return super().propose(observation, config, step_index)

    with pytest.raises(TypeError):
        run_episode(TurnsInvalid({}), short_trace(), config=SHORT)
    with pytest.raises(TypeError):
        run_benchmark_suite([TurnsInvalid({})], [short_trace()], config=SHORT, output_dir=tmp_path)
    assert list(tmp_path.iterdir()) == []


def _late(*extra_traces):
    return lambda good: ([good], [short_trace(), *extra_traces], {})


NAN_TRACE = WorkloadTrace("nan_params", {"x": float("nan")}, np.full(100, 1.0))
PREDICTABLE_FAILURES = [
    pytest.param(_late(workloads.constant()), ValueError, id="later-trace-too-long"),
    pytest.param(_late(workloads.constant(horizon=99)), ValueError, id="later-trace-too-short"),
    pytest.param(_late(np.ones(100)), TypeError, id="later-trace-wrong-type"),
    pytest.param(_late(NAN_TRACE), ValueError, id="later-trace-nan-params"),
    pytest.param(_late(short_trace("a/b", 200.0)), ValueError, id="later-trace-unsafe-name"),
    pytest.param(_late(short_trace()), ValueError, id="later-trace-duplicate"),
    pytest.param(lambda good: ([good, object()], [short_trace()], {}), TypeError, id="later-controller-wrong-type"),
    pytest.param(lambda good: ([good, StaticBaseline], [short_trace()], {}), TypeError, id="later-controller-class"),
    pytest.param(lambda good: ([good, ParamsController({"x": b"a"})], [short_trace()], {}), TypeError, id="later-controller-bytes"),
    pytest.param(lambda good: ([good, NamedController("  ")], [short_trace()], {}), ValueError, id="later-controller-blank-name"),
    pytest.param(lambda good: ([good, NamedController(b"x")], [short_trace()], {}), (TypeError, ValueError), id="later-controller-bytes-name"),
    pytest.param(lambda good: ([good], [short_trace()], {"seed": -1}), ValueError, id="negative-seed"),
    pytest.param(lambda good: ([good], [short_trace()], {"seed": True}), ValueError, id="bool-seed"),
    pytest.param(lambda good: ([good], [short_trace()], {"seed": 1.0}), ValueError, id="float-seed"),
    pytest.param(lambda good: ([good], [short_trace()], {"config": dataclasses.replace(SHORT, slo_ms=Fraction(250))}),
                 TypeError, id="config-not-json-native"),
    pytest.param(lambda good: ([good], [short_trace()], {"config": None}), ValueError, id="default-config-horizon"),
]


@pytest.mark.parametrize("build, error", PREDICTABLE_FAILURES)
def test_predictable_failures_never_create_the_output_directory(tmp_path, build, error):
    good = ScriptedController()
    controllers, traces, options = build(good)
    with pytest.raises(error):
        run_benchmark_suite(controllers, traces, **{"config": SHORT, "output_dir": tmp_path / "new" / "nested", **options})
    assert list(tmp_path.iterdir()) == [] and good.log == []


@pytest.mark.parametrize("build, error", PREDICTABLE_FAILURES)
def test_predictable_failures_leave_an_existing_output_directory_unchanged(tmp_path, build, error):
    output = tmp_path / "out"
    (output / "sub").mkdir(parents=True)
    (output / "sentinel.txt").write_text("precious")
    (output / "sub" / "deep.json").write_text("{}")
    good, before = ScriptedController(), directory_state(output)
    controllers, traces, options = build(good)
    with pytest.raises(error):
        run_benchmark_suite(controllers, traces, **{"config": SHORT, "output_dir": output, **options})
    assert directory_state(output) == before and good.log == []


def test_a_wrong_horizon_is_found_before_any_episode_runs_without_output_too():
    good = ScriptedController()
    with pytest.raises(ValueError, match="horizon"):
        run_benchmark_suite([good], [short_trace(), workloads.constant()], config=SHORT)
    assert good.log == []


@pytest.mark.parametrize("controllers, traces, error", [
    pytest.param([], [workloads.constant()], ValueError, id="trace-without-controllers"),
    pytest.param([object()], [], TypeError, id="controller-without-traces"),
    pytest.param([NamedController("a/b")], [], ValueError, id="unsafe-name-without-traces"),
])
def test_every_entry_is_checked_even_when_no_episode_would_use_it(tmp_path, controllers, traces, error):
    with pytest.raises(error):
        run_benchmark_suite(controllers, traces, config=SHORT, output_dir=tmp_path / "out")
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def git_project(tmp_path, monkeypatch):
    """A clean one-commit repository that the runner treats as the project root."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    for name in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)  # ignore the developer's own hooks, signing and ignores
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    root = (tmp_path / "project").resolve()
    root.mkdir()
    (root / "code.py").write_text("x = 1\n")
    for command in (("init", "-q"), ("add", "-A"),
                    ("-c", "user.name=t", "-c", "user.email=t@example.invalid", "commit", "-q", "-m", "base")):
        subprocess.run(["git", *command], cwd=root, check=True, capture_output=True)
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", root)
    return root


def head_commit(root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def test_a_run_does_not_report_its_own_output_as_a_change(git_project):
    traces = [short_trace(rate=rate) for rate in (100.0, 200.0, 300.0)]
    manifests = run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=git_project / "manifests")
    expected = {"commit": head_commit(git_project), "dirty": False}
    assert [manifest["git"] for manifest in manifests] == [expected] * 3
    written = [json.loads(path.read_text())["git"] for path in (git_project / "manifests").iterdir()]
    assert written == [expected] * 3


def test_manifests_are_byte_identical_inside_and_outside_the_repository(git_project, tmp_path):
    traces = [short_trace(rate=rate) for rate in (100.0, 200.0)]
    run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=tmp_path / "outside")
    run_benchmark_suite([StaticBaseline()], traces, config=SHORT, output_dir=git_project / "inside")
    assert snapshot(tmp_path / "outside") == snapshot(git_project / "inside")


def test_every_manifest_of_a_run_shares_the_state_at_its_start(git_project):
    class EditsTrackedCode(ScriptedController):
        def reset(self):
            (git_project / "code.py").write_text("x = 2\n")

    manifests = run_benchmark_suite([EditsTrackedCode()], [short_trace(), short_trace(rate=200.0)], config=SHORT)
    assert [manifest["git"]["dirty"] for manifest in manifests] == [False, False]
    later = run_benchmark_suite([StaticBaseline()], [short_trace()], config=SHORT)
    assert later[0]["git"] == {"commit": head_commit(git_project), "dirty": True}


def test_changes_made_before_a_run_are_reported_as_dirty(git_project):
    (git_project / "untracked.py").write_text("y = 1\n")
    manifests = run_benchmark_suite([StaticBaseline()], [short_trace()], config=SHORT)
    assert manifests[0]["git"] == {"commit": head_commit(git_project), "dirty": True}


def test_a_parent_repository_is_never_attributed_to_the_project(git_project, monkeypatch):
    nested = git_project / "nested"
    nested.mkdir()
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", nested)
    assert run_episode(StaticBaseline(), short_trace(), config=SHORT)["git"] == {"commit": "unavailable", "dirty": None}


def test_missing_git_never_stops_a_run(monkeypatch):
    monkeypatch.setenv("PATH", "")
    assert run_episode(StaticBaseline(), short_trace(), config=SHORT)["git"] == {"commit": "unavailable", "dirty": None}


def test_a_repository_without_commits_has_no_commit_to_report(git_project, tmp_path, monkeypatch):
    empty = (tmp_path / "empty").resolve()
    empty.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=empty, check=True, capture_output=True)
    monkeypatch.setattr(runner_module, "PROJECT_ROOT", empty)
    # Here "git rev-parse HEAD" fails but still prints "HEAD".
    assert run_episode(StaticBaseline(), short_trace(), config=SHORT)["git"] == {"commit": "unavailable", "dirty": None}


def test_a_failing_status_leaves_the_dirty_flag_unknown(git_project):
    (git_project / ".git" / "index").write_bytes(b"not an index")
    assert run_episode(StaticBaseline(), short_trace(), config=SHORT)["git"] == {
        "commit": head_commit(git_project), "dirty": None,
    }


def test_git_information_is_not_shared_between_manifests(git_project):
    first, second = run_benchmark_suite([StaticBaseline()], [short_trace(), short_trace(rate=200.0)], config=SHORT)
    first["git"]["dirty"] = "changed by the caller"
    assert second["git"]["dirty"] is False


def test_latency_summary_is_ordered():
    summary = run_episode(SimHPABaseline(), workloads.flash_crowd())["metrics"]["substep_latency_proxy_ms"]
    assert summary["p50"] <= summary["p95"] <= summary["p99"] <= summary["max"]
    assert summary["mean"] <= summary["max"]


def test_manifest_schema_and_json_round_trip():
    trace = workloads.flash_crowd()
    manifest = run_episode(SimHPABaseline(), trace, seed=7)
    assert set(manifest) == {
        "manifest_schema_version", "spec_versions", "git", "seed", "config", "controller", "workload", "metrics",
    }
    assert manifest["manifest_schema_version"] == "1.0.0"
    assert manifest["spec_versions"] == {"simulator": "1.0.0", "baselines": "1.0.0"}
    assert set(manifest["git"]) == {"commit", "dirty"} and isinstance(manifest["git"]["commit"], str)
    assert manifest["git"]["dirty"] in (True, False, None)
    assert manifest["seed"] == 7
    assert manifest["config"] == json.loads(json.dumps(dataclasses.asdict(SimulatorConfig())))
    assert manifest["controller"] == {"name": "sim_hpa", "params": {"target_utilization": 0.7, "tolerance": 0.1}}
    assert manifest["workload"] == {
        "name": "flash_crowd",
        "params": {"base_rate": 100.0, "peak_rate": 900.0, "start_time": 300, "duration": 120, "horizon": 1800},
        "sha256": trace.sha256, "horizon": 1800, "min_rps": 100.0, "max_rps": 900.0,
        "mean_rps": pytest.approx((1680 * 100.0 + 120 * 900.0) / 1800),
    }
    assert set(manifest["metrics"]) == METRIC_KEYS
    assert json.loads(json.dumps(manifest)) == manifest


def test_repeated_runs_are_byte_identical(tmp_path):
    controllers = [StaticBaseline(), SimHPABaseline(), TunedReactiveThresholdController()]
    traces = [workloads.constant(horizon=100), workloads.seeded_bursty_poisson(horizon=100)]
    first = run_benchmark_suite(controllers, traces, config=SHORT, output_dir=tmp_path / "a")
    second = run_benchmark_suite(controllers, traces, config=SHORT, output_dir=tmp_path / "b")
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    names = sorted(path.name for path in (tmp_path / "a").iterdir())
    assert len(names) == 6 and names == sorted(path.name for path in (tmp_path / "b").iterdir())
    for name in names:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
        assert json.loads((tmp_path / "a" / name).read_text()) in first


def test_seed_is_recorded_and_physics_stays_deterministic():
    one = run_episode(TunedReactiveThresholdController(), workloads.constant(horizon=100), config=SHORT, seed=1)
    two = run_episode(TunedReactiveThresholdController(), workloads.constant(horizon=100), config=SHORT, seed=2)
    assert (one["seed"], two["seed"]) == (1, 2)
    assert one["metrics"] == two["metrics"]


def test_duplicate_output_names_are_rejected(tmp_path):
    with pytest.raises(ValueError):
        run_benchmark_suite(
            [StaticBaseline(), StaticBaseline()], [workloads.constant(horizon=100)], config=SHORT, output_dir=tmp_path,
        )


# ------------------------------------------------------------ 4. isolation
@pytest.mark.parametrize("module", STAGE_1B_MODULES)
def test_stage_1b_modules_import_nothing_forbidden(module):
    source = (ROOT / module).read_text()
    roots = {name.split(".")[0] for name in imported_modules(ROOT / module)}
    assert not FORBIDDEN & roots
    assert "importlib" not in roots and "__import__" not in source


def test_clean_interpreter_loads_no_forbidden_package():
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]); import sim.workloads, sim.baselines, sim.runner; "
        "print(sorted({name.split('.')[0] for name in sys.modules} & set(sys.argv[2:])))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(ROOT), *sorted(FORBIDDEN)], capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
