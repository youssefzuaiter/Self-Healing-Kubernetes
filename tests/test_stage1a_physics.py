"""Stage 1A acceptance tests. Item numbers refer to SPEC_01 Section 34.

Expected values are derived by hand from the spec, not from the code under test.
A row index in ``run()`` output equals the simulation time when run from reset.
"""
from __future__ import annotations

import copy
import dataclasses
import functools
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sim.shield as shield_module  # noqa: E402
from sim.config import SimulatorConfig  # noqa: E402
from sim.env import (  # noqa: E402
    MicroserviceMeshEnv,
    SimulationInvariantError,
    check_conservation,
    contention_multiplier,
    end_to_end_failure,
    interval_utilization,
)
from sim.shield import ApprovedAction, convert_proposal, round_component  # noqa: E402

GATEWAY, AUTH, CATALOG, ORDER, DATABASE = range(5)
SCALABLE = [GATEWAY, AUTH, CATALOG, ORDER]
MU = (100.0, 120.0, 80.0, 50.0, 300.0)
HOLD = (0, 0, 0, 0)
OVERLOAD = 5000.0


# ------------------------------------------------------------------ helpers
def make_env(trace=None, **overrides) -> MicroserviceMeshEnv:
    env = MicroserviceMeshEnv(trace, dataclasses.replace(SimulatorConfig(), **overrides))
    env.reset(seed=0)
    return env


def trace_of(*head: float) -> np.ndarray:
    trace = np.zeros(1800)
    trace[: len(head)] = head
    return trace


def constant(rate: float) -> np.ndarray:
    return np.full(1800, rate)


def act(env, raw=HOLD):
    return env.step(env.shield.approve(raw))


def run(env, steps: int, schedule=None) -> dict:
    """Run control steps and stack their per-substep rows."""
    schedule = schedule or {}
    infos = [act(env, schedule.get(k, HOLD))[4]["substeps"] for k in range(steps)]
    return {key: np.concatenate([info[key] for info in infos]) for key in infos[0]}


def delta(node: int, value: int) -> tuple:
    proposal = [0, 0, 0, 0]
    proposal[node] = value
    return tuple(proposal)


def without_replicas(node: int) -> dict:
    minimum, initial = [1, 1, 1, 1, 1], [2, 2, 2, 2, 1]
    minimum[node], initial[node] = 0, 0
    return {"min_replicas": tuple(minimum), "initial_replicas": tuple(initial)}


def fill_cluster(env) -> None:
    """Three all-+2 proposals reach the 28-pod quota: 9 -> 17 -> 25 -> 28."""
    for _ in range(3):
        act(env, (2, 2, 2, 2))


def digest(env) -> dict:
    return {
        "time": env.time, "step": env.step_index, "pointer": env.trace_pointer,
        "queue": env.queue, "ready": env.ready, "pending": env.pending,
        "terminating": env.terminating, "desired": env.desired, "cooldown": env.cooldown,
        "boot": env.boot_register, "term": env.termination_register,
        "downstream": env.downstream_buffer, "retry": env.retry_buffer,
        "fallback": env.shield.fallback_count, "features": env._features.copy(),
    }


def ledger(env) -> dict:
    """The environment-owned token ledger: nonce -> (step index, deltas, consumed)."""
    return {nonce: (record.step_index, record.deltas, record.consumed) for nonce, record in env._tokens.items()}


def assert_same_state(before: dict, after: dict) -> None:
    for key, value in before.items():
        np.testing.assert_array_equal(value, after[key], err_msg=key)


def _scenarios() -> dict:
    time = np.arange(1800.0)
    spike = constant(100.0)
    spike[300:360] = 4000.0
    return {
        "default": None, "idle": np.zeros(1800), "overload": constant(OVERLOAD), "spike": spike,
        "ramp": np.linspace(0.0, 3000.0, 1800), "sine": 800.0 + 700.0 * np.sin(time / 60.0),
    }


SCENARIOS = _scenarios()


class Episode(NamedTuple):
    observations: list
    rewards: list
    rows: dict


@functools.lru_cache(maxsize=None)
def episode(name: str) -> Episode:
    """A full 360-step episode driven by seeded random proposals."""
    env = MicroserviceMeshEnv(SCENARIOS[name])
    observation, _ = env.reset(seed=0)
    observations, rewards, infos = [observation], [], []
    for proposal in np.random.default_rng(7).integers(-1, 3, size=(360, 4)):
        observation, reward, _, _, info = act(env, proposal)
        observations.append(observation)
        rewards.append(reward)
        infos.append(info["substeps"])
    rows = {key: np.concatenate([info[key] for info in infos]) for key in infos[0]}
    return Episode(observations, rewards, rows)


every_scenario = pytest.mark.parametrize("name", list(SCENARIOS))


# ------------------------------------------------- physics: items 1 to 14
@every_scenario
def test_flow_conservation_holds_for_every_node_and_substep(name):
    rows = episode(name).rows
    residual = (rows["queue_start"] + rows["arrivals_new"] + rows["arrivals_retry"]
                - rows["completed"] - rows["overflow"] - rows["timeout"] - rows["queue_end"])
    assert residual.shape == (1800, 5)
    assert np.abs(residual).max() <= 1e-9
    np.testing.assert_array_equal(rows["queue_start"][1:], rows["queue_end"][:-1])


def test_overload_scenario_exercises_every_loss_path():
    rows = episode("overload").rows
    assert rows["overflow"].sum() > 0 and rows["timeout"].sum() > 0
    assert rows["arrivals_retry"].sum() > 0


@pytest.mark.parametrize("error, raises", [
    (0.0, False), (0.9e-9, False), (-0.9e-9, False), (1.1e-9, True), (-1.1e-9, True), (1.0, True),
])
def test_conservation_diagnostic_raises_only_beyond_tolerance(error, raises):
    # 10 + 5 + 1 - 6 - 2 - 3 = 5 is the balanced next queue.
    args = [np.full(5, value) for value in (10.0, 5.0, 1.0, 6.0, 2.0, 3.0, 5.0 + error)]
    if raises:
        with pytest.raises(SimulationInvariantError, match="conservation"):
            check_conservation(*args, tolerance=1e-9)
    else:
        check_conservation(*args, tolerance=1e-9)


@pytest.mark.parametrize("arrival, accepted, overflow", [
    (1999.0, 1999.0, 0.0), (2000.0, 2000.0, 0.0), (2001.0, 2000.0, 1.0), (5000.0, 2000.0, 3000.0),
])
def test_admission_boundary_at_queue_capacity(arrival, accepted, overflow):
    rows = run(make_env(trace_of(arrival)), 1)
    assert rows["accepted"][0, GATEWAY] == accepted
    assert rows["overflow"][0, GATEWAY] == overflow


def test_same_substep_service_does_not_create_admission_room():
    rows = run(make_env(constant(OVERLOAD)), 1)
    assert rows["queue_end"][0, GATEWAY] == 1800.0
    assert rows["accepted"][1, GATEWAY] == 200.0  # 2000 - 1800, not 400
    assert rows["overflow"][1, GATEWAY] == 4800.0


@every_scenario
def test_queues_stay_within_capacity_and_non_negative(name):
    rows = episode(name).rows
    assert rows["queue_end"].min() >= 0.0
    assert np.all(rows["queue_end"] <= np.array([2000, 2000, 1500, 1000, 3000]) + 1e-9)


@pytest.mark.parametrize("t, age", [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])
def test_fifo_serves_the_oldest_bucket_first(t, age):
    rows = run(make_env(constant(OVERLOAD)), 1)
    expected = np.zeros(5)
    expected[age] = 200.0
    np.testing.assert_array_equal(rows["served_by_age"][t, GATEWAY], expected)


def test_fifo_spills_remaining_capacity_into_the_next_bucket():
    rows = run(make_env(trace_of(500.0, 100.0)), 1)
    np.testing.assert_array_equal(rows["served_by_age"][1, GATEWAY], [0, 200, 0, 0, 0])
    np.testing.assert_array_equal(rows["served_by_age"][2, GATEWAY], [0, 100, 100, 0, 0])


# One downstream node at a time is made slow and small under 100 req/s of Gateway
# traffic. It then receives 100 (Auth), 80 (Catalog), 20 (Order) or 90 (Database)
# per substep. node: (service rate per pod, queue capacity, first arrival substep)
SLOW_NODE = {AUTH: (10.0, 50.0, 1), CATALOG: (10.0, 50.0, 1), ORDER: (2.0, 15.0, 1), DATABASE: (20.0, 50.0, 2)}


def run_with_slow_node(node: int) -> tuple:
    rate, size, start = SLOW_NODE[node]
    rates, sizes = list(SimulatorConfig().service_rate), list(SimulatorConfig().queue_capacity)
    rates[node], sizes[node] = rate, size
    rows = run(make_env(constant(100.0), service_rate=tuple(rates), queue_capacity=tuple(sizes)), 1)
    return rows, start


@pytest.mark.parametrize("node, accepted, overflow", [
    pytest.param(AUTH, [50, 20], [50, 80], id="Auth"),
    pytest.param(CATALOG, [50, 20], [30, 60], id="Catalog"),
    pytest.param(ORDER, [15, 4], [5, 16], id="Order"),
    pytest.param(DATABASE, [50, 20], [40, 70], id="Database"),
])
def test_every_downstream_node_admits_only_free_space_and_counts_overflow(node, accepted, overflow):
    # First arrival: free space is the full queue. Next substep: only what is left
    # after the first substep's service, not what this substep will serve.
    rows, start = run_with_slow_node(node)
    assert rows["accepted"][:start, node].sum() == 0
    np.testing.assert_allclose(rows["accepted"][start:start + 2, node], accepted, atol=1e-9)
    np.testing.assert_allclose(rows["overflow"][start:start + 2, node], overflow, atol=1e-9)


@pytest.mark.parametrize("node, served", [
    pytest.param(AUTH, [[20, 0, 0, 0, 0], [0, 20, 0, 0, 0], [0, 10, 10, 0, 0]], id="Auth"),
    pytest.param(CATALOG, [[20, 0, 0, 0, 0], [0, 20, 0, 0, 0], [0, 10, 10, 0, 0]], id="Catalog"),
    pytest.param(ORDER, [[4, 0, 0, 0, 0], [0, 4, 0, 0, 0], [0, 0, 4, 0, 0]], id="Order"),
    pytest.param(DATABASE, [[20, 0, 0, 0, 0], [0, 20, 0, 0, 0], [0, 10, 10, 0, 0]], id="Database"),
])
def test_every_downstream_node_serves_its_oldest_work_first(node, served):
    rows, start = run_with_slow_node(node)
    np.testing.assert_allclose(rows["served_by_age"][start:start + 3, node], served, atol=1e-9)


def test_completions_propagate_by_multiplicity_one_substep_per_hop():
    rows = run(make_env(trace_of(100.0)), 1)
    np.testing.assert_allclose(rows["arrivals_new"][:4], [
        [100, 0, 0, 0, 0],
        [0, 100, 80, 20, 0],
        [0, 0, 0, 0, 90],
        [0, 0, 0, 0, 0],
    ], atol=1e-12)


def test_fractional_quantities_propagate_without_rounding():
    rows = run(make_env(trace_of(1.0)), 1)
    np.testing.assert_allclose(rows["arrivals_new"][1], [0, 1.0, 0.8, 0.2, 0], atol=1e-12)
    assert rows["arrivals_new"][2, DATABASE] == pytest.approx(0.9)


@pytest.mark.parametrize("silent_node, database_arrivals", [(AUTH, 60.0), (CATALOG, 50.0), (ORDER, 70.0)])
def test_each_database_edge_has_its_own_multiplicity(silent_node, database_arrivals):
    rows = run(make_env(trace_of(100.0), **without_replicas(silent_node)), 1)
    assert rows["arrivals_new"][2, DATABASE] == pytest.approx(database_arrivals)


def test_children_receive_parent_completions_not_parent_arrivals():
    rows = run(make_env(constant(OVERLOAD)), 1)
    assert rows["arrivals_new"][1, AUTH] == 200.0


def test_request_expires_at_the_end_of_its_fifth_second():
    rows = run(make_env(trace_of(100.0), **without_replicas(GATEWAY)), 2)
    np.testing.assert_array_equal(rows["timeout"][:, GATEWAY], [0, 0, 0, 0, 100, 0, 0, 0, 0, 50])
    np.testing.assert_array_equal(rows["queue_end"][:, GATEWAY], [100, 100, 100, 100, 0, 50, 50, 50, 50, 0])


def test_work_served_from_bucket_four_does_not_time_out():
    rows = run(make_env(constant(OVERLOAD)), 1)
    np.testing.assert_array_equal(rows["timeout"][:, GATEWAY], [0, 0, 0, 0, 1000.0])  # 1200 - 200


@pytest.mark.parametrize("node, window, expired", [
    # Scalable nodes have no ready pods: a one-shot arrival at t=1 expires at t=window,
    # and half of it returns as a retry that expires one window later.
    pytest.param(AUTH, 5, {5: 100.0}, id="Auth-W5"),
    pytest.param(AUTH, 3, {3: 100.0, 6: 50.0, 9: 25.0}, id="Auth-W3"),
    pytest.param(CATALOG, 5, {5: 80.0}, id="Catalog-W5"),
    pytest.param(CATALOG, 3, {3: 80.0, 6: 40.0, 9: 20.0}, id="Catalog-W3"),
    pytest.param(ORDER, 5, {5: 20.0}, id="Order-W5"),
    pytest.param(ORDER, 3, {3: 20.0, 6: 10.0, 9: 5.0}, id="Order-W3"),
    # The database serves 1 req/s: 90 arrive at t=2, and 1 is served each second.
    pytest.param(DATABASE, 5, {6: 85.0}, id="Database-W5"),
    pytest.param(DATABASE, 3, {4: 87.0, 7: 40.5}, id="Database-W3"),
])
def test_every_downstream_node_expires_work_after_its_own_timeout(node, window, expired):
    timeouts = [5, 5, 5, 5, 5]
    timeouts[node] = window
    overrides = {"service_rate": (100, 120, 80, 50, 1)} if node == DATABASE else without_replicas(node)
    rows = run(make_env(trace_of(100.0), timeout_seconds=tuple(timeouts), **overrides), 2)
    expected = np.zeros(10)
    for t, amount in expired.items():
        expected[t] = amount
    np.testing.assert_allclose(rows["timeout"][:, node], expected, atol=1e-9)


def test_retries_arrive_one_substep_after_timeouts_at_half_volume():
    rows = run(make_env(trace_of(100.0), **without_replicas(GATEWAY)), 3)
    expected = np.zeros(15)
    expected[5], expected[10] = 50.0, 25.0
    np.testing.assert_array_equal(rows["arrivals_retry"][:, GATEWAY], expected)


@every_scenario
def test_retries_return_to_the_same_service_as_half_of_timeouts(name):
    rows = episode(name).rows
    np.testing.assert_array_equal(rows["arrivals_retry"][1:], 0.5 * rows["timeout"][:-1])
    np.testing.assert_array_equal(rows["arrivals_retry"][0], np.zeros(5))


def test_retries_are_subject_to_normal_admission():
    overrides = {**without_replicas(GATEWAY), "queue_capacity": (100, 2000, 1500, 1000, 3000)}
    rows = run(make_env(constant(100.0), **overrides), 2)
    assert rows["arrivals_retry"][5, GATEWAY] == 50.0
    assert rows["accepted"][5, GATEWAY] == 100.0
    assert rows["overflow"][5, GATEWAY] == 50.0


def test_overflow_is_never_retried():
    rows = run(make_env(constant(OVERLOAD)), 2)
    assert rows["overflow"][:5, GATEWAY].sum() == 22200.0  # 3000 + 4 * 4800
    np.testing.assert_array_equal(rows["arrivals_retry"][:5, GATEWAY], np.zeros(5))
    assert rows["arrivals_retry"][5, GATEWAY] == 500.0  # 0.5 * 1000 timed out at t=4


def test_interval_failure_ratio_is_volume_weighted():
    env = make_env(trace_of(1000, 100, 100, 100, 100), queue_capacity=(100, 2000, 1500, 1000, 3000))
    observation, _, _, _, info = act(env)
    np.testing.assert_allclose(info["substeps"]["failure_ratio"][:, GATEWAY], [0.9, 0, 0, 0, 0])
    assert observation["node_features"][GATEWAY, 7] == pytest.approx(900.0 / 1400.0)  # not 0.18
    assert info["interval_failure_ratio"][GATEWAY] == pytest.approx(900.0 / 1400.0)


def test_failure_ratio_is_zero_without_resolved_work():
    observation, _, _, _, info = act(make_env(np.zeros(1800)))
    np.testing.assert_array_equal(info["substeps"]["failure_ratio"], np.zeros((5, 5)))
    np.testing.assert_array_equal(observation["node_features"][:, 7], np.zeros(5))


def test_fractional_traffic_failure_ratio_uses_the_floored_denominator():
    # completed 0.2 and failed 0.3: 0.3 / max(1.0, 0.5) = 0.3, not 0.3 / 0.5 = 0.6.
    env = make_env(trace_of(0.5), queue_capacity=(0.2, 2000, 1500, 1000, 3000))
    observation, _, _, _, info = act(env)
    rows = info["substeps"]
    assert rows["completed"][0, GATEWAY] == pytest.approx(0.2)
    assert rows["overflow"][0, GATEWAY] == pytest.approx(0.3)
    assert rows["failure_ratio"][0, GATEWAY] == pytest.approx(0.3)
    assert info["interval_failure_ratio"][GATEWAY] == pytest.approx(0.3)
    assert observation["node_features"][GATEWAY, 7] == pytest.approx(0.3)


def test_failure_ratio_when_resolved_work_exceeds_one():
    # completed 1.0 and failed 0.3: 0.3 / max(1.0, 1.3) = 0.3 / 1.3, about 0.230769.
    env = make_env(trace_of(1.3), queue_capacity=(1.0, 2000, 1500, 1000, 3000))
    observation, _, _, _, info = act(env)
    rows = info["substeps"]
    assert rows["completed"][0, GATEWAY] == pytest.approx(1.0)
    assert rows["overflow"][0, GATEWAY] == pytest.approx(0.3)
    assert rows["failure_ratio"][0, GATEWAY] == pytest.approx(0.230769, abs=1e-6)
    assert info["interval_failure_ratio"][GATEWAY] == pytest.approx(0.230769, abs=1e-6)
    assert observation["node_features"][GATEWAY, 7] == pytest.approx(0.230769, abs=1e-6)


@pytest.mark.parametrize("arrival, capacity, expected", [
    (0.5, 0.5, 0.0),    # nothing failed
    (0.5, 0.2, 0.3),    # resolved 0.5: the floor applies
    (1.0, 0.4, 0.6),    # resolved exactly 1.0: floored and plain ratios agree
    (2.0, 0.5, 0.75),   # resolved 2.0: the floor is inactive
    (1.3, 1.0, 0.3 / 1.3),  # completed 1.0, failed 0.3: about 0.230769
])
def test_substep_failure_ratio_floor_boundary(arrival, capacity, expected):
    env = make_env(trace_of(arrival), queue_capacity=(capacity, 2000, 1500, 1000, 3000))
    rows = run(env, 1)
    assert rows["failure_ratio"][0, GATEWAY] == pytest.approx(expected)


@pytest.mark.parametrize("head, interval_ratio, mean_of_substeps", [
    pytest.param((0.3, 0.3, 0.3, 0, 0), 0.6, 0.12, id="sum-below-one"),
    pytest.param((0.6, 0.6, 0.6, 0.6, 0.6), 2.5 / 3.0, 0.5, id="each-below-one-sum-above"),
    pytest.param((10.1, 0.1, 0.1, 0.1, 0.1), 10.0 / 10.5, (10.0 / 10.1) / 5, id="one-large-substep"),
])
def test_interval_ratio_is_summed_failed_over_floored_summed_resolved(head, interval_ratio, mean_of_substeps):
    # A 0.1 queue and ample capacity: each substep completes 0.1 and overflows the rest.
    env = make_env(trace_of(*head), queue_capacity=(0.1, 2000, 1500, 1000, 3000))
    observation, _, _, _, info = act(env)
    assert info["substeps"]["failure_ratio"][:, GATEWAY].mean() == pytest.approx(mean_of_substeps)
    assert info["interval_failure_ratio"][GATEWAY] == pytest.approx(interval_ratio)
    assert observation["node_features"][GATEWAY, 7] == pytest.approx(interval_ratio)


def test_floored_failure_ratio_feeds_the_end_to_end_proxy_and_reward():
    env = make_env(trace_of(0.5), queue_capacity=(0.2, 2000, 1500, 1000, 3000))
    _, reward, _, _, info = act(env)
    np.testing.assert_allclose(info["substeps"]["e2e_failure"], [0.3, 0, 0, 0, 0], atol=1e-12)
    assert reward == pytest.approx(-(5.0 * 0.06 + 0.5 * 9 / 28))


def test_served_age_latency_does_not_use_the_failure_ratio_floor():
    # K = 0.2. At t=1 all 0.2 served is one second old: 1000 ms, not 1000 * 0.2 / max(1, 0.2).
    rows = run(make_env(trace_of(0.5), service_rate=(0.1, 120, 80, 50, 300)), 1)
    assert rows["served_by_age"][1, GATEWAY, 1] == pytest.approx(0.2)
    assert rows["node_latency_ms"][1, GATEWAY] == pytest.approx(1005.0)


@every_scenario
def test_failure_ratios_and_proxy_stay_within_unit_interval(name):
    rows = episode(name).rows
    assert 0.0 <= rows["failure_ratio"].min() and rows["failure_ratio"].max() <= 1.0
    assert 0.0 <= rows["e2e_failure"].min() and rows["e2e_failure"].max() <= 1.0


@pytest.mark.parametrize("success, expected", [
    ((1, 1, 1, 1, 1), 0.0),
    ((0, 1, 1, 1, 1), 1.0),
    ((1, 0, 1, 1, 1), 1.0),
    ((1, 1, 0, 1, 1), 0.8),   # only the 20% Order branch survives
    ((1, 1, 1, 0, 1), 0.2),   # only the 80% Catalog branch survives
    ((1, 1, 1, 1, 0), 1.0),
    ((0.5, 1, 1, 1, 1), 0.5),
    ((1, 1, 1, 1, 0.5), 1 - 0.5 ** 0.3 * (0.8 * 0.5 ** 0.5 + 0.2 * 0.5)),
])
def test_end_to_end_failure_proxy(success, expected):
    assert end_to_end_failure(np.array(success, dtype=float)) == pytest.approx(expected)


@pytest.mark.parametrize("pressure", [0.0, 0.25, 0.49, 0.5])
def test_no_contention_at_or_below_the_threshold(pressure):
    assert contention_multiplier(np.array([2.5]), pressure * 3000.0, 3000.0, 0.5)[0] == 1.0


def test_contention_begins_just_above_the_threshold():
    assert contention_multiplier(np.array([2.5]), 1503.0, 3000.0, 0.5)[0] < 1.0


@pytest.mark.parametrize("kappa, pressure, expected", [
    (1.0, 1.0, 1 / 1.5), (1.5, 1.0, 1 / 1.75), (2.5, 1.0, 1 / 2.25),
    (2.5, 0.75, 1 / 1.625), (1.0, 0.6, 1 / 1.1), (0.0, 1.0, 1.0),
])
def test_contention_coefficients(kappa, pressure, expected):
    result = contention_multiplier(np.array([kappa]), pressure * 3000.0, 3000.0, 0.5)[0]
    assert result == pytest.approx(expected)


def test_capacity_uses_each_kappa_and_the_start_of_substep_database_queue():
    env = make_env(service_rate=(100, 120, 80, 50, 1), queue_capacity=(2000, 2000, 1500, 1000, 100))
    rows = run(env, 1)
    # 90 calls reach the database during t=2 but must not affect capacity until t=3.
    np.testing.assert_array_equal(rows["capacity"][2], [200, 240, 160, 100, 1])
    assert rows["queue_start"][3, DATABASE] == 89.0
    np.testing.assert_allclose(rows["capacity"][3], [200, 240 / 1.39, 160 / 1.585, 100 / 1.975, 1])


@pytest.mark.parametrize("node", SCALABLE)
def test_zero_ready_replicas_give_zero_capacity_and_no_completions(node):
    rows = run(make_env(constant(50.0), **without_replicas(node)), 2)
    assert rows["arrivals_new"][:, node].sum() > 0
    np.testing.assert_array_equal(rows["capacity"][:, node], np.zeros(10))
    np.testing.assert_array_equal(rows["completed"][:, node], np.zeros(10))


@pytest.mark.parametrize("completed, capacity, expected", [
    (50.0, 100.0, 0.5), (100.0, 100.0, 1.0), (0.0, 100.0, 0.0), (0.0, 0.0, 0.0), (5.0, 0.0, 1.0),
])
def test_interval_utilization(completed, capacity, expected):
    assert interval_utilization(np.array([completed]), np.array([capacity]))[0] == expected


@pytest.mark.parametrize("t, expected_ms", [(0, 1505.0), (1, 1005.0), (2, 1505.0), (3, 5.0)])
def test_gateway_latency_in_milliseconds(t, expected_ms):
    # t=0 backlog 300/200 s; t=1 backlog and age both 1 s; t=2 mean age 1.5 s; t=3 idle.
    rows = run(make_env(trace_of(500.0, 100.0)), 1)
    assert rows["node_latency_ms"][t, GATEWAY] == pytest.approx(expected_ms)


def test_backlog_latency_is_capped_at_the_timeout_window():
    rows = run(make_env(constant(OVERLOAD)), 1)
    assert rows["node_latency_ms"][2, GATEWAY] == pytest.approx(5005.0)  # 1800/200 = 9 s, capped at 5


def test_backlog_latency_divides_by_at_least_one_when_capacity_is_zero():
    rows = run(make_env(trace_of(0.5), **without_replicas(GATEWAY)), 1)
    assert rows["node_latency_ms"][0, GATEWAY] == pytest.approx(505.0)


def test_idle_latency_is_base_latency_along_the_topology():
    rows = run(make_env(), 1)
    np.testing.assert_allclose(rows["node_latency_ms"][4], [5, 8, 15, 25, 10])
    assert rows["e2e_latency_ms"][4] == pytest.approx(39.0)  # 5+8+3+0.8(20)+0.2(35)


def test_end_to_end_latency_includes_gateway_delay():
    rows = run(make_env(trace_of(500.0, 100.0)), 1)
    np.testing.assert_allclose(rows["e2e_latency_ms"], [1539, 1039, 1539, 39, 39])


# ----------------------------------------- replica lifecycle: items 15 to 21
@pytest.mark.parametrize("node", SCALABLE)
@pytest.mark.parametrize("k", [0, 1, 7])
def test_pod_is_pending_through_t19_and_ready_at_t20(node, k):
    rows = run(make_env(), k + 6, {k: delta(node, 1)})
    t = 5 * k
    np.testing.assert_array_equal(rows["pending"][t:t + 20, node], np.ones(20))
    np.testing.assert_array_equal(rows["ready"][t:t + 20, node], np.full(20, 2))
    assert rows["pending"][t + 20, node] == 0
    assert rows["ready"][t + 20, node] == 3
    assert rows["capacity"][t + 19, node] == 2 * MU[node]
    assert rows["capacity"][t + 20, node] == 3 * MU[node]
    np.testing.assert_array_equal(rows["occupancy"][t:t + 21], np.full(21, 10))


@pytest.mark.parametrize("t", [0, 3, 13])
def test_boot_register_walks_left_from_slot_19_at_any_substep(t):
    env = make_env(substeps_per_control_step=1, control_horizon=1800)
    assert env.boot_register.shape == (5, 20)
    for _ in range(t):
        act(env)
    act(env, delta(GATEWAY, 1))
    for elapsed in range(1, 20):
        assert env.boot_register[GATEWAY, 19 - elapsed] == 1
        assert env.boot_register[GATEWAY].sum() == 1
        assert env.ready[GATEWAY] == 2
        act(env)
    assert env.boot_register[GATEWAY].sum() == 0
    assert env.ready[GATEWAY] == 3
    assert env.time == t + 20


@pytest.mark.parametrize("node", SCALABLE)
@pytest.mark.parametrize("k", [0, 1, 7])
def test_terminating_pod_uses_quota_through_t4_and_exits_at_t5(node, k):
    rows = run(make_env(), k + 2, {k: delta(node, -1)})
    t = 5 * k
    np.testing.assert_array_equal(rows["terminating"][t:t + 5, node], np.ones(5))
    np.testing.assert_array_equal(rows["occupancy"][t:t + 5], np.full(5, 9))
    assert rows["terminating"][t + 5, node] == 0
    assert rows["occupancy"][t + 5] == 8
    assert rows["capacity"][t, node] == MU[node]  # a draining pod serves nothing
    if k:
        assert rows["capacity"][t - 1, node] == 2 * MU[node]


@pytest.mark.parametrize("t", [0, 3, 13])
def test_termination_register_walks_left_from_slot_4_at_any_substep(t):
    env = make_env(substeps_per_control_step=1, control_horizon=1800)
    assert env.termination_register.shape == (5, 5)
    for _ in range(t):
        act(env)
    act(env, delta(GATEWAY, -1))
    for elapsed in range(1, 5):
        assert env.termination_register[GATEWAY, 4 - elapsed] == 1
        assert env.cluster_occupancy == 9
        act(env)
    assert env.termination_register[GATEWAY].sum() == 0
    assert env.cluster_occupancy == 8
    assert env.time == t + 5


def test_cancellation_removes_the_newest_pending_pod_first():
    env = make_env(cooldown_control_steps=0)
    act(env, (1, 0, 0, 0))   # t=0: the older pod
    act(env, (2, 0, 0, 0))   # t=5: two newer pods
    act(env, (-1, 0, 0, 0))  # t=10: must cancel a t=5 pod
    expected = np.zeros(20, dtype=int)
    expected[4], expected[9] = 1, 1
    np.testing.assert_array_equal(env.boot_register[GATEWAY], expected)
    assert (env.ready[GATEWAY], env.terminating[GATEWAY], env.desired[GATEWAY]) == (2, 0, 4)
    rows = run(env, 3)  # row i is t = 15 + i
    assert rows["ready"][4, GATEWAY] == 2   # t=19
    assert rows["ready"][5, GATEWAY] == 3   # t=20: the older pod
    assert rows["ready"][9, GATEWAY] == 3   # t=24
    assert rows["ready"][10, GATEWAY] == 4  # t=25: the surviving newer pod


def test_scale_down_cancels_pending_before_terminating_ready_pods():
    env = make_env(cooldown_control_steps=0)
    act(env, (1, 0, 0, 0))
    rows = run(env, 1, {0: (-1, 0, 0, 0)})
    np.testing.assert_array_equal(rows["ready"][:, GATEWAY], np.full(5, 2))
    np.testing.assert_array_equal(rows["terminating"][:, GATEWAY], np.zeros(5))
    assert env.desired[GATEWAY] == 2
    rows = run(env, 1, {0: (-1, 0, 0, 0)})  # nothing left to cancel
    np.testing.assert_array_equal(rows["ready"][:, GATEWAY], np.ones(5))
    np.testing.assert_array_equal(rows["terminating"][:, GATEWAY], np.ones(5))


def test_scale_down_beyond_pending_cancels_then_terminates_the_remainder():
    env = make_env(cooldown_control_steps=0, action_min=-2)
    act(env, (1, 0, 0, 0))
    rows = run(env, 1, {0: (-2, 0, 0, 0)})
    assert (rows["pending"][0, GATEWAY], rows["ready"][0, GATEWAY]) == (0, 1)
    assert rows["terminating"][0, GATEWAY] == 1
    assert env.desired[GATEWAY] == 1


def test_terminating_pods_are_never_cancelled_or_restored():
    env = make_env(cooldown_control_steps=0, substeps_per_control_step=1, control_horizon=1800)
    act(env, (-1, 0, 0, 0))
    act(env, (1, 0, 0, 0))
    act(env, (-1, 0, 0, 0))  # cancels the new pod, leaves the draining one alone
    assert (env.ready[GATEWAY], env.pending[GATEWAY], env.terminating[GATEWAY]) == (1, 0, 1)


def test_cancellation_leaves_occupancy_from_the_first_substep():
    env = make_env(cooldown_control_steps=0)
    act(env, (1, 0, 0, 0))
    assert env.cluster_occupancy == 10
    rows = run(env, 1, {0: (-1, 0, 0, 0)})
    np.testing.assert_array_equal(rows["occupancy"], np.full(5, 9))


def test_cancellation_releases_quota_for_a_same_step_scale_up():
    env = make_env(cooldown_control_steps=0)
    fill_cluster(env)
    assert env.cluster_occupancy == 28
    info = act(env, (1, 0, 0, -1))[4]
    np.testing.assert_array_equal(info["approved_delta"], [1, 0, 0, -1])
    assert env.cluster_occupancy == 28


def test_terminating_pods_hold_quota_until_they_exit():
    env = make_env()
    fill_cluster(env)
    run(env, 5)
    assert env.cooldown[ORDER] == 0 and env.cooldown[GATEWAY] == 1
    info = act(env, (1, 0, 0, -1))[4]  # t=40: Order drains, but its pod still counts
    np.testing.assert_array_equal(info["approved_delta"], [0, 0, 0, -1])
    np.testing.assert_array_equal(info["substeps"]["occupancy"], np.full(5, 28))
    info = act(env, (1, 0, 0, 0))[4]   # t=45: the pod has exited
    np.testing.assert_array_equal(info["approved_delta"], [1, 0, 0, 0])


# --------------------------------------------- safety shield: items 22 to 36
MALFORMED = [
    pytest.param(np.zeros(3), id="short"), pytest.param(np.zeros(5), id="long"),
    pytest.param(np.zeros((4, 1)), id="column"), pytest.param(np.zeros((1, 4)), id="row"),
    pytest.param(np.zeros((2, 2)), id="square"), pytest.param(1.0, id="scalar"),
    pytest.param([], id="empty"), pytest.param(None, id="none"),
    pytest.param([[1, 2], [3]], id="ragged"),
]
NON_NUMERIC = [
    pytest.param(["a", "b", "c", "d"], id="strings"), pytest.param([1, "2", 0, 0], id="one-string"),
    pytest.param([1, None, 0, 0], id="one-none"), pytest.param({"a": 1}, id="dict"),
    pytest.param(np.array([True, False, True, False]), id="bool-array"),
    pytest.param([True, 0, 0, 0], id="one-bool"), pytest.param(np.ones(4, dtype=complex), id="complex"),
    pytest.param(np.array([1, 2, 0, 0], dtype=object), id="object"), pytest.param(object(), id="object-instance"),
]
NON_FINITE = [
    pytest.param(tuple(bad if i == position else 2.0 for i in range(4)), id=f"{bad}-at-{position}")
    for bad in (np.nan, np.inf, -np.inf) for position in range(4)
]


@pytest.mark.parametrize("raw", MALFORMED + NON_NUMERIC + NON_FINITE)
def test_invalid_proposal_is_rejected_whole_and_counted_once(raw):
    env = make_env()
    token = env.shield.approve(raw)
    assert token.deltas == HOLD
    assert env.shield.fallback_count == 1
    np.testing.assert_array_equal(env.step(token)[4]["approved_delta"], np.zeros(4))
    np.testing.assert_array_equal(env.desired, [2, 2, 2, 2, 1])


@pytest.mark.parametrize("raw", [(1, 0, -1, 2), [0.0, 0.0, 0.0, 0.0], np.array([1, 2, 0, 0]),
                                 np.array([0.5, 1.5, 0, 0], dtype=np.float32)])
def test_valid_proposal_is_not_a_fallback(raw):
    env = make_env()
    env.shield.approve(raw)
    assert env.shield.fallback_count == 0


@pytest.mark.parametrize("raw, rounded", [
    (0.5, 0), (1.5, 2), (2.5, 2), (-0.5, 0), (3.0, 3), (-1.5, -2), (-2.5, -2), (0.49, 0), (0.51, 1),
])
def test_half_to_even_rounding_before_clipping(raw, rounded):
    result = round_component(raw)
    assert result == rounded
    assert type(result) is int


@pytest.mark.parametrize("raw, clipped", [
    (-1e9, -1), (-2.0, -1), (-1.0, -1), (-0.5, 0), (0.0, 0), (0.5, 0),
    (1.0, 1), (1.5, 2), (2.0, 2), (2.5, 2), (3.0, 2), (1e9, 2),
])
def test_conversion_clips_to_the_allowed_action_set(raw, clipped):
    assert convert_proposal(np.full(4, raw), SimulatorConfig()) == (clipped,) * 4


def test_shield_applies_rounding_per_component():
    assert make_env().shield.approve(np.array([0.5, 1.5, 2.5, -0.5])).deltas == (0, 2, 2, 0)


def test_scale_down_stops_at_minimum_replicas():
    env = make_env()
    act(env, (-1, 0, 0, 0))
    info = act(env, (-1, 0, 0, 0))[4]
    np.testing.assert_array_equal(info["approved_delta"], np.zeros(4))
    assert (env.desired[GATEWAY], env.ready[GATEWAY]) == (1, 1)


def test_scale_up_is_truncated_at_maximum_replicas():
    env = make_env(max_replicas=(3, 10, 10, 10, 1))
    assert act(env, (2, 0, 0, 0))[4]["approved_delta"][GATEWAY] == 1
    assert act(env, (2, 0, 0, 0))[4]["approved_delta"][GATEWAY] == 0
    assert env.desired[GATEWAY] == 3


def test_default_maximum_is_ten_replicas():
    env = make_env(cluster_quota=41)
    run(env, 4, {k: (2, 0, 0, 0) for k in range(4)})
    assert env.desired[GATEWAY] == 10
    assert act(env, (2, 0, 0, 0))[4]["approved_delta"][GATEWAY] == 0


def test_scale_down_is_processed_before_a_lower_indexed_scale_up():
    # Gateway precedes Order in arbitration order, yet Order's cancellation funds it.
    env = make_env(cooldown_control_steps=0, cluster_quota=10)
    act(env, (0, 0, 0, 1))
    assert env.cluster_occupancy == 10
    assert env.shield.approve((1, 0, 0, -1)).deltas == (1, 0, 0, -1)


@pytest.mark.parametrize("proposal, free, expected", [
    ((2, 2, 2, 2), 8, (2, 2, 2, 2)), ((2, 2, 2, 2), 7, (2, 2, 2, 1)), ((2, 2, 2, 2), 5, (2, 1, 1, 1)),
    ((2, 2, 2, 2), 4, (1, 1, 1, 1)), ((2, 2, 2, 2), 3, (1, 1, 1, 0)), ((2, 2, 2, 2), 1, (1, 0, 0, 0)),
    ((2, 2, 2, 2), 0, (0, 0, 0, 0)), ((0, 2, 0, 2), 3, (0, 2, 0, 1)), ((2, 0, 0, 1), 2, (1, 0, 0, 1)),
    ((2, 1, 0, 0), 2, (1, 1, 0, 0)), ((1, 0, 2, 0), 2, (1, 0, 1, 0)),
])
def test_round_robin_allocates_one_pod_per_service_per_round(proposal, free, expected):
    assert make_env(cluster_quota=9 + free).shield.approve(proposal).deltas == expected


def test_round_robin_skips_a_service_at_its_maximum():
    env = make_env(max_replicas=(3, 10, 10, 10, 1), cluster_quota=15)
    assert env.shield.approve((2, 2, 2, 2)).deltas == (1, 2, 2, 1)


def test_default_quota_of_28_includes_the_database_pod():
    env = make_env()
    fill_cluster(env)
    np.testing.assert_array_equal(env.desired, [7, 7, 7, 6, 1])
    assert env.cluster_occupancy == 28
    assert env.shield.approve((2, 2, 2, 2)).deltas == HOLD


@every_scenario
def test_cluster_quota_is_never_exceeded(name):
    assert episode(name).rows["occupancy"].max() <= 28


def test_cooldown_counts_down_once_per_interval_after_the_scale_up_interval():
    env = make_env()
    history, features = [], []
    for step in range(8):
        observation = act(env, (1, 0, 0, 0) if step == 0 else HOLD)[0]
        history.append(int(env.cooldown[GATEWAY]))
        features.append(observation["node_features"][GATEWAY, 8])
    assert history == [6, 5, 4, 3, 2, 1, 0, 0]
    np.testing.assert_allclose(features, np.array(history) / 6)


@pytest.mark.parametrize("node", SCALABLE)
def test_scale_down_is_blocked_until_cooldown_reaches_zero(node):
    env = make_env()
    act(env, delta(node, 1))
    for _ in range(6):
        assert act(env, delta(node, -1))[4]["approved_delta"][node] == 0
    assert act(env, delta(node, -1))[4]["approved_delta"][node] == -1


def test_a_new_scale_up_resets_cooldown_without_decrementing_that_interval():
    env = make_env()
    run(env, 4, {0: (1, 0, 0, 0), 3: (1, 0, 0, 0)})
    assert env.cooldown[GATEWAY] == 6
    act(env)
    assert env.cooldown[GATEWAY] == 5


def test_cooldown_is_per_service():
    env = make_env()
    act(env, (1, 0, 0, 0))
    assert act(env, (0, -1, 0, 0))[4]["approved_delta"][AUTH] == -1


def test_partial_scale_up_sets_cooldown_and_denied_scale_up_does_not():
    env = make_env(cluster_quota=10)
    act(env, (2, 2, 0, 0))  # one free pod: Gateway +1, Auth denied
    assert (env.cooldown[GATEWAY], env.cooldown[AUTH]) == (6, 0)


def test_scale_downs_do_not_start_cooldown():
    env = make_env(initial_replicas=(4, 2, 2, 2, 1))
    for _ in range(3):
        assert act(env, (-1, 0, 0, 0))[4]["approved_delta"][GATEWAY] == -1
    assert env.cooldown[GATEWAY] == 0


def test_cooldown_blocks_cancellation():
    env = make_env(cooldown_control_steps=2)
    act(env, (1, 0, 0, 0))
    assert act(env, (-1, 0, 0, 0))[4]["approved_delta"][GATEWAY] == 0
    assert env.pending[GATEWAY] == 1


def test_default_cooldown_outlasts_boot_so_cancellation_cannot_occur():
    env = make_env()
    act(env, (1, 0, 0, 0))
    while env.pending[GATEWAY] > 0:
        assert act(env, (-1, 0, 0, 0))[4]["approved_delta"][GATEWAY] == 0
    assert env.cooldown[GATEWAY] > 0


@every_scenario
def test_database_never_scales(name):
    rows = episode(name).rows
    np.testing.assert_array_equal(rows["ready"][:, DATABASE], np.ones(1800))
    assert rows["pending"][:, DATABASE].sum() == 0 and rows["terminating"][:, DATABASE].sum() == 0


def test_a_fifth_action_component_cannot_address_the_database():
    env = make_env()
    assert env.shield.approve((0, 0, 0, 0, 2)).deltas == HOLD
    assert env.shield.fallback_count == 1


@pytest.mark.parametrize("overrides, warmup, deltas", [
    pytest.param({}, None, (3, 0, 0, 0), id="above-action-set"),
    pytest.param({}, None, (-2, 0, 0, 0), id="below-action-set"),
    pytest.param({}, None, (0, 0, 0, 0, 1), id="database-component"),
    pytest.param({"initial_replicas": (1, 2, 2, 2, 1)}, None, (-1, 0, 0, 0), id="below-minimum"),
    pytest.param({"max_replicas": (2, 10, 10, 10, 1)}, None, (1, 0, 0, 0), id="above-maximum"),
    pytest.param({"cluster_quota": 9}, None, (1, 0, 0, 0), id="above-quota"),
    pytest.param({}, (1, 0, 0, 0), (-1, 0, 0, 0), id="during-cooldown"),
])
def test_environment_independently_rejects_an_illegal_approved_vector(monkeypatch, overrides, warmup, deltas):
    env = make_env(**overrides)
    if warmup:
        act(env, warmup)
    monkeypatch.setattr(shield_module, "arbitrate", lambda *args, **kwargs: deltas)
    token = env.shield.approve(HOLD)
    before = digest(env)
    with pytest.raises(PermissionError):
        env.step(token)
    assert_same_state(before, digest(env))
    with pytest.raises(PermissionError, match="consumed"):
        env.step(token)


class FakeApproved(ApprovedAction):
    """A subclass must not pass the exact-type check."""


INVALID_ACTIONS = {
    "raw-array": lambda env, token: np.zeros(4),
    "raw-list": lambda env, token: [0, 0, 0, 0],
    "raw-tuple": lambda env, token: (0, 0, 0, 0),
    "none": lambda env, token: None,
    "int": lambda env, token: 0,
    "string": lambda env, token: token.nonce,
    "dict": lambda env, token: dataclasses.asdict(token),
    "unknown-nonce": lambda env, token: ApprovedAction(env.env_id, env.step_index, "forged", HOLD),
    "random-nonce": lambda env, token: ApprovedAction(env.env_id, env.step_index, uuid.uuid4().hex, HOLD),
    "unhashable-nonce": lambda env, token: ApprovedAction(env.env_id, env.step_index, ["x"], HOLD),
    "rebuilt-from-fields": lambda env, token: ApprovedAction(token.env_id, token.step_index, token.nonce, token.deltas),
    "edited-deltas": lambda env, token: dataclasses.replace(token, deltas=(2, 2, 2, 2)),
    "edited-step": lambda env, token: dataclasses.replace(token, step_index=token.step_index + 1),
    "edited-env": lambda env, token: dataclasses.replace(token, env_id="other"),
    "shallow-copy": lambda env, token: copy.copy(token),
    "deep-copy": lambda env, token: copy.deepcopy(token),
    "duck-type": lambda env, token: SimpleNamespace(**dataclasses.asdict(token)),
    "subclass": lambda env, token: FakeApproved(token.env_id, token.step_index, token.nonce, token.deltas),
}


@pytest.mark.parametrize("build", INVALID_ACTIONS.values(), ids=INVALID_ACTIONS.keys())
def test_invalid_action_is_rejected_without_changing_state(build):
    env = make_env(constant(OVERLOAD))
    run(env, 3, {0: (1, 0, 0, 0)})
    token = env.shield.approve((0, 1, 0, 0))
    before = digest(env)
    tokens_before = ledger(env)
    with pytest.raises(PermissionError):
        env.step(build(env, token))
    assert_same_state(before, digest(env))
    assert ledger(env) == tokens_before
    # A failed attempt must not burn the genuine token.
    np.testing.assert_array_equal(env.step(token)[4]["approved_delta"], [0, 1, 0, 0])


def test_token_mutated_in_place_is_rejected():
    env = make_env()
    token = env.shield.approve(HOLD)
    object.__setattr__(token, "deltas", (2, 2, 2, 2))
    with pytest.raises(PermissionError):
        env.step(token)
    np.testing.assert_array_equal(env.desired, [2, 2, 2, 2, 1])


def test_unused_token_from_a_previous_step_is_stale():
    env = make_env()
    stale = env.shield.approve((2, 2, 2, 2))
    act(env)
    before = digest(env)
    tokens_before = ledger(env)
    with pytest.raises(PermissionError, match="stale"):
        env.step(stale)
    assert_same_state(before, digest(env))
    assert ledger(env) == tokens_before


def test_only_one_of_two_same_step_tokens_can_be_used():
    env = make_env()
    first, second = env.shield.approve(HOLD), env.shield.approve((1, 0, 0, 0))
    env.step(first)
    with pytest.raises(PermissionError, match="stale"):
        env.step(second)


def test_token_from_before_reset_is_unknown():
    env = make_env()
    token = env.shield.approve(HOLD)
    env.reset(seed=0)
    with pytest.raises(PermissionError, match="unknown"):
        env.step(token)


def test_reused_token_is_rejected_without_changing_state():
    env = make_env()
    token = env.shield.approve((1, 0, 0, 0))
    env.step(token)
    before = digest(env)
    tokens_before = ledger(env)
    with pytest.raises(PermissionError):
        env.step(token)
    assert_same_state(before, digest(env))
    assert ledger(env) == tokens_before


def test_token_stays_consumed_when_processing_raises_after_consumption():
    env = make_env()
    run(env, 360)
    token = env.shield.approve(HOLD)
    with pytest.raises(ValueError):
        env.step(token)
    with pytest.raises(PermissionError, match="consumed"):
        env.step(token)


def test_token_from_another_environment_is_rejected():
    home, other = make_env(), make_env()
    token = home.shield.approve((1, 0, 0, 0))
    before = digest(other)
    tokens_before = ledger(other)
    with pytest.raises(PermissionError, match="different environment"):
        other.step(token)
    assert_same_state(before, digest(other))
    assert ledger(other) == tokens_before
    home.step(token)  # still valid at home
    assert home.desired[GATEWAY] == 3


def test_approving_changes_no_simulator_state_until_the_token_is_used():
    env = make_env()
    before = digest(env)
    env.shield.approve((2, 2, 2, 2))
    assert_same_state(before, digest(env))


# ---------------------------------------------- environment: items 37 to 46
@pytest.mark.parametrize("trace", [
    pytest.param(np.zeros(1799), id="1799"), pytest.param(np.zeros(1801), id="1801"),
    pytest.param(np.zeros((1800, 1)), id="column"), pytest.param(np.zeros((1, 1800)), id="row"),
    pytest.param(np.zeros((2, 900)), id="2d"), pytest.param(np.array(5.0), id="0d"),
    pytest.param(np.zeros(0), id="empty"), pytest.param(list(range(1800)), id="list"),
    pytest.param(100.0, id="float"), pytest.param("trace", id="string"),
    pytest.param(np.full(1800, "a"), id="string-dtype"), pytest.param(np.zeros(1800, dtype=bool), id="bool"),
    pytest.param(np.zeros(1800, dtype=complex), id="complex"), pytest.param(np.zeros(1800, dtype=object), id="object"),
])
def test_trace_with_wrong_shape_or_type_is_rejected_at_construction(trace):
    with pytest.raises(ValueError):
        MicroserviceMeshEnv(trace)


@pytest.mark.parametrize("index", [0, 900, 1799])
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1.0, -1e-12])
def test_trace_with_invalid_value_is_rejected_at_construction(index, bad):
    trace = np.ones(1800)
    trace[index] = bad
    with pytest.raises(ValueError):
        MicroserviceMeshEnv(trace)


@pytest.mark.parametrize("trace", [np.zeros(1800), np.ones(1800, dtype=np.int64), np.full(1800, 1e5)],
                         ids=["zeros", "integers", "large"])
def test_valid_trace_is_accepted_and_drives_external_arrivals(trace):
    rows = run(make_env(trace), 1)
    np.testing.assert_array_equal(rows["external"], trace[:5])


def test_none_trace_is_constant_100_for_the_whole_horizon():
    np.testing.assert_array_equal(run(make_env(), 360)["external"], np.full(1800, 100.0))


def test_trace_is_read_in_order_for_the_whole_horizon():
    trace = SCENARIOS["ramp"]
    np.testing.assert_array_equal(run(make_env(trace), 360)["external"], trace)


def test_later_edits_to_the_callers_trace_do_not_affect_the_run():
    trace = np.full(1800, 7.0)
    env = make_env(trace)
    trace[:] = 999.0
    np.testing.assert_array_equal(run(env, 1)["external"], np.full(5, 7.0))


def test_reset_returns_the_trace_pointer_to_zero():
    trace = SCENARIOS["ramp"]
    env = make_env(trace)
    run(env, 3)
    assert env.trace_pointer == 15
    env.reset(seed=0)
    assert env.trace_pointer == 0
    np.testing.assert_array_equal(run(env, 1)["external"], trace[:5])


@pytest.mark.parametrize("trace", [None, np.ones(1800)], ids=["constant", "array"])
def test_traffic_exhaustion_raises_at_index_1800_without_mutating_state(trace):
    env = make_env(trace)
    run(env, 360)
    before = digest(env)
    with pytest.raises(ValueError, match="1800"):
        act(env, (2, 2, 2, 2))
    assert_same_state(before, digest(env))


@pytest.mark.parametrize("trace", [None, np.ones(1800)], ids=["constant", "array"])
def test_episode_terminates_exactly_at_the_360th_control_step(trace):
    env = make_env(trace)
    for step in range(1, 361):
        _, _, terminated, truncated, _ = act(env)
        assert terminated is (step == 360)
        assert truncated is False
    assert (env.time, env.step_index) == (1800, 360)


def test_observation_shapes_and_topology_matrix():
    env = make_env()
    observation, _ = env.reset(seed=0)
    assert observation["node_features"].shape == (5, 9)
    assert observation["call_multiplicity"].shape == (5, 5)
    assert env.observation_space.contains(observation)
    expected = [[0, 1.0, 0.8, 0.2, 0], [0, 0, 0, 0, 0.3], [0, 0, 0, 0, 0.5], [0, 0, 0, 0, 1.0], [0, 0, 0, 0, 0]]
    np.testing.assert_array_equal(observation["call_multiplicity"], expected)
    observation["call_multiplicity"][0, 1] = 9.0  # a caller's edit must not reach the env
    np.testing.assert_array_equal(act(env)[0]["call_multiplicity"], expected)


def test_reset_observation_has_no_interval_history():
    observation, _ = make_env().reset(seed=0)
    features = observation["node_features"]
    np.testing.assert_array_equal(features[:, [0, 1, 2, 3, 5, 6, 7, 8]], np.zeros((5, 8)))
    np.testing.assert_allclose(features[:, 4], [0.2, 0.2, 0.2, 0.2, 1.0])


def test_first_interval_features_match_hand_values():
    features = act(make_env())[0]["node_features"]
    # Gateway: 100 req/s of 1000; 500 served of 1000 capacity; 2 of 10 pods; 5 ms.
    np.testing.assert_allclose(features[GATEWAY], [0.1, 0, 0, 0.5, 0.2, 0, 0.005, 0, 0])
    # Database: 90 req/s for 3 of 5 substeps; 270 served of 1500 capacity; 1 of 1 pod.
    np.testing.assert_allclose(features[DATABASE], [0.18, 0, 0, 0.18, 1.0, 0, 0.01, 0, 0])


def test_retry_queue_pending_and_saturated_features():
    env = make_env(trace_of(100.0), **without_replicas(GATEWAY))
    act(env)
    assert act(env)[0]["node_features"][GATEWAY, 1] == pytest.approx(0.01)  # 50 over 5 s of 1000
    features = act(make_env(constant(OVERLOAD)), (2, 0, 0, 0))[0]["node_features"]
    assert features[GATEWAY, 2] == pytest.approx(0.4)   # 800 of 2000
    assert features[GATEWAY, 3] == 1.0
    assert features[GATEWAY, 5] == pytest.approx(0.2)   # 2 of 10
    assert features[GATEWAY, 6] == 1.0                  # latency above 1000 ms clips


def test_latency_feature_is_the_mean_over_the_interval():
    features = act(make_env(trace_of(500.0, 100.0)))[0]["node_features"]
    # Gateway latency is 1505, 1005, 1505, 5, 5 ms: mean 805 ms, not the last 5 ms.
    assert features[GATEWAY, 6] == pytest.approx(0.805)


@every_scenario
def test_observations_are_finite_and_within_bounds(name):
    env = MicroserviceMeshEnv()
    for observation in episode(name).observations:
        features = observation["node_features"]
        assert np.all(np.isfinite(features))
        assert features.min() >= 0.0 and features.max() <= 1.0
        assert env.observation_space.contains(observation)


@every_scenario
def test_rewards_are_finite_and_non_positive(name):
    rewards = np.array(episode(name).rewards)
    assert rewards.shape == (360,)
    assert np.all(np.isfinite(rewards)) and rewards.max() <= 0.0


def test_idle_reward_is_the_resource_cost_only():
    assert act(make_env())[1] == pytest.approx(-0.5 * 9 / 28)


@pytest.mark.parametrize("proposal, quota, scaling_cost, occupancy", [
    ((2, 2, 2, 2), 28, 1.0, 17), ((1, 0, 0, 0), 28, 0.125, 10),
    ((-1, 0, 0, 0), 28, 0.125, 9),   # a draining pod still costs resources
    ((2, 2, 2, 2), 10, 0.125, 10),   # only the approved pod is charged
])
def test_scaling_cost_uses_approved_changes_once_per_interval(proposal, quota, scaling_cost, occupancy):
    _, reward, _, _, info = act(make_env(cluster_quota=quota), proposal)
    assert info["reward_components"]["scaling_cost"] == scaling_cost
    assert reward == pytest.approx(-(0.5 * occupancy / quota + 0.2 * scaling_cost))


def test_failure_cost_is_the_mean_proxy_over_the_interval():
    env = make_env(trace_of(1000, 100, 100, 100, 100), queue_capacity=(100, 2000, 1500, 1000, 3000))
    _, reward, _, _, info = act(env)
    assert info["reward_components"]["failure_cost"] == pytest.approx(0.18)  # 0.9 at t=0 only
    assert reward == pytest.approx(-(5.0 * 0.18 + 0.5 * 9 / 28))


def test_slo_violation_is_capped_per_substep_then_averaged():
    _, reward, _, _, info = act(make_env(trace_of(500.0, 100.0)))
    # L = 1539, 1039, 1539, 39, 39. Raw violations 5.156 cap at 5; 789/250 = 3.156.
    np.testing.assert_allclose(info["substeps"]["slo_violation"], [5.0, 3.156, 5.0, 0.0, 0.0])
    assert info["reward_components"]["slo_violation"] == pytest.approx(2.6312)
    assert reward == pytest.approx(-(2.0 * 2.6312 + 0.5 * 9 / 28))


@pytest.mark.parametrize("name", ["sine", "overload"])
def test_identical_inputs_reproduce_identical_trajectories(name):
    def rollout(env, seed):
        observation, _ = env.reset(seed=seed)
        record = [observation["node_features"]]
        for proposal in np.random.default_rng(11).integers(-1, 3, size=(360, 4)):
            observation, reward, _, _, info = act(env, proposal)
            record += [observation["node_features"], reward, *info["substeps"].values()]
        return record

    env = MicroserviceMeshEnv(SCENARIOS[name])
    first, replay = rollout(env, 3), rollout(env, 3)
    second_instance = rollout(MicroserviceMeshEnv(SCENARIOS[name]), 3)
    other_seed = rollout(MicroserviceMeshEnv(SCENARIOS[name]), 4)  # physics has no randomness
    for other in (replay, second_instance, other_seed):
        for left, right in zip(first, other):
            np.testing.assert_array_equal(left, right)


def test_reset_clears_all_state_after_a_messy_episode():
    trace = constant(OVERLOAD)
    fresh = digest(make_env(trace))
    env = make_env(trace)
    run(env, 12, {0: (2, 2, 2, 2), 1: (-1, 0, 0, 0), 2: (2, 0, 2, 0)})
    env.shield.approve(np.nan)
    env.reset(seed=0)
    assert_same_state(fresh, digest(env))
    assert fresh["fallback"] == 0
    assert ledger(env) == {}


@pytest.mark.parametrize("corrupt", [
    pytest.param(lambda env: env._queue.__setitem__((1, 2), -1.0), id="negative-queue"),
    pytest.param(lambda env: env._queue.__setitem__((0, 0), np.nan), id="nan-queue"),
    pytest.param(lambda env: env._queue.__setitem__((0, 1), 5000.0), id="queue-over-capacity"),
    pytest.param(lambda env: env._ready.__setitem__(0, -1), id="negative-replicas"),
    pytest.param(lambda env: env._desired.__setitem__(1, 5), id="desired-mismatch"),
])
def test_diagnostics_fail_loudly_instead_of_clipping(corrupt):
    env = make_env()
    corrupt(env)
    with pytest.raises(SimulationInvariantError):
        act(env)


def test_quota_diagnostic_fires_on_corrupted_occupancy():
    env = make_env()
    env._ready[0] += 30
    env._desired[0] += 30
    with pytest.raises(SimulationInvariantError, match="quota"):
        act(env)


def test_conservation_is_checked_inside_the_running_simulation():
    env = make_env(constant(OVERLOAD))
    env._nodes[AUTH] = GATEWAY  # credits Gateway's timeouts to Auth at t=4
    with pytest.raises(SimulationInvariantError, match="conservation"):
        act(env)


def test_default_config_reproduces_the_spec_table():
    config = SimulatorConfig()
    assert config.service_rate == MU
    assert config.queue_capacity == (2000, 2000, 1500, 1000, 3000)
    assert config.base_latency_ms == (5, 8, 15, 25, 10)
    assert config.min_replicas == (1, 1, 1, 1, 1)
    assert config.max_replicas == (10, 10, 10, 10, 1)
    assert config.initial_replicas == (2, 2, 2, 2, 1)
    assert config.boot_seconds == (20,) * 5
    assert config.termination_seconds == (5,) * 5
    assert config.timeout_seconds == (5,) * 5
    assert config.contention_kappa == (0, 1.0, 1.5, 2.5, 0)
    assert (config.substeps_per_control_step, config.control_horizon, config.internal_horizon) == (5, 360, 1800)
    assert (config.cluster_quota, config.cooldown_control_steps, config.queue_age_buckets) == (28, 6, 5)


def test_default_conservation_tolerance_is_the_spec_value():
    assert SimulatorConfig().conservation_tolerance == 1e-9


def test_config_is_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SimulatorConfig().cluster_quota = 99


@pytest.mark.parametrize("overrides", [
    {"service_rate": (100, 120, 80, 50)},
    {"service_rate": (0, 120, 80, 50, 300)},
    {"queue_capacity": (0, 2000, 1500, 1000, 3000)},
    {"initial_replicas": (11, 2, 2, 2, 1)},
    {"initial_replicas": (0, 2, 2, 2, 1)},
    {"max_replicas": (10, 10, 10, 10, 2)},
    {"timeout_seconds": (0, 5, 5, 5, 5)},
    {"boot_seconds": (0, 20, 20, 20, 20)},
    {"cluster_quota": 8},
    {"substeps_per_control_step": 0},
    {"call_multiplicity": ((0.0,) * 5,) * 4},
])
def test_invalid_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        dataclasses.replace(SimulatorConfig(), **overrides)


NAN, INF = float("nan"), float("inf")
INVALID_SCALARS = {
    "substeps_per_control_step": [0, -1, 1.5, INF, True, "5"],
    "control_horizon": [0, -1, 360.0, INF],
    "default_traffic_rps": [-1.0, NAN, INF, True, "100"],
    "db_contention_threshold": [-0.1, 1.1, NAN, INF],
    "capacity_multiplier": [0.0, -1.0, NAN, INF],
    "retry_coefficient": [-0.1, 1.1, NAN, INF],
    "conservation_tolerance": [0.0, -1e-9, NAN, INF],
    "cluster_quota": [8, 28.0, INF],
    "cooldown_control_steps": [-1, 1.5, INF],
    "action_min": [1, 0.5, NAN],
    "action_max": [-1, 1.5, NAN],
    "slo_ms": [0.0, -1.0, NAN, INF],
    "slo_violation_cap": [-1.0, NAN, INF],
    "weight_slo": [-0.1, NAN, INF],
    "weight_failure": [-0.1, NAN, INF],
    "weight_resources": [-0.1, NAN, INF],
    "weight_scaling": [-0.1, NAN, INF],
    "scaling_cost_norm": [0.0, -1.0, NAN, INF],
}
INVALID_FIRST_NODE_VALUES = {
    "service_rate": [0, -1, NAN, INF, True],
    "queue_capacity": [0, -1, NAN, INF],
    "base_latency_ms": [-1, NAN, INF],
    "contention_kappa": [-1, NAN, INF],
    "min_replicas": [-1, 3, 1.5],
    "max_replicas": [1, 0, 1.5],
    "initial_replicas": [0, 11, 1.5],
    "boot_seconds": [0, -1, 1.5],
    "termination_seconds": [0, -1, 1.5],
    "timeout_seconds": [0, -1, 1.5],
}


def _with_first_call_multiplicity(value) -> tuple:
    default = SimulatorConfig().call_multiplicity
    return ((value,) + default[0][1:],) + default[1:]


@pytest.mark.parametrize("field, value", [
    pytest.param(field, value, id=f"{field}={value!r}")
    for field, values in INVALID_SCALARS.items() for value in values
])
def test_invalid_scalar_config_value_is_rejected(field, value):
    with pytest.raises(ValueError):
        dataclasses.replace(SimulatorConfig(), **{field: value})


@pytest.mark.parametrize("field, value", [
    pytest.param(field, value, id=f"{field}={value!r}")
    for field, values in INVALID_FIRST_NODE_VALUES.items() for value in values
])
def test_invalid_per_node_config_value_is_rejected(field, value):
    default = getattr(SimulatorConfig(), field)
    with pytest.raises(ValueError):
        dataclasses.replace(SimulatorConfig(), **{field: (value,) + default[1:]})


@pytest.mark.parametrize("overrides", [
    pytest.param({"service_rate": [100, 120, 80, 50, 300]}, id="list-not-tuple"),
    pytest.param({"min_replicas": (1, 1, 1, 1, 0)}, id="database-minimum"),
    pytest.param({"call_multiplicity": _with_first_call_multiplicity(-0.1)}, id="negative-multiplicity"),
    pytest.param({"call_multiplicity": _with_first_call_multiplicity(NAN)}, id="nan-multiplicity"),
    pytest.param({"call_multiplicity": _with_first_call_multiplicity(INF)}, id="inf-multiplicity"),
    pytest.param({"call_multiplicity": tuple(list(row) for row in SimulatorConfig().call_multiplicity)}, id="list-rows"),
])
def test_invalid_structured_config_is_rejected(overrides):
    with pytest.raises(ValueError):
        dataclasses.replace(SimulatorConfig(), **overrides)


@pytest.mark.parametrize("overrides", [
    {"default_traffic_rps": 0.0}, {"db_contention_threshold": 0.0}, {"db_contention_threshold": 1.0},
    {"retry_coefficient": 0.0}, {"retry_coefficient": 1.0}, {"slo_violation_cap": 0.0},
    {"weight_slo": 0.0}, {"weight_failure": 0.0}, {"weight_resources": 0.0}, {"weight_scaling": 0.0},
    {"action_min": 0}, {"action_max": 0}, {"cooldown_control_steps": 0},
    {"base_latency_ms": (0, 0, 0, 0, 0)}, {"contention_kappa": (0, 0, 0, 0, 0)},
    {"cluster_quota": np.int64(28)}, {"slo_ms": np.float64(250.0)},
    {"service_rate": (np.float32(100), 120, 80, 50, 300)},
])
def test_valid_boundary_config_is_accepted_and_runs(overrides):
    assert np.isfinite(act(make_env(**overrides))[1])


def test_per_node_timeout_is_honoured():
    timeouts = (3, 5, 5, 5, 5)
    rows = run(make_env(trace_of(100.0), timeout_seconds=timeouts, **without_replicas(GATEWAY)), 1)
    np.testing.assert_array_equal(rows["timeout"][:, GATEWAY], [0, 0, 100, 0, 0])


def test_stage_isolation_keeps_forbidden_packages_out():
    assert not {"torch", "stable_baselines3", "kubernetes"} & set(sys.modules)
