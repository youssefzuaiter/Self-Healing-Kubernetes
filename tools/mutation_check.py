#!/usr/bin/env python3
"""Reproducible mutation check for the Stage 1A simulator.

Run from anywhere, in an environment with NumPy, Gymnasium and pytest:

    python tools/mutation_check.py                 # every mutant
    python tools/mutation_check.py --group config  # one mutant per config field
    python tools/mutation_check.py --jobs 1        # no parallelism

Each mutant makes one exact-text edit to a temporary copy of sim/ and tests/, then
runs the full pytest suite on that copy. A mutant is killed when the suite fails.
The working tree is never changed. Results print in list order, so runs are
comparable. Exit code 0 means the baseline passed and every mutant was killed.

The mutants are hand-picked, not generated. A clean run is evidence, not proof.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, Optional

ROOT = Path(__file__).resolve().parents[1]
ENV, SHIELD, CONFIG = "sim/env.py", "sim/shield.py", "sim/config.py"
TIMEOUT_SECONDS = 180
SPEC_TABLE_TEST = "test_default_config_reproduces_the_spec_table"


class Mutant(NamedTuple):
    name: str
    path: str
    old: str
    new: str


LOGIC = [
    # Propagation, retries, admission, FIFO, timeouts
    Mutant("transposed topology", ENV, "completed @ self._multiplicity", "completed @ self._multiplicity.T"),
    Mutant("downstream arrivals dropped", ENV, "arrivals_new = self._next_downstream.copy()",
           "arrivals_new = self._next_downstream.copy() * 0.0"),
    Mutant("two-substep propagation", ENV, "        self._next_downstream = completed @ self._multiplicity\n",
           "        self._next_downstream, self._held = "
           "getattr(self, '_held', np.zeros(NUM_NODES)), completed @ self._multiplicity\n"),
    Mutant("overflow retried", ENV, "cfg.retry_coefficient * timeout", "cfg.retry_coefficient * (timeout + overflow)"),
    Mutant("LIFO service", ENV, "for age in range(self._queue.shape[1] - 1, -1, -1):",
           "for age in range(self._queue.shape[1]):"),
    Mutant("service creates room", ENV, "np.maximum(0.0, self._q_max - queue_start)",
           "np.maximum(0.0, self._q_max - queue_start + capacity)"),
    Mutant("expired work not cleared", ENV, "        residual[self._nodes, self._window - 1] = 0.0\n", ""),
    Mutant("bucket 0 not cleared", ENV, "        self._queue[:, 0] = 0.0\n", ""),
    Mutant("timeout read from wrong bucket", ENV, "timeout = residual[self._nodes, self._window - 1].copy()",
           "timeout = residual[self._nodes, self._window - 2].copy()"),
    # Contention and capacity
    Mutant("contention sees current arrivals", ENV, "self._kappa, queue_start[DATABASE], self._q_max",
           "self._kappa, queue_start[DATABASE] + arrivals_new[DATABASE], self._q_max"),
    Mutant("no contention threshold", ENV, "max(0.0, pressure - threshold)", "max(0.0, pressure)"),
    Mutant("pending pods serve", ENV, "capacity = self._ready * self._mu",
           "capacity = (self._ready + self.pending) * self._mu"),
    Mutant("terminating pods serve", ENV, "capacity = self._ready * self._mu",
           "capacity = (self._ready + self.terminating) * self._mu"),
    # Failure ratio floor
    Mutant("floor removed from feature 7", ENV, "features[:, 7] = failure_ratio(failed, completed + failed)",
           "features[:, 7] = safe_ratio(failed, completed + failed)"),
    Mutant("floor removed from substep ratio", ENV, "ratio = failure_ratio(failed, completed + failed)",
           "ratio = safe_ratio(failed, completed + failed)"),
    Mutant("floor removed from info ratio", ENV, '"interval_failure_ratio": failure_ratio(failed,',
           '"interval_failure_ratio": safe_ratio(failed,'),
    Mutant("floor of 0.5", ENV, "return failed / np.maximum(1.0, resolved)", "return failed / np.maximum(0.5, resolved)"),
    Mutant("floor of 2.0", ENV, "return failed / np.maximum(1.0, resolved)", "return failed / np.maximum(2.0, resolved)"),
    Mutant("interval floored per substep", ENV, "features[:, 7] = failure_ratio(failed, completed + failed)",
           'features[:, 7] = failed / np.maximum(1.0, rows["completed"] + rows["overflow"] + rows["timeout"]).sum(axis=0)'),
    Mutant("interval is mean of substep ratios", ENV, "features[:, 7] = failure_ratio(failed, completed + failed)",
           'features[:, 7] = rows["failure_ratio"].mean(axis=0)'),
    Mutant("served age uses the floor", ENV, "served_delay = 1000.0 * safe_ratio(served @ self._ages, completed)",
           "served_delay = 1000.0 * failure_ratio(served @ self._ages, completed)"),
    # Latency and proxies
    Mutant("age off by one", ENV, "served @ self._ages", "served @ (self._ages + 1)"),
    Mutant("backlog divides by raw capacity", ENV, "np.maximum(1.0, capacity)", "(capacity + 1e-9)"),
    Mutant("backlog not capped", ENV, "np.minimum(self._window, queue_end / np.maximum(1.0, capacity))",
           "(queue_end / np.maximum(1.0, capacity))"),
    Mutant("served delay in seconds", ENV, "served_delay = 1000.0 * safe_ratio", "served_delay = 1.0 * safe_ratio"),
    Mutant("branch weights swapped in failure", ENV, "(0.8 * p2 * p4 ** 0.5 + 0.2 * p3 * p4 ** 1.0)",
           "(0.2 * p2 * p4 ** 0.5 + 0.8 * p3 * p4 ** 1.0)"),
    Mutant("auth database exponent", ENV, "p0 * p1 * p4 ** 0.3 *", "p0 * p1 * p4 ** 0.5 *"),
    Mutant("auth database latency weight", ENV, "l0 + l1 + 0.3 * l4", "l0 + l1 + 0.5 * l4"),
    # Reward
    Mutant("signed scaling cost", ENV, "float(np.abs(approved).sum())", "float(approved.sum())"),
    Mutant("scaling cost charged per substep", ENV, "/ cfg.scaling_cost_norm  # once", "* 5 / cfg.scaling_cost_norm  # once"),
    Mutant("uncapped SLO violation", ENV,
           "np.array(min(cfg.slo_violation_cap, max(0.0, (e2e_latency - cfg.slo_ms) / cfg.slo_ms)))",
           "np.array(max(0.0, (e2e_latency - cfg.slo_ms) / cfg.slo_ms))"),
    Mutant("SLO cost uses the maximum", ENV, 'rows["slo_violation"].mean()', 'rows["slo_violation"].max()'),
    Mutant("failure cost uses the maximum", ENV, 'rows["e2e_failure"].mean()', 'rows["e2e_failure"].max()'),
    Mutant("resource cost ignores the database", ENV, "occupancy / cfg.cluster_quota", "(occupancy - 1) / cfg.cluster_quota"),
    # Lifecycle
    Mutant("boot slot off by one", ENV, "self.config.boot_seconds[node] - 1] += count",
           "self.config.boot_seconds[node] - 2] += count"),
    Mutant("termination slot off by one", ENV, "self.config.termination_seconds[node] - 1] += remaining",
           "self.config.termination_seconds[node] - 2] += remaining"),
    Mutant("oldest-first cancellation", ENV, "for slot in range(self._boot_register.shape[1] - 1, -1, -1):",
           "for slot in range(self._boot_register.shape[1]):"),
    Mutant("ready pod not removed", ENV, "        self._ready[node] -= remaining\n", ""),
    Mutant("desired not reduced", ENV, "        self._desired[node] -= count\n", ""),
    Mutant("cooldown not set", ENV, "        self._cooldown[node] = self.config.cooldown_control_steps\n", ""),
    Mutant("cooldown decrements in scale-up interval", ENV,
           "np.where(approved > 0, current, np.maximum(current - 1, 0))", "np.maximum(current - 1, 0)"),
    # Tokens and independent validation
    Mutant("token not consumed", ENV, "        record.consumed = True  # Section 30.1 step 7: before anything can raise\n", ""),
    Mutant("no identity check", ENV, "if record.token is not action or (action.step_index", "if (action.step_index"),
    Mutant("no field check", ENV,
           "if record.token is not action or (action.step_index, action.deltas) != (record.step_index, record.deltas):",
           "if record.token is not action:"),
    Mutant("no stale check", ENV, "if record.step_index != self._step_index:", "if False:"),
    Mutant("no consumed check", ENV, "if record.consumed:", "if False:"),
    Mutant("no environment check", ENV, "if action.env_id != self._env_id:", "if False:"),
    Mutant("no independent validation", ENV, "        self._validate_approved(approved)\n", ""),
    Mutant("no quota revalidation", ENV,
           "if scale_ups > 0 and self.cluster_occupancy - cancelled + scale_ups > cfg.cluster_quota:", "if False:"),
    Mutant("no cooldown revalidation", ENV, "if np.any((approved < 0) & (self._cooldown[:NUM_SCALABLE] > 0)):", "if False:"),
    Mutant("ledger survives reset", ENV, "self._tokens: dict[str, _TokenRecord] = {}", 'self._tokens = getattr(self, "_tokens", {})'),
    # Horizon, trace, reset
    Mutant("horizon probe removed", ENV, "        self._peek_traffic()  # a post-horizon step raises here, before any mutation\n", ""),
    Mutant("exhaustion off by one", ENV, "if index >= self.config.internal_horizon:", "if index > self.config.internal_horizon:"),
    Mutant("terminates a step late", ENV, "terminated = self._step_index >= self.config.control_horizon",
           "terminated = self._step_index > self.config.control_horizon"),
    Mutant("trace shape unchecked", ENV, "if trace.shape != (length,):", "if False:"),
    Mutant("trace finiteness unchecked", ENV, 'if not np.all(np.isfinite(values)):\n        raise ValueError("traffic_trace',
           'if False:\n        raise ValueError("traffic_trace'),
    Mutant("trace not copied", ENV, "values = trace.astype(np.float64)", "values = np.asarray(trace, dtype=np.float64)"),
    Mutant("negative trace allowed", ENV, "if np.any(values < 0.0):", "if np.any(values < -1.0):"),
    Mutant("retry buffer survives reset", ENV, "        self._next_retry = np.zeros(NUM_NODES)\n",
           '        self._next_retry = getattr(self, "_next_retry", np.zeros(NUM_NODES))\n'),
    Mutant("fallback survives reset", ENV, "        self.shield.reset()\n", ""),
    # Observation
    Mutant("observation not clipped", ENV, "return np.clip(features, 0.0, 1.0)", "return features"),
    Mutant("arrival feature sums", ENV, 'rows["arrivals_new"].mean(axis=0)', 'rows["arrivals_new"].sum(axis=0)'),
    Mutant("retry feature sums", ENV, 'rows["arrivals_retry"].mean(axis=0)', 'rows["arrivals_retry"].sum(axis=0)'),
    Mutant("queue feature wrong scale", ENV, "features[:, 2] = self._queue.sum(axis=1) / self._q_max",
           "features[:, 2] = self._queue.sum(axis=1) / 1000.0"),
    Mutant("ready feature wrong scale", ENV, "features[:, 4] = self._ready / self._r_max", "features[:, 4] = self._ready / 5.0"),
    Mutant("pending feature reads terminating", ENV, "features[:, 5] = self.pending / self._r_max",
           "features[:, 5] = self.terminating / self._r_max"),
    Mutant("latency feature uses last substep", ENV, 'rows["node_latency_ms"].mean(axis=0) / 1000.0',
           'rows["node_latency_ms"][-1] / 1000.0'),
    Mutant("cooldown feature wrong scale", ENV, "max(1, self.config.cooldown_control_steps)", "5"),
    # Diagnostics
    Mutant("conservation check not wired", ENV,
           '        check_conservation(\n            record["queue_start"], record["arrivals_new"], '
           'record["arrivals_retry"], record["completed"],\n            record["overflow"], record["timeout"], '
           'record["queue_end"], tolerance,\n        )\n', ""),
    Mutant("served work not checked", ENV, " and np.all(served >= 0.0)", ""),
    # Shield
    Mutant("cooldown ignored", SHIELD, "if change >= 0 or cooldown[node] > 0:", "if change >= 0:"),
    Mutant("termination frees quota", SHIELD, "occupancy -= min(allowed, int(pending[node]))", "occupancy -= allowed"),
    Mutant("cancellation keeps quota", SHIELD, "            occupancy -= min(allowed, int(pending[node]))\n", ""),
    Mutant("reverse round-robin", SHIELD, "for node in range(NUM_SCALABLE):\n            at_maximum",
           "for node in reversed(range(NUM_SCALABLE)):\n            at_maximum"),
    Mutant("maximum off by one", SHIELD, ">= config.max_replicas[node]", "> config.max_replicas[node]"),
    Mutant("minimum ignored", SHIELD, "min(-change, int(desired[node]) - config.min_replicas[node])", "-change"),
    Mutant("half-up rounding", SHIELD, "return int(np.round(value))", "return int(np.floor(value + 0.5))"),
    Mutant("fallback not counted", SHIELD, "            self._fallback_count += 1\n", ""),
    Mutant("bool list accepted", SHIELD,
           "if isinstance(raw, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in raw):", "if False:"),
    Mutant("non-numeric dtype accepted", SHIELD,
           "if not (np.issubdtype(vector.dtype, np.integer) or np.issubdtype(vector.dtype, np.floating)):",
           "if vector.dtype.kind in 'US':"),
    # Gaps closed after the Stage 1B readiness check
    Mutant("LIFO everywhere except the Gateway", ENV,
           "        for age in range(self._queue.shape[1] - 1, -1, -1):\n"
           "            served[:, age] = np.minimum(self._queue[:, age], remaining)\n"
           "            remaining -= served[:, age]\n",
           "        for age in range(self._queue.shape[1] - 1, -1, -1):\n"
           "            served[0, age] = min(self._queue[0, age], remaining[0])\n"
           "            remaining[0] -= served[0, age]\n"
           "        for age in range(self._queue.shape[1]):\n"
           "            served[1:, age] = np.minimum(self._queue[1:, age], remaining[1:])\n"
           "            remaining[1:] -= served[1:, age]\n"),
    Mutant("Gateway timeout window used for every node", ENV,
           "        timeout = residual[self._nodes, self._window - 1].copy()\n"
           "        residual[self._nodes, self._window - 1] = 0.0\n",
           "        timeout = residual[self._nodes, self._window[GATEWAY] - 1].copy()\n"
           "        residual[self._nodes, self._window[GATEWAY] - 1] = 0.0\n"),
    Mutant("Gateway queue capacity used for every node", ENV, "np.maximum(0.0, self._q_max - queue_start)",
           "np.maximum(0.0, self._q_max[GATEWAY] - queue_start)"),
    Mutant("invalid token resets the cached observation", ENV,
           '        """Section 30.1 steps 1-6. Raises PermissionError and changes nothing."""\n',
           '        """Section 30.1 steps 1-6. Raises PermissionError and changes nothing."""\n'
           "        self._features = self._build_features(None)\n"),
    Mutant("unknown nonce written into the ledger", ENV,
           '            raise PermissionError("token has an unknown nonce")\n',
           "            if isinstance(action.nonce, str):\n"
           "                self._tokens[action.nonce] = _TokenRecord(action, action.step_index, action.deltas, True)\n"
           '            raise PermissionError("token has an unknown nonce")\n'),
    Mutant("stale rejection deletes the ledger record", ENV,
           '            raise PermissionError(f"stale token from step',
           '            del self._tokens[action.nonce]\n            raise PermissionError(f"stale token from step'),
    # Config validation: type rules
    Mutant("infinite numbers accepted", CONFIG, " and math.isfinite(value)", ""),
    Mutant("bool accepted as number", CONFIG, "isinstance(value, Real) and not isinstance(value, bool)", "isinstance(value, Real)"),
    Mutant("bool accepted as integer", CONFIG, "isinstance(value, Integral) and not isinstance(value, bool)", "isinstance(value, Integral)"),
    Mutant("float accepted as integer", CONFIG, "return isinstance(value, Integral) and not isinstance(value, bool)",
           "return isinstance(value, Real) and not isinstance(value, bool)"),
    Mutant("list accepted for per-node field", CONFIG, "isinstance(values, tuple) and len(values) == NUM_NODES", "len(values) == NUM_NODES"),
    Mutant("list rows accepted in topology", CONFIG, "all(isinstance(row, tuple) and len(row) == NUM_NODES for row in matrix)",
           "all(len(row) == NUM_NODES for row in matrix)"),
    # Config validation: each range rule removed
    *[Mutant(f"{name} range rule removed", CONFIG, f"        {line}\n", "") for name, line in (
        ("substeps", '_require(self.substeps_per_control_step >= 1, "substeps_per_control_step must be >= 1")'),
        ("control_horizon", '_require(self.control_horizon >= 1, "control_horizon must be >= 1")'),
        ("default_traffic_rps", '_require(self.default_traffic_rps >= 0, "default_traffic_rps must be >= 0")'),
        ("service_rate", '_require(all(rate > 0 for rate in self.service_rate), "service_rate must be positive")'),
        ("queue_capacity", '_require(all(size > 0 for size in self.queue_capacity), "queue_capacity must be positive")'),
        ("base_latency_ms", '_require(all(latency >= 0 for latency in self.base_latency_ms), "base_latency_ms must be >= 0")'),
        ("contention_kappa", '_require(all(kappa >= 0 for kappa in self.contention_kappa), "contention_kappa must be >= 0")'),
        ("call_multiplicity", '_require(all(value >= 0 for row in self.call_multiplicity for value in row), "call_multiplicity must be >= 0")'),
        ("cluster_quota", '_require(sum(self.initial_replicas) <= self.cluster_quota, "initial replicas exceed cluster_quota")'),
        ("db_contention_threshold", '_require(0 <= self.db_contention_threshold <= 1, "db_contention_threshold must be in [0, 1]")'),
        ("capacity_multiplier", '_require(self.capacity_multiplier > 0, "capacity_multiplier must be positive")'),
        ("retry_coefficient", '_require(0 <= self.retry_coefficient <= 1, "retry_coefficient must be in [0, 1]")'),
        ("conservation_tolerance", '_require(self.conservation_tolerance > 0, "conservation_tolerance must be positive")'),
        ("cooldown", '_require(self.cooldown_control_steps >= 0, "cooldown_control_steps must be >= 0")'),
        ("action set", '_require(self.action_min <= 0 <= self.action_max, "the action set must contain 0")'),
        ("slo_ms", '_require(self.slo_ms > 0, "slo_ms must be positive")'),
        ("slo_violation_cap", '_require(self.slo_violation_cap >= 0, "slo_violation_cap must be >= 0")'),
        ("scaling_cost_norm", '_require(self.scaling_cost_norm > 0, "scaling_cost_norm must be positive")'),
    )],
    Mutant("replica order rule removed", CONFIG, "0 <= low <= start <= high and high >= 1", "True"),
    Mutant("database replica rule removed", CONFIG, "self.min_replicas[DATABASE] == self.initial_replicas[DATABASE] == self.max_replicas[DATABASE] == 1", "True"),
    Mutant("delay rule removed", CONFIG, 'all(value >= 1 for value in getattr(self, name))', "True"),
    Mutant("weight rule removed", CONFIG, "getattr(self, name) >= 0", "True"),
    # Range rules loosened by one boundary
    Mutant("retry coefficient may exceed 1", CONFIG, "0 <= self.retry_coefficient <= 1", "0 <= self.retry_coefficient"),
    Mutant("retry coefficient may be negative", CONFIG, "0 <= self.retry_coefficient <= 1", "self.retry_coefficient <= 1"),
    Mutant("threshold may exceed 1", CONFIG, "0 <= self.db_contention_threshold <= 1", "0 <= self.db_contention_threshold"),
    Mutant("threshold may be negative", CONFIG, "0 <= self.db_contention_threshold <= 1", "self.db_contention_threshold <= 1"),
    Mutant("zero capacity multiplier allowed", CONFIG, "self.capacity_multiplier > 0", "self.capacity_multiplier >= 0"),
    Mutant("zero tolerance allowed", CONFIG, "self.conservation_tolerance > 0", "self.conservation_tolerance >= 0"),
    Mutant("zero SLO allowed", CONFIG, "self.slo_ms > 0", "self.slo_ms >= 0"),
    Mutant("zero scaling norm allowed", CONFIG, "self.scaling_cost_norm > 0", "self.scaling_cost_norm >= 0"),
    # Rules that must not become too strict
    Mutant("zero traffic rejected", CONFIG, "self.default_traffic_rps >= 0", "self.default_traffic_rps > 0"),
    Mutant("retry coefficient 1 rejected", CONFIG, "0 <= self.retry_coefficient <= 1", "0 <= self.retry_coefficient < 1"),
    Mutant("threshold 1 rejected", CONFIG, "0 <= self.db_contention_threshold <= 1", "0 <= self.db_contention_threshold < 1"),
    Mutant("zero cap rejected", CONFIG, "self.slo_violation_cap >= 0", "self.slo_violation_cap > 0"),
    Mutant("zero weight rejected", CONFIG, "getattr(self, name) >= 0", "getattr(self, name) > 0"),
]

# One mutant per SimulatorConfig field. The name is the field name.
CONFIG_FIELDS = [
    Mutant("substeps_per_control_step", CONFIG, "substeps_per_control_step: int = 5", "substeps_per_control_step: int = 4"),
    Mutant("control_horizon", CONFIG, "control_horizon: int = 360", "control_horizon: int = 359"),
    Mutant("default_traffic_rps", CONFIG, "default_traffic_rps: float = 100.0", "default_traffic_rps: float = 101.0"),
    Mutant("service_rate", CONFIG, "(100.0, 120.0, 80.0, 50.0, 300.0)", "(101.0, 120.0, 80.0, 50.0, 300.0)"),
    Mutant("queue_capacity", CONFIG, "(2000.0, 2000.0, 1500.0, 1000.0, 3000.0)", "(2001.0, 2000.0, 1500.0, 1000.0, 3000.0)"),
    Mutant("base_latency_ms", CONFIG, "(5.0, 8.0, 15.0, 25.0, 10.0)", "(6.0, 8.0, 15.0, 25.0, 10.0)"),
    Mutant("min_replicas", CONFIG, "min_replicas: tuple[int, ...] = (1, 1, 1, 1, 1)", "min_replicas: tuple[int, ...] = (2, 1, 1, 1, 1)"),
    Mutant("max_replicas", CONFIG, "max_replicas: tuple[int, ...] = (10, 10, 10, 10, 1)", "max_replicas: tuple[int, ...] = (9, 10, 10, 10, 1)"),
    Mutant("initial_replicas", CONFIG, "initial_replicas: tuple[int, ...] = (2, 2, 2, 2, 1)", "initial_replicas: tuple[int, ...] = (3, 2, 2, 2, 1)"),
    Mutant("boot_seconds", CONFIG, "boot_seconds: tuple[int, ...] = (20, 20, 20, 20, 20)", "boot_seconds: tuple[int, ...] = (21, 20, 20, 20, 20)"),
    Mutant("termination_seconds", CONFIG, "termination_seconds: tuple[int, ...] = (5, 5, 5, 5, 5)", "termination_seconds: tuple[int, ...] = (6, 5, 5, 5, 5)"),
    Mutant("timeout_seconds", CONFIG, "timeout_seconds: tuple[int, ...] = (5, 5, 5, 5, 5)", "timeout_seconds: tuple[int, ...] = (6, 5, 5, 5, 5)"),
    Mutant("contention_kappa", CONFIG, "(0.0, 1.0, 1.5, 2.5, 0.0)", "(0.0, 1.1, 1.5, 2.5, 0.0)"),
    Mutant("call_multiplicity", CONFIG, "(0.0, 1.0, 0.8, 0.2, 0.0),", "(0.0, 1.0, 0.7, 0.2, 0.0),"),
    Mutant("db_contention_threshold", CONFIG, "db_contention_threshold: float = 0.50", "db_contention_threshold: float = 0.40"),
    Mutant("capacity_multiplier", CONFIG, "capacity_multiplier: float = 1.0", "capacity_multiplier: float = 1.1"),
    Mutant("retry_coefficient", CONFIG, "retry_coefficient: float = 0.5", "retry_coefficient: float = 0.6"),
    Mutant("conservation_tolerance", CONFIG, "conservation_tolerance: float = 1e-9", "conservation_tolerance: float = 1e-6"),
    Mutant("cluster_quota", CONFIG, "cluster_quota: int = 28", "cluster_quota: int = 29"),
    Mutant("cooldown_control_steps", CONFIG, "cooldown_control_steps: int = 6", "cooldown_control_steps: int = 5"),
    Mutant("action_min", CONFIG, "action_min: int = -1", "action_min: int = -2"),
    Mutant("action_max", CONFIG, "action_max: int = 2", "action_max: int = 3"),
    Mutant("slo_ms", CONFIG, "slo_ms: float = 250.0", "slo_ms: float = 300.0"),
    Mutant("slo_violation_cap", CONFIG, "slo_violation_cap: float = 5.0", "slo_violation_cap: float = 6.0"),
    Mutant("weight_slo", CONFIG, "weight_slo: float = 2.0", "weight_slo: float = 3.0"),
    Mutant("weight_failure", CONFIG, "weight_failure: float = 5.0", "weight_failure: float = 4.0"),
    Mutant("weight_resources", CONFIG, "weight_resources: float = 0.5", "weight_resources: float = 0.4"),
    Mutant("weight_scaling", CONFIG, "weight_scaling: float = 0.2", "weight_scaling: float = 0.3"),
    Mutant("scaling_cost_norm", CONFIG, "scaling_cost_norm: float = 8.0", "scaling_cost_norm: float = 9.0"),
]


class Result(NamedTuple):
    mutant: Mutant
    status: str
    failures: list


def run_suite(mutant: Optional[Mutant]) -> Result:
    """Run the suite on a temporary copy, applying the mutant if one is given."""
    with tempfile.TemporaryDirectory(prefix="stage1a-mutant-") as directory:
        root = Path(directory)
        for folder in ("sim", "tests"):
            shutil.copytree(ROOT / folder, root / folder, ignore=shutil.ignore_patterns("__pycache__"))
        if mutant is not None:
            target = root / mutant.path
            source = target.read_text()
            if source.count(mutant.old) != 1:
                return Result(mutant, "BAD PATTERN", [])
            target.write_text(source.replace(mutant.old, mutant.new))
        try:
            completed = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "--tb=no", "-p", "no:cacheprovider"],
                cwd=root, capture_output=True, text=True, timeout=TIMEOUT_SECONDS,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        except subprocess.TimeoutExpired:
            return Result(mutant, "killed (timeout)", [])
    if completed.returncode == 0:
        return Result(mutant, "SURVIVED", [])
    failures = re.findall(r"^(?:FAILED|ERROR) \S*?::(\S+)", completed.stdout, flags=re.M)
    return Result(mutant, "killed" if failures else "killed (collection error)", failures)


def describe(result: Result) -> str:
    """Prefer a behavioral test name over the spec-table check."""
    functions = list(dict.fromkeys(failure.split("[")[0] for failure in result.failures))
    behavioral = [name for name in functions if name != SPEC_TABLE_TEST]
    first = behavioral[0] if behavioral else (functions[0] if functions else "-")
    return f"{len(result.failures):4d} failed  {first}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", choices=("logic", "config", "all"), default="all")
    parser.add_argument("--jobs", type=int, default=min(6, os.cpu_count() or 1))
    args = parser.parse_args()

    baseline = run_suite(None)
    if baseline.status != "SURVIVED":
        print(f"Baseline suite must pass first. Status: {baseline.status}, failures: {baseline.failures}")
        return 2
    print("Baseline suite passed.\n")

    groups = {"logic": LOGIC, "config": CONFIG_FIELDS}
    selected = groups if args.group == "all" else {args.group: groups[args.group]}
    clean = True
    for name, mutants in selected.items():
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(run_suite, mutants))
        print(f"== {name} ==")
        for result in results:
            print(f"{result.status:26} {result.mutant.name:42} {describe(result)}")
        killed = sum(result.status.startswith("killed") for result in results)
        print(f"\n{name}: {killed}/{len(results)} killed\n")
        clean = clean and killed == len(results)
    return 0 if clean else 1


if __name__ == "__main__":
    sys.exit(main())
