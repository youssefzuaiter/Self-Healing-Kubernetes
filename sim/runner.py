"""Stage 1B reproducible experiment runner (docs/SPEC_02_BASELINES.md Section 3).

Every episode length comes from the config: this module contains no literal horizon.
Run the full benchmark with ``python -m sim.runner [--output-dir DIR]``.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import errno
import json
import math
import os
import subprocess
import unicodedata
from numbers import Integral
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from sim.baselines import BaseController, SimHPABaseline, StaticBaseline, TunedReactiveThresholdController
from sim.config import SimulatorConfig
from sim.env import MicroserviceMeshEnv
from sim.shield import convert_proposal
from sim.workloads import WorkloadTrace, benchmark_suite

__all__ = ["run_episode", "run_benchmark_suite", "json_safe", "format_table", "main"]

MANIFEST_SCHEMA_VERSION = "1.0.0"
SPEC_VERSIONS = {"simulator": "1.0.0", "baselines": "1.0.0"}
# Each reward component and the config weight that multiplies it in _reward().
COST_WEIGHTS = (
    ("slo_violation", "weight_slo"),
    ("failure_cost", "weight_failure"),
    ("resource_cost", "weight_resources"),
    ("scaling_cost", "weight_scaling"),
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNSAFE_NAME_PARTS = ("/", "\\", "..", "\x00")
MAX_FILE_NAME_BYTES = 255
JSON_LEAVES = (type(None), bool, int, float, str)


def _json_key(key: Any) -> str:
    if type(key) is not str:
        raise TypeError(f"manifest keys must be plain strings, got key {key!r} of type {type(key).__name__}")
    return key


def json_safe(value: Any) -> Any:
    """Convert to strictly JSON-native types, or raise (SPEC_02 Section 3.3).

    Mappings, lists, tuples, sets, NumPy arrays and NumPy scalars are converted, and
    everything inside them is checked too. Keys must be exactly ``str``, and leaves
    exactly None, bool, int, finite float or str. Nothing is ever stringified.
    """
    if isinstance(value, Mapping):
        return {_json_key(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, frozenset, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        unwrapped = value.item()
        if isinstance(unwrapped, np.generic):  # for example longdouble, which has no Python equivalent
            raise TypeError(f"manifest values must be JSON-native, got {type(unwrapped).__name__}")
        return json_safe(unwrapped)
    if type(value) not in JSON_LEAVES:
        raise TypeError(f"manifest values must be JSON-native, got {type(value).__name__}")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"manifest values must be finite, got {value}")
    return value


def _git_info() -> dict[str, Any]:
    """HEAD commit and dirty flag, only when the project root is the repository top level."""
    def git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.SubprocessError):
            return None
        return result.stdout.strip() if result.returncode == 0 else None

    top_level = git("rev-parse", "--show-toplevel")
    commit = git("rev-parse", "HEAD")
    if top_level is None or commit is None or Path(top_level).resolve() != PROJECT_ROOT:
        return {"commit": "unavailable", "dirty": None}
    status = git("status", "--porcelain")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def _summary(latencies: np.ndarray) -> dict[str, float]:
    p50, p95, p99 = np.percentile(latencies, [50, 95, 99])
    return {"mean": float(latencies.mean()), "p50": float(p50), "p95": float(p95), "p99": float(p99),
            "max": float(latencies.max())}


def _check_controller(controller: Any) -> None:
    if not isinstance(controller, BaseController):
        raise TypeError(f"controller must be a BaseController, got {type(controller).__name__}")
    json_safe({"name": controller.name, "params": controller.params})


def _check_trace(trace: Any, cfg: SimulatorConfig) -> None:
    if not isinstance(trace, WorkloadTrace):
        raise TypeError(f"trace must be a WorkloadTrace, got {type(trace).__name__}")
    if trace.values.shape != (cfg.internal_horizon,):
        raise ValueError(
            f"trace '{trace.name}' has {trace.values.shape[0]} substeps, but the config's horizon "
            f"is {cfg.internal_horizon} substeps"
        )
    json_safe({"name": trace.name, "params": trace.params})


def _check_run(cfg: SimulatorConfig, seed: Any) -> None:
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError(f"seed must be an integer >= 0, got {seed!r}")
    json_safe(dataclasses.asdict(cfg))


def run_episode(controller: BaseController, trace: WorkloadTrace, config: Optional[SimulatorConfig] = None,
                seed: int = 0) -> dict[str, Any]:
    """Run one controller on one trace for the config's full horizon and return its manifest."""
    cfg = config if config is not None else SimulatorConfig()
    _check_controller(controller)
    _check_trace(trace, cfg)
    _check_run(cfg, seed)

    controller.reset()
    env = MicroserviceMeshEnv(traffic_trace=trace.values, config=cfg)
    observation, _ = env.reset(seed=int(seed))

    rewards: list[float] = []
    components: dict[str, list[float]] = {key: [] for key, _ in COST_WEIGHTS}
    total_costs: list[float] = []
    latencies, failures, occupancy = [], [], []
    scaling_events = overrides = 0
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        raw = controller.propose(observation, env.config, env.step_index)
        converted = convert_proposal(raw, env.config)
        token = env.shield.approve(raw)
        if converted is not None and token.deltas != converted:
            overrides += 1
        observation, reward, terminated, truncated, info = env.step(token)
        rewards.append(float(reward))
        for key, _ in COST_WEIGHTS:
            components[key].append(float(info["reward_components"][key]))
        total_costs.append(float(info["reward_components"]["total_cost"]))
        latencies.append(info["substeps"]["e2e_latency_ms"])
        failures.append(info["substeps"]["e2e_failure"])
        occupancy.append(info["substeps"]["occupancy"])
        scaling_events += int(np.abs(info["approved_delta"]).sum())

    if len(rewards) != cfg.control_horizon:
        raise RuntimeError(f"episode ran {len(rewards)} control steps, expected {cfg.control_horizon}")

    latency = np.concatenate(latencies)
    failure = np.concatenate(failures)
    unweighted = {key: float(np.sum(values)) for key, values in components.items()}
    weighted = {key: getattr(cfg, weight) * unweighted[key] for key, weight in COST_WEIGHTS}
    weighted["total_cost"] = float(np.sum(total_costs))
    total_reward = float(np.sum(rewards))
    values = trace.values
    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "spec_versions": dict(SPEC_VERSIONS),
        "git": _git_info(),
        "seed": int(seed),
        "config": dataclasses.asdict(cfg),
        "controller": {"name": controller.name, "params": controller.params},
        "workload": {
            "name": trace.name, "params": trace.params, "sha256": trace.sha256, "horizon": int(len(values)),
            "min_rps": float(values.min()), "max_rps": float(values.max()), "mean_rps": float(values.mean()),
        },
        "metrics": {
            "control_steps_executed": len(rewards),
            "substeps_executed": int(latency.size),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "total_reward": total_reward,
            "mean_reward": total_reward / len(rewards),
            "unweighted_cost_sums": unweighted,
            "weighted_cost_sums": weighted,
            # The SPEC_01 Section 24 modeled latency proxy per substep, not empirical request percentiles.
            "substep_latency_proxy_ms": _summary(latency),
            "mean_e2e_failure_proxy": float(failure.mean()),
            "max_e2e_failure_proxy": float(failure.max()),
            "slo_violation_substeps": int(np.sum(latency > cfg.slo_ms)),
            "total_pod_seconds": int(np.sum(np.concatenate(occupancy))),
            "total_scaling_events": scaling_events,
            "shield_arbitration_override_count": overrides,
            # Cumulative within the episode (SPEC_02 Section 0.1): take the last value, never sum.
            "shield_fallback_count": int(info["fallback_count"]),
        },
    }
    return json_safe(manifest)


def _check_name(kind: str, name: Any) -> None:
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} must be a non-empty string, got {name!r}")
    if any(part in name for part in UNSAFE_NAME_PARTS):
        raise ValueError(f"{kind} {name!r} contains a path separator, '..' or a null byte")


def _plan_destinations(controllers: Sequence[BaseController], traces: Sequence[WorkloadTrace],
                       directory: Path) -> list[Path]:
    """Section 3.2: check every output file before anything is run, created or written."""
    for controller in controllers:
        _check_name("controller name", controller.name)
    for trace in traces:
        _check_name("workload name", trace.name)
    root, destinations, seen = directory.resolve(), [], set()
    for controller in controllers:
        for trace in traces:
            file_name = f"{controller.name}__{trace.name}__{trace.sha256[:12]}.json"
            _check_name("file name", file_name)
            if len(file_name.encode("utf-8")) > MAX_FILE_NAME_BYTES:  # unencodable names raise ValueError too
                raise ValueError(f"file name {file_name!r} is longer than {MAX_FILE_NAME_BYTES} bytes")
            destination = directory / file_name
            if destination.resolve().parent != root:
                raise ValueError(f"{destination} is outside the output directory {directory}")
            # Some file systems ignore case and Unicode normalization, so compare without them.
            key = unicodedata.normalize("NFC", file_name).casefold()
            if key in seen:
                raise ValueError(f"two episodes would both write {file_name}")
            seen.add(key)
            destinations.append(destination)
    for destination in destinations:
        if os.path.lexists(destination):
            raise FileExistsError(errno.EEXIST, "manifest already exists", str(destination))
    return destinations


def _write_new_manifest(destination: Path, manifest: Mapping[str, Any]) -> None:
    """Create ``destination`` exclusively. A failed write removes only the file created here."""
    payload = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if json.loads(payload) != manifest:
        raise ValueError(f"the manifest for {destination.name} does not survive a JSON round trip")
    handle = destination.open("x", encoding="utf-8")  # FileExistsError here leaves nothing to clean up
    created = None
    try:
        with handle as stream:
            created = os.lstat(destination)
            stream.write(payload)
    except BaseException:
        with contextlib.suppress(OSError):  # the write error below matters more than a failed cleanup
            if created is not None and os.path.samestat(created, os.lstat(destination)):
                destination.unlink()
        raise


def run_benchmark_suite(controllers: Iterable[BaseController], traces: Iterable[WorkloadTrace],
                        config: Optional[SimulatorConfig] = None, seed: int = 0,
                        output_dir: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Run every controller on every trace, controllers in the outer loop (Section 3.2).

    Everything that can be known to be wrong is found before the first episode runs
    and before the file system is touched.
    """
    controllers, traces = list(controllers), list(traces)
    cfg = config if config is not None else SimulatorConfig()
    _check_run(cfg, seed)
    for controller in controllers:
        _check_controller(controller)
    for trace in traces:
        _check_trace(trace, cfg)
    destinations = None
    if output_dir is not None:
        directory = Path(output_dir)
        destinations = iter(_plan_destinations(controllers, traces, directory))
    git_info = _git_info()  # read once, before this run's own output can look like a change
    if destinations is not None:
        directory.mkdir(parents=True, exist_ok=True)
    manifests = []
    for controller in controllers:
        for trace in traces:
            manifest = run_episode(controller, trace, config=cfg, seed=seed)
            manifest["git"] = dict(git_info)
            manifests.append(manifest)
            if destinations is not None:
                _write_new_manifest(next(destinations), manifest)
    return manifests


def format_table(manifests: Sequence[Mapping[str, Any]]) -> str:
    """A Markdown table of the main measured metrics, one row per episode."""
    header = ("controller", "workload", "total reward", "latency proxy p95 ms", "latency proxy p99 ms",
              "SLO-violating substeps", "mean failure proxy", "pod-seconds", "scaling events",
              "shield overrides", "fallbacks")
    rows = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for manifest in manifests:
        metrics, latency = manifest["metrics"], manifest["metrics"]["substep_latency_proxy_ms"]
        cells = (manifest["controller"]["name"], manifest["workload"]["name"], f"{metrics['total_reward']:.2f}",
                 f"{latency['p95']:.1f}", f"{latency['p99']:.1f}", str(metrics["slo_violation_substeps"]),
                 f"{metrics['mean_e2e_failure_proxy']:.4f}", str(metrics["total_pod_seconds"]),
                 str(metrics["total_scaling_events"]), str(metrics["shield_arbitration_override_count"]),
                 str(metrics["shield_fallback_count"]))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Stage 1B baselines on the benchmark workloads.")
    parser.add_argument("--output-dir", help="write one JSON manifest per episode here")
    parser.add_argument("--seed", type=int, default=0, help="environment reset seed")
    parser.add_argument("--workload-seed", type=int, default=42, help="seed for the Poisson workload")
    args = parser.parse_args(argv)
    controllers = [StaticBaseline(), SimHPABaseline(), TunedReactiveThresholdController()]
    manifests = run_benchmark_suite(controllers, benchmark_suite(args.workload_seed), seed=args.seed,
                                    output_dir=args.output_dir)
    print(format_table(manifests))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
