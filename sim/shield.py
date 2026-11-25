"""Deterministic safety shield and ApprovedAction token (SPEC_01 Sections 26-30).

The shield turns a raw proposal into an approved delta vector and issues a token.
It reads simulator state but never writes it: state changes only in ``env.step``.

Security limit: the token stops API-level bypasses. It is not unforgeable against
arbitrary code running in the same Python process, which can reach private state.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

import numpy as np

from sim.config import NUM_SCALABLE, SimulatorConfig

if TYPE_CHECKING:
    from sim.env import MicroserviceMeshEnv

HOLD = (0,) * NUM_SCALABLE


@dataclass(frozen=True)
class ApprovedAction:
    """A one-use capability token. Consumed status lives in the environment.

    Building one by hand gains nothing: ``env.step`` accepts only the exact object
    recorded in its own ledger when the shield issued it.
    """

    env_id: str
    step_index: int
    nonce: str
    deltas: tuple[int, ...]


def round_component(value: Any) -> int:
    """Section 26.1: NumPy half-to-even rounding, before clipping."""
    return int(np.round(value))


def convert_proposal(raw: Any, config: SimulatorConfig) -> Optional[tuple[int, ...]]:
    """Convert a raw proposal to clipped integer deltas, or None if invalid.

    Validity is all-or-nothing (Section 26.1). Booleans count as non-numeric.
    """
    if isinstance(raw, (list, tuple)) and any(isinstance(item, (bool, np.bool_)) for item in raw):
        return None
    try:
        vector = np.asarray(raw)
    except (TypeError, ValueError):
        return None
    if vector.shape != (NUM_SCALABLE,):
        return None
    if not (np.issubdtype(vector.dtype, np.integer) or np.issubdtype(vector.dtype, np.floating)):
        return None
    if not np.all(np.isfinite(vector)):
        return None
    return tuple(min(max(round_component(item), config.action_min), config.action_max) for item in vector)


def arbitrate(
    proposal: Sequence[int],
    desired: Sequence[int],
    pending: Sequence[int],
    cooldown: Sequence[int],
    occupancy: int,
    config: SimulatorConfig,
) -> tuple[int, ...]:
    """Apply replica bounds, cooldown and quota to a converted proposal."""
    approved = [0] * NUM_SCALABLE

    # Scale-downs first (Section 28). Only cancelled pods free quota now; a
    # terminated pod stays in occupancy until its timer ends (Section 16).
    for node, change in enumerate(proposal):
        if change >= 0 or cooldown[node] > 0:
            continue
        allowed = min(-change, int(desired[node]) - config.min_replicas[node])
        if allowed > 0:
            approved[node] = -allowed
            occupancy -= min(allowed, int(pending[node]))

    # Round-robin scale-ups: one pod per service per round (Section 28).
    free = config.cluster_quota - occupancy
    wanted = [max(change, 0) for change in proposal]
    allocated = True
    while free > 0 and allocated:
        allocated = False
        for node in range(NUM_SCALABLE):
            at_maximum = int(desired[node]) + approved[node] >= config.max_replicas[node]
            if free > 0 and wanted[node] > 0 and not at_maximum:
                approved[node] += 1
                wanted[node] -= 1
                free -= 1
                allocated = True
    return tuple(approved)


class SafetyShield:
    """Approves proposals for exactly one environment."""

    def __init__(self, env: "MicroserviceMeshEnv") -> None:
        self._env = env
        self._fallback_count = 0

    @property
    def fallback_count(self) -> int:
        """Proposals rejected whole since the last reset."""
        return self._fallback_count

    def reset(self) -> None:
        self._fallback_count = 0

    def approve(self, raw_proposal: Any) -> ApprovedAction:
        """Convert, arbitrate and issue a token for the current control step."""
        env = self._env
        proposal = convert_proposal(raw_proposal, env.config)
        if proposal is None:
            self._fallback_count += 1
            proposal = HOLD
        deltas = arbitrate(proposal, env.desired, env.pending, env.cooldown, env.cluster_occupancy, env.config)
        token = ApprovedAction(env.env_id, env.step_index, uuid.uuid4().hex, deltas)
        env._record_issued_token(token)
        return token
