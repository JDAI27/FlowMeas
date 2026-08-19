"""Validation-tier vocabulary for the split exact-small / scalable-large
evaluation design.

The existing ``evaluator_mode`` axis answers *what kind of evaluator runs*
(exact full-state vs. scalable structural).  This module adds an orthogonal
``validation_tier`` axis that answers *what quality of reference the run
produces*:

  - ``exact``           – full-state exact expectation; ground truth for
                          small systems.  Paired with ``EVALUATOR_MODE_EXACT_SMALL``.
  - ``structural``      – structural circuit metrics only (no scalar energy
                          reference).  Acceptable for pilot benchmarking;
                          the default tier for ``EVALUATOR_MODE_SCALABLE_LARGE``
                          until DMRG is wired in.
  - ``dmrg_reference``  – DMRG-backed scalar energy reference, no full-state
                          allocation.  Required for the final
                          "50+ qubit Hubbard ready" claim.

Keeping the tier dimension orthogonal to the evaluator mode means
(structural report) and (DMRG path) can plug in without
re-architecting the dispatch surface — supplies an actual scalar
``dmrg_reference_energy`` (an ``Optional[float]``), and the tier
mechanically becomes ``dmrg_reference`` only when that scalar is present.
A bare capability flag is intentionally NOT enough to promote the tier:
that would let a structural-only report advertise scalar-backed
final-readiness while still emitting ``energy_estimate: null`` (finding).

Implementation note: the evaluator-mode literals below are intentionally
hardcoded copies of ``code.main.EVALUATOR_MODE_*`` to avoid importing
``main`` at module load (``main`` is large and itself imports modules
that import this one indirectly via reports). A rename of the ``main``
constants must be mirrored here or the tier dispatch silently desyncs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Tier constants
# ---------------------------------------------------------------------------

EVALUATION_TIER_EXACT = "exact"
EVALUATION_TIER_STRUCTURAL = "structural"
EVALUATION_TIER_DMRG_REFERENCE = "dmrg_reference"

VALID_EVALUATION_TIERS = frozenset(
    {
        EVALUATION_TIER_EXACT,
        EVALUATION_TIER_STRUCTURAL,
        EVALUATION_TIER_DMRG_REFERENCE,
    }
)


# Mirror of ``code.main.EVALUATOR_MODE_*``; cross-validated by tests.
_EVALUATOR_MODE_EXACT_SMALL = "exact_small"
_EVALUATOR_MODE_SCALABLE_LARGE = "scalable_large"


# ---------------------------------------------------------------------------
# Tier specs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EvaluationTierSpec:
    """Immutable description of a validation-tier contract.

    Fields:
      tier
        The canonical string name (one of ``VALID_EVALUATION_TIERS``).
      provides_scalar_energy
        Whether the tier yields a scalar energy reference suitable for
        side-by-side comparison with the trained policy's estimates.
      sufficient_for_final_readiness_claim
        Whether a run using this tier is, on its own, enough to back a
        final "50+ qubit Hubbard ready" claim.  ``structural`` is NOT
        sufficient.
      description
        Short human-readable summary used in reports + logs.
    """

    tier: str
    provides_scalar_energy: bool
    sufficient_for_final_readiness_claim: bool
    description: str


_TIER_SPECS: Dict[str, EvaluationTierSpec] = {
    EVALUATION_TIER_EXACT: EvaluationTierSpec(
        tier=EVALUATION_TIER_EXACT,
        provides_scalar_energy=True,
        sufficient_for_final_readiness_claim=True,
        description=(
            "Full-state exact expectation; reference truth for small systems."
        ),
    ),
    EVALUATION_TIER_STRUCTURAL: EvaluationTierSpec(
        tier=EVALUATION_TIER_STRUCTURAL,
        provides_scalar_energy=False,
        sufficient_for_final_readiness_claim=False,
        description=(
            "Structural circuit metrics only; no scalar energy reference. "
            "Acceptable for pilot benchmarking, but the final readiness "
            "claim on the frozen benchmark workload requires a DMRG-backed "
            "reference."
        ),
    ),
    EVALUATION_TIER_DMRG_REFERENCE: EvaluationTierSpec(
        tier=EVALUATION_TIER_DMRG_REFERENCE,
        provides_scalar_energy=True,
        sufficient_for_final_readiness_claim=True,
        description=(
            "DMRG-backed scalar energy reference; no full-state allocation. "
            "Required for the final '50+ qubit Hubbard ready' claim."
        ),
    ),
}


def get_tier_spec(tier: str) -> EvaluationTierSpec:
    """Return the spec for a tier; raises ``ValueError`` on unknown tiers."""
    try:
        return _TIER_SPECS[tier]
    except KeyError as err:
        raise ValueError(
            f"Unknown evaluation tier {tier!r}; valid tiers: "
            f"{sorted(VALID_EVALUATION_TIERS)}"
        ) from err


# ---------------------------------------------------------------------------
# Mode → tier selection
# ---------------------------------------------------------------------------

def select_validation_tier(
    *,
    evaluator_mode: str,
    dmrg_reference_energy: Any = None,
) -> str:
    """Map an evaluator mode + actual DMRG scalar to the validation tier.

    Mapping:
      - ``exact_small``     → ``exact``       (always)
      - ``scalable_large``  → ``dmrg_reference`` iff ``dmrg_reference_energy``
                              normalizes to a non-None finite float (i.e. an
                              actual scalar reference has been computed and
                              attached)
                              else ``structural``

    The ``dmrg_reference_energy`` argument is intentionally typed ``Any``:
    callers may pass values from heterogeneous sources (config attribute,
    JSON-loaded sidecar, side-channel dict).  The function runs
    ``coerce_dmrg_reference_energy`` on the input so it cannot be tricked
    into promoting the tier by ``False``, ``"false"``, ``NaN``, or any
    non-numeric placeholder — that hardening previously lived only in the
    ``get_validation_tier_metadata`` wrapper and could be bypassed by
    direct callers (third-round review finding).

    Raises ``ValueError`` on unknown ``evaluator_mode``.
    """
    if evaluator_mode == _EVALUATOR_MODE_EXACT_SMALL:
        return EVALUATION_TIER_EXACT
    if evaluator_mode == _EVALUATOR_MODE_SCALABLE_LARGE:
        # Coerce inside the selector so direct callers cannot bypass the
        # scalar-presence contract.  ``coerce_dmrg_reference_energy`` is
        # defined later in this module; Python resolves the reference at
        # call time, so the forward reference is safe.
        normalized = coerce_dmrg_reference_energy(dmrg_reference_energy)
        return (
            EVALUATION_TIER_DMRG_REFERENCE
            if normalized is not None
            else EVALUATION_TIER_STRUCTURAL
        )
    raise ValueError(
        f"Unknown evaluator_mode {evaluator_mode!r}; expected "
        f"{_EVALUATOR_MODE_EXACT_SMALL!r} or "
        f"{_EVALUATOR_MODE_SCALABLE_LARGE!r}."
    )


def coerce_dmrg_reference_energy(value: Any) -> Optional[float]:
    """Normalise a candidate DMRG-reference energy to ``Optional[float]``.

    Returns ``None`` for ``None``, ``NaN``, ``inf``, booleans, or anything
    that cannot be cast to a finite float (e.g. strings like ``'false'``,
    an unset attribute, or a non-numeric placeholder).  This is the only
    sanctioned way to derive a DMRG scalar from a heterogeneous source
    (config attribute, JSON-loaded field, side-channel dict) and replaces
    the older bare-bool ``dmrg_reference_available`` flag.

    Boolean rejection is explicit because in Python ``bool`` is a subclass
    of ``int`` — without this check ``float(False)`` would yield ``0.0``
    and promote the tier to ``dmrg_reference`` from a JSON ``false`` value
    (second-round review finding).
    """
    if value is None:
        return None
    if isinstance(value, bool):
        # Reject explicitly: bool ⊂ int in Python, so float(False)==0.0
        # would otherwise be accepted as a valid scalar reference.
        return None
    try:
        energy = float(value)
    except (TypeError, ValueError):
        return None
    # NaN/inf are not scalar references.
    if energy != energy or energy in (float("inf"), float("-inf")):
        return None
    return energy


def get_validation_tier_metadata(
    *,
    evaluator_mode: str,
    dmrg_reference_energy: Optional[float] = None,
) -> Dict[str, Any]:
    """Return tier metadata suitable for inclusion in evaluation reports
    and hyperparameter records.

    The returned dict is purely additive over ``get_evaluator_mode_metadata``:
    consumers that don't know about validation tiers simply ignore the
    new keys.

    ``dmrg_reference_available`` in the returned dict is derived from
    whether ``dmrg_reference_energy`` is non-None — it is therefore an
    accurate "a scalar reference is attached" signal, not a capability
    flag that can lie about scalar presence.
    """
    energy = coerce_dmrg_reference_energy(dmrg_reference_energy)
    tier = select_validation_tier(
        evaluator_mode=evaluator_mode,
        dmrg_reference_energy=energy,
    )
    spec = get_tier_spec(tier)
    return {
        "validation_tier": tier,
        "provides_scalar_energy": spec.provides_scalar_energy,
        "sufficient_for_final_readiness_claim": (
            spec.sufficient_for_final_readiness_claim
        ),
        "tier_description": spec.description,
        "dmrg_reference_available": energy is not None,
        "dmrg_reference_energy": energy,
    }
