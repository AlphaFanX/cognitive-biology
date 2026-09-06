"""growth_program.py -- THE EMBRYONIC GROWTH-PROGRAM HEAD (cycle 19 of the embryo loop, 2026-09-01).

The composition instrument (medic.stage_composition, on the ordinal-fixed Carnegie reference) showed the
model's organ shares are STAGE-INDEPENDENT -- the condensation loop recruits every point organ to a flat
ORG_TARGET fraction -- while the real ladder MOVES: the heart is 12% of the embryo at CS10 and 1.5% at
CS23 (the body outgrows it), the liver balloons 0.3% -> 4.2% -> 8.3% (fetal haematopoiesis). This head
makes the targets CLOCK-GATED: per-family target fractions as piecewise-linear functions of the PRC2
clock, the same clock that already gates differentiation -- the master-gene growth bursts (heart NKX2-5
early expansion; liver the HGF/Runx1 haematopoietic balloon) read as measured canonical constants.

Constants are MEASURED (Amsterdam atlas voxel fractions per Carnegie stage; the bp3d-constants class, at
the embryo end of the ledger). The PRC2<->CS map is read off the movie's own clock schedule (frame stage
strings; deterministic in simulate).

MECHANISM SEMANTICS (honest limits):
* RECRUIT-ONLY: when the staged target falls below the family's current share, recruitment FREEZES and
  the family dilutes as the body keeps growing -- organs are outgrown, never de-differentiated. In a
  cloud whose births taper late, dilution is partial: the late-window EXCESS shrinks but does not vanish.
* Targets before a family exists (pre-sprout / pre-condensation-gate) cannot act -- emergence TIMING is
  its own mechanism (the unlock schedule), not this head.
* WIRED FAMILIES: Heart + Liver only (the two clean, well-segmented, large signals). Kidney/lung/spleen
  excesses and the early-gut deficit wait for the honest full-cloud census read (cloud_census.json) and
  their own cycles. Unwired families keep ORG_TARGET/GUT_TARGET.

Used by unified_embryo's condensation loop: target(fate, prc2, default) -> fraction of born.
"""
from __future__ import annotations
import numpy as np

# mean PRC2 of the movie's cloud frames mapped to each Carnegie stage (the deterministic clock schedule
# read from the shipped movie; the frame->stage rule is canon_frame_score._pick's even split).
PRC2_OF_CS = {
    "CS09": 0.590, "CS10": 0.535, "CS11": 0.487, "CS12": 0.445, "CS13": 0.410,
    "CS15": 0.375, "CS16": 0.345, "CS17": 0.317, "CS18": 0.293, "CS20": 0.273,
    "CS21": 0.250, "CS23": 0.233,
}

# measured organ-group voxel fractions of the labelled embryo (data/organ_cascade/stage_composition.json,
# reference side, ordinal-fixed loader) -- the full ladder recorded for reference; only STAGED_FAMILIES
# are wired into the build.
LADDER = {
    "Heart": {"CS09": 0.047, "CS10": 0.121, "CS11": 0.110, "CS12": 0.100, "CS13": 0.077,
              "CS15": 0.048, "CS16": 0.037, "CS17": 0.021, "CS18": 0.023, "CS20": 0.016,
              "CS21": 0.014, "CS23": 0.015},
    "Liver": {"CS12": 0.003, "CS13": 0.011, "CS15": 0.042, "CS16": 0.036, "CS17": 0.030,
              "CS18": 0.039, "CS20": 0.057, "CS21": 0.050, "CS23": 0.083},
    "Kidney": {"CS12": 0.0063, "CS13": 0.0047, "CS15": 0.0029, "CS16": 0.0031, "CS17": 0.0034,
               "CS18": 0.0044, "CS20": 0.0033, "CS21": 0.0039, "CS23": 0.0058},
    "Spleen": {"CS17": 0.0001, "CS18": 0.0001, "CS20": 0.0001, "CS21": 0.0001, "CS23": 0.0001},
    # LIMB (cycle 68 -- the ladder's next customer, the cycle-66 pool correction): the limb-program
    # knobs were fitted at n=9k (limb 6.8% after the 2026-07-30 HESTA rebalance) but bud CONVERSION
    # is geometric, so at the movie's 120k the family drifted to 2.3% and the autopod draw took ~90%
    # of the below-knee pool. SINGLE MEASURED ANCHOR, declared: HESTA CS12-13 limb share 4.7%
    # (limb_search.json target_frac_hesta; the Amsterdam atlas labels only limb BONES -- object
    # mismatch -- and MOSTA has no limb region label, so no staged whole-limb ladder exists yet).
    # Held flat across the window; a rising tail awaits a measured source.
    "Limb Bud": {"CS13": 0.047, "CS23": 0.047},
}
STAGED_FAMILIES = ("Heart", "Liver", "Kidney", "Spleen", "Limb Bud")

# REPRESENTABILITY FLOOR (cycle 23): the measured embryonic kidney is 0.3-0.6% of the embryo and the
# spleen ~0.01% -- real, but below the model's cell-count resolution (a scoreable bean needs ~400
# cells/side; the adult organs are shaped from these cells). Targets clamp to the floor, shrinking the
# kidney's ~4x excess (1.6% -> ~0.8%) and holding the spleen at its scoreable size (the kidney-staged
# rebalance halved it to 0.46% and the adult trace collapsed 84 -> 71). An honest instrument limit,
# not anatomy.
FLOOR = {"Kidney": 0.008, "Spleen": 0.007}

_INTERP = {}


def _table(fam):
    if fam not in _INTERP:
        pts = sorted((PRC2_OF_CS[cs], f) for cs, f in LADDER[fam].items() if cs in PRC2_OF_CS)
        _INTERP[fam] = (np.array([p for p, _ in pts]), np.array([f for _, f in pts]))
    return _INTERP[fam]


def target(fam, prc2, default=None):
    """Staged target fraction-of-born for `fam` at clock position prc2 (falls back to `default` for
    unwired families). PRC2 DECREASES through development, so the ladder is indexed descending."""
    if fam not in STAGED_FAMILIES:
        return default
    x, y = _table(fam)
    return max(float(np.interp(prc2, x, y)), FLOOR.get(fam, 0.0))


if __name__ == "__main__":
    for fam in STAGED_FAMILIES:
        print(fam, " ".join(f"{cs}:{100*target(fam, p):.1f}%" for cs, p in PRC2_OF_CS.items()))
