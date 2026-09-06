"""digital_ray_head.py -- THE DIGITAL RAYS AND THEIR INTERZONES (cycle 82f, 2026-09-06): the autopod
completion ladder's condensation rungs, run INSIDE simulate on the limb-bud cells.

The completion instrument (medic.foot_completion) had read PLATE at every stage and RAYS NEVER since the
allocation landed (cycle 68): the paddle had the cells but nothing condensed them. The rungs are cell
behaviours with master genes and clock windows, so they belong in the growth loop, not in a display pass:

  RAYS       SOX9 digital-ray condensation, CS17 -> CS20 (prc2 0.317 -> 0.273): the paddle's distal cells
             sort into five rays across the fan (the lateral-inhibition wavelength of the limb head sets
             the count) and each ray condenses toward its own centreline.
  INTERZONES GDF5 joint interzones, CS18 -> CS21 (prc2 0.293 -> 0.250): two planes at 1/3 and 2/3 of the
             autopod's length empty out -- the joint rows that make three phalangeal segments.
  SEPARATION follows from the rays' own condensation (the interdigital cells are the rays' recruits here;
             BMP-driven interdigital apoptosis is the successor once the deaths run in simulate).

BILATERAL BY CONSTRUCTION: the scored and displayed clouds are _symmetrize'd (every cell folded to |z| and
reflected), so each side carries BOTH paddles superimposed -- rays laid independently per side blurred
each other (the first read: 2-4 flickering peaks). Each limb pair is therefore patterned as ONE folded
paddle (z -> |z|), the same fan axis, ray centrelines and interzone planes on both sides, and the
displacement is mirrored back per cell. Frame-agnostic: the folded pair's own principal axis (oriented
away from the body centroid) is proximodistal; the fan is the in-plane principal axis of the distal
cells. Deterministic (sorting only). Read-only on every non-limb cell. Runs AFTER the step's mechanical
relaxation (the same-fate cadherin spring of the paddle restored a 10%/step pull completely between
steps), so the emitted frame and the next step start condensed.
"""
from __future__ import annotations
import numpy as np

RAY_OPEN, RAY_FULL = 0.317, 0.273     # CS17 -> CS20 (growth_program.PRC2_OF_CS)
SEG_OPEN, SEG_FULL = 0.293, 0.250     # CS18 -> CS21
N_RAYS = 5
DISTAL_FRAC = 0.12                    # the autopod = the distal 12% of the bud along its own axis
GAIN_RAY, GAIN_SEG = 0.70, 0.70       # per-step fractions (measured at 120k: 0.10 was restored completely
#                                       between steps by the relaxation; 0.45 left within/total 0.10-0.12
#                                       and 3-5 peaks; 0.70 takes it below the peak-resolution floor)
SEG_PULL = 0.6                        # per-call fraction of the distance to the segment (phalanx) centre
#                                       (measured: 0.6 -> hand free-digits at CS23 on two frames, foot
#                                       separation with 5 rays / segments 2; 0.8 was no better for the
#                                       foot and cost the hand one of its free-digit frames -- the foot's
#                                       segment rung is the open item, not this knob)


def bud_axis(B, body_c):
    """proximodistal unit axis of one bud: PCA-1 of its cells, pointing away from the body centroid."""
    C = B - B.mean(0)
    ax = np.linalg.svd(C, full_matrices=False)[2][0]
    if float((B.mean(0) - body_c) @ ax) < 0:
        ax = -ax
    return ax


IZ_HALF = 0.10                        # gap-mode interzone half-width, fraction of the autopod length
FRAME_RAY, FRAME_SEG, FRAME_PULL = 0.80, 1.0, None   # the per-FRAME transform (movie cloud loop + run_full):
#                                       a pure function of the raw cloud and the clock, applied once per
#                                       frame at the window's strength -- no relaxation to fight, no
#                                       build-side perturbation (the in-simulate variant was measured and
#                                       reverted, see unified_embryo). 0.80 keeps 20% of each cell's
#                                       in-plane offset = the ray's own width.


def apply_frame(P, fid, prc2, limb_id):
    """The movie/instrument path: condense this FRAME's paddles by the clock's strength. In place on P."""
    return apply(P, fid, prc2, limb_id, gain_ray=FRAME_RAY, gain_seg=FRAME_SEG, seg_pull=FRAME_PULL)


def apply(P, fid, prc2, limb_id, gain_ray=GAIN_RAY, gain_seg=GAIN_SEG, seg_pull=SEG_PULL):
    """In place on P (the positions of the born cells). Returns the number of autopod cells acted on."""
    if prc2 > RAY_OPEN:
        return 0
    lm = np.where(fid == limb_id)[0]
    if len(lm) < 80:
        return 0
    s_ray = gain_ray * float(np.clip((RAY_OPEN - prc2) / (RAY_OPEN - RAY_FULL), 0.0, 1.0))
    s_seg = gain_seg * float(np.clip((SEG_OPEN - prc2) / (SEG_OPEN - SEG_FULL), 0.0, 1.0))
    x = P[:, 0]
    a = (x - x.min()) / (np.ptp(x) + 1e-9)
    body_c = P.mean(0).copy()
    body_c[2] = abs(body_c[2])
    mirror = np.array([1.0, 1.0, -1.0])
    moved = 0
    for half in (a[lm] < 0.5, a[lm] >= 0.5):                 # the two limb pairs (fore / hind by AP half)
        bi = lm[half]
        if len(bi) < 80:
            continue
        sz = np.sign(P[bi, 2]); sz[sz == 0] = 1.0            # each cell's real side
        B = P[bi].copy(); B[:, 2] = np.abs(B[:, 2])          # the folded pair = one paddle
        ax = bud_axis(B, body_c)
        t = (B - B.mean(0)) @ ax
        am = t >= np.quantile(t, 1.0 - DISTAL_FRAC)
        A = bi[am]
        if len(A) < 15:
            continue
        sA = sz[am]
        PA = B[am]
        tA = t[am]
        Cp = PA - PA.mean(0)
        Cp = Cp - np.outer(Cp @ ax, ax)                      # in-plane (perpendicular to the bud axis)
        fan = np.linalg.svd(Cp, full_matrices=False)[2][0]
        fan = fan - float(fan @ ax) * ax
        fan = fan / (np.linalg.norm(fan) + 1e-12)
        thick = np.cross(ax, fan)
        u = Cp @ fan
        v = Cp @ thick
        t0 = float(tA.min()); L = float(np.ptp(tA)) + 1e-9   # common interzone planes for the whole autopod
        groups = np.array_split(np.argsort(u), N_RAYS)       # five contiguous rays across the fan
        for g in groups:
            if len(g) < 2:
                continue
            du = u[g].mean() - u[g]
            dv = v[g].mean() - v[g]
            d = du[:, None] * fan + dv[:, None] * thick      # folded-frame displacement
            d[:, 2] *= sA[g]                                 # mirror the ML component back per cell
            P[A[g]] += s_ray * d
        if s_seg > 0.0:
            if seg_pull is None:
                # GAP MODE (the per-frame path): the two joint interzones at 1/3 and 2/3 of the autopod
                # empty out exactly -- cells within IZ_HALF of a plane move to its edge -- and the
                # paddle keeps its extent (centre-compaction shortened it, so the instrument's distal
                # 12% re-selected a mix of the distal anlage and the limb below it: segments read 1).
                for fr in (1.0 / 3.0, 2.0 / 3.0):
                    dz = tA - (t0 + fr * L)
                    near = np.abs(dz) < IZ_HALF * L
                    if near.any():
                        push = np.sign(dz[near] + 1e-12) * (IZ_HALF * L - np.abs(dz[near]))
                        d = push[:, None] * ax[None, :]
                        d[:, 2] *= sA[near]
                        P[A[near]] += s_seg * d
            else:
                # PULL MODE (the per-step path, measured against the relaxation): every autopod cell
                # compacts along the ray toward the centre of its third -- three cartilage anlagen per
                # ray -- which empties the two joint rows between them.
                k = np.clip(((tA - t0) / L * 3.0).astype(int), 0, 2)
                tc = t0 + (k + 0.5) * L / 3.0
                d = (seg_pull * (tc - tA))[:, None] * ax[None, :]
                d[:, 2] *= sA
                P[A] += s_seg * d
        moved += len(A)
    return moved
