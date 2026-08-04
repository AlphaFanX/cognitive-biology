"""
Phase 1 -- REGIONALIZED PROLIFERATION anchored to the neural signaling centers.
==============================================================================
Replaces the old monotonic `head_expand` boost (head_grade = 1 + 1.2*clip(1 - a/0.38)*ceph), which
peaks at the very anterior TIP and so BALLOONS the forebrain: growth was maximal at a->0 with no
regional structure and no isthmic organizer. Real brain proportioning is set by DIFFERENTIAL
PROLIFERATION under a few signaling centres, each a named gene program -- so the growth-rate
multiplier is literally the SUM of those organizer gradients (the "anchor in the genome"), and their
relative strengths are the calibratable knobs (g_K-anchor pattern: mechanism genome-derived, scale
atlas-tuned).

Organizers (AP coordinate a in [0,1], 0 = anterior/rostral .. 1 = posterior/caudal; DV d in [0,1],
0 = ventral/floor .. 1 = dorsal/roof):
  - ANTERIOR NEURAL RIDGE (ANR): FGF8 + Foxg1/Six3 -> telencephalon/forebrain growth. Gaussian at
    the rostral tip.
  - ISTHMIC ORGANIZER: the Otx2(anterior)/Gbx2(posterior) boundary INDUCES FGF8 at the isthmus, which
    drives midbrain + anterior-hindbrain (cerebellar) proliferation and the mid-hindbrain constriction.
    A gaussian at the MHB -- this is the term the old monotonic boost lacked, and it redistributes
    growth CAUDALLY off the forebrain tip.  (Harada 2016; Otx2/Gbx2->Fgf8, isthmus-organizer reviews.)
  - DV FLOOR/ROOF: Shh from the floor plate (ventral) and Wnt/Bmp from the roof (dorsal) set a mild
    dorso-ventral proliferation gradient.

Refs: Fgf8 isthmus Harada 2016 (Dev Growth Differ 58:437); Otx2/Gbx2 precede Fgf8 (PMID 11231064);
isthmus organizer review (PMID 16111543); Gbx2/Fgf8 MHB boundary (PMC3026416).
"""
from __future__ import annotations
import numpy as np

# --- organizer AP positions (neural-tube regionalization) ---
A_ANR = 0.03      # anterior neural ridge (rostral forebrain)
A_ISTH = 0.17     # isthmic organizer = Otx2/Gbx2 boundary (midbrain <-> hindbrain)
W_ANR = 0.065     # gaussian width of the ANR FGF8 field
W_ISTH = 0.055    # gaussian width of the isthmic FGF8 field

# --- organizer strengths (the calibratable knobs; the g_K-anchor scale) ---
# ISTHMUS deliberately >= ANR so growth is NOT all piled on the anterior tip -- this is the correction
# that deflates the forebrain blob and gives the midbrain/cerebellar bulge its real relative size.
S_ANR = 0.85      # FGF8 (ANR) / Foxg1-Six3  -> forebrain  (calibrated 07-20)
S_ISTH = 1.40     # FGF8 (isthmus)           -> midbrain + anterior hindbrain  (calibrated 07-20)
S_SHH = 0.22      # Shh floor plate (ventral) proliferation
S_WNT = 0.12      # Wnt/Bmp roof (dorsal) proliferation


def region_growth(a, d, ceph, strength=1.0):
    """Multiplicative proliferation field for the cephalic neural domain. Returns g >= 1 for cephalic
    cells (`ceph` in [0,1]) and ~1 elsewhere. a = AP [0 rostral..1 caudal]; d = DV [0 ventral..1 dorsal].

    g = 1 + strength * (ANR + isthmus + 0.5*(Shh_ventral + Wnt_dorsal)) * ceph
    """
    a = np.asarray(a, np.float32); d = np.asarray(d, np.float32); ceph = np.asarray(ceph, np.float32)
    anr = S_ANR * np.exp(-((a - A_ANR) / W_ANR) ** 2)          # forebrain FGF8/Foxg1/Six3
    isth = S_ISTH * np.exp(-((a - A_ISTH) / W_ISTH) ** 2)      # isthmic FGF8 (Otx2/Gbx2 boundary)
    dv = S_SHH * (1.0 - d) + S_WNT * d                         # Shh floor (ventral) + Wnt/Bmp roof (dorsal)
    return 1.0 + strength * (anr + isth + 0.5 * dv) * ceph


def region_of(a):
    """Coarse AP label for reporting/calibration (not used in the sim loop)."""
    a = np.asarray(a)
    lab = np.full(a.shape, "trunk", dtype=object)
    lab[a < 0.09] = "forebrain"
    lab[(a >= 0.09) & (a < A_ISTH)] = "midbrain"
    lab[(a >= A_ISTH) & (a < 0.30)] = "hindbrain"
    return lab
