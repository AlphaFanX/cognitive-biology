"""
The homologous mammalian skeleton -- the liftable, multiscale anatomical kernel.

Each bone is a NAMED, HOMOLOGOUS part (shared across mammals; species differ by deformation,
not by the part list). Every placed bone carries the three things that make the coarse model
REFINABLE BACK TO CELLS ("liftable"):
  (1) genome parameters  -- the knobs that scaled it (so the same genome drives both scales);
  (2) cell-fate makeup   -- the lineage that builds it (for the lifting infill);
  (3) boundary           -- the canonical capsule (the attractor target the cell-NCA relaxes into).

Restriction R (cells -> part) and lifting P (part -> cells) are a matched pair with R(P(part)) ~ part.

Axial counts are FROZEN (7 cervical, 13 thoracic, 6 lumbar, sacrum, caudal x tail_count); species
differ by per-segment SCALE. Run: venv_win_new/Scripts/python.exe -m menagerie.skeleton
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .genome import (Genome, CERVICAL_COUNT, THORACIC_COUNT, LUMBAR_COUNT,
                     SACRAL_COUNT, BASE_CAUDAL)
from .decoder import L_CERV, L_TRUNK, L_SACR, L_CAUD, LIMB_BASE, GIRTH_BASE

# fate labels (the liftable cell-fate makeup, one per bone class)
FATE_AXIAL = "sclerotome -> chondrocyte -> osteoblast (endochondral)"
FATE_LIMB  = "lateral-plate mesenchyme -> chondrocyte -> osteoblast (endochondral)"
FATE_SKULL = "cranial neural crest + mesoderm (intramembranous/endochondral)"


@dataclass
class Bone:
    name: str
    homology: str          # Uberon-style homology label (species-neutral)
    a: np.ndarray          # proximal end (world)
    b: np.ndarray          # distal end (world)
    radius: float
    fate: str
    knobs: dict            # (1) genome parameters that scaled this bone
    side: str = "axial"

    def boundary(self):    # (3) canonical capsule = the attractor target for lifting
        return dict(a=self.a.tolist(), b=self.b.tolist(), radius=float(self.radius))

    def length(self):
        return float(np.linalg.norm(self.b - self.a))


# ---------------------------------------------------------------- forward kinematics
def build_skeleton(g: Genome):
    """Instantiate the homologous skeleton, deformed by the genome. Returns list[Bone]."""
    if g.posture == "biped":
        return _build_biped(g)
    if g.body_plan == "finned":
        return _build_finned(g)
    bs = g.body_size
    bones: list[Bone] = []
    kn = lambda *ks: {k: getattr(g, k) for k in ks}

    # limb segment lengths (fore slightly longer -> the mammalian sloping back)
    limb = LIMB_BASE * g.limb_len * bs
    fore = limb * 1.06
    hind = limb * 0.94
    rlimb = 0.05 * g.limb_gracility * bs                      # bone thickness
    shoulder_z = fore
    hip_z = hind
    vr = 0.6 * GIRTH_BASE * g.trunk_girth * bs * 0.45         # vertebra body radius

    # --- thoracic + lumbar spine: shoulder (x=0) backward to hip ---
    x = 0.0
    spine_pts = []
    for i in range(THORACIC_COUNT):
        c = np.array([x, 0, shoulder_z + (hip_z - shoulder_z) * (i / (THORACIC_COUNT + LUMBAR_COUNT))])
        spine_pts.append(("T%d" % (i + 1), c))
        x -= L_TRUNK * g.trunk_len * bs
    for i in range(LUMBAR_COUNT):
        f = (THORACIC_COUNT + i) / (THORACIC_COUNT + LUMBAR_COUNT)
        c = np.array([x, 0, shoulder_z + (hip_z - shoulder_z) * f])
        spine_pts.append(("L%d" % (i + 1), c))
        x -= L_TRUNK * g.trunk_len * bs
    hip_x = x
    for nm, c in spine_pts:
        b2 = c + np.array([-L_TRUNK * g.trunk_len * bs * 0.9, 0, 0])
        bones.append(Bone(nm, "vertebra", c, b2, vr, FATE_AXIAL, kn("trunk_len", "body_size")))

    # --- ribs: paired arcs from each thoracic vertebra, curving ventral ---
    girth = GIRTH_BASE * g.trunk_girth * bs
    for i in range(0, THORACIC_COUNT, 1):
        _, c = spine_pts[i]
        for s in (+1, -1):
            mid = c + np.array([-0.02, s * girth * 0.9, -girth * 0.5])
            end = c + np.array([0.05, s * girth * 0.35, -girth * 1.25])
            bones.append(Bone("rib%d%s" % (i + 1, "R" if s > 0 else "L"), "rib",
                              c, mid, 0.018 * bs, FATE_AXIAL, kn("trunk_girth", "body_size"),
                              side="R" if s > 0 else "L"))
            bones.append(Bone("rib%d%s_v" % (i + 1, "R" if s > 0 else "L"), "rib",
                              mid, end, 0.016 * bs, FATE_AXIAL, kn("trunk_girth", "body_size"),
                              side="R" if s > 0 else "L"))

    # --- sacrum + caudal (tail), from hip backward and drooping ---
    ta = np.deg2rad(28)
    xc, zc = hip_x, hip_z
    n_caud = max(1, round(BASE_CAUDAL * g.tail_count))
    sac0 = np.array([xc, 0, zc])
    for i in range(SACRAL_COUNT):
        nxt = np.array([xc - L_SACR * bs * np.cos(ta * 0.4), 0, zc - L_SACR * bs * np.sin(ta * 0.4)])
        bones.append(Bone("S%d" % (i + 1), "sacral vertebra", np.array([xc, 0, zc]), nxt,
                          vr * 1.1, FATE_AXIAL, kn("body_size")))
        xc, zc = nxt[0], nxt[2]
    for i in range(n_caud):
        L = L_CAUD * g.tail_len * bs
        nxt = np.array([xc - L * np.cos(ta), 0, max(0.02, zc - L * np.sin(ta))])
        bones.append(Bone("Ca%d" % (i + 1), "caudal vertebra", np.array([xc, 0, zc]), nxt,
                          vr * (0.6 * (1 - i / n_caud) + 0.12), FATE_AXIAL,
                          kn("tail_len", "tail_count", "body_size")))
        xc, zc = nxt[0], nxt[2]

    # --- cervical spine: shoulder forward and up (7 FROZEN) ---
    na = np.deg2rad(8 + 74 * g.neck_raise)
    xc, zc = 0.0, shoulder_z
    for i in range(CERVICAL_COUNT):
        L = L_CERV * g.cervical_elong * bs
        nxt = np.array([xc + L * np.cos(na), 0, zc + L * np.sin(na)])
        bones.append(Bone("C%d" % (CERVICAL_COUNT - i), "cervical vertebra",
                          np.array([xc, 0, zc]), nxt, vr * 0.85, FATE_AXIAL,
                          kn("cervical_elong", "body_size")))
        xc, zc = nxt[0], nxt[2]
    neck_tip = np.array([xc, 0, zc])

    # --- skull + mandible at the neck tip ---
    sk = 0.28 * g.skull_size * bs
    fwd = np.array([np.cos(na), 0, np.sin(na)])
    skull_end = neck_tip + fwd * (sk * (1.0 + 0.9 * (g.snout_len - 1)))
    bones.append(Bone("cranium", "cranium", neck_tip, skull_end, 0.55 * sk, FATE_SKULL,
                      kn("skull_size", "snout_len", "body_size")))
    bones.append(Bone("mandible", "mandible", neck_tip + np.array([0, 0, -0.25 * sk]),
                      skull_end + np.array([0, 0, -0.35 * sk]), 0.16 * sk, FATE_SKULL,
                      kn("skull_size", "snout_len", "body_size")))

    # --- mineralized cranial appendages (horn cores, tusks) ---
    lat = np.array([0, 1.0, 0])
    if g.horn_mode == "paired" and g.horn_size > 0:          # buffalo horns / giraffe ossicones (bony core)
        for s in (+1, -1):
            base = neck_tip + fwd * (0.15 * sk) + s * lat * (0.18 * sk) + np.array([0, 0, 0.10 * sk])
            tip = base + (fwd * 0.05 + s * lat * 0.55 + np.array([0, 0, 0.6])) * g.horn_size * bs
            bones.append(Bone("horn core %s" % ("R" if s > 0 else "L"), "horn core (os cornu)",
                              base, tip, 0.045 * bs, "cranial neural crest (os cornu / ossicone)",
                              kn("horn_size", "body_size"), side="R" if s > 0 else "L"))
    elif g.horn_mode == "single_median" and g.horn_size > 0:  # rhino: dermal keratin horn on the nasal midline (LR node)
        base = skull_end + np.array([0, 0, 0.06 * sk])
        tip = base + (fwd * 0.20 + np.array([0, 0, 0.7])) * g.horn_size * bs
        bones.append(Bone("nasal horn (keratin)", "nasal horn (dermal keratin)",
                          base, tip, 0.06 * bs, "dermal / epidermal keratin (integument)",
                          kn("horn_size", "body_size")))
    if g.dentition == "tusks" and g.tusk_len > 0:            # elephant: tusks = elongated incisors (ivory)
        for s in (+1, -1):
            base = skull_end + s * lat * (0.09 * sk) + np.array([0, 0, -0.10 * sk])
            tip = base + fwd * (0.85 * g.tusk_len * bs) + np.array([0, 0, -0.28 * g.tusk_len * bs])
            bones.append(Bone("tusk %s" % ("R" if s > 0 else "L"), "tusk (incisor, ivory)",
                              base, tip, 0.028 * bs, "odontogenic (dentine / enamel)",
                              kn("tusk_len", "body_size"), side="R" if s > 0 else "L"))

    # --- girdles + limbs (paired) ---
    def limb_chain(top, seg_lengths, seg_names, homols, side, splay):
        p = np.array(top, float)
        down = np.array([0.02, splay, -1.0]); down /= np.linalg.norm(down)
        for L, nm, hm in zip(seg_lengths, seg_names, homols):
            q = p + down * L
            bones.append(Bone("%s_%s" % (nm, side), hm, p.copy(), q.copy(), rlimb,
                              FATE_LIMB, kn("limb_len", "limb_gracility", "body_size"), side=side))
            down = np.array([0.04, splay * 0.3, -1.0]); down /= np.linalg.norm(down)
            p = q

    splay = 0.5 if g.posture == "sprawling" else 0.06
    T1 = spine_pts[0][1]
    Lx = hip_x
    for s in (+1, -1):
        sd = "R" if s > 0 else "L"
        # pectoral girdle (scapula) + forelimb
        sc_top = T1 + np.array([0.05, s * 0.10 * bs, 0.02])
        sc_bot = T1 + np.array([-0.02, s * 0.14 * bs, -0.18 * fore])
        bones.append(Bone("scapula_%s" % sd, "scapula", sc_top, sc_bot, 0.03 * bs, FATE_LIMB,
                          kn("limb_gracility", "body_size"), side=sd))
        limb_chain(sc_bot, [fore * 0.38, fore * 0.40, fore * 0.22],
                   ["humerus", "radius-ulna", "manus"], ["humerus", "radius/ulna", "manus (autopod)"],
                   sd, s * splay)
        # pelvic girdle + hindlimb
        pel_top = np.array([Lx + 0.06, s * 0.10 * bs, hip_z + 0.02])
        pel_bot = np.array([Lx - 0.04, s * 0.14 * bs, hip_z - 0.16 * hind])
        bones.append(Bone("pelvis_%s" % sd, "hip bone (pelvis)", pel_top, pel_bot, 0.035 * bs,
                          FATE_LIMB, kn("body_size"), side=sd))
        limb_chain(pel_bot, [hind * 0.40, hind * 0.42, hind * 0.18],
                   ["femur", "tibia-fibula", "pes"], ["femur", "tibia/fibula", "pes (autopod)"],
                   sd, s * splay)

    return bones


# ---------------------------------------------------------------- bipedal plan (human)
def _build_biped(g: Genome):
    """The same homologous roster in an upright, bipedal configuration (Homo). Legs weight-bearing
    to the ground; spine vertical; arms hanging from the shoulders; tail vestigial. Sexual
    dimorphism sets the girdle widths (male: broader shoulders; female: broader pelvis)."""
    bs = g.body_size
    bones: list[Bone] = []
    kn = lambda *ks: {k: getattr(g, k) for k in ks}
    male = g.sex == "male"; female = g.sex == "female"

    leg = LIMB_BASE * g.limb_len * bs                       # long, weight-bearing
    arm = leg * 0.80
    rlimb = 0.05 * g.limb_gracility * bs
    vr = 0.6 * GIRTH_BASE * g.trunk_girth * bs * 0.45
    shw = 0.16 * bs * (1.30 if male else 0.95 if female else 1.0)   # shoulder half-width
    hpw = 0.13 * bs * (0.95 if male else 1.30 if female else 1.0)   # pelvis half-width
    # a vertical stacked spine is far shorter than the quadruped's body-length spacing:
    # compact the torso/neck so the biped is leg-dominant (human proportions)
    tf, nf = 0.30, 0.28
    seg_tr = L_TRUNK * g.trunk_len * bs * tf
    seg_cv = L_CERV * g.cervical_elong * bs * nf

    pelvis_z = leg
    hip_c = np.array([0, 0, pelvis_z])
    # sacrum + vestigial tail
    bones.append(Bone("S1", "sacral vertebra", hip_c, hip_c + np.array([-0.03 * bs, 0, -0.04 * bs]),
                      vr * 1.1, FATE_AXIAL, kn("body_size")))
    for i in range(SACRAL_COUNT - 1):
        bones.append(Bone("S%d" % (i + 2), "sacral vertebra", hip_c + np.array([0, 0, -0.02 * i * bs]),
                          hip_c + np.array([-0.03 * bs, 0, -0.03 * bs]), vr, FATE_AXIAL, kn("body_size")))
    for i in range(max(1, round(BASE_CAUDAL * g.tail_count * 0.25))):   # coccyx (vestigial)
        z = pelvis_z - 0.03 * (i + 1) * bs
        bones.append(Bone("Ca%d" % (i + 1), "caudal vertebra", np.array([-0.04 * bs, 0, z]),
                          np.array([-0.05 * bs, 0, z - 0.03 * bs]), vr * 0.3, FATE_AXIAL,
                          kn("tail_count", "body_size")))

    # spine UP: lumbar -> thoracic -> cervical, gentle S-curve in x
    z = pelvis_z
    spine = []
    def stack(prefix, count, seg, hom, lean):
        nonlocal z
        for i in range(count):
            c = np.array([lean * np.sin(np.pi * i / count) * bs, 0, z])
            spine.append((f"{prefix}{i+1}", c))
            z += seg
    stack("L", LUMBAR_COUNT, seg_tr, "vertebra", +0.06)   # lordosis
    t1_z = z
    stack("T", THORACIC_COUNT, seg_tr, "vertebra", -0.05)  # kyphosis
    thorax_top = np.array([0, 0, z])
    stack("C", CERVICAL_COUNT, seg_cv, "cervical vertebra", +0.04)
    neck_tip = np.array([0, 0, z])
    for nm, c in spine:
        hom = "cervical vertebra" if nm.startswith("C") else "vertebra"
        knb = kn("cervical_elong", "body_size") if nm.startswith("C") else kn("trunk_len", "body_size")
        bones.append(Bone(nm, hom, c, c + np.array([0, 0, -0.6 * seg_tr]), vr, FATE_AXIAL, knb))

    # ribs around the thoracic region (forward + lateral)
    girth = GIRTH_BASE * g.trunk_girth * bs
    thor = [(nm, c) for nm, c in spine if nm.startswith("T")]
    for i, (nm, c) in enumerate(thor):
        for s in (+1, -1):
            mid = c + np.array([girth * 0.6, s * girth * 0.85, -0.02])
            end = c + np.array([girth * 1.0, s * girth * 0.3, -0.05])
            bones.append(Bone("rib%d%s" % (i + 1, "R" if s > 0 else "L"), "rib", c, mid, 0.017 * bs,
                              FATE_AXIAL, kn("trunk_girth", "body_size"), side="R" if s > 0 else "L"))
            bones.append(Bone("rib%d%s_v" % (i + 1, "R" if s > 0 else "L"), "rib", mid, end, 0.015 * bs,
                              FATE_AXIAL, kn("trunk_girth", "body_size"), side="R" if s > 0 else "L"))

    # skull + mandible on top of the cervical column
    sk = 0.28 * g.skull_size * bs
    up = np.array([0, 0, 1.0])
    skull_end = neck_tip + up * (sk * 0.75)
    bones.append(Bone("cranium", "cranium", neck_tip, skull_end, 0.6 * sk, FATE_SKULL,
                      kn("skull_size", "snout_len", "body_size")))
    bones.append(Bone("mandible", "mandible", neck_tip + np.array([0.15 * sk, 0, 0.15 * sk]),
                      neck_tip + np.array([0.4 * sk, 0, 0.05 * sk]), 0.16 * sk, FATE_SKULL,
                      kn("skull_size", "body_size")))

    # legs (hindlimb) straight down to the ground; arms (forelimb) hanging from the shoulders
    def chain(top, total, fracs, names, homols, side, dirn):
        p = np.array(top, float); d = np.array(dirn, float); d /= np.linalg.norm(d)
        for fr, nm, hm in zip(fracs, names, homols):
            q = p + d * (total * fr)
            bones.append(Bone("%s_%s" % (nm, side), hm, p.copy(), q.copy(), rlimb, FATE_LIMB,
                              kn("limb_len", "limb_gracility", "body_size"), side=side))
            p = q

    for s in (+1, -1):
        sd = "R" if s > 0 else "L"
        # pelvis + leg
        hip = np.array([0, s * hpw, pelvis_z])
        bones.append(Bone("pelvis_%s" % sd, "hip bone (pelvis)", hip_c + np.array([0, 0, 0.02]),
                          hip, 0.04 * bs, FATE_LIMB, kn("body_size"), side=sd))
        chain(hip, leg, [0.48, 0.40, 0.12], ["femur", "tibia-fibula", "pes"],
              ["femur", "tibia/fibula", "pes (autopod)"], sd, [0.02, s * 0.04, -1.0])
        # shoulder (scapula) + arm hanging
        sho = np.array([0, s * shw, thorax_top[2] - 0.02 * bs])
        bones.append(Bone("scapula_%s" % sd, "scapula", thorax_top + np.array([0, s * shw * 0.5, 0.0]),
                          sho, 0.03 * bs, FATE_LIMB, kn("limb_gracility", "body_size"), side=sd))
        chain(sho, arm, [0.40, 0.36, 0.24], ["humerus", "radius-ulna", "manus"],
              ["humerus", "radius/ulna", "manus (autopod)"], sd, [0.03, s * 0.12, -1.0])

    return bones


# ---------------------------------------------------------------- finned plan (fish)
def _build_finned(g: Genome):
    """A limbless FINNED body plan (fish). Deep homology: the paired fins are the SAME
    appendage module as tetrapod limbs (driven by limb_len / limb_gracility, Shh/Fgf8 AER +
    Hox), just expressed as fins instead of weight-bearing legs; the median fins (dorsal /
    anal / caudal) are axial fin-fold fields driven by the trunk / tail program. Streamlined
    horizontal axis, no raised neck. Organs (eyes, gut...) place on the same homologous frame."""
    bs = g.body_size
    bones: list[Bone] = []
    kn = lambda *ks: {k: getattr(g, k) for k in ks}
    vr = 0.6 * GIRTH_BASE * g.trunk_girth * bs * 0.45
    girth = GIRTH_BASE * g.trunk_girth * bs

    # --- horizontal thoraco-lumbar axis: shoulder (x=0) backward ---
    x = 0.0
    spine_pts = []
    for i in range(THORACIC_COUNT):
        spine_pts.append(("T%d" % (i + 1), np.array([x, 0, 0.0]))); x -= L_TRUNK * g.trunk_len * bs
    for i in range(LUMBAR_COUNT):
        spine_pts.append(("L%d" % (i + 1), np.array([x, 0, 0.0]))); x -= L_TRUNK * g.trunk_len * bs
    hip_x = x
    smap = dict(spine_pts)
    for nm, c in spine_pts:
        bones.append(Bone(nm, "vertebra", c, c + np.array([-L_TRUNK * g.trunk_len * bs * 0.9, 0, 0]),
                          vr, FATE_AXIAL, kn("trunk_len", "body_size")))

    # --- ribs (fish have ribs), lighter, curving ventral ---
    for i in range(0, THORACIC_COUNT, 2):
        c = spine_pts[i][1]
        for s in (+1, -1):
            end = c + np.array([0.02, s * girth * 0.55, -girth * 0.75])
            bones.append(Bone("rib%d%s" % (i + 1, "R" if s > 0 else "L"), "rib", c, end,
                              0.014 * bs, FATE_AXIAL, kn("trunk_girth", "body_size"),
                              side="R" if s > 0 else "L"))

    # --- sacral + caudal (tail axis) straight back ---
    xc = hip_x
    n_caud = max(1, round(BASE_CAUDAL * g.tail_count))
    for i in range(SACRAL_COUNT):
        nxt = np.array([xc - L_SACR * bs, 0, 0.0])
        bones.append(Bone("S%d" % (i + 1), "sacral vertebra", np.array([xc, 0, 0]), nxt,
                          vr * 1.1, FATE_AXIAL, kn("body_size"))); xc = nxt[0]
    for i in range(n_caud):
        nxt = np.array([xc - L_CAUD * g.tail_len * bs, 0, 0.0])
        bones.append(Bone("Ca%d" % (i + 1), "caudal vertebra", np.array([xc, 0, 0]), nxt,
                          vr * (0.6 * (1 - i / n_caud) + 0.12), FATE_AXIAL,
                          kn("tail_len", "tail_count", "body_size"))); xc = nxt[0]
    tail_tip = np.array([xc, 0, 0.0])

    # --- short flat cervical + skull at the front (no neck raise) ---
    xc = 0.0
    for i in range(CERVICAL_COUNT):
        nxt = np.array([xc + L_CERV * g.cervical_elong * bs, 0, 0.0])
        bones.append(Bone("C%d" % (CERVICAL_COUNT - i), "cervical vertebra", np.array([xc, 0, 0]),
                          nxt, vr * 0.85, FATE_AXIAL, kn("cervical_elong", "body_size"))); xc = nxt[0]
    neck_tip = np.array([xc, 0, 0.0])
    sk = 0.28 * g.skull_size * bs
    skull_end = neck_tip + np.array([sk * (1.0 + 0.9 * (g.snout_len - 1)), 0, 0])
    bones.append(Bone("cranium", "cranium", neck_tip, skull_end, 0.55 * sk, FATE_SKULL,
                      kn("skull_size", "snout_len", "body_size")))
    bones.append(Bone("mandible", "mandible", neck_tip + np.array([0, 0, -0.25 * sk]),
                      skull_end + np.array([0, 0, -0.30 * sk]), 0.16 * sk, FATE_SKULL,
                      kn("skull_size", "snout_len", "body_size")))

    # --- FINS = the same paired-appendage module (limb genes), expressed as fins ---
    FATE_FIN = "fin field (paired appendage / median fin-fold; Shh/Fgf8 AER + Hox = limb homolog)"
    ray_r = 0.011 * bs
    finL = LIMB_BASE * g.limb_len * bs
    spread = 0.9 * g.limb_gracility

    def fan(origin, direction, span, n, length, side, homology, nm, knset):
        direction = np.asarray(direction, float); direction /= np.linalg.norm(direction)
        span = np.asarray(span, float); span /= (np.linalg.norm(span) + 1e-9)
        for k in range(n):
            f = (k / (n - 1) - 0.5) if n > 1 else 0.0
            d = direction + span * (f * spread * 1.5); d /= np.linalg.norm(d)
            bones.append(Bone(f"{nm} ray{k+1}{side}", homology, np.asarray(origin, float).copy(),
                              np.asarray(origin, float) + d * length, ray_r, FATE_FIN, knset,
                              side=side))

    kfin = kn("limb_len", "limb_gracility", "body_size")
    T1c = spine_pts[0][1]
    for s in (+1, -1):                              # pectoral fin = forelimb homolog
        sd = "R" if s > 0 else "L"
        base = T1c + np.array([-0.02, s * girth * 0.7, -girth * 0.45])
        fan(base, [-0.5, s * 0.85, -0.35], [-1.0, 0, 0.4], 6, finL, sd,
            "pectoral fin (paired appendage; forelimb homolog)", "pectoral fin", kfin)
    Lc = smap.get("L2", spine_pts[THORACIC_COUNT][1])
    for s in (+1, -1):                              # pelvic fin = hindlimb homolog
        sd = "R" if s > 0 else "L"
        base = Lc + np.array([0, s * girth * 0.45, -girth * 0.7])
        fan(base, [-0.3, s * 0.7, -0.7], [-1.0, 0, 0.2], 5, finL * 0.7, sd,
            "pelvic fin (paired appendage; hindlimb homolog)", "pelvic fin", kfin)

    # median fins (axial fin-fold): dorsal (up), anal (down), caudal (tail fan)
    kmed = kn("body_size")
    dbases = [spine_pts[i][1] for i in range(THORACIC_COUNT - 2, THORACIC_COUNT + LUMBAR_COUNT - 2)
              if 0 <= i < len(spine_pts)][:6]
    for j, c in enumerate(dbases):
        top = c + np.array([0, 0, girth * (1.0 + 0.5 * np.sin(np.pi * (j + 1) / (len(dbases) + 1)))])
        bones.append(Bone(f"dorsal fin ray{j+1}", "median fin (dorsal; axial fin-fold field)",
                          c + np.array([0, 0, girth * 0.4]), top, ray_r, FATE_FIN, kmed, side="axial"))
    abase = spine_pts[min(len(spine_pts) - 1, THORACIC_COUNT + LUMBAR_COUNT - 1)][1]
    for j in range(4):
        c = abase + np.array([-0.12 * bs * j, 0, 0])
        bones.append(Bone(f"anal fin ray{j+1}", "median fin (anal; axial fin-fold field)",
                          c + np.array([0, 0, -girth * 0.4]),
                          c + np.array([0, 0, -girth * (0.85 + 0.25 * np.sin(np.pi * j / 4))]),
                          ray_r, FATE_FIN, kmed, side="axial"))
    cf_len = 0.6 * bs * (0.6 + 0.5 * g.tail_len)   # caudal (tail) fin fan
    for k in range(9):
        f = k / 8.0 - 0.5
        d = np.array([-1.0, 0, f * 2.4]); d /= np.linalg.norm(d)
        bones.append(Bone(f"caudal fin ray{k+1}", "median fin (caudal; axial + tail program)",
                          tail_tip.copy(), tail_tip + d * cf_len, ray_r, FATE_FIN,
                          kn("tail_len", "body_size"), side="axial"))
    return bones


# ---------------------------------------------------------------- render
def render_points(bones, seed=0, bone_rgb=(0.92, 0.90, 0.84)):
    rng = np.random.default_rng(seed)
    P, C = [], []
    for bn in bones:
        L = bn.length()
        npts = max(10, int(120 * L * bn.radius / 0.01))
        npts = min(npts, 500)
        t = rng.random(npts)
        axis = bn.a[None] + t[:, None] * (bn.b - bn.a)[None]
        d = rng.normal(size=(npts, 3)); d[:, 0] *= 0.3
        d -= (d * (bn.b - bn.a)).sum(1, keepdims=True) * (bn.b - bn.a) / (L**2 + 1e-9)
        d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
        pts = axis + d * bn.radius * (0.6 + 0.4 * rng.random(npts))[:, None]
        P.append(pts); C.append(np.tile(bone_rgb, (npts, 1)))
    return np.vstack(P).astype(np.float32), np.vstack(C).astype(np.float32)


# ---------------------------------------------------------------- multiscale operators
def lift(bone: Bone, n=800, seed=0):
    """P: part -> cells. Fill the bone's canonical boundary with a cell cloud whose fate is the
    bone's cell-fate makeup (a representative fine state consistent with the coarse part)."""
    rng = np.random.default_rng(seed)
    L = bone.length(); axhat = (bone.b - bone.a) / (L + 1e-9)
    t = rng.random(n)
    ax = bone.a[None] + t[:, None] * (bone.b - bone.a)[None]
    d = rng.normal(size=(n, 3))
    d -= (d @ axhat)[:, None] * axhat[None]
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-9
    r = bone.radius * np.sqrt(rng.random(n))                 # solid fill
    cells = ax + d * r[:, None]
    return cells.astype(np.float32), bone.fate


def restrict(cells: np.ndarray):
    """R: cells -> part. Recover the coarse capsule (axis, length, radius) by averaging."""
    c = cells.mean(0)
    X = cells - c
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    axhat = Vt[0]
    proj = X @ axhat
    a = c + axhat * proj.min(); b = c + axhat * proj.max()
    radial = X - proj[:, None] * axhat[None]
    radius = float(np.sqrt((radial ** 2).sum(1)).mean() * 1.35)   # mean->capsule radius
    return dict(a=a, b=b, length=float(proj.max() - proj.min()), radius=radius, axis=axhat)


def consistency(bone: Bone, n=1500):
    """R(P(part)) ~ part: lift the bone to cells, restrict back, report the round-trip error."""
    cells, _ = lift(bone, n=n)
    rec = restrict(cells)
    dl = abs(rec["length"] - bone.length()) / (bone.length() + 1e-9)
    dr = abs(rec["radius"] - bone.radius) / (bone.radius + 1e-9)
    return dict(length_err=dl, radius_err=dr, n_cells=n)


if __name__ == "__main__":
    from .targets import reference_genome
    for sp in ["base_vertebrate", "giraffe", "elephant"]:
        g = reference_genome(sp) if sp != "base_vertebrate" else Genome()
        bones = build_skeleton(g)
        cerv = [b for b in bones if b.homology == "cervical vertebra"]
        print(f"{sp:16s} {len(bones):3d} bones | {len(cerv)} cervicals "
              f"(len each {cerv[0].length():.2f}m) | neck {sum(b.length() for b in cerv):.2f}m")
    # multiscale round-trip on one bone
    g = reference_genome("giraffe")
    femur = [b for b in build_skeleton(g) if b.name.startswith("femur")][0]
    print("round-trip femur:", consistency(femur))
