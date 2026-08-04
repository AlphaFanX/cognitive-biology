"""
The muscle layer: functional Hill-type muscle groups on the homologous skeleton.

A muscle is a NAMED homologous part defined FUNCTIONALLY -- an origin and an insertion on named
bones, a line of action, and the joint it actuates -- which is exactly the musculoskeletal object
a central pattern generator drives. Each muscle carries the liftable payload (genome parameters,
cell-fate makeup = myogenic, boundary = the fusiform belly), so it too is refinable back to cells.

~36 functional groups (coarse). Paired limb muscles are mirrored L/R; axial muscles are midline.
Run: venv_win_new/Scripts/python.exe -m menagerie.muscles
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .genome import Genome
from .skeleton import build_skeleton, Bone

FATE_AXIAL_M = "myotome (somite) -> myoblast -> myofibre"
FATE_LIMB_M  = "dermomyotome -> migratory limb myoblast -> myofibre"

# (name, origin_bone, o_frac, insertion_bone, i_frac, function, joint, paired)
# The principal named muscles of Gray's Anatomy, region by region, each with its ORIGIN and INSERTION bone
# (the audit scores the bone, not the exact facet). Origins Gray's places on the sternum/clavicle are routed to
# rib1/scapula, which the model's skeleton carries. Coarse groups above are superseded by their named members.
MUSCLE_PLAN = [
    # --- head / jaw / neck ---
    ("masseter",            "cranium", 0.4, "mandible", 0.6, "jaw-close", "jaw", False),
    ("temporalis",          "cranium", 0.2, "mandible", 0.5, "jaw-close", "jaw", False),
    ("pterygoid",           "cranium", 0.5, "mandible", 0.4, "jaw-close", "jaw", False),
    ("sternocleidomastoid", "rib1", 0.0, "cranium", 0.2, "rotate",   "neck", True),
    ("splenius capitis",    "T3", 0.0, "cranium", 0.3, "extensor", "neck", True),
    ("semispinalis cervicis","T2", 0.1, "C6", 0.5, "extensor", "neck", True),
    ("longus colli",        "T2", 0.0, "C4", 0.5, "flexor",   "neck", False),
    ("scalenes",            "rib1", 0.2, "C5", 0.4, "flexor",   "neck", True),
    # --- trunk wall / abdomen ---
    ("erector spinae",      "S1", 0.0, "T1", 0.0, "extensor", "spine", False),
    ("multifidus",          "S1", 0.2, "L1", 0.4, "extensor", "spine", False),
    ("rectus abdominis",    "pelvis", 0.5, "rib5", 0.5, "flexor", "trunk", True),
    ("external oblique",    "rib5", 0.6, "pelvis", 0.3, "rotate", "trunk", True),
    ("internal oblique",    "pelvis", 0.2, "rib10", 0.5, "rotate", "trunk", True),
    ("transversus abdominis","L3", 0.2, "rib10", 0.4, "compress", "trunk", True),
    ("quadratus lumborum",  "pelvis", 0.1, "rib13", 0.3, "lateral-flex", "trunk", True),
    ("external intercostal","rib5", 0.5, "rib6", 0.5, "inspire", "trunk", True),
    ("diaphragm",           "L1", 0.0, "rib7", 0.6, "inspire", "trunk", False),
    # --- shoulder girdle ---
    ("trapezius",           "T4", 0.0, "scapula", 0.1, "elevate", "shoulder", True),
    ("rhomboid",            "T2", 0.1, "scapula", 0.5, "retract", "shoulder", True),
    ("levator scapulae",    "C2", 0.2, "scapula", 0.2, "elevate", "shoulder", True),
    ("serratus anterior",   "rib4", 0.5, "scapula", 0.6, "protract", "shoulder", True),
    ("pectoralis major",    "rib3", 0.4, "humerus", 0.25, "adduct", "shoulder", True),
    ("pectoralis minor",    "rib3", 0.5, "scapula", 0.15, "depress", "shoulder", True),
    ("latissimus dorsi",    "L3", 0.2, "humerus", 0.2, "adduct", "shoulder", True),
    # --- rotator cuff + arm ---
    ("deltoid",             "scapula", 0.4, "humerus", 0.4, "abduct", "shoulder", True),
    ("supraspinatus",       "scapula", 0.2, "humerus", 0.3, "abduct", "shoulder", True),
    ("infraspinatus",       "scapula", 0.3, "humerus", 0.35, "ext-rotate", "shoulder", True),
    ("teres major",         "scapula", 0.6, "humerus", 0.3, "adduct", "shoulder", True),
    ("teres minor",         "scapula", 0.5, "humerus", 0.35, "ext-rotate", "shoulder", True),
    ("subscapularis",       "scapula", 0.45, "humerus", 0.3, "int-rotate", "shoulder", True),
    ("coracobrachialis",    "scapula", 0.75, "humerus", 0.5, "flexor", "shoulder", True),
    ("biceps brachii",      "scapula", 0.8, "radius-ulna", 0.25, "flexor", "elbow", True),
    ("brachialis",          "humerus", 0.5, "radius-ulna", 0.2, "flexor", "elbow", True),
    ("triceps brachii",     "scapula", 0.9, "radius-ulna", 0.05, "extensor", "elbow", True),
    ("anconeus",            "humerus", 0.9, "radius-ulna", 0.12, "extensor", "elbow", True),
    # --- forearm / hand ---
    ("brachioradialis",     "humerus", 0.85, "manus", 0.1, "flexor", "elbow", True),
    ("pronator teres",      "humerus", 0.92, "radius-ulna", 0.5, "pronate", "wrist", True),
    ("supinator",           "humerus", 0.9, "radius-ulna", 0.4, "supinate", "wrist", True),
    ("flexor carpi radialis","humerus", 0.95, "manus", 0.3, "flexor", "wrist", True),
    ("flexor carpi ulnaris","humerus", 0.95, "manus", 0.35, "flexor", "wrist", True),
    ("palmaris longus",     "humerus", 0.96, "manus", 0.4, "flexor", "wrist", True),
    ("flexor digitorum",    "radius-ulna", 0.2, "manus", 0.6, "flexor", "wrist", True),
    ("extensor carpi radialis","humerus", 0.9, "manus", 0.3, "extensor", "wrist", True),
    ("extensor carpi ulnaris","humerus", 0.9, "manus", 0.35, "extensor", "wrist", True),
    ("extensor digitorum",  "radius-ulna", 0.2, "manus", 0.6, "extensor", "wrist", True),
    # --- hip / gluteal ---
    ("iliopsoas",           "L4", 0.0, "femur", 0.2, "flexor", "hip", True),
    ("gluteus maximus",     "pelvis", 0.3, "femur", 0.25, "extensor", "hip", True),
    ("gluteus medius",      "pelvis", 0.2, "femur", 0.15, "abduct", "hip", True),
    ("gluteus minimus",     "pelvis", 0.25, "femur", 0.12, "abduct", "hip", True),
    ("tensor fasciae latae","pelvis", 0.15, "tibia-fibula", 0.1, "abduct", "hip", True),
    ("piriformis",          "S2", 0.0, "femur", 0.15, "ext-rotate", "hip", True),
    # --- thigh: adductors, quadriceps, hamstrings ---
    ("pectineus",           "pelvis", 0.5, "femur", 0.3, "adduct", "hip", True),
    ("adductor longus",     "pelvis", 0.55, "femur", 0.5, "adduct", "hip", True),
    ("adductor magnus",     "pelvis", 0.5, "femur", 0.6, "adduct", "hip", True),
    ("gracilis",            "pelvis", 0.55, "tibia-fibula", 0.2, "adduct", "hip", True),
    ("sartorius",           "pelvis", 0.1, "tibia-fibula", 0.15, "flexor", "hip", True),
    ("rectus femoris",      "pelvis", 0.6, "tibia-fibula", 0.15, "extensor", "knee", True),
    ("vastus lateralis",    "femur", 0.3, "tibia-fibula", 0.15, "extensor", "knee", True),
    ("vastus medialis",     "femur", 0.35, "tibia-fibula", 0.15, "extensor", "knee", True),
    ("vastus intermedius",  "femur", 0.4, "tibia-fibula", 0.15, "extensor", "knee", True),
    ("biceps femoris",      "pelvis", 0.4, "tibia-fibula", 0.2, "flexor", "knee", True),
    ("semitendinosus",      "pelvis", 0.4, "tibia-fibula", 0.25, "flexor", "knee", True),
    ("semimembranosus",     "pelvis", 0.42, "tibia-fibula", 0.2, "flexor", "knee", True),
    # --- leg / foot ---
    ("gastrocnemius",       "femur", 0.85, "pes", 0.1, "plantarflex", "ankle", True),
    ("soleus",              "tibia-fibula", 0.4, "pes", 0.1, "plantarflex", "ankle", True),
    ("plantaris",           "femur", 0.83, "pes", 0.1, "plantarflex", "ankle", True),
    ("popliteus",           "femur", 0.8, "tibia-fibula", 0.25, "flexor", "knee", True),
    ("tibialis anterior",   "tibia-fibula", 0.3, "pes", 0.35, "dorsiflex", "ankle", True),
    ("tibialis posterior",  "tibia-fibula", 0.4, "pes", 0.4, "invert", "ankle", True),
    ("peroneus longus",     "tibia-fibula", 0.5, "pes", 0.3, "evert", "ankle", True),
    ("extensor digitorum longus","tibia-fibula", 0.35, "pes", 0.6, "dorsiflex", "ankle", True),
    ("flexor digitorum longus","tibia-fibula", 0.45, "pes", 0.6, "plantarflex", "ankle", True),
    # --- tail ---
    ("caudal extensor",     "S2", 0.0, "Ca1", 0.5, "extensor", "tail", False),
]


@dataclass
class Muscle:
    name: str
    origin: np.ndarray
    insertion: np.ndarray
    function: str          # flexor / extensor / ...
    joint: str             # the joint it actuates (the CPG hook)
    fate: str
    knobs: dict
    side: str = "axial"

    def length(self):
        return float(np.linalg.norm(self.insertion - self.origin))

    def boundary(self):    # fusiform belly = the attractor target for lifting
        return dict(origin=self.origin.tolist(), insertion=self.insertion.tolist(),
                    radius=0.10 * self.length())


def _bone_map(bones):
    return {b.name: b for b in bones}


def _pt(bmap, name, frac):
    """A point at fraction `frac` along a bone (resolves L/R suffixes by the caller)."""
    b = bmap.get(name)
    if b is None:                       # tolerate missing (e.g., rib index) -> nearest fallback
        cand = [k for k in bmap if k.startswith(name)]
        if not cand:
            return None
        b = bmap[cand[0]]
    return b.a + frac * (b.b - b.a)


def build_muscles(g: Genome, bones=None):
    if bones is None:
        bones = build_skeleton(g)
    bmap = _bone_map(bones)
    kn = lambda: {k: getattr(g, k) for k in ("body_size", "limb_gracility")}
    out = []
    for (name, ob, of, ib, ifr, func, joint, paired) in MUSCLE_PLAN:
        sides = [("R",), ("L",)] if paired else [("",)]
        for (sd,) in sides:
            o = _pt(bmap, ob + (("_" + sd) if paired and ob not in ("T4","rib4") else ""), of) \
                if paired else _pt(bmap, ob, of)
            # robust resolution: for paired, suffix limb bones; axial bones stay unsuffixed
            o = _resolve(bmap, ob, of, sd, paired)
            i = _resolve(bmap, ib, ifr, sd, paired)
            if o is None or i is None:
                continue
            fate = FATE_LIMB_M if joint in ("shoulder","elbow","wrist","hip","knee","ankle") else FATE_AXIAL_M
            out.append(Muscle(name + (f" {sd}" if paired else ""), o, i, func, joint, fate,
                              kn(), side=sd if paired else "axial"))

    # elephant trunk: a muscular hydrostat (no bone) -- a proboscis muscle from the skull
    if g.nose == "trunk" and g.proboscis_len > 0:
        cr = bmap.get("cranium")
        if cr is not None:
            fwd = (cr.b - cr.a) / (np.linalg.norm(cr.b - cr.a) + 1e-9)
            origin = cr.b.copy()
            tip = origin + fwd * (0.6 * g.proboscis_len * g.body_size) \
                + np.array([0, 0, -1.0]) * (0.9 * g.proboscis_len * g.body_size)
            out.append(Muscle("proboscis (trunk, muscular hydrostat)", origin, tip,
                              "hydrostat", "trunk-nose",
                              "proboscis facial muscle (muscular hydrostat)", kn()))
    return out


def _resolve(bmap, bone, frac, side, paired):
    """Resolve a bone name to a point, adding the L/R suffix only for limb/paired bones present."""
    limb_bones = ("scapula","humerus","radius-ulna","manus","pelvis","femur","tibia-fibula","pes")
    nm = bone
    if paired and bone in limb_bones:
        nm = f"{bone}_{side}"
    b = bmap.get(nm)
    if b is None:
        cand = [k for k in bmap if k.startswith(bone)]
        if not cand:
            return None
        b = bmap[cand[0]]
    return b.a + frac * (b.b - b.a)


def render_points(muscles, seed=1, rgb=(0.72, 0.18, 0.18)):
    rng = np.random.default_rng(seed)
    P, C = [], []
    for m in muscles:
        L = m.length(); n = max(12, int(80 * L))
        n = min(n, 240)
        t = rng.random(n)
        mid = (m.origin + m.insertion) / 2
        bow = np.cross(m.insertion - m.origin, [0, 0, 1.0])
        nb = np.linalg.norm(bow)
        bow = bow / nb * 0.06 * L if nb > 1e-6 else 0
        axis = (1 - t)[:, None] * m.origin[None] + t[:, None] * m.insertion[None] \
               + (np.sin(np.pi * t))[:, None] * (bow if np.ndim(bow) else 0)
        rad = 0.10 * L * np.sin(np.pi * t) ** 0.6 + 0.01
        pts = axis + rng.normal(size=(n, 3)) * rad[:, None]
        P.append(pts); C.append(np.tile(rgb, (n, 1)))
    return np.vstack(P).astype(np.float32), np.vstack(C).astype(np.float32)


if __name__ == "__main__":
    from .targets import reference_genome
    for sp in ["base_vertebrate", "giraffe"]:
        g = reference_genome(sp) if sp != "base_vertebrate" else Genome()
        ms = build_muscles(g)
        joints = sorted(set(m.joint for m in ms))
        print(f"{sp:16s} {len(ms):2d} muscles across joints {joints}")
    ms = build_muscles(reference_genome("giraffe"))
    print("sample:", ms[0].name, "L=%.2f" % ms[0].length(), ms[0].function, "|", ms[6].name, ms[6].joint)
