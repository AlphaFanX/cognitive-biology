"""subhead_program.py -- the sub-head PROGRAM: Gray's structures as a TABLE, not code. (2026-08-29)

Miles's reduction: a structure is 4 coordinates + a clock + a master gene (Paper: The Structure Program),
so the whole roster is a TABLE and one generic engine instantiates every row. Adding a Gray's structure =
adding a ROW here, never editing the heads. The engine does the two things the telencephalon audit proved
are required:

  (1) INHERIT every parent map. A child fate is registered in the value maps (adhesion / ECM / voltage /
      clock), the packing group, and the scorecard composite by INHERITING its parent -- so a child is
      electrically and developmentally its parent until it diverges. This is why the split does not silently
      fall out of a map (the DV_GROUP bug) or read as a different tissue (the Vm bug).
  (2) ORDER on the parent's OWN operator. The children are laid along the family coordinate of the parent's
      gap-junction operator -- SHELL = whitened radial depth (concentric organs), CHAIN = Fiedler geodesic
      (tubes) -- exactly the four-families machinery, applied one level down.

Row 1 is the kidney's cortico-medullary shells (the worklist's #1: gap 26).
"""
from __future__ import annotations
import numpy as np

# (name, radial_lo, radial_hi, master_gene). SHELL fracs are 0=centre .. 1=surface; CHAIN fracs are the
# geodesic 0=one pole .. 1=the other.
SPECS = [
    dict(parent="Kidney", family="shell", children=[
        ("Renal Cortex",  0.62, 1.00, "Six2/Wt1"),     # outer shell: the nephron-bearing cortex
        ("Renal Medulla", 0.30, 0.62, "Umod"),          # middle: loops of Henle + collecting ducts (pyramids)
        ("Renal Pelvis",  0.00, 0.30, "Gata3/Ret"),     # inner core: the collecting space / ureteric tree
    ]),
    # Spinal cord: a chain along its own geodesic = the cranio-caudal neuraxis; the four levels are Hox-
    # colinear bands along it (thoracic the longest region). Orientation of the geodesic pole may flip;
    # the split is four ordered levels regardless, and they fold back into Spinal Cord for scoring.
    # orient=(axis, sign): the geodesic's rank-0 pole is fixed anatomically (cycle 82) -- the Fiedler vector
    # is defined only up to sign, so without this the children flipped end-for-end between builds (the
    # build_base nondeterminism the phase-bisect convicted). Sacral = caudal = low AP (head at +x).
    # CORD = AXIS family (cycle 82): at 45k cells the cord's kNN Laplacian is not one connected chain, so
    # its Fiedler vector was a component indicator, not the neuraxis -- Sacral and Cervical landed at the
    # SAME height (x 0.856 vs 0.878 on the standing body). The Hox-colinear levels ARE bands along the
    # cranio-caudal axis, and the build_base cord is straight (flex 0), so the axis split is the honest one.
    dict(parent="Spinal Cord", family="axis", axis=0, children=[
        ("Sacral Cord",   0.00, 0.18, "Hoxd10/Hoxd12"),
        ("Lumbar Cord",   0.18, 0.42, "Hoxc10"),
        ("Thoracic Cord", 0.42, 0.72, "Hoxc8"),
        ("Cervical Cord", 0.72, 1.00, "Hoxc5/Hoxb4"),
    ]),
    # Liver: a chain along its transverse (left-right) long axis; the right lobe is the largest, the
    # caudate the smallest. A packed abdominal viscus -> the lobes inherit Liver's packing group and
    # fold back into the Liver composite, so the split adds names without moving the organ.
    dict(parent="Liver", family="chain", orient=(2, +1), children=[          # right lobe = +z (laterality sign)
        ("Caudate Lobe",  0.00, 0.14, "Tbx3"),
        ("Left Hepatic Lobe",  0.14, 0.42, "Hlx"),
        ("Right Hepatic Lobe", 0.42, 1.00, "Hlx"),
    ]),
    # Bladder: a chain along the supero-inferior axis; dome at the apex, trigone at the base/neck.
    dict(parent="Bladder", family="chain", orient=(0, +1), children=[        # trigone inferior = low AP
        ("Bladder Trigone", 0.00, 0.25, "Tbx18/Shh"),
        ("Bladder Body",    0.25, 0.70, "Uroplakin"),
        ("Bladder Dome",    0.70, 1.00, "Shh"),
    ]),
    # Pancreas: a chain along its own long axis; head (duodenal) - body - tail (splenic).
    dict(parent="Pancreas", family="chain", orient=(2, -1), children=[       # head duodenal (right, +z) -> tail splenic (left, -z)
        ("Pancreatic Head", 0.00, 0.42, "Pdx1"),
        ("Pancreatic Body", 0.42, 0.75, "Pdx1"),
        ("Pancreatic Tail", 0.75, 1.00, "Pdx1"),
    ]),
    # Thymus: bilobed -> left and right lobe, split on the medio-lateral axis.
    dict(parent="Thymus", family="axis", axis=2, children=[
        ("Left Thymic Lobe",  0.00, 0.50, "Foxn1"),
        ("Right Thymic Lobe", 0.50, 1.00, "Foxn1"),
    ]),
    # Meninges: the three concentric coverings -- dura (outer) / arachnoid / pia (inner, on the cord surface).
    dict(parent="Meninges", family="shell", children=[
        ("Pia Mater",       0.00, 0.33, "Foxc1"),       # inner, against the neuraxis
        ("Arachnoid Mater", 0.33, 0.66, "Foxc1"),
        ("Dura Mater",      0.66, 1.00, "Foxc1"),        # outer, the tough layer
    ]),
    # Gonad: cortex (outer, germ-cell layer) over medulla (inner) -- a concentric shell.
    dict(parent="Gonad", family="shell", children=[
        ("Gonadal Medulla", 0.00, 0.50, "Sox9/Nr5a1"),
        ("Gonadal Cortex",  0.50, 1.00, "Wnt4/Foxl2"),
    ]),
    # Foregut: the proximal gut tube as a chain -- oesophagus - stomach - duodenum from cranial to caudal.
    dict(parent="Foregut", family="chain", orient=(0, -1), children=[        # oesophagus cranial = high AP
        ("Oesophagus",       0.00, 0.33, "Sox2"),
        ("Stomach",          0.33, 0.70, "Barx1"),
        ("Duodenum",         0.70, 1.00, "Pdx1"),
    ]),
    # Lung: a BRANCHING TREE -> the five lobes (right superior/middle/inferior, left superior/inferior).
    # The relabel clusters the lung on its own branch modes; the primary order axis is medio-lateral so the
    # two lungs group, then cranio-caudal within each. (Which cluster maps to which lobe is best-effort.)
    dict(parent="Lung", family="tree", axis=2, children=[
        ("Right Superior Lobe", 0.0, 1.0, "Sox9"),
        ("Right Middle Lobe",   0.0, 1.0, "Sox9"),
        ("Right Inferior Lobe", 0.0, 1.0, "Sox9"),
        ("Left Superior Lobe",  0.0, 1.0, "Sox9"),
        ("Left Inferior Lobe",  0.0, 1.0, "Sox9"),
    ]),
    # NOTE: the BRAIN-region axis splits (Midbrain -> tectum/tegmentum [DV]; Cerebellum -> vermis/
    # hemispheres [LR]; Forebrain-diencephalon -> thalamus/hypothalamus [DV]) are DEFERRED. They read
    # cleanly at the built body (integrity 13/13, even splits), and the axis engine handles them, but
    # merely registering their child fates in FATES perturbs the grown body via the developmental-head
    # brain coupling (isolated: +Midbrain 86331->90308, +Cerebellum ->80570), the same name-branching
    # sensitivity the telencephalon audit found. Because these children are post-build RELABELS (never
    # in fate_of), that coupling is a spurious side effect, not real biology, so they wait for the
    # per-head brain treatment (register the child in the brain heads it should share, exempt the rest --
    # e.g. the eye evaginates from diencephalon, not telencephalon). The axis family is validated on the
    # peripheral Thymus, which does not move the body (-98 cells).
]


def _children():
    for sp in SPECS:
        for (name, lo, hi, master) in sp["children"]:
            yield sp, name, lo, hi, master


def extend_value_maps(ADH=None, ECM=None, VM_OF=None):
    """Children inherit the parent's adhesion / ECM / voltage (call before FATES is built for ADH/ECM,
    and again for VM_OF once it exists)."""
    for sp, name, lo, hi, master in _children():
        p = sp["parent"]
        if ADH is not None and p in ADH:
            ADH.setdefault(name, ADH[p])
        if ECM is not None and p in ECM:
            ECM.setdefault(name, ECM[p])
        if VM_OF is not None and p in VM_OF:
            VM_OF.setdefault(name, VM_OF[p])


def extend_clock(FATE_PRC2):
    for sp, name, lo, hi, master in _children():
        FATE_PRC2.setdefault(name, FATE_PRC2.get(sp["parent"], 0.34))   # suborgan unlocks with its parent


def extend_dv_group(DV_GROUP):
    for sp, name, lo, hi, master in _children():
        DV_GROUP.setdefault(name, DV_GROUP.get(sp["parent"], sp["parent"]))  # child packs with its organ


def composites():
    """parent -> (parent, *children): the scorecard folds the children back into the whole organ."""
    return {sp["parent"]: tuple([sp["parent"]] + [c[0] for c in sp["children"]]) for sp in SPECS}


def children_of(parent):
    """the sub-head names a parent was split into (empty if unsplit)."""
    return [c[0] for sp in SPECS if sp["parent"] == parent for c in sp["children"]]


def expand_names(names):
    """append the sub-head children for any parent named -- so a name-keyed organ set (integrity audit,
    condense, packing) still finds a parent's cells after the split relabels them into its children.
    This is the general form of the 'register the split in every fate-keyed map' rule."""
    out = list(names)
    for n in names:
        out.extend(children_of(n))
    return out


def apply(base, F, FIDX):
    """Order each parent's cells along its family coordinate and relabel into the children. In place-safe
    (returns a new F). Runs in build_base after the organ heads, like the heart-tube chamber assignment."""
    from medic.suborgan_attractor import gj_laplacian, fiedler
    F = F.copy()
    for sp in SPECS:
        pid = FIDX.get(sp["parent"])
        if pid is None:
            continue
        idx = np.where(F == pid)[0]
        if len(idx) < 30:
            continue
        pts = base[idx].astype(float)
        if sp["family"] == "tree":
            # a branching network: its sub-parts are spatial TERRITORIES (branches), not a 1-D order, so
            # they cannot be banded on one coordinate. The branches ARE the degenerate low modes of the
            # structure's own operator (why the cascade reads a tree as a SHELL). So spectral-embed the
            # gap-junction Laplacian on its first K-1 branch modes, cluster into K territories, and name
            # them in canonical spatial order (primary axis, LR tie-break). Exact per-branch anatomical
            # naming (which lobe is which) is a canonical-map refinement; the territories are read from the
            # operator.
            from scipy.cluster.vq import kmeans2
            K = len(sp["children"])
            c = pts.mean(0); C = pts - c                              # whiten the cloud so territories are
            ev, U = np.linalg.eigh(C.T @ C / len(C) + 1e-9 * np.eye(3))  # shape-normalised (branches of a
            emb = (C @ U) / np.sqrt(ev + 1e-9)                        # supplied region = spatial territories)
            _, lab = kmeans2(emb, K, seed=0, minit="++", missing="warn")
            oa = sp.get("axis", 0)
            present = [j for j in range(K) if (lab == j).any()]
            cent = {j: pts[lab == j].mean(0) for j in present}
            order = sorted(present, key=lambda j: (cent[j][oa], cent[j][2]))  # primary axis, LR tie-break
            for rank_j, j in enumerate(order):
                cid = FIDX.get(sp["children"][rank_j][0])
                if cid is not None:
                    F[idx[lab == j]] = cid
            continue
        if sp["family"] == "shell":
            c = pts.mean(0); C = pts - c
            ev, U = np.linalg.eigh(C.T @ C / len(C) + 1e-9 * np.eye(3))
            w = (C @ U) / np.sqrt(ev + 1e-9)                       # whiten ellipsoid -> concentric spheres
            rad = np.linalg.norm(w, axis=1)                        # radial depth
            r = np.argsort(np.argsort(rad)).astype(float) / (len(rad) - 1 + 1e-9)  # RANK 0=centre..1=surface
            #  (rank, not raw radius, so the bands are cell-fraction shells robust to a few outlier cells)
        elif sp["family"] == "axis":
            # order along a named BODY axis (0=AP cranio-caudal, 1=DV ventral..dorsal, 2=LR). This is the
            # anatomical-axis split the four-families machinery needs for dorso-ventral parts (tectum vs
            # tegmentum) and medio-lateral parts (cerebellar vermis vs hemispheres) that no geodesic orders.
            v = pts[:, sp.get("axis", 1)]
            r = np.argsort(np.argsort(v)).astype(float) / (len(v) - 1 + 1e-9)   # rank 0=low..1=high along axis
        else:                                                      # chain
            s = fiedler(gj_laplacian(pts, np.ones(len(pts))))
            # POLE CONVENTION (cycle 82): a Fiedler vector is defined up to sign, and eigsh started from
            # the process-global random state, so the geodesic's pole -- hence which end is child 0 --
            # flipped between builds (in-process AND cross-process; the label multiset was identical,
            # the assignment was not = the ORDER-ONLY signature). Anchor the pole to the spec's body
            # axis: rank increases with sign * coordinate. fiedler() now also takes a seeded start.
            oax, osg = sp.get("orient", (0, +1))
            q = pts[:, oax]
            if np.std(q) > 0 and np.std(s) > 0 and osg * np.corrcoef(s, q)[0, 1] < 0:
                s = 1.0 - s
            r = np.argsort(np.argsort(s)).astype(float) / (len(s) - 1 + 1e-9)  # geodesic rank 0..1
        for (name, lo, hi, master) in sp["children"]:
            cid = FIDX.get(name)
            if cid is None:
                continue
            sel = (r >= lo) & ((r <= hi) if hi >= 1.0 else (r < hi))
            F[idx[sel]] = cid
    return F


if __name__ == "__main__":
    print("SUB-HEAD PROGRAM -- the roster as a table")
    for sp in SPECS:
        print(f"  {sp['parent']:10s} [{sp['family']}] -> " +
              ", ".join(f"{c[0]} ({c[3]})" for c in sp["children"]))
    print(f"\n{sum(len(sp['children']) for sp in SPECS)} sub-heads in {len(SPECS)} specs. Add a structure = add a row.")
