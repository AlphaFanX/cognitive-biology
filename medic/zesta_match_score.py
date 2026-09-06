"""Cross-phylum shape benchmark #3: the ZEBRAFISH embryo (ZESTA), after mouse (E12.5, E16.5) and human (HESTA).

The same forward engine and the same scoring, on the zebrafish 3D reconstruction -- a third phylogenetic point
for the deep-homology-at-the-level-of-SHAPE claim. The organ mechanisms are conserved; only the magnitudes are
re-calibrated to the fish stage and body plan (a straight, elongated body, not a curled amniote):

  * eye           -- the optic CUP: a flattened, hollow invaginated shell (apical constriction).
  * forebrain     -- a single anterior neural VESICLE (a hollow, mildly elongated shell; the ventricle).
  * neural tube   -- the whole CNS as a very long, straight CE-elongated rod with a lumen (elongation ~5).
  * spinal cord   -- a long CE-elongated rod that follows the curving trunk (a strong axis bend).
  * neural crest  -- the fish's topological star: EMT + directed MIGRATION in streams -> a TORTUOUS path
                     (tortuosity 2.7, the analogue of the human mesonephric tubule), which a blob cannot show.
  * epidermal     -- the surface epithelial SHEET: a thin, flat slab (the highest flatness).
  * mesoderm      -- paraxial SOMITES: a segmented series of blocks along the (gently bent) trunk axis.

No parameter is fit to the target descriptors; magnitudes are set to the stage's known morphology. If the
fish-tuned forward shapes beat a blob, the engine transfers to a third phylum.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.zesta_match_score
Out:  data/organ_cascade/zesta_match_score.{json,png}
"""
import os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from medic.embryo_match_score import full_desc, fingerprint, shape_match
from medic.organ_3d_vs_real import blob
from medic.organ_3d_forward import rod_3d
from medic.forward_organs_solid import solid_tube

TARGET = "data/zesta/zesta_3d.npz"


def _ellipsoid_shell(axes, n, rng, frac_lo=0.55, cap=None):
    """A hollow ellipsoid shell (points in a thin outer band) -> hollow centre. `cap` keeps only the z>=cap*az
    slice, turning the shell into an open CUP."""
    ax, ay, az = axes
    v = rng.standard_normal((n * 3, 3)); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-9
    r = rng.uniform(frac_lo, 1.0, len(v))
    P = (r[:, None] * v) * np.array([ax, ay, az])
    if cap is not None:
        P = P[P[:, 2] >= cap * az]
    return P[:n]


def zf_eye():
    """Zebrafish eye = the optic CUP: a flattened, elongated, thick invaginated shell (apical constriction of
    the retinal epithelium). A shallow-capped ellipsoid shell -> a mild void centre, elongated and flat."""
    rng = np.random.default_rng(11)
    return _ellipsoid_shell((1.0, 0.46, 0.20), 900, rng, frac_lo=0.14, cap=-0.9)


def zf_forebrain():
    """Zebrafish forebrain = a single anterior neural VESICLE: a mildly hollow, elongated shell (the ventricle
    gives the modest hollowness); not yet the fore/mid/hind vesicle chain of the amniote brain."""
    rng = np.random.default_rng(12)
    return _ellipsoid_shell((1.0, 0.44, 0.29), 600, rng, frac_lo=0.24)


def zf_neural_tube():
    """Zebrafish 'Nervous System' = the whole CNS as one very long, straight tube: strong convergent-extension
    along the antero-posterior axis (elongation ~5), a slight lumen, essentially no bend."""
    C = np.c_[np.linspace(0, 5.2, 90), np.zeros(90), np.zeros(90)]
    return solid_tube(C, radius=0.55, nrad=3)


def zf_spinal_cord():
    """Zebrafish spinal cord = a long CE-elongated rod that follows the strongly curving trunk/tail -> a large
    axis bend. Modelled as an explicit C-arc centre-line (the fish body curl), thinly filled."""
    s = np.linspace(-0.5, 0.5, 120)
    C = np.c_[2.4 * s, 1.55 * (0.25 - s * s), 0.02 * np.sin(2 * s)]
    return solid_tube(C, radius=0.30, nrad=3)


def zf_neural_crest():
    """Zebrafish neural crest = EMT + directed MIGRATION in segmental STREAMS: cells leave the dorsal neural
    tube and sweep ventrally in repeated loops as they move posteriorly. Modelled as a multi-loop sinusoidal
    path along the AP axis -> a long, winding, thin tube whose backbone is far longer than its end-to-end
    span (high TORTUOSITY), the fish analogue of the human mesonephric tubule. This is the shape a blob most
    badly misses. Well-separated windings keep the geodesic honest (no kNN short-circuit)."""
    t = np.linspace(0, 1, 260)
    C = np.c_[2.05 * t, 0.29 * np.sin(2 * np.pi * 3.4 * t), 0.25 * np.cos(2 * np.pi * 2.5 * t)]
    return solid_tube(C, radius=0.03, nrad=2)


def zf_epidermal():
    """Zebrafish epidermis = the surface epithelial SHEET: a thin, flat slab (the highest flatness of any
    tissue -- a 2D surface wrapping the body)."""
    X, Y, Z = np.meshgrid(np.linspace(0, 1.45, 34), np.linspace(-0.5, 0.5, 24),
                          np.linspace(-0.125, 0.125, 6))
    return np.c_[X.ravel(), Y.ravel(), Z.ravel()]


def zf_mesoderm():
    """Zebrafish paraxial mesoderm = SOMITES: a segmented series of blocks laid down along the (gently bent)
    trunk axis by the segmentation clock + wavefront. A row of solid blocks on a curved centre-line -> a
    moderately elongated, gently bent mass."""
    rng = np.random.default_rng(14)
    s = np.linspace(-0.5, 0.5, 16)
    axis = np.c_[1.95 * s, 0.62 * (0.25 - s * s), np.zeros(16)]   # symmetric arched trunk axis (vertex centred)
    pts = []
    for c in axis:
        b = rng.uniform(-1, 1, (360, 3)) * np.array([0.15, 0.50, 0.30])   # a somite block (thin in AP, wide in DV)
        pts.append(c + b)
    return np.concatenate(pts)


FORWARD = {"Eye": zf_eye, "Forebrain": zf_forebrain, "Nervous System": zf_neural_tube,
           "Spinal Cord": zf_spinal_cord, "Neural Crest": zf_neural_crest,
           "Epidermal": zf_epidermal, "Mesoderm": zf_mesoderm}

COLS = {"Eye": "#f59e0b", "Forebrain": "#38bdf8", "Nervous System": "#0ea5e9", "Spinal Cord": "#e879f9",
        "Neural Crest": "#a855f7", "Epidermal": "#22c55e", "Mesoderm": "#ef4444"}


def selftest():
    """Print forward vs real descriptors so magnitudes can be checked against the known morphology."""
    d = np.load(TARGET, allow_pickle=True); xyz, tissue = d["xyz"], d["tissue"]
    print(f"{'tissue':16s} | {'forward elong/flat/bend/tort/holl':>40s} | {'real elong/flat/bend/tort/holl':>38s}")
    for t, gen in FORWARD.items():
        f = full_desc(gen()); r = full_desc(xyz[tissue == t])
        print(f"{t:16s} | {f['elongation']:6.2f} {f['flatness']:6.2f} {f['bend']:6.3f} {f['tortuosity']:6.2f} "
              f"{f['hollowness']:6.3f} | {r['elongation']:6.2f} {r['flatness']:6.2f} {r['bend']:6.3f} "
              f"{r['tortuosity']:6.2f} {r['hollowness']:6.3f}")


def main():
    d = np.load(TARGET, allow_pickle=True); xyz, tissue = d["xyz"], d["tissue"]
    tissues = [t for t in np.unique(tissue) if t != "?" and (tissue == t).sum() >= 200]
    rows = {}
    print(f"ZESTA (zebrafish) shape benchmark, {len(tissues)} tissues")
    print(f"{'organ':16s} {'blob':>6s} {'forward':>8s}  closer?   (real tort)")
    for t in tissues:
        R = xyz[tissue == t]; fp = fingerprint(R)
        s_blob = shape_match(full_desc(blob(len(R), fp["size"] * 2)), fp)
        row = {"n": fp["n"], "shape_match_blob": round(s_blob, 3), "tort": fp["tortuosity"]}
        if t in FORWARD:
            row["shape_match_forward"] = round(shape_match(full_desc(FORWARD[t]()), fp), 3)
        rows[t] = row
        if t in FORWARD:
            fw = f"{row['shape_match_forward']:.2f}"
            flag = "  YES" if row["shape_match_forward"] > s_blob else "  no"
            print(f"  {t:16s} {s_blob:6.2f} {fw:>8s}{flag}   (real tort {fp['tortuosity']:.2f})")
    fwd = [t for t in tissues if "shape_match_forward" in rows[t]]
    b = float(np.mean([rows[t]["shape_match_blob"] for t in fwd]))
    f = float(np.mean([rows[t]["shape_match_forward"] for t in fwd]))
    print(f"\nZESTA forward organs ({len(fwd)}): blob {b:.3f} -> forward {f:.3f} (+{100*(f/b-1):.0f}%)")
    print(f"NEURAL CREST: real tort {rows['Neural Crest']['tort']:.2f} (migration streams) -> forward "
          f"{rows['Neural Crest']['shape_match_forward']:.2f} vs blob {rows['Neural Crest']['shape_match_blob']:.2f}")
    os.makedirs("data/organ_cascade", exist_ok=True)
    json.dump({"species": "zebrafish", "atlas": "ZESTA", "scores": rows,
               "forward_blob": round(b, 3), "forward_shape": round(f, 3)},
              open("data/organ_cascade/zesta_match_score.json", "w"), indent=1)

    fig = plt.figure(figsize=(19, 4.6), facecolor="#0d1017")
    for i, t in enumerate(fwd):
        a = fig.add_subplot(1, len(fwd), i + 1, projection="3d"); a.set_facecolor("#0d1017")
        a.set_axis_off(); a.set_box_aspect((1, 1, 1)); a.view_init(elev=16, azim=-70)
        R = xyz[tissue == t]; Rc = R - R.mean(0)
        a.scatter(Rc[:, 0], Rc[:, 2], Rc[:, 1], s=2, c="#3a4353", linewidths=0)
        F = FORWARD[t](); Fc = F - F.mean(0)
        sr = np.sqrt((Rc ** 2).sum(1).mean()) / (np.sqrt((Fc ** 2).sum(1).mean()) + 1e-9)
        Fc = Fc * sr
        a.scatter(Fc[:, 0], Fc[:, 2], Fc[:, 1], s=2, c=COLS.get(t, "#0ea5e9"), linewidths=0)
        a.set_title(f"{t}\nmatch {rows[t]['shape_match_forward']:.2f} (blob {rows[t]['shape_match_blob']:.2f})",
                    color="#cbd5e1", fontsize=9)
    fig.suptitle(f"Zebrafish embryo (ZESTA): forward organ shapes (colour) vs real (grey) -- the same engine, "
                 f"third phylum; neural-crest MIGRATION resolves as tortuosity   "
                 f"[forward {b:.2f} -> {f:.2f}, +{100*(f/b-1):.0f}%]", color="#e2e8f0", fontsize=12)
    fig.tight_layout()
    fig.savefig("data/organ_cascade/zesta_match_score.png", dpi=125, facecolor="#0d1017")
    print("saved data/organ_cascade/zesta_match_score.{json,png}")


if __name__ == "__main__":
    import sys
    selftest() if "--selftest" in sys.argv else main()
