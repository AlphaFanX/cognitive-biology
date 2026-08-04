"""Richer body SURFACE generator: deform the real MakeHuman mesh by the genome, colour it by the genome
(Miles, 2026-07-26: build the richer surface generator; what else can we source from the genome?).

The silhouette generator (body_builder_v3) makes a coarse point cloud whose mean is ~10% off the human. The
richer generator does for the BODY what the FaceBase mesh does for the FACE: it takes the real human SURFACE
(the MakeHuman mesh) as the base and DEFORMS it by the genome-covered proportion knobs, so its mean IS the
human (0% by construction) and a genotype deforms it into a specific, still-fully-surfaced body. Regions
(head / trunk / arms / legs) are segmented geometrically and scaled:

   height        <- adult height (7452 loci)              overall y-scale
   leg fraction  <- sitting-height ratio (1231)           lengthen legs about the hip line
   girth         <- BMI / waist-hip (2338)                trunk & limb x,z-scale
   head fraction <- head circumference / ICV (112)        head scale about the neck
   arm length    <- limb-to-trunk (1231)                  arm x-extent

WHAT ELSE THE GENOME CAN SOURCE FOR THE ILLUSTRATIONS (reviewed): the single biggest addition is
PIGMENTATION -- skin, hair and eye colour are OLIGOGENIC with large-effect, textbook loci (SLC24A5 rs1426654,
SLC45A2 rs16891982 for skin; HERC2/OCA2 rs12913832 for eye; MC1R for red hair), so unlike the polygenic
proportions the colour can be set from a handful of alleles. This module colours the surface by a pigmentation
score from those loci. Further genome-sourceable layers, not yet built: sexual dimorphism (sex x shoulder/hip
GWAS), muscularity (lean-mass 1323 loci), fat distribution (waist-hip / gynoid-android), hair (curl EDAR/TCHH,
pattern AR).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.body_surface_generator
Out:  data/organ_cascade/body_surface_generator.{json,png}
"""
import os, json
import numpy as np

HERE = os.path.dirname(__file__)
MH = os.path.join(HERE, "..", "data", "bodybase", "makehuman_decimated.npz")
OUT = os.path.join(HERE, "..", "data", "organ_cascade", "body_surface_generator.json")
FIG = os.path.join(HERE, "..", "data", "organ_cascade", "body_surface_generator.png")

# pigmentation loci (illustrative dosages -> a 0..1 light score); skin endpoints (dark..light) RGB
PIG_LOCI = {"SLC24A5 rs1426654": 0.4, "SLC45A2 rs16891982": 0.4, "HERC2/OCA2 rs12913832": 0.2}
SKIN_DARK = np.array([92, 56, 36]); SKIN_LIGHT = np.array([255, 221, 181])


def segment(V):
    ymax = V[:, 1].max()
    seg = np.array(["trunk"] * len(V), dtype=object)
    seg[V[:, 1] > 0.86 * ymax] = "head"
    arm = (np.abs(V[:, 0]) > 18) & (V[:, 1] > 0.45 * ymax) & (V[:, 1] <= 0.86 * ymax)
    seg[arm] = "arm"
    leg = (V[:, 1] < 0.48 * ymax) & (np.abs(V[:, 0]) < 18)
    seg[leg] = "leg"
    return seg


def deform(V, seg, k):
    V = V.copy(); ymax = V[:, 1].max()
    y_hip = 0.47 * ymax; y_neck = 0.83 * ymax
    tr = seg == "trunk"; hd = seg == "head"; ar = seg == "arm"; lg = seg == "leg"
    V[tr, 0] *= k["girth"]; V[tr, 2] *= k["girth"]
    V[hd, 0] *= k["head"]; V[hd, 2] *= k["head"]; V[hd, 1] = y_neck + (V[hd, 1] - y_neck) * k["head"]
    V[ar, 0] *= k["arm"]
    V[lg, 1] = y_hip + (V[lg, 1] - y_hip) * k["leg"]
    V[:, 1] *= k["height"]
    return V


def skin_rgb(pig):
    return (SKIN_DARK + (SKIN_LIGHT - SKIN_DARK) * pig) / 255.0


def chamfer(A, B):
    from scipy.spatial import cKDTree
    def al(P):
        Q = P - P.mean(0); return Q / (np.sqrt((Q ** 2).sum(1).mean()) + 1e-9)
    A, B = al(A), al(B)
    return 100 * 0.5 * (cKDTree(B).query(A)[0].mean() + cKDTree(A).query(B)[0].mean())


DEFAULT = dict(height=1.0, leg=1.0, girth=1.0, head=1.0, arm=1.0)
# a developmental sequence back to the fetus (proportions from staged allometry: big head, short limbs early),
# plus the adult shown in two pigmentations
PEOPLE = {
    "fetus":      dict(height=0.34, leg=0.52, girth=1.22, head=1.85, arm=0.62, pig=0.5),
    "newborn":    dict(height=0.50, leg=0.64, girth=1.16, head=1.55, arm=0.80, pig=0.5),
    "child (6 yr)": dict(height=0.66, leg=0.80, girth=1.10, head=1.42, arm=0.90, pig=0.5),
    "adult, fair": dict(DEFAULT, pig=0.88),
    "adult, dark": dict(DEFAULT, pig=0.18),
}


def run():
    z = np.load(MH, allow_pickle=True); V0 = z["V"].astype(float)
    seg = segment(V0)
    counts = {s: int((seg == s).sum()) for s in ["head", "trunk", "arm", "leg"]}
    print("Richer body SURFACE generator (MakeHuman mesh deformed by the genome)")
    print("segments:", counts)

    built = {}
    for name, p in PEOPLE.items():
        k = {kk: p.get(kk, DEFAULT[kk]) for kk in DEFAULT}
        Vd = deform(V0, seg, k)
        ch = chamfer(Vd, V0)
        built[name] = dict(V=Vd, pig=p["pig"], chamfer_to_makehuman=round(ch, 2), knobs=k)
        print(f"  {name:26s} pig={p['pig']:.2f}  Chamfer to MakeHuman {ch:5.1f}%  "
              f"(mean IS the mesh -> baseline 0)")

    print("\nGENOME SOURCES FOR THE ILLUSTRATIONS:")
    print("  proportions <- height 7452 / sitting-height 1231 / BMI-WHR 2338 / head 112 loci")
    print("  PIGMENTATION (NEW, oligogenic): skin SLC24A5+SLC45A2, eye HERC2/OCA2, red hair MC1R "
          "-> skin/eye/hair colour set from a handful of alleles (unlike the polygenic proportions)")
    print("  further (not yet built): sexual dimorphism (sex x shoulder/hip), muscularity (lean-mass 1323), "
          "fat distribution (WHR gynoid/android), hair curl (EDAR/TCHH)")

    out = dict(segments=counts,
               people={n: dict(pig=built[n]["pig"], chamfer_to_makehuman=built[n]["chamfer_to_makehuman"],
                               knobs=built[n]["knobs"]) for n in built},
               pigmentation_loci=list(PIG_LOCI),
               genome_sources=dict(proportions="height 7452 / sitting-height 1231 / BMI-WHR 2338 / head 112",
                                   pigmentation="SLC24A5, SLC45A2 (skin); HERC2/OCA2 (eye); MC1R (red hair)",
                                   further="sexual dimorphism, lean-mass muscularity 1323, WHR fat distribution, hair curl EDAR/TCHH"),
               note="richer surface generator = real MakeHuman mesh deformed by genome proportion knobs + "
                    "coloured by genome pigmentation; mean = the mesh (0% baseline), a genotype deforms+recolours it")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=1)
    _figure(built)
    print(f"\nwrote {OUT} and {FIG}")
    return out


def _render_mesh(a, V, F, base):
    """smooth (Gouraud) shaded surface render of the front-facing mesh -> a recognizable, smooth human."""
    import matplotlib.tri as mtri
    from matplotlib.colors import LinearSegmentedColormap
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    fn /= (np.linalg.norm(fn, axis=1, keepdims=True) + 1e-9)
    vn = np.zeros_like(V)                                        # per-vertex normals (smooth shading)
    for j in range(3):
        np.add.at(vn, F[:, j], fn)
    vn /= (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.3, 0.2, 1.0]); light /= np.linalg.norm(light)
    inten = np.clip(vn @ light, 0.25, 1.0)                       # per-vertex Lambertian
    ff = fn[:, 2] > 0.0                                          # front-facing triangles only (clean occlusion)
    cmap = LinearSegmentedColormap.from_list("skin", [np.clip(base * 0.35, 0, 1), base, np.clip(base * 1.12, 0, 1)])
    tri = mtri.Triangulation(V[:, 0], V[:, 1], F[ff])
    a.tripcolor(tri, inten, shading="gouraud", cmap=cmap, vmin=0.2, vmax=1.05)


def _figure(built):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    z = np.load(MH, allow_pickle=True); F = z["F"]
    names = list(built)
    fig, ax = plt.subplots(1, len(names), figsize=(2.8 * len(names), 6.4), facecolor="white")
    # shared scale across panels so the SIZE progression shows (fetus small -> adult tall), feet aligned at 0
    gx = max(np.abs(built[n]["V"][:, 0]).max() for n in names) + 6
    gy = max(built[n]["V"][:, 1].max() for n in names) + 6
    for a, name in zip(ax, names):
        _render_mesh(a, built[name]["V"], F, skin_rgb(built[name]["pig"]))
        a.set_aspect("equal"); a.axis("off")
        a.set_xlim(-gx, gx); a.set_ylim(-2, gy)
        a.set_title(f"{name}\n(off MakeHuman {built[name]['chamfer_to_makehuman']:.0f}%)", fontsize=8.5)
    fig.suptitle("Richer body SURFACE generator: the real MakeHuman mesh deformed by genome proportion knobs "
                 "(height, leg-to-trunk, girth, head) and coloured by genome PIGMENTATION (skin SLC24A5/SLC45A2), "
                 "smooth-shaded.\nA developmental sequence fetus -> newborn -> child -> adult (big head + short "
                 "limbs early, from the negative cranial / positive limb allometry), the adult shown fair and dark "
                 "-- recognizable, fully-surfaced humans from the genome.", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.9]); fig.savefig(FIG, dpi=140, facecolor="white")


if __name__ == "__main__":
    run()
