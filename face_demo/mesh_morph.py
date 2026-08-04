"""Dense morph of the FaceBase mean mesh from GWAS dosages.

Anchors a handful of GWAS facial regions geometrically on the mean mesh, then propagates
each anchor's displacement to all 43k vertices via a smooth Gaussian field. Turns the
landmark-level adapter (Sigma dosage*beta*e) into a deformation of the real face surface.

Mesh frame (confirmed from data): +y = up, +z = anterior, x = lateral.
Magnitudes (scale fractions) are now grounded in the Xiong et al. 2025 C-GWAS per-allele effect
sizes (see deformers() and medic/ground_face_betas.py); directions are from published GWAS.
EDAR(chin) and PAX3(nasion) carry real per-population freqs.
"""
import os, numpy as np

MESH = r"C:\Users\jacobsme\cognimed\face_demo\data\meanface.npz"

def load():
    d = np.load(MESH)
    return d["V"].astype(float), d["F"].astype(int), d["HL"].astype(float)

def anchors(V):
    cx = np.median(V[:,0]); size = np.linalg.norm(V.max(0)-V.min(0))
    nose = V[np.argmax(V[:,2])]                      # pronasale = most anterior
    ny = nose[1]
    # chin: in the low band BELOW the nose (excludes the more-anterior nose tip), central,
    # then the most anterior point there -> pogonion
    m = (V[:,1] < ny - 0.07*size) & (np.abs(V[:,0]-cx) < 0.10*size)
    if not m.any():   # fallback: lowest central vertex
        m = (np.abs(V[:,0]-cx) < 0.10*size) & (V[:,1] < np.percentile(V[:,1],8))
    chin = V[m][np.argmax(V[m][:,2])]
    # nasion: central, above nose tip toward brow, least anterior (bridge root depression)
    m = (np.abs(V[:,0]-cx) < 0.05*size) & (V[:,1] > ny+0.06*size) & (V[:,1] < ny+0.22*size)
    nasion = V[m][np.argmin(V[m][:,2])]
    # Lateral wings/bridge as SYMMETRIC constructed points (mirrored across the midline cx).
    # Deriving L/R independently from the mesh gave asymmetric, too-high anchors that tore the
    # brow/eye region when pushed laterally; these are anchor points for a smooth field, so a
    # clean mirrored pair on the nose is both correct and stable.
    bx = 0.050*size; by = 0.62*nasion[1] + 0.38*ny; bz = 0.55*nasion[2] + 0.45*nose[2]  # upper-mid nasal bridge
    bridge_L = np.array([cx-bx, by, bz]); bridge_R = np.array([cx+bx, by, bz])
    ax_ = 0.10*size; ay = ny + 0.025*size; az = nose[2] - 0.06*size           # alar wings, beside/above the tip
    alare_L = np.array([cx-ax_, ay, az]); alare_R = np.array([cx+ax_, ay, az])
    return dict(nose=nose, chin=chin, nasion=nasion, alare_L=alare_L, alare_R=alare_R,
                bridge_L=bridge_L, bridge_R=bridge_R, size=size, cx=cx)

def _u(v): v=np.asarray(v,float); return v/(np.linalg.norm(v)+1e-9)

import os as _os, json as _json
_DATA = _os.path.join(_os.path.dirname(__file__), "..", "data")


def _load_json(name):
    try:
        return _json.load(open(_os.path.join(_DATA, name)))
    except Exception:
        return {}


def _betas():
    """gene -> |beta|, read LIVE from the catalog->model adapter table (medic/gene_trait_adapter.py).
    Forward genes (chin/nasion/tip) use their region effect from data/adapter_table.json; the lateral
    WIDENING genes (bridge/alae) use the real alare-alare nose-width betas from data/nose_width_betas.json.
    So the figure is driven by the table, not by hard-coded scales (fallbacks apply if a file is absent)."""
    fwd = {}
    t = _load_json("adapter_table.json")
    for k in ("face.chin", "face.nasion", "face.nasal_tip"):
        v = t.get(k, {})
        if isinstance(v.get("beta"), (int, float)):
            fwd[v["gene"]] = abs(v["beta"])
    wb = t.get("face.nose_width", {}).get("width_betas", {})   # real alare-alare nose-width betas
    lat = {g: abs(b) for g, b in wb.items()}
    return fwd, lat


def deformers(A):
    """gene -> list of (anchor, unit_direction, scale_per_allele, sigma_frac).

    DIRECTIONS from the facial-shape literature (Adhikari 2016). MAGNITUDES are read LIVE from the
    catalog->model adapter table: scale = gain*|beta|. Forward effects use gain 0.25 on the region beta;
    lateral (widening) effects use gain 0.50 on the real nose-WIDTH beta (a widening is visually amplified,
    so it is shown at reduced gain). Tight sigma keeps the nose widening local. Xiong C-GWAS (Zenodo
    13730680) via medic/gene_trait_adapter.py -> data/adapter_table.json + nose_width_betas.json."""
    fwd, lat = _betas()
    GF, GL = 0.25, 0.50
    def scf(g, fb): return GF * fwd[g] if g in fwd else fb
    def scl(g, fb): return GL * lat[g] if g in lat else fb
    return {
        "EDAR":  [(A["chin"],   _u([0,0,1]), scf("EDAR", 0.0248), 0.10)],     # chin protrusion
        "PAX3":  [(A["nasion"], _u([0,0,1]), scf("PAX3", 0.0254), 0.09)],     # nasion depth
        "DCHS2": [(A["nose"],   _u([0,1,0.3]), scf("DCHS2", 0.0288), 0.08)],  # nasal tip
        "RUNX2": [(A["bridge_L"],_u([-1,0,0]),scl("RUNX2",0.0095),0.032),(A["bridge_R"],_u([1,0,0]),scl("RUNX2",0.0095),0.032)],  # bridge
        "GLI3":  [(A["alare_L"],_u([-1,0,0]),scl("GLI3",0.0115),0.034),(A["alare_R"],_u([1,0,0]),scl("GLI3",0.0115),0.034)],      # alae
        "PAX1":  [(A["alare_L"],_u([-1,0,0]),scl("PAX1",0.0025),0.034),(A["alare_R"],_u([1,0,0]),scl("PAX1",0.0025),0.034)],      # alae
    }

def morph(V, A, dosages, exaggerate=1.0, sigma_frac=0.11):
    """Return deformed vertices. dosages: dict gene->allele dosage (0..2 or 2*freq)."""
    s = A["size"]
    D = np.zeros_like(V)
    defs = deformers(A)
    for gene, dose in dosages.items():
        for item in defs.get(gene, []):
            anchor, direction, scale = item[0], item[1], item[2]
            sigma = (item[3] if len(item) > 3 else sigma_frac) * s   # per-deformer sigma
            dist2 = ((V-anchor)**2).sum(1)
            w = np.exp(-dist2/(2*sigma**2))            # smooth local field
            D += (dose*scale*s*exaggerate) * w[:,None]*direction
    return V + D
