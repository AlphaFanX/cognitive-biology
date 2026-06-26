"""Dense morph of the FaceBase mean mesh from GWAS dosages.

Anchors a handful of GWAS facial regions geometrically on the mean mesh, then propagates
each anchor's displacement to all 43k vertices via a smooth Gaussian field. Turns the
landmark-level adapter (Sigma dosage*beta*e) into a deformation of the real face surface.

Mesh frame (confirmed from data): +y = up, +z = anterior, x = lateral.
Magnitudes (scale fractions) are illustrative placeholders pending real beta_k; directions
are grounded in published GWAS. EDAR(chin) and PAX3(nasion) carry real per-population freqs.
"""
import os, numpy as np

# FaceBase mean-face mesh (third-party data, not redistributed here; see data/README.md).
# Defaults to face_demo/data/meanface.npz next to this file; override with $FACEBASE_MEANFACE.
MESH = os.environ.get(
    "FACEBASE_MEANFACE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "meanface.npz"),
)

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
    # alare wings: near nose-base height, lateral extremes within nasal width
    m = (np.abs(V[:,1]-ny) < 0.08*size) & (np.abs(V[:,0]-cx) < 0.20*size)
    al = V[m]; alare_L = al[np.argmin(al[:,0])]; alare_R = al[np.argmax(al[:,0])]
    # bridge points: lateral to nasion, same height
    m = (np.abs(V[:,1]-nasion[1]) < 0.05*size) & (np.abs(V[:,0]-cx) < 0.10*size)
    br = V[m]; bridge_L = br[np.argmin(br[:,0])]; bridge_R = br[np.argmax(br[:,0])]
    return dict(nose=nose, chin=chin, nasion=nasion, alare_L=alare_L, alare_R=alare_R,
                bridge_L=bridge_L, bridge_R=bridge_R, size=size, cx=cx)

def _u(v): v=np.asarray(v,float); return v/(np.linalg.norm(v)+1e-9)

def deformers(A):
    """gene -> list of (anchor_point, unit_direction, scale_fraction_per_allele)."""
    s = A["size"]
    return {
        "EDAR":  [(A["chin"],   _u([0,0,1]), 0.030)],                 # chin protrusion
        "PAX3":  [(A["nasion"], _u([0,0,1]), 0.022)],                 # nasion depth
        "DCHS2": [(A["nose"],   _u([0,1,0.3]), 0.022)],               # nasal tip up
        "RUNX2": [(A["bridge_L"],_u([-1,0,0]),0.015),(A["bridge_R"],_u([1,0,0]),0.015)],
        "GLI3":  [(A["alare_L"],_u([-1,0,0]),0.018),(A["alare_R"],_u([1,0,0]),0.018)],
        "PAX1":  [(A["alare_L"],_u([-1,0,0]),0.012),(A["alare_R"],_u([1,0,0]),0.012)],
    }

def morph(V, A, dosages, exaggerate=1.0, sigma_frac=0.11):
    """Return deformed vertices. dosages: dict gene->allele dosage (0..2 or 2*freq)."""
    s = A["size"]; sigma = sigma_frac*s
    D = np.zeros_like(V)
    defs = deformers(A)
    for gene, dose in dosages.items():
        for anchor, direction, scale in defs.get(gene, []):
            dist2 = ((V-anchor)**2).sum(1)
            w = np.exp(-dist2/(2*sigma**2))            # smooth local field
            D += (dose*scale*s*exaggerate) * w[:,None]*direction
    return V + D
