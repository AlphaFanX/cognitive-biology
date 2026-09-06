"""
Basic-vertebrate development for the BROWSER — grow from one cell to ~100k cells, WITH limb buds
and organ buds (heart, otic vesicles, eyes), a late body flexure, exported as a frames JSON + a
three.js viewer you can rotate (OrbitControls), play/pause, scrub, and recolour (anatomy/voltage).

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.basic_vertebrate_browser
Then: cd cognimed && python -m http.server 8903 --directory data
      open http://localhost:8903/basic_vertebrate_viewer.html
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

from medic.unified_embryo import simulate, _symmetrize, VMIN, VMAX, FATES, FIDX

N_END = 100000
N_RENDER = 16000
JSON = Path("data/movie/basic_vertebrate_frames.json")
HTML = Path("data/basic_vertebrate_viewer.html")

# anatomy palette (per fate); the buds pop: limb=green, eye=cyan, heart=red, otic=gold
ANAT = {"Forebrain": (0.30, 0.46, 0.95), "Eye": (0.20, 0.85, 1.00), "Nervous System": (0.36, 0.55, 0.95),
        "Spinal Cord": (0.46, 0.62, 0.92), "Neural Crest": (0.66, 0.42, 0.86), "Mesoderm": (0.92, 0.56, 0.46),
        "Somite": (0.96, 0.66, 0.42), "Epidermal": (0.82, 0.86, 0.90), "Hypoblast": (0.86, 0.76, 0.46),
        "Yolk Syncytial Layer": (0.90, 0.80, 0.40), "Blastodisc": (0.72, 0.74, 0.78),
        "Proliferative Like Cell": (0.66, 0.68, 0.72), "Limb Bud": (0.28, 0.86, 0.46),
        "Heart": (0.93, 0.16, 0.22), "Otic": (1.00, 0.82, 0.20), "Liver": (0.72, 0.34, 0.62),
        "Lung": (0.55, 0.78, 0.90), "Pancreas": (0.80, 0.82, 0.30), "Gut": (0.82, 0.60, 0.40),
        "Rib": (0.94, 0.94, 0.86), "Kidney": (0.66, 0.28, 0.46), "Muscle": (0.86, 0.42, 0.42),
        "Notochord": (0.60, 0.82, 0.72), "Skin": (0.96, 0.82, 0.74),
        "Cartilage": (0.80, 0.86, 0.92), "DRG": (0.72, 0.30, 0.90),
        "Sympathetic": (0.90, 0.45, 0.85), "Vessel": (0.80, 0.10, 0.20),
        # leaf heads added 2026-07-18
        "Meninges": (0.58, 0.50, 0.78), "Connective": (0.78, 0.70, 0.58), "Jaw": (0.86, 0.62, 0.70),
        "Choroid": (0.40, 0.78, 0.86), "Gonad": (0.90, 0.52, 0.60),
        # brain subheads (blue family, anterior -> posterior gets deeper)
        "Midbrain": (0.34, 0.52, 0.94), "Hindbrain": (0.40, 0.62, 0.90), "Cerebellum": (0.52, 0.72, 0.94),
        # 6 remaining MOSTA heads
        "Mesothelium": (0.66, 0.74, 0.68), "Mesentery": (0.72, 0.66, 0.52), "Mucosa": (0.88, 0.72, 0.52),
        "HeadMes": (0.70, 0.62, 0.52), "Branchial": (0.80, 0.56, 0.62), "Blood": (0.74, 0.10, 0.14),
        # organ subheads + new organ heads (07-18)
        "Atrium": (0.95, 0.34, 0.40), "Ventricle": (0.85, 0.12, 0.18), "Outflow": (0.99, 0.52, 0.44),
        "LiverHaem": (0.80, 0.20, 0.34), "Foregut": (0.88, 0.68, 0.44), "Hindgut": (0.68, 0.48, 0.32),
        "Nephron": (0.58, 0.30, 0.54), "Retina": (0.24, 0.76, 0.86), "Adrenal": (0.94, 0.72, 0.30),
        "Thymus": (0.76, 0.82, 0.58), "Spleen": (0.58, 0.16, 0.30),
        "Bladder": (0.86, 0.76, 0.56), "Adipose": (0.96, 0.86, 0.54), "OlfactoryBulb": (0.42, 0.56, 0.92)}
ANAT_LIST = [list(ANAT[f]) for f in FATES]


def shape_limbs(Q, fate, f, limb_id):
    """Outgrow the four limb buds into flattened PADDLES with digit rays (late). Per limb: flatten
    dorsoventrally, extend distally, and groove the distal plate into 3 digits (autopod)."""
    isb = fate == limb_id
    if f < 0.55 or not isb.any():
        return Q
    Q = Q.copy()
    prog = (f - 0.55) / 0.45
    idx = np.where(isb)[0]
    P = Q[idx]
    xmid = np.median(P[:, 0])
    for fore in (True, False):
        for side in (+1, -1):
            sel = ((P[:, 0] < xmid) if fore else (P[:, 0] >= xmid)) & (np.sign(P[:, 2]) == side)
            if sel.sum() < 12:
                continue
            L = P[sel]
            cen = L.mean(0)
            # buds are ALREADY separated by lateral inhibition (see unified_embryo); just gently
            # compact each field, then flatten -> outgrow -> notch, per limb.
            L[:, 0] = cen[0] + (L[:, 0] - cen[0]) * (1 - 0.25 * prog)
            dz = np.clip(np.abs(L[:, 2]) - np.abs(L[:, 2]).min(), 0, None)
            u = dz / (dz.max() + 1e-9)                      # 0 proximal .. 1 distal
            ymid = np.median(L[:, 1])
            L[:, 1] = ymid + (L[:, 1] - ymid) * (1 - 0.55 * prog)     # flatten into a paddle
            L[:, 2] += side * 0.13 * prog * (0.35 + 0.65 * u)       # outgrow distally
            xc = L[:, 0].mean(); xspan = np.ptp(L[:, 0]) + 1e-9
            tx = (L[:, 0] - xc) / xspan
            centers = np.array([-0.34, 0.0, 0.34])
            snap = centers[np.argmin(np.abs(tx[:, None] - centers[None, :]), 1)] * xspan + xc
            w = np.clip((u - 0.55) / 0.45, 0, 1) * prog * 0.5      # subtle digit notches at the tip
            L[:, 0] = L[:, 0] * (1 - w) + snap * w
            P[sel] = L
    Q[idx] = P
    return Q


def flex(Q, f):
    """Late cephalo-caudal flexure: a GENTLE cephalic curve (not a sharp C) so the organs
    read as a clean anteroposterior sequence ALONG the trunk instead of a folded boomerang."""
    bend = np.radians(24.0) * float(np.clip((f - 0.42) / 0.58, 0, 1))
    if bend < 1e-6:
        return Q
    x, y, z = Q[:, 0], Q[:, 1], Q[:, 2]
    L = np.ptp(x) + 1e-9
    b = bend / L
    s = x - x.min()
    th = b * s
    cx = np.sin(th) / b
    cy = (np.cos(th) - 1.0) / b
    nx, ny = -np.sin(th), np.cos(th)
    h = y - np.median(y)
    return np.stack([cx + h * nx, cy + h * ny, z], 1).astype(np.float32)


def _frame(Ps):
    """Electric-body frame of a cell cloud: AP eigenmode (rank), LR eigenmode (node=midline), LR corr."""
    n = len(Ps)
    nb = cKDTree(Ps).query(Ps, k=11)[1][:, 1:]
    rr = np.repeat(np.arange(n), nb.shape[1]); cc = nb.ravel()
    W = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(n, n)).tocsr(); W = ((W + W.T) > 0).astype(float)
    L = diags(np.asarray(W.sum(1)).ravel()) - W
    vv, UU = eigsh(L, k=14, which="SM"); UU = UU[:, np.argsort(vv)]

    def best(coord):
        b, bi = 0.0, 1
        for i in range(1, 14):
            cabs = abs(np.corrcoef(UU[:, i], coord)[0, 1])
            if cabs > b:
                b, bi = cabs, i
        return bi, b
    iAP, _ = best(Ps[:, 0]); iLR, cLR = best(Ps[:, 2])
    apm = UU[:, iAP] * (1 if np.corrcoef(UU[:, iAP], Ps[:, 0])[0, 1] >= 0 else -1)
    lrm = UU[:, iLR] / (np.abs(UU[:, iLR]).max() + 1e-9)
    apr = np.argsort(np.argsort(apm)).astype(np.float32) / max(1, n - 1)
    return apr, lrm, cLR


# genome-derived fore/hind AP levels (fossil-record Hox enhancers) -- same as the main sim
try:
    from medic.limb_genome_frame import genome_limb_frame as _glf
    _FORE_AP, _HIND_AP = _glf(1.0)["fore_ap"], _glf(1.0)["hind_ap"]
except Exception:
    _FORE_AP, _HIND_AP = 0.294, 0.755


def width_sweep(Ps, c, scale, n_w=7, n_pts=20000):
    """Precompute the FISH->TETRAPOD (amphibian) sweep: thin -> wide body; at each width, recompute
    the electric-body frame and mark the limb cells (Hox AP level x LR-mode antinode). Thin body:
    no LR eigenmode -> no separated limbs; wide body: LR mode present -> four limbs on the antinodes."""
    if len(Ps) > n_pts:
        sel = np.random.default_rng(1).choice(len(Ps), n_pts, replace=False); Ps = Ps[sel]
    base = Ps - Ps.mean(0)
    fracs = np.linspace(0.15, 1.0, n_w)
    frames = []
    for s in fracs:
        P = base.copy(); P[:, 2] *= s
        apr, lrm, cLR = _frame(P)
        dv = (P[:, 1] - P[:, 1].min()) / (np.ptp(P[:, 1]) + 1e-9)
        hox = np.exp(-((apr - _FORE_AP) / 0.06) ** 2) + np.exp(-((apr - _HIND_AP) / 0.06) ** 2)
        # limbs ONLY where a genuine LR bilateral eigenmode exists (cLR high): a thin fish has
        # no LR mode (cLR~0) -> limbless, however wide the |lrm| noise looks; a wide tetrapod does.
        lr_present = cLR > 0.35
        limb = lr_present & (np.abs(lrm) > 0.45) & (dv >= 0.26) & (dv <= 0.62) & (hox > 0.4)
        Q = (P - P.mean(0)) * scale
        # GROW the marked limb cells into visible BUDS (else they are flat coloured patches on the
        # trunk and read as specks): push them laterally off the midline + a touch ventrally, scaled
        # by how strong the LR mode is -- so as you widen the body the buds also physically emerge.
        if limb.any():
            grow = 0.55 * min(1.0, cLR)
            sgn = np.sign(Q[:, 2] + 1e-9)
            Q[limb, 2] += sgn[limb] * grow
            Q[limb, 1] -= 0.30 * grow
        frames.append(dict(w=round(float(s), 2), lrcorr=round(float(cLR), 2),
                           xyz=[round(float(x), 3) for x in Q.ravel()],
                           limb=[int(x) for x in limb]))
        print(f"    width {s:.2f}  LR|corr| {cLR:.2f}  limb cells {int(limb.sum())}")
    return [round(float(x), 2) for x in fracs], frames


# the width slider now DRIVES the movie: grow the full development at several widths
# (fish -> tetrapod). convergent_ext high = strong Wnt-PCP = narrow = fish; low = wide = tetrapod.
WIDTH_CE = [5.5, 2.4, 1.5, 1.0]        # fish (narrow, limbless at 26k) ... tetrapod (wide)
NE_W = 26000                           # cells per width-movie (smaller than N_END so 4 movies fit)
RENDER_W = 5200


def export():
    from medic.limb_genome_frame import genome_limb_frame
    LIMB = FIDX["Limb Bud"]
    raws = []
    for ce in WIDTH_CE:
        print(f"growing development at convergent_ext={ce} (pcp={genome_limb_frame(ce)['pcp']:.2f}) ...")
        frames, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=NE_W, limb_buds=True,
                             convergent_ext=ce)
        sym = [_symmetrize(P, V, F) for (_, _, _, P, V, F) in frames]
        raws.append((ce, frames, sym))

    # shared center + scale from the widest (tetrapod, last) so every width sits in one frame
    Pf = raws[-1][2][-1][0]
    c = Pf.mean(0); c[2] = 0.0
    scale = 1.7 / (0.5 * max(np.ptp(Pf[:, 0]), np.ptp(Pf[:, 1]), np.ptp(Pf[:, 2])) + 1e-9)
    procs = []
    for ce, frames, sym in raws:
        nfr = len(frames)
        procs.append([flex(shape_limbs((Ps - c) * scale, Fs, fi / (nfr - 1), LIMB), fi / (nfr - 1))
                      for fi, (Ps, _, Fs) in enumerate(sym)])
    cc = procs[-1][-1].mean(0)                                   # common recentre (tetrapod)
    R = max(float(np.abs(p[-1] - cc).max()) for p in procs)

    HI_FATES = set(FIDX[n] for n in ("Limb Bud", "Eye", "Heart", "Otic", "Liver", "Lung",
                                     "Pancreas", "Gut", "Rib", "Kidney", "Muscle",
                                     "Cartilage", "DRG", "Sympathetic", "Vessel",
                                     "Jaw", "Choroid", "Gonad", "Meninges",
                                     "Branchial", "Blood", "Mesentery", "Mucosa",
                                     "Atrium", "Ventricle", "Outflow", "LiverHaem", "Foregut", "Hindgut",
                                     "Nephron", "Retina", "Adrenal", "Thymus", "Spleen",
                                     "Bladder", "Adipose", "OlfactoryBulb"))
    movies = []
    for (ce, frames, sym), proc in zip(raws, procs):
        rng = np.random.default_rng(0)
        limbN = int((sym[-1][2] == LIMB).sum())
        out = []
        for fi, ((born, t_hpf, prc2, _, _, _), (Ps, Vs, Fs)) in enumerate(zip(frames, sym)):
            Q = proc[fi] - cc; n = len(Q)
            if n > RENDER_W:
                # NEVER subsample organ/limb cells away (else sparse posterior organs vanish):
                # keep all highlight cells, subsample only the background to fill the budget.
                is_hi = np.isin(Fs, list(HI_FATES))
                hi_i = np.where(is_hi)[0]; bg_i = np.where(~is_hi)[0]
                keep_bg = max(0, min(len(bg_i), RENDER_W - len(hi_i)))
                sel = np.concatenate([hi_i, rng.choice(bg_i, keep_bg, replace=False)])
                Q, Vs, Fs = Q[sel], Vs[sel], Fs[sel]
            out.append(dict(stage=f"N = {born:,} cells   ·   {t_hpf:.0f} hpf   ·   PRC2 {prc2:.2f}",
                            xyz=[round(float(x), 2) for x in Q.ravel()],
                            vm=[round(float(x)) for x in Vs],
                            fate=[int(x) for x in Fs]))
        pcp = genome_limb_frame(ce)["pcp"]
        kind = "fish (limbless)" if limbN < 20 else "tetrapod"
        movies.append(dict(ce=round(float(ce), 2), pcp=round(float(pcp), 2),
                           limbN=limbN, kind=kind, frames=out))
        print(f"  -> ce={ce} pcp={pcp:.2f}: {limbN} limb cells ({kind})")

    doc = dict(display="Basic vertebrate · grown from one cell (NCA + LGM) · width drives development",
               vmin=VMIN, vmax=VMAX, R=round(R, 3), anat=ANAT_LIST,
               hi=[FIDX[n] for n in ("Limb Bud", "Eye", "Heart", "Otic", "Liver", "Lung",
                                     "Pancreas", "Gut", "Rib", "Kidney", "Muscle", "Notochord",
                                     "Cartilage", "DRG", "Sympathetic", "Vessel",
                                     "Jaw", "Choroid", "Gonad", "Meninges",
                                     "Branchial", "Blood", "Mesentery", "Mucosa", "Mesothelium", "HeadMes",
                                     "Atrium", "Ventricle", "Outflow", "LiverHaem", "Foregut", "Hindgut",
                                     "Nephron", "Retina", "Adrenal", "Thymus", "Spleen",
                                     "Bladder", "Adipose", "OlfactoryBulb")],
               skin=FIDX["Skin"],
               movies=movies)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(JSON, "w"))
    print(f"saved {JSON}  ({len(movies)} width-movies x {len(movies[0]['frames'])} frames, "
          f"{JSON.stat().st_size/1e6:.1f} MB)")
    HTML.write_text(VIEWER, encoding="utf-8")
    print(f"saved {HTML}")


VIEWER = """<!doctype html><html><head><meta charset="utf-8"><title>Basic vertebrate — development</title>
<style>
  html,body{margin:0;height:100%;background:#0d1017;color:#cbd5e1;font:13px system-ui;overflow:hidden}
  #ui{position:fixed;top:10px;left:12px;z-index:3;background:#0d1017cc;padding:9px 12px;border-radius:9px;max-width:72%}
  #ui b{color:#e8eef4}#stage{color:#7dd3fc;font-size:14px;margin-top:3px}#legend{color:#8091a8;margin-top:4px}
  #key{margin-top:5px}#key span{margin-right:11px;white-space:nowrap}
  #bar{position:fixed;bottom:12px;left:12px;right:12px;z-index:3;display:flex;align-items:center;gap:10px;
       background:#0d1017cc;padding:8px 12px;border-radius:9px}
  #wbar{position:fixed;bottom:58px;left:12px;right:12px;z-index:3;display:flex;align-items:center;gap:10px;
        background:#0d1017cc;padding:7px 12px;border-radius:9px}
  button{font:13px system-ui;background:#1b2130;color:#e2e8f0;border:1px solid #33405a;border-radius:6px;padding:4px 12px;cursor:pointer}
  button.on{background:#2b6cb0;border-color:#2b6cb0}input[type=range]{flex:1}
  .dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}
</style></head>
<body>
<div id="ui"><b>Basic vertebrate — grown from one cell (NCA + LGM)</b>
<div id="legend">4 heads + limb &amp; organ buds &nbsp;·&nbsp; drag to rotate, scrub the timeline</div>
<div id="key"></div>
<div id="stage">loading…</div></div>
<div id="wbar"><span style="color:#8aa0b4;white-space:nowrap">body width — fish → tetrapod:</span>
  <input id="wslider" type="range" min="0" max="0" value="0" step="1">
  <span id="wlabel" style="color:#7dd3fc;white-space:nowrap">drag to widen the body → limbs emerge in the SAME development</span></div>
<div id="bar">
  <button id="play">⏸ pause</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <button id="mode" class="on">anatomy</button>
  <button id="skinb" class="on">skin</button>
  <button id="rot">↻ auto-rotate</button>
</div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const params=new URLSearchParams(location.search);
let cur=0, playing=params.get('pause')?false:true, last=0, FRAME_MS=95, mode='anat', wmode=false;
const sc=new THREE.Scene();
const cam=new THREE.PerspectiveCamera(50, innerWidth/innerHeight, 0.01, 500);
const rn=new THREE.WebGLRenderer({antialias:true, preserveDrawingBuffer:true});
rn.setSize(innerWidth,innerHeight); rn.setPixelRatio(devicePixelRatio); document.body.appendChild(rn.domElement);
sc.add(new THREE.AmbientLight(0xffffff,0.72));
const _dl=new THREE.DirectionalLight(0xffffff,0.85); _dl.position.set(0.6,1,0.8); sc.add(_dl);
const ctrl=new OrbitControls(cam, rn.domElement); ctrl.enableDamping=true; ctrl.autoRotateSpeed=1.1;
let DATA=null, pts=[], vmin=0, vmax=1, nf=0;
const UNC=[0.5,0.55,0.6];
function vcol(vm){ let t=(vm-vmin)/(vmax-vmin+1e-9); t=Math.max(0,Math.min(1,t));
  let a=[0.23,0.32,0.78],b=[0.93,0.93,0.93],c=[0.82,0.14,0.16];
  if(t<0.5){let u=t*2;return[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,a[2]+(b[2]-a[2])*u];}
  let u=(t-0.5)*2;return[b[0]+(c[0]-b[0])*u,b[1]+(c[1]-b[1])*u,b[2]+(c[2]-b[2])*u]; }
function acol(fate){ return fate<0?UNC:DATA.anat[fate]; }
const _SPH=new THREE.SphereGeometry(1,7,6);              // each cell = a small shaded sphere (GPU-instanced)
function _fill(mesh,pos,col,size){ const dm=new THREE.Object3D(), cc=new THREE.Color();
  for(let i=0;i<pos.length/3;i++){ dm.position.set(pos[3*i],pos[3*i+1],pos[3*i+2]); dm.scale.setScalar(size); dm.updateMatrix();
    mesh.setMatrixAt(i,dm.matrix); cc.setRGB(col[3*i],col[3*i+1],col[3*i+2]); mesh.setColorAt(i,cc); }
  mesh.instanceMatrix.needsUpdate=true; if(mesh.instanceColor)mesh.instanceColor.needsUpdate=true; return mesh; }
function mk(pos,col,size){ return _fill(new THREE.InstancedMesh(_SPH,new THREE.MeshLambertMaterial({}),pos.length/3),pos,col,size); }
function mkt(pos,col,size){ return _fill(new THREE.InstancedMesh(_SPH,      // translucent skin envelope
  new THREE.MeshLambertMaterial({transparent:true,opacity:0.24,depthWrite:false}),pos.length/3),pos,col,size); }
let wIdx=0, skinOn=true;                     // which width-movie; whether the skin envelope shows
function build(i){
  const fr=DATA.movies[wIdx].frames[i];
  for(const p of pts){sc.remove(p);p.geometry.dispose();p.material.dispose();}
  const N=fr.vm.length, hi=new Set(DATA.hi||[]), SKIN=DATA.skin;
  const bp=[],bc=[],hp=[],hc=[],kp=[],kc=[];
  for(let k=0;k<N;k++){
    const f=fr.fate[k], c=(mode==='anat')?acol(f):vcol(fr.vm[k]);
    if(mode==='anat' && f===SKIN){ kp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); kc.push(c[0],c[1],c[2]); }
    else if(mode==='anat' && hi.has(f)){ hp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); hc.push(c[0],c[1],c[2]); }
    else { bp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); bc.push(c[0],c[1],c[2]); }
  }
  pts=[mk(bp,bc,0.016)]; if(hp.length) pts.push(mk(hp,hc,0.030));
  if(kp.length && skinOn) pts.push(mkt(kp,kc,0.028));
  for(const p of pts) sc.add(p);
  const mv=DATA.movies[wIdx];
  document.getElementById('stage').textContent=fr.stage;
  document.getElementById('wlabel').textContent=
    'width '+Math.round(100/mv.pcp*0.25)+'%  ·  '+mv.kind+'  ·  '+mv.limbN+' limb cells';
  document.getElementById('slider').value=i;
}
const KEY=[['Limb','#47db76'],['Eye','#33d8ff'],['Heart','#ee2938'],['Ear','#ffd23a'],
           ['Liver','#b857a3'],['Lung','#8cc7e6'],['Pancreas','#ccd14d'],['Gut','#d19a66'],
           ['Rib','#f0f0db'],['Kidney','#a8497a'],['Muscle','#db6b6b'],['Notochord','#99d1b8'],
           ['Cartilage','#ccdbeb'],['DRG','#b84de6'],['Sympath','#e673d9'],['Vessel','#cc1a33'],
           ['Meninges','#9480c7'],['Connect','#c7b294'],['Jaw','#db9eb2'],['Choroid','#66c7db'],['Gonad','#e58599'],
           ['Midbrain','#5785f0'],['Hindbrain','#669ee5'],['Cerebellum','#85b8f0'],
           ['Atrium','#f2576b'],['Ventricle','#d91f2e'],['Outflow','#fc8570'],['LivHaem','#cc3357'],['Foregut','#e0ad70'],['Hindgut','#ad7a52'],
           ['Nephron','#944d8a'],['Retina','#3dc2db'],['Adrenal','#f0b84d'],['Thymus','#c2d194'],['Spleen','#94294d'],
           ['Bladder','#dbc28f'],['Adipose','#f5db8a'],['OlfBulb','#6b8feb'],
           ['Mesothel','#a8bdad'],['Mesentery','#b8a885'],['Mucosa','#e0b885'],['HeadMes','#b39e85'],['Branchial','#cc8f9e'],['Blood','#bd1a24'],
           ['Skin','#f5d1bd'],['Neural','#5b78e8']];
fetch('movie/basic_vertebrate_frames.json').then(r=>r.json()).then(d=>{
  DATA=d; vmin=d.vmin; vmax=d.vmax; nf=d.movies[0].frames.length;
  document.getElementById('slider').max=nf-1;
  document.getElementById('wslider').max=d.movies.length-1;
  document.getElementById('wslider').value=d.movies.length-1; wIdx=d.movies.length-1;  // start on the tetrapod
  document.getElementById('key').innerHTML=KEY.map(k=>`<span><i class="dot" style="background:${k[1]}"></i>${k[0]}</span>`).join('');
  const R=d.R||2.0; cam.position.set(R*0.15,R*0.30,R*2.35); ctrl.target.set(0,0,0);  // lateral: trunk head->tail across screen
  cur=params.get('f')?Math.min(nf-1,Math.max(0,parseInt(params.get('f')))):0;
  build(cur); syncPlay();
});
const playBtn=document.getElementById('play'),rotBtn=document.getElementById('rot'),
      modeBtn=document.getElementById('mode'),slider=document.getElementById('slider');
function syncPlay(){ playBtn.textContent=playing?'⏸ pause':'▶ play'; }
playBtn.onclick=()=>{playing=!playing;syncPlay();if(playing)build(cur);};
rotBtn.onclick=()=>{ctrl.autoRotate=!ctrl.autoRotate;rotBtn.classList.toggle('on',ctrl.autoRotate);};
modeBtn.onclick=()=>{ mode=(mode==='anat')?'volt':'anat'; modeBtn.textContent=(mode==='anat')?'anatomy':'voltage';
  modeBtn.classList.toggle('on',mode==='anat'); build(cur); };
slider.oninput=()=>{playing=false;syncPlay();cur=parseInt(slider.value);build(cur);};
const skinBtn=document.getElementById('skinb');
skinBtn.onclick=()=>{skinOn=!skinOn;skinBtn.classList.toggle('on',skinOn);build(cur);};
const wslider=document.getElementById('wslider');
wslider.oninput=()=>{wIdx=parseInt(wslider.value);build(cur);};   // same timeline point, different width
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
function loop(t){ requestAnimationFrame(loop);
  if(DATA&&playing&&t-last>FRAME_MS){last=t;cur=(cur+1)%nf;build(cur);}
  if(!window.__freeze) ctrl.update();
  rn.render(sc,cam); }
requestAnimationFrame(loop);
// paper-figure camera hook: window.__view('side'|'top'|'oblique') freezes the loop and poses the camera
// (x = head->tail, y = dorsal up, z = left-right); side = look along z, top = look down y.
window.__view=(v,roll)=>{ const R=(DATA&&DATA.R)||2.0; window.__freeze=true; ctrl.autoRotate=false; playing=false;
  const rr=(roll==null?(v==='side'?-12.5:0):roll)*Math.PI/180;           // roll levels the sagittal flexure curl (side default -12.5)
  if(v==='side'){ cam.up.set(-Math.sin(rr),Math.cos(rr),0); cam.position.set(0,0,R*2.6); }
  else if(v==='top'){ cam.up.set(0,0,-1); cam.position.set(0,R*2.6,0); }
  else { cam.up.set(0,1,0); cam.position.set(R*1.55,R*0.98,R*1.85); }   // true 3/4 oblique (shows depth)
  cam.lookAt(0,0,0); rn.render(sc,cam); };
</script></body></html>"""


if __name__ == "__main__":
    export()
