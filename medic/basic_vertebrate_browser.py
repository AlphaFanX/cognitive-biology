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
        "Heart": (0.93, 0.16, 0.22), "Otic": (1.00, 0.82, 0.20)}
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
    """Late cephalo-caudal flexure: curl the AP axis into a C, ramping over the second half."""
    bend = np.radians(64.0) * float(np.clip((f - 0.42) / 0.58, 0, 1))
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


def width_sweep(Ps, c, scale, n_w=7, n_pts=8000):
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
        hox = np.exp(-((apr - 0.20) / 0.05) ** 2) + np.exp(-((apr - 0.44) / 0.05) ** 2)
        limb = (np.abs(lrm) > 0.45) & (dv >= 0.26) & (dv <= 0.62) & (hox > 0.4)
        Q = (P - P.mean(0)) * scale
        frames.append(dict(w=round(float(s), 2), lrcorr=round(float(cLR), 2),
                           xyz=[round(float(x), 3) for x in Q.ravel()],
                           limb=[int(x) for x in limb]))
        print(f"    width {s:.2f}  LR|corr| {cLR:.2f}  limb cells {int(limb.sum())}")
    return [round(float(x), 2) for x in fracs], frames


def export():
    print(f"growing basic vertebrate: 1 cell -> {N_END} cells, WITH limb + organ buds ...")
    frames, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=N_END, limb_buds=True,
                         convergent_ext=1.0, verbose=True)
    sym = [_symmetrize(P, V, F) for (_, _, _, P, V, F) in frames]
    Pf = sym[-1][0]
    c = Pf.mean(0); c[2] = 0.0
    scale = 1.7 / (0.5 * max(np.ptp(Pf[:, 0]), np.ptp(Pf[:, 1]), np.ptp(Pf[:, 2])) + 1e-9)

    nfr = len(frames)
    LIMB = FIDX["Limb Bud"]
    proc = []
    for fi, (Ps, _, Fs) in enumerate(sym):
        f = fi / (nfr - 1)
        Q = shape_limbs((Ps - c) * scale, Fs, f, LIMB)
        proc.append(flex(Q, f))
    cc = proc[-1].mean(0)
    R = float(np.abs(proc[-1] - cc).max())
    rng = np.random.default_rng(0)

    out = []
    for fi, ((born, t_hpf, prc2, _, _, _), (Ps, Vs, Fs)) in enumerate(zip(frames, sym)):
        Q = proc[fi] - cc
        n = len(Q)
        if n > N_RENDER:
            sel = rng.choice(n, N_RENDER, replace=False)
            Q, Vs, Fs = Q[sel], Vs[sel], Fs[sel]
        out.append(dict(stage=f"N = {born:,} cells   ·   {t_hpf:.0f} hpf   ·   PRC2 {prc2:.2f}",
                        xyz=[round(float(x), 3) for x in Q.ravel()],
                        vm=[round(float(x), 1) for x in Vs],
                        fate=[int(x) for x in Fs]))
    print("computing the amphibian width sweep (fish -> tetrapod) ...")
    wfracs, wframes = width_sweep(sym[-1][0], c, scale)
    doc = dict(display="Basic vertebrate · grown from one cell (NCA + LGM, 4 heads + limb & organ buds)",
               vmin=VMIN, vmax=VMAX, R=round(R, 3), anat=ANAT_LIST,
               hi=[FIDX[n] for n in ("Limb Bud", "Eye", "Heart", "Otic")],
               width=dict(fracs=wfracs, frames=wframes), frames=out)
    JSON.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(JSON, "w"))
    print(f"saved {JSON}  ({len(out)} frames, {JSON.stat().st_size/1e6:.1f} MB)")
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
  <span id="wlabel" style="color:#7dd3fc;white-space:nowrap">drag to widen the body → the limbs separate</span></div>
<div id="bar">
  <button id="play">⏸ pause</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <button id="mode" class="on">anatomy</button>
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
sc.add(new THREE.AmbientLight(0xffffff,1));
const ctrl=new OrbitControls(cam, rn.domElement); ctrl.enableDamping=true; ctrl.autoRotateSpeed=1.1;
let DATA=null, pts=[], vmin=0, vmax=1, nf=0;
const UNC=[0.5,0.55,0.6];
function vcol(vm){ let t=(vm-vmin)/(vmax-vmin+1e-9); t=Math.max(0,Math.min(1,t));
  let a=[0.23,0.32,0.78],b=[0.93,0.93,0.93],c=[0.82,0.14,0.16];
  if(t<0.5){let u=t*2;return[a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u,a[2]+(b[2]-a[2])*u];}
  let u=(t-0.5)*2;return[b[0]+(c[0]-b[0])*u,b[1]+(c[1]-b[1])*u,b[2]+(c[2]-b[2])*u]; }
function acol(fate){ return fate<0?UNC:DATA.anat[fate]; }
function mk(pos,col,size){
  const g=new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col,3));
  return new THREE.Points(g, new THREE.PointsMaterial({size, vertexColors:true, sizeAttenuation:true}));
}
function build(i){
  const fr=DATA.frames[i];
  for(const p of pts){sc.remove(p);p.geometry.dispose();p.material.dispose();}
  const N=fr.vm.length, hi=new Set(DATA.hi||[]);
  const bp=[],bc=[],hp=[],hc=[];
  for(let k=0;k<N;k++){
    const c=(mode==='anat')?acol(fr.fate[k]):vcol(fr.vm[k]);
    if(mode==='anat' && hi.has(fr.fate[k])){ hp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); hc.push(c[0],c[1],c[2]); }
    else { bp.push(fr.xyz[3*k],fr.xyz[3*k+1],fr.xyz[3*k+2]); bc.push(c[0],c[1],c[2]); }
  }
  pts=[mk(bp,bc,0.030)]; if(hp.length) pts.push(mk(hp,hc,0.060));
  for(const p of pts) sc.add(p);
  document.getElementById('stage').textContent=fr.stage;
  document.getElementById('slider').value=i;
}
function buildWidth(i){
  const fr=DATA.width.frames[i];
  for(const p of pts){sc.remove(p);p.geometry.dispose();p.material.dispose();}
  const N=fr.limb.length, bp=[],bc=[],hp=[],hc=[];
  for(let k=0;k<N;k++){ const x=fr.xyz[3*k],y=fr.xyz[3*k+1],z=fr.xyz[3*k+2];
    if(fr.limb[k]){hp.push(x,y,z);hc.push(0.16,0.86,0.30);} else {bp.push(x,y,z);bc.push(0.60,0.64,0.70);} }
  pts=[mk(bp,bc,0.028)]; if(hp.length) pts.push(mk(hp,hc,0.062));
  for(const p of pts) sc.add(p);
  document.getElementById('wlabel').textContent='width '+Math.round(fr.w*100)+'%  ·  LR |corr| '+fr.lrcorr.toFixed(2)+(fr.lrcorr<0.5?'  —  no left-right mode: limbless (fish)':'  —  left-right mode present: limbs (tetrapod)');
}
const KEY=[['Limb bud','#47db76'],['Eye','#33d8ff'],['Heart','#ee2938'],['Otic (ear)','#ffd23a'],
           ['Neural','#5b78e8'],['Somite/meso','#f0985e'],['Yolk','#e6cc66']];
fetch('movie/basic_vertebrate_frames.json').then(r=>r.json()).then(d=>{
  DATA=d; vmin=d.vmin; vmax=d.vmax; nf=d.frames.length;
  document.getElementById('slider').max=nf-1;
  document.getElementById('wslider').max=(d.width?d.width.frames.length-1:0);
  document.getElementById('key').innerHTML=KEY.map(k=>`<span><i class="dot" style="background:${k[1]}"></i>${k[0]}</span>`).join('');
  const R=d.R||2.0; cam.position.set(R*1.15,R*0.65,R*1.5); ctrl.target.set(0,0,0);
  cur=params.get('f')?Math.min(nf-1,Math.max(0,parseInt(params.get('f')))):0;
  build(cur); syncPlay();
});
const playBtn=document.getElementById('play'),rotBtn=document.getElementById('rot'),
      modeBtn=document.getElementById('mode'),slider=document.getElementById('slider');
function syncPlay(){ playBtn.textContent=playing?'⏸ pause':'▶ play'; }
playBtn.onclick=()=>{playing=!playing;if(playing)wmode=false;syncPlay();if(playing)build(cur);};
rotBtn.onclick=()=>{ctrl.autoRotate=!ctrl.autoRotate;rotBtn.classList.toggle('on',ctrl.autoRotate);};
modeBtn.onclick=()=>{ mode=(mode==='anat')?'volt':'anat'; modeBtn.textContent=(mode==='anat')?'anatomy':'voltage';
  modeBtn.classList.toggle('on',mode==='anat'); if(!wmode) build(cur); };
slider.oninput=()=>{wmode=false;playing=false;syncPlay();cur=parseInt(slider.value);build(cur);};
const wslider=document.getElementById('wslider');
wslider.oninput=()=>{wmode=true;playing=false;syncPlay();buildWidth(parseInt(wslider.value));};
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
function loop(t){ requestAnimationFrame(loop);
  if(DATA&&playing&&!wmode&&t-last>FRAME_MS){last=t;cur=(cur+1)%nf;build(cur);}
  ctrl.update(); rn.render(sc,cam); }
requestAnimationFrame(loop);
</script></body></html>"""


if __name__ == "__main__":
    export()
