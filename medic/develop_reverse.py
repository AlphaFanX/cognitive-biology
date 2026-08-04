"""A: the reverse developmental scrubber -- ONE clock coordinate tau in [0,1], run both ways.

tau = 0  zygote  ...  blastocyst ... embryo ... fetus ... infant ... tau = 1  adult.

The model already develops along a single coupled clock (telomere -> PRC2 -> fate unlock / division /
electric-body axes, in medic.unified_embryo.simulate). This exposes that clock as a scrubbable coordinate
and, crucially, runs it BACKWARD past the blastula: as tau falls the coupled schedules reverse together --
    * fate commitment  -> 0   (cells de-differentiate to blastomeres: PRC2 re-engages the embryonic stratum)
    * the AP / LR electric-body axes collapse -> the elongated body balls up into a RADIAL blastocyst
      (the frameless state the electric-body model predicts: no low eigenmode without elongation)
    * the body-fold un-curls
    * cell number N -> 1
Forward (tau -> 1) it matures: the fold relaxes, allometry shifts (cranium:face), size grows.

This is honestly the model's OWN coordinate reversed, not a validated adult mouse: we transform one
developed body rather than re-growing every adult organ. The blastocyst end is principled (axis collapse);
the adult end is allometric maturation of the existing body. Emits a frames JSON + a tau-scrubber viewer.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m medic.develop_reverse
Serve: cd cognimed && venv_win_new/Scripts/python.exe -m http.server 8904 --directory data
       -> http://localhost:8904/reverse_clock_viewer.html
"""
import os, json
import numpy as np
from medic.unified_embryo import simulate, _flex, V_NEUTRAL, FATES, VMIN, VMAX

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "..", "data", "movie", "reverse_clock_frames.json")
VIEW = os.path.join(HERE, "..", "data", "reverse_clock_viewer.html")

TAU_DEV = 0.52          # tau of the developed anchor body (late organogenesis / fetus)
N_RENDER = 6000
N_BLAST = 140           # cells in the rendered blastocyst
NFRAMES = 49


def stage_label(tau):
    if tau < 0.03:  return "zygote / cleavage"
    if tau < 0.10:  return "morula / blastocyst"
    if tau < 0.28:  return "gastrula / early embryo"
    if tau < TAU_DEV: return "organogenesis / fetus"
    if tau < 0.80:  return "late fetus / infant"
    return "juvenile -> adult"


def build():
    # --- the developed anchor body (grow once) ----------------------------------------------------
    frames, meta = simulate(use_ecm=True, seed=0, n_start=1, n_end=9000, limb_buds=True,
                            convergent_ext=1.0, head_expand=True, apical_fold=0.0)
    _, _, _, P0, vm0, fid0 = frames[-1]
    P0 = np.asarray(P0, np.float32); vm0 = np.asarray(vm0, np.float32); fid0 = np.asarray(fid0, np.int32)
    if len(P0) > N_RENDER:
        keep = np.random.RandomState(0).choice(len(P0), N_RENDER, replace=False)
        P0, vm0, fid0 = P0[keep], vm0[keep], fid0[keep]
    N0 = len(P0)
    P0 = P0 - P0.mean(0)
    span = P0.std(0) + 1e-6                              # AP/DV/ML extents of the developed body
    # a stable de-commit order (outermost / last-born decommit first as the clock reverses)
    order = np.argsort(np.random.RandomState(1).rand(N0))

    out_frames = []
    for k in range(NFRAMES):
        tau = k / (NFRAMES - 1)
        # ---- N(tau): cell number, 1 at zygote -> N0 at the developed anchor, flat after ------------
        if tau <= TAU_DEV:
            ramp = (tau / TAU_DEV) ** 1.4
            n = max(1, int(round(N_BLAST + (N0 - N_BLAST) * ramp))) if tau > 0.02 else max(1, int(round(1 + (N_BLAST - 1) * (tau / 0.02))))
        else:
            n = N0
        n = int(np.clip(n, 1, N0))
        idx = order[:n]
        P = P0[idx].copy(); vm = vm0[idx].copy(); fid = fid0[idx].copy()

        # ---- AXIS COLLAPSE (tau < TAU_DEV): the AP/LR electric-body modes vanish -> radial ball ------
        # weight of collapse grows as the clock winds back below the anchor.
        wcol = float(np.clip((TAU_DEV - tau) / TAU_DEV, 0.0, 1.0)) ** 1.1
        if wcol > 0:
            # anisotropic contraction toward the mediolateral (thinnest) scale => body balls up
            r_ball = 0.35 * float(span.min())
            sx = 1.0 - wcol * (1.0 - r_ball / span[0])
            sy = 1.0 - wcol * (1.0 - r_ball / span[1])
            sz = 1.0 - wcol * (1.0 - r_ball / (span[2] + 1e-6))
            P = P * np.array([sx, sy, sz], np.float32)
            # blend toward a filled sphere so it reads as a morula/blastocyst, not a flat disc
            rng = np.random.RandomState(7)
            u = rng.normal(size=(n, 3)); u /= (np.linalg.norm(u, axis=1, keepdims=True) + 1e-9)
            rad = r_ball * (rng.uniform(size=(n, 1)) ** (1 / 3.0))
            P = (1 - 0.85 * wcol) * P + (0.85 * wcol) * (u * rad)

        # ---- FATE de-commitment: committed fraction falls with the clock -----------------------------
        c = float(np.clip(tau / TAU_DEV, 0.0, 1.0))
        rank = np.argsort(np.argsort(order[:n]))         # stable per-cell rank within this frame
        uncommitted = rank > int(c * n)
        fid = fid.copy(); fid[uncommitted] = -1
        vm = vm.copy(); vm[uncommitted] = V_NEUTRAL

        # ---- FOLD: a fetal C-curl that peaks mid-gestation, 0 at blastocyst and adult ---------------
        fold = 2.2 * float(np.exp(-((tau - 0.42) / 0.16) ** 2)) if 0.10 < tau < 0.80 else 0.0
        if fold > 1e-3:
            P = _flex(P, fold)

        # ---- ADULT allometry (tau > TAU_DEV): cranium:face shift + overall growth --------------------
        if tau > TAU_DEV:
            g = (tau - TAU_DEV) / (1 - TAU_DEV)
            scale = 1.0 + 0.25 * g                        # overall maturation growth
            head = P[:, 0] < np.percentile(P[:, 0], 25)   # anterior quartile = cranium
            P = P * scale
            P[head, 1] *= (1.0 - 0.12 * g)                # cranium grows less in DV (face lengthens) - allometry

        out_frames.append(dict(
            tau=round(tau, 3), stage=stage_label(tau), n=n,
            xyz=[round(float(x), 3) for x in P.ravel()],
            vm=[round(float(v), 1) for v in vm],
            fate=[int(f) for f in fid]))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(frames=out_frames, fates=FATES, vmin=VMIN, vmax=VMAX, tau_dev=TAU_DEV), open(OUT, "w"))
    write_viewer()
    print(f"wrote {NFRAMES} tau frames (zygote->adult): {OUT}")
    for f in out_frames[::8]:
        print(f"  tau={f['tau']:.2f}  n={f['n']:5d}  {f['stage']}")


def write_viewer():
    html = r"""<!doctype html><html><head><meta charset=utf-8><title>Reverse developmental clock</title>
<style>body{margin:0;background:#0a0d12;color:#cdd;font:13px system-ui;overflow:hidden}
#hud{position:fixed;top:10px;left:12px;z-index:9}#hud b{font-size:16px;color:#8ecbff}
#ctl{position:fixed;bottom:12px;left:12px;right:12px;z-index:9;display:flex;gap:10px;align-items:center}
input[type=range]{flex:1}button{background:#1b2430;color:#cde;border:1px solid #2f3f52;border-radius:5px;padding:5px 10px;cursor:pointer}
.tag{color:#7fd1a0}</style></head><body>
<div id=hud><b id=stage>...</b><br>tau <span id=tau>0</span> &nbsp; cells <span id=n>0</span>
<br><span class=tag>left = zygote &nbsp;|&nbsp; right = adult</span></div>
<div id=ctl><button id=play>reverse ◄</button><button id=fwd>forward ►</button>
<input id=s type=range min=0 max=0 value=0><button id=col>colour: anatomy</button></div>
<script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
<script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
<script>
let D,frames,mode='fate',cur=0,dir=0;
const PAL=['#6cc','#e55','#5c8','#fc4','#7af','#c8f','#8f8','#fa8','#88f','#cf6','#f6c','#6ff'];
fetch('movie/reverse_clock_frames.json').then(r=>r.json()).then(d=>{D=d;frames=d.frames;
 s.max=frames.length-1;init();draw(0);});
const scene=new THREE.Scene();const cam=new THREE.PerspectiveCamera(55,innerWidth/innerHeight,.01,100);
cam.position.set(0,0,4);const rnd=new THREE.WebGLRenderer({antialias:true});rnd.setSize(innerWidth,innerHeight);
document.body.appendChild(rnd.domElement);const ctr=new THREE.OrbitControls(cam,rnd.domElement);
scene.add(new THREE.AmbientLight(0xffffff,.9));let pts;
function vcol(v,lo,hi){let t=(v-lo)/(hi-lo);t=Math.max(0,Math.min(1,t));
 return new THREE.Color(t<.5?0x3355ff:0xff3322).lerp(new THREE.Color(0xffffff),1-Math.abs(t-.5)*2);}
function init(){const g=new THREE.SphereGeometry(1,6,6);const m=new THREE.MeshBasicMaterial();
 pts=new THREE.InstancedMesh(g,m,20000);scene.add(pts);}
function draw(i){cur=i;const f=frames[i];const N=f.n;const xyz=f.xyz,vm=f.vm,fa=f.fate;
 const dummy=new THREE.Object3D();const R=0.016;
 for(let k=0;k<N;k++){dummy.position.set(xyz[k*3],xyz[k*3+1],xyz[k*3+2]);dummy.scale.setScalar(R);dummy.updateMatrix();
  pts.setMatrixAt(k,dummy.matrix);
  let c;if(fa[k]<0)c=new THREE.Color(0x556);else if(mode==='vm')c=vcol(vm[k],D.vmin,D.vmax);
  else c=new THREE.Color(PAL[fa[k]%PAL.length]);pts.setColorAt(k,c);}
 pts.count=N;pts.instanceMatrix.needsUpdate=true;if(pts.instanceColor)pts.instanceColor.needsUpdate=true;
 stage.textContent=f.stage;tau.textContent=f.tau.toFixed(2);n.textContent=N;s.value=i;}
play.onclick=()=>dir=dir===-1?0:-1;fwd.onclick=()=>dir=dir===1?0:1;
col.onclick=()=>{mode=mode==='fate'?'vm':'fate';col.textContent='colour: '+(mode==='fate'?'anatomy':'voltage');draw(cur);};
s.oninput=()=>{dir=0;draw(+s.value);};
let t=0;function loop(){requestAnimationFrame(loop);t++;if(dir&&t%2===0){let i=cur+dir;
 if(i<0)i=0;if(i>=frames.length)i=frames.length-1;if(i!==cur)draw(i);}ctr.update();rnd.render(scene,cam);}
loop();addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rnd.setSize(innerWidth,innerHeight);});
</script></body></html>"""
    open(VIEW, "w", encoding="utf-8").write(html)


if __name__ == "__main__":
    build()
