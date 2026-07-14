"""
Interactive three.js viewer for the menagerie — the NCA-grown bodies WITH the genome-grounded limbs.
====================================================================================================

Each species is grown from one cell by the unified NCA+LGM embryo (medic.unified_embryo) with the
genome-grounded limb frame active (medic.limb_genome_frame): fore/hind Hox levels from real Hox
colinearity, and the body width from the species' Wnt-PCP convergent-extension value. A wide tetrapod
admits the left-right electric-body eigenmode -> four limbs on its antinodes; the finned fish
(convergent_ext from the measured zebrafish/mouse ratio) tapers -> no LR mode -> limbless. Then the
genome's proportions deform the generic body late (von Baer). Rotate, pick a species, toggle limbs.

Run:  cd cognimed && venv_win_new/Scripts/python.exe -m menagerie.viewer_limbed
Then: cd cognimed && python -m http.server 8903 --directory data
      open http://localhost:8903/menagerie_viewer.html
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

import medic.unified_embryo as ue
from medic.unified_embryo import FATES, FIDX
from .grow_nca import species_deform, species_convergent_ext
from .genome import Genome
from .targets import reference_genome, BIG_SEVEN

# anatomy palette per fate; the buds pop: limb=green, eye=cyan, heart=red, otic=gold
ANAT = {"Forebrain": (0.30, 0.46, 0.95), "Eye": (0.20, 0.85, 1.00), "Nervous System": (0.36, 0.55, 0.95),
        "Spinal Cord": (0.46, 0.62, 0.92), "Neural Crest": (0.66, 0.42, 0.86), "Mesoderm": (0.90, 0.60, 0.50),
        "Somite": (0.96, 0.66, 0.42), "Epidermal": (0.80, 0.84, 0.88), "Hypoblast": (0.86, 0.76, 0.46),
        "Yolk Syncytial Layer": (0.90, 0.80, 0.40), "Blastodisc": (0.72, 0.74, 0.78),
        "Proliferative Like Cell": (0.66, 0.68, 0.72), "Limb Bud": (0.16, 0.86, 0.30),
        "Heart": (0.93, 0.16, 0.22), "Otic": (1.00, 0.82, 0.20)}
ANAT_LIST = [list(ANAT.get(f, (0.7, 0.72, 0.76))) for f in FATES]
LIMB = FIDX["Limb Bud"]


def grow_species(g, seed=0, n_render=14000):
    ce = species_convergent_ext(g)
    frames, _ = ue.simulate(use_ecm=True, seed=seed, limb_buds=True, convergent_ext=ce)
    born, t, prc2, P, V, F = frames[-1]
    Ps, Vs, Fs = ue._symmetrize(P, V, F)          # bilateral mirror across the midline
    Ps = species_deform(Ps, g)                    # von Baer: late species proportions
    if len(Ps) > n_render:
        sel = np.random.default_rng(0).choice(len(Ps), n_render, replace=False)
        Ps, Fs = Ps[sel], Fs[sel]
    # atlas (x fore-aft, y dorsoventral, z mediolateral) -> three.js (y up)
    xyz = np.stack([Ps[:, 0], Ps[:, 1], Ps[:, 2]], 1)
    xyz = xyz - xyz.mean(0)
    return {"xyz": [round(float(v), 3) for v in xyz.ravel()],
            "fate": [int(x) for x in Fs],
            "nlimb": int((Fs == LIMB).sum()),
            "conv_ext": round(float(ce), 2),
            "body_plan": getattr(g, "body_plan", "tetrapod")}


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Cognitive Biology — Menagerie (grown)</title>
<style>html,body{margin:0;height:100%;background:#0d1017;color:#cbd5e1;font:13px system-ui;overflow:hidden}
#ui{position:fixed;top:10px;left:12px;z-index:2;background:#0d1017cc;padding:9px 12px;border-radius:9px;max-width:74%}
#ui b{color:#e8eef4}#sub{color:#8091a8;margin-top:3px}#stat{color:#7dd3fc;margin-top:4px}
select,button{font:13px system-ui;background:#1b2130;color:#e2e8f0;border:1px solid #33405a;border-radius:6px;padding:4px 9px;cursor:pointer}
button.on{background:#2b6cb0;border-color:#2b6cb0}
#key{margin-top:5px}#key span{margin-right:11px;white-space:nowrap}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px;vertical-align:-1px}</style></head>
<body><div id="ui"><b>Cognitive Biology — the Menagerie, grown from the genome</b>
<div id="sub">each animal grown from one cell (NCA + LGM); limbs emerge from the electric-body eigenmode when the Wnt-PCP width admits it</div>
<div style="margin-top:6px">species <select id="sp"></select>&nbsp;
<button id="mode" class="on">anatomy</button>
<button id="limbs" class="on">highlight limbs</button>
<label style="margin-left:6px"><input type="checkbox" id="rot" checked> rotate</label></div>
<div id="key"></div><div id="stat">loading…</div></div>
<script type="importmap">{"imports":{"three":"https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js",
"three/addons/":"https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/"}}</script>
<script type="module">
import * as THREE from 'three';import{OrbitControls}from 'three/addons/controls/OrbitControls.js';
const DATA=__DATA__;let mode='anat',hl=true;
const sc=new THREE.Scene();const cam=new THREE.PerspectiveCamera(50,innerWidth/innerHeight,0.01,300);
const rn=new THREE.WebGLRenderer({antialias:true});rn.setSize(innerWidth,innerHeight);
rn.setPixelRatio(devicePixelRatio);document.body.appendChild(rn.domElement);
sc.add(new THREE.AmbientLight(0xffffff,1));
const ct=new OrbitControls(cam,rn.domElement);ct.enableDamping=true;ct.autoRotate=true;ct.autoRotateSpeed=0.9;
const LIMB=__LIMB__, ANAT=DATA.anat;let pts=[];
function acol(f){return ANAT[f]||[0.7,0.72,0.76];}
function mk(p,c,s){const g=new THREE.BufferGeometry();
  g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));
  g.setAttribute('color',new THREE.Float32BufferAttribute(c,3));
  return new THREE.Points(g,new THREE.PointsMaterial({size:s,vertexColors:true,sizeAttenuation:true}));}
function show(name){
  for(const p of pts){sc.remove(p);p.geometry.dispose();p.material.dispose();}
  const d=DATA.sp[name];const N=d.fate.length;const bp=[],bc=[],hp=[],hc=[];
  for(let k=0;k<N;k++){const isL=d.fate[k]===LIMB;
    const c=(mode==='anat')?acol(d.fate[k]):[0.6,0.64,0.7];
    if(hl&&isL){hp.push(d.xyz[3*k],d.xyz[3*k+1],d.xyz[3*k+2]);hc.push(0.16,0.86,0.30);}
    else{bp.push(d.xyz[3*k],d.xyz[3*k+1],d.xyz[3*k+2]);bc.push(c[0],c[1],c[2]);}}
  pts=[mk(bp,bc,0.028)];if(hp.length)pts.push(mk(hp,hc,0.060));
  for(const p of pts)sc.add(p);
  let R=0;for(let k=0;k<bp.length;k+=3){R=Math.max(R,Math.hypot(bp[k],bp[k+1],bp[k+2]));}
  cam.position.set(R*1.4,R*0.6,R*1.8);ct.target.set(0,0,0);
  const tag=d.body_plan==='finned'?'finned — NO limbs (fish)':(d.nlimb+' limb-bud cells (tetrapod)');
  document.getElementById('stat').textContent=name+'  ·  body plan '+d.body_plan+'  ·  Wnt-PCP conv_ext '+d.conv_ext+'  ·  '+tag;}
const KEY=[['Limb bud','#29db4d'],['Eye','#33d8ff'],['Heart','#ee2938'],['Otic','#ffd23a'],['Neural','#5b78e8'],['Somite/meso','#e69a66']];
document.getElementById('key').innerHTML=KEY.map(k=>`<span><i class="dot" style="background:${k[1]}"></i>${k[0]}</span>`).join('');
const sel=document.getElementById('sp');
Object.keys(DATA.sp).forEach(k=>{const o=document.createElement('option');o.value=k;o.textContent=k;sel.appendChild(o);});
sel.onchange=()=>show(sel.value);
const mb=document.getElementById('mode'),lb=document.getElementById('limbs');
mb.onclick=()=>{mode=(mode==='anat')?'plain':'anat';mb.textContent=(mode==='anat')?'anatomy':'plain';mb.classList.toggle('on',mode==='anat');show(sel.value);};
lb.onclick=()=>{hl=!hl;lb.classList.toggle('on',hl);show(sel.value);};
document.getElementById('rot').onchange=e=>ct.autoRotate=e.target.checked;
addEventListener('resize',()=>{cam.aspect=innerWidth/innerHeight;cam.updateProjectionMatrix();rn.setSize(innerWidth,innerHeight);});
show(Object.keys(DATA.sp)[0]);
(function a(){requestAnimationFrame(a);ct.update();rn.render(sc,cam);})();
</script></body></html>"""


def main():
    species = [("basic vertebrate", Genome())]
    for sp in BIG_SEVEN:
        species.append((sp, reference_genome(sp)))
    species.append(("human male", reference_genome("human_male")))
    species.append(("human female", reference_genome("human_female")))
    species.append(("mouse", reference_genome("mouse")))
    species.append(("zebrafish", reference_genome("zebrafish")))

    data = {"anat": ANAT_LIST, "sp": {}}
    for label, g in species:
        d = grow_species(g)
        data["sp"][label] = d
        print(f"  {label:20s} conv_ext {d['conv_ext']:.2f}  {d['body_plan']:8s}  limbs {d['nlimb']}")

    html = HTML.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__LIMB__", str(LIMB))
    out = Path("data/menagerie_viewer.html")
    out.write_text(html, encoding="utf-8")
    print(f"\nsaved {out}  ({out.stat().st_size/1024:.0f} KB)")
    print("serve: python -m http.server 8903 --directory data -> http://localhost:8903/menagerie_viewer.html")


if __name__ == "__main__":
    main()
