"""viewer_tabs.py -- augment the generated human_movie_viewer.html with the extra tabs + reveal toggle.

human_movie.build() writes the viewer from the VIEWER template string, which would WIPE any tabs patched onto the
file. So the tabs live HERE and build() calls augment() after writing the HTML -> the GRAY'S tab, the NCA+LLM tab,
the big bottom-left plate box, and the manual anatomy-REVEAL toggle survive every re-render.

Data the tabs load (produced separately): data/movie/grays_parts.json, data/movie/nca_llm.json, data/grays_plates/*.
"""
from __future__ import annotations
from pathlib import Path

CSS = """
  /* --- extra tabs --- */
  #graysPanel h3,#ncaPanel h3{margin:0 0 4px;font-size:13px;color:#e8eef4}
  #gsum{color:#8091a8;font-size:10px;margin-bottom:6px}
  #gsearch,#nsearch{width:100%;box-sizing:border-box;background:#141a26;border:1px solid #2a3547;color:#cbd5e1;border-radius:6px;padding:4px 7px;font:11px system-ui;margin-bottom:6px}
  .gcat{color:#cbd5e1;font-size:11px;font-weight:600;letter-spacing:.03em;margin:8px 0 2px;border-top:1px solid #263043;padding-top:5px;display:flex;justify-content:space-between}
  .gcat .cs{color:#6b7688;font-weight:400}
  .gpart{display:flex;align-items:center;gap:6px;padding:2px 5px;border-radius:5px;cursor:pointer;font-size:11px}
  .gpart:hover{background:#1b2233}.gpart.sel{background:#24466e}
  .gpart .sd{width:8px;height:8px;border-radius:50%;flex:0 0 auto}
  .gpart .gn{flex:1;color:#dbe4ef;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .gpart .gsc{font-variant-numeric:tabular-nums;font-size:10px}
  .gpart .tf{color:#8091a8;font-size:9.5px}
  #gdetail,#ndetail{border-top:1px solid #263043;margin-top:8px;padding-top:7px}
  #gdetail .dt,#ndetail .dt{font-size:13px;color:#e8eef4;font-weight:600}
  #gdetail .dm,#ndetail .dm{color:#8091a8;font-size:10px;margin-bottom:5px}
  #gdetail .ck{display:flex;gap:6px;margin:3px 0;font-size:11px;line-height:1.3}
  #gdetail .ck .mk{flex:0 0 auto;font-weight:700}
  #gdetail .ck.pass .mk{color:#4ade80}#gdetail .ck.fail .mk{color:#f87171}
  #gdetail .ck .cd{color:#93a0b5}#gdetail .cf{color:#c9d3e0;font-weight:500}
  #narch{display:flex;flex-wrap:wrap;gap:3px;margin:4px 0 6px}
  .achip{font-size:9.5px;padding:2px 6px;border-radius:5px;background:#182234;color:#8fb4e0;cursor:pointer;border:1px solid #223049;white-space:nowrap}
  .achip:hover{background:#20304a}.achip.sel{background:#2b6cb0;color:#fff;border-color:#2b6cb0}
  .vgrid{display:grid;grid-template-columns:auto 1fr;gap:2px 8px;font-size:11px;margin:2px 0}
  .vgrid .vk{color:#8091a8}.vgrid .vv{color:#dbe4ef;font-variant-numeric:tabular-nums}
  .vsub{color:#c9d3e0;font-weight:600;font-size:11px;margin:7px 0 2px}
  .knob{background:#161d2b;border-radius:5px;padding:4px 6px;margin:3px 0;font-size:10.5px}
  .knob .kg{color:#ffd23a;font-weight:600}.knob .kb{color:#7dd3fc}.knob .kd{color:#93a0b5}
  #gplateBox{position:fixed;left:12px;bottom:60px;z-index:4;display:none;background:#0d1017ee;padding:9px;border-radius:10px;max-width:min(40vw,480px);box-shadow:0 6px 22px #000a}
  #gplateBox img{width:100%;border-radius:6px;background:#e8e4dc;display:block}
  #gplateBox .cap{color:#a8b8cc;font-size:11px;margin-top:6px;font-style:italic}
</style>"""

# the JS for the two data tabs + reveal toggle, injected before the render loop
JS = r"""
// ============================ EXTRA TABS + REVEAL TOGGLE ============================
let graysMode=false, GDATA=null, gCtx=null, gSelMesh=null, gSelKey=null;
let ncaMode=false, NDATA=null, ncaCtx=null, nSelMesh=null, nSelKey=null, nArchSel=null;
let revealOn=false;
function mkGrayCloud(flat,color,size,op){ const n=flat.length/3,P=new Float32Array(n*3);
  for(let i=0;i<n;i++){P[3*i]=flat[3*i+2];P[3*i+1]=flat[3*i];P[3*i+2]=flat[3*i+1];}
  const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.BufferAttribute(P,3));
  return new THREE.Points(g,new THREE.PointsMaterial({color:new THREE.Color(color[0],color[1],color[2]),size:size,sizeAttenuation:true,transparent:true,opacity:op==null?1:op})); }
function frameCam(c,r){ const cx=c[2],cy=c[0],cz=c[1]; ctrl.target.set(cx,cy,cz); cam.position.set(cx+r*0.3,cy+r*0.05,cz+r*3.2); }
function scoreColor(s){ const t=Math.max(0,Math.min(1,(s-0.5)/0.5));
  return `rgb(${Math.round(248-(248-74)*t)},${Math.round(113+(222-113)*t)},${Math.round(113+(128-113)*t)})`; }
function rgbOf(c){ return `rgb(${Math.round(c[0]*255)},${Math.round(c[1]*255)},${Math.round(c[2]*255)})`; }
// ---- Gray's ----
function buildGrayContext(){ const all=[]; for(const p of GDATA.parts) for(const v of p.xyz) all.push(v);
  gCtx=mkGrayCloud(all,[0.40,0.46,0.56],0.006,0.45); sc.add(gCtx); }
function renderRoster(){
  const q=((document.getElementById('gsearch')||{}).value||'').toLowerCase();
  const byCat={}; for(const p of GDATA.parts){(byCat[p.cat]=byCat[p.cat]||[]).push(p);}
  let html=`<h3>Gray's — every named part vs its checklist</h3><div id="gsum">${GDATA.n_parts} parts · mean ${GDATA.mean_score} · ${GDATA.fully_passing} fully pass · ${GDATA.n_plates} with plate</div><input id="gsearch" placeholder="filter parts…" value="${q}">`;
  for(const cat of GDATA.cat_order){ let list=(byCat[cat]||[]).filter(p=>!q||p.key.toLowerCase().includes(q)); if(!list.length)continue;
    const avg=GDATA.by_category[cat]; html+=`<div class="gcat">${cat} <span class="cs">${list.length} · ${avg==null?'':avg.toFixed(2)}</span></div>`;
    list.sort((a,b)=>a.score-b.score||a.key.localeCompare(b.key));
    for(const p of list){ const col=scoreColor(p.score);
      html+=`<div class="gpart${p.key===gSelKey?' sel':''}" data-k="${encodeURIComponent(p.key)}"><span class="sd" style="background:${col}"></span><span class="gn">${p.name}${p.plate?' 🖼':''}</span><span class="gsc" style="color:${col}">${p.score.toFixed(2)}</span></div>`; } }
  html+=`<div id="gdetail"></div>`; const gp=document.getElementById('graysPanel'); gp.innerHTML=html;
  gp.querySelectorAll('.gpart').forEach(el=>el.onclick=()=>selectPart(decodeURIComponent(el.dataset.k)));
  const se=document.getElementById('gsearch'); if(se){se.oninput=renderRoster; if(q){se.focus();se.setSelectionRange(q.length,q.length);}}
  if(gSelKey){const p=GDATA.parts.find(x=>x.key===gSelKey); if(p) renderPartDetail(p);} }
function renderPartDetail(p){ const dd=document.getElementById('gdetail'); if(!dd)return;
  let h=`<div class="dt">${p.name} <span style="color:${scoreColor(p.score)}">${p.passed}/${p.nfeat}</span></div><div class="dm">${p.key} · ${p.ncells} cells · ${p.cat}</div>`;
  for(const f of p.feats) h+=`<div class="ck ${f.ok?'pass':'fail'}"><span class="mk">${f.ok?'✓':'✗'}</span><span><span class="cf">${f.f}</span><br><span class="cd">${f.d}</span></span></div>`;
  dd.innerHTML=h; const box=document.getElementById('gplateBox');
  if(p.plate){box.style.display='block';box.innerHTML=`<img src="grays_plates/${p.plate}" alt="Gray's plate"><div class="cap">${p.plate_cap||''}</div>`;}else{box.style.display='none';box.innerHTML='';} }
function selectPart(key){ gSelKey=key; const p=GDATA.parts.find(x=>x.key===key); if(!p)return;
  if(gSelMesh){sc.remove(gSelMesh);gSelMesh.geometry.dispose();gSelMesh.material.dispose();}
  gSelMesh=mkGrayCloud(p.xyz,GDATA.cat_color[p.cat]||[0.9,0.9,0.9],0.02,1.0); sc.add(gSelMesh);
  let cx=0,cy=0,cz=0,n=p.xyz.length/3; for(let i=0;i<n;i++){cx+=p.xyz[3*i+2];cy+=p.xyz[3*i];cz+=p.xyz[3*i+1];}
  ctrl.target.set(cx/n,cy/n,cz/n); renderRoster(); }
function enterGrays(){ graysMode=true; playing=false; syncPlay(); document.getElementById('grays').classList.add('on');
  document.getElementById('moviePanel').style.display='none'; document.getElementById('graysPanel').style.display='block';
  for(const p of pts) p.visible=false; if(skinMesh) skinMesh.visible=false; for(const o of organMeshes) o.visible=false;
  const go=()=>{ if(!gCtx) buildGrayContext(); else gCtx.visible=true; if(gSelMesh) gSelMesh.visible=true; frameCam(GDATA.center,GDATA.radius); renderRoster(); };
  if(GDATA){go();} else {document.getElementById('graysPanel').innerHTML='<h3>Gray\'s</h3><div id="gsum">loading 294 parts…</div>';
    fetch('movie/grays_parts.json?v='+Date.now()).then(r=>r.json()).then(d=>{GDATA=d;go();});} }
function exitGrays(){ graysMode=false; document.getElementById('grays').classList.remove('on');
  document.getElementById('graysPanel').style.display='none'; document.getElementById('moviePanel').style.display='block';
  if(gCtx) gCtx.visible=false; if(gSelMesh) gSelMesh.visible=false;
  const box=document.getElementById('gplateBox'); if(box){box.style.display='none';box.innerHTML='';} build(cur); }
// ---- NCA+LLM ----
function buildNcaContext(){ const all=[]; for(const h of NDATA.heads) for(const v of h.xyz) all.push(v);
  ncaCtx=mkGrayCloud(all,[0.40,0.46,0.56],0.006,0.40); sc.add(ncaCtx); }
function vgrid(obj){ let h='<div class="vgrid">'; for(const k in obj){const v=obj[k];
  h+=`<span class="vk">${k}</span><span class="vv">${(v&&typeof v==='object')?JSON.stringify(v):(v==null?'—':v)}</span>`;} return h+'</div>'; }
function knobHtml(k){ return `<div class="knob"><span class="kg">${k.gene}</span> → ${k.trait||k.knob} ${k.beta!=null?`<span class="kb">β=${k.beta} ${k.unit||''}</span>`:''}<div class="kd">${k.snp||''} ${k.direction||''} · ${k.source||''}</div></div>`; }
function renderHeadRoster(){ const q=((document.getElementById('nsearch')||{}).value||'').toLowerCase();
  let html=`<h3>NCA + LLM — the model internals</h3><div id="gsum">${NDATA.n_heads} heads · ${NDATA.n_knobs} adapter knobs · click a layer or a head</div><div id="narch">`;
  NDATA.architecture.forEach((L,i)=>{html+=`<span class="achip${i===nArchSel?' sel':''}" data-i="${i}">${L.layer}</span>`;});
  html+=`</div><input id="nsearch" placeholder="filter heads / TF…" value="${q}">`;
  const byKind={}; for(const h of NDATA.heads){(byKind[h.kind]=byKind[h.kind]||[]).push(h);}
  for(const kind of NDATA.kind_order){ let list=(byKind[kind]||[]).filter(h=>!q||h.name.toLowerCase().includes(q)||(h.master_tf||'').toLowerCase().includes(q)); if(!list.length)continue;
    const c=rgbOf(NDATA.kind_color[kind]); html+=`<div class="gcat">${kind} <span class="cs">${list.length}</span></div>`;
    for(const h of list){ html+=`<div class="gpart${h.name===nSelKey?' sel':''}" data-k="${encodeURIComponent(h.name)}"><span class="sd" style="background:${c}"></span><span class="gn">${h.name}<span class="tf">${h.master_tf&&h.master_tf!=='-'?' · '+h.master_tf:''}</span></span><span class="gsc" style="color:#7f8ca3">${h.n_cells}</span></div>`; } }
  html+=`<div id="ndetail"></div>`; const np=document.getElementById('ncaPanel'); np.innerHTML=html;
  np.querySelectorAll('.achip').forEach(el=>el.onclick=()=>selectArch(+el.dataset.i));
  np.querySelectorAll('.gpart').forEach(el=>el.onclick=()=>selectHead(decodeURIComponent(el.dataset.k)));
  const se=document.getElementById('nsearch'); if(se){se.oninput=renderHeadRoster; if(q){se.focus();se.setSelectionRange(q.length,q.length);}}
  if(nSelKey){const h=NDATA.heads.find(x=>x.name===nSelKey); if(h) renderHeadDetail(h);} else if(nArchSel!=null) renderArchDetail(NDATA.architecture[nArchSel]); }
function renderArchDetail(L){ const dd=document.getElementById('ndetail'); if(!dd)return;
  let h=`<div class="dt">${L.layer}</div><div class="dm">${L.role}</div><div style="font-size:11px;color:#aeb9c9;margin-bottom:5px">${L.detail}</div><div class="vsub">values</div>`+vgrid(L.values);
  if(L.layer.indexOf('MLP')>=0||L.layer.indexOf('LGM')>=0){h+=`<div class="vsub">LGM adapter — GWAS knobs (real β)</div>`; for(const k of NDATA.adapter_knobs) h+=knobHtml(k);} dd.innerHTML=h; }
function renderHeadDetail(h){ const dd=document.getElementById('ndetail'); if(!dd)return;
  let html=`<div class="dt">${h.name} <span style="font-size:10px;color:#8091a8">${h.kind}${h.parent?' ◂ '+h.parent:''}</span></div><div class="dm">master TF ${h.master_tf} · AlphaGenome: ${h.ag_tissue||'—'} · ${h.n_cells} cells${h.implemented?'':' · (not yet in build)'}</div><div class="vsub">inner-NCA parameters (actual values)</div>`+
    vgrid({'Vm set-point (mV)':h.vm_setpoint_mV,'cadherin ADH':h.adhesion,'integrin ECM':h.ecm,'PRC2 unlock':h.prc2_unlock,'prolif weight':h.prolif_weight,'organ target (frac)':h.org_target,'paired':h.paired})+
    `<div class="vsub">electric-frame address (measured centroid)</div>`+vgrid({'AP':h.ap,'DV':h.dv,'LR (ML)':h.lr});
  if(h.knobs&&h.knobs.length){html+=`<div class="vsub">LGM adapter knobs on this head</div>`; for(const k of h.knobs) html+=knobHtml(k);} dd.innerHTML=html; }
function selectArch(i){ nArchSel=i; nSelKey=null; if(nSelMesh){sc.remove(nSelMesh);nSelMesh.geometry.dispose();nSelMesh.material.dispose();nSelMesh=null;} renderHeadRoster(); }
function selectHead(name){ nSelKey=name; nArchSel=null; const h=NDATA.heads.find(x=>x.name===name); if(!h)return;
  if(nSelMesh){sc.remove(nSelMesh);nSelMesh.geometry.dispose();nSelMesh.material.dispose();}
  nSelMesh=mkGrayCloud(h.xyz,NDATA.kind_color[h.kind]||[0.9,0.9,0.9],0.02,1.0); sc.add(nSelMesh);
  let cx=0,cy=0,cz=0,n=h.xyz.length/3; for(let i=0;i<n;i++){cx+=h.xyz[3*i+2];cy+=h.xyz[3*i];cz+=h.xyz[3*i+1];}
  ctrl.target.set(cx/n,cy/n,cz/n); renderHeadRoster(); }
function enterNca(){ ncaMode=true; playing=false; syncPlay(); document.getElementById('nca').classList.add('on');
  document.getElementById('moviePanel').style.display='none'; document.getElementById('ncaPanel').style.display='block';
  for(const p of pts) p.visible=false; if(skinMesh) skinMesh.visible=false; for(const o of organMeshes) o.visible=false;
  const go=()=>{ if(!ncaCtx) buildNcaContext(); else ncaCtx.visible=true; if(nSelMesh) nSelMesh.visible=true; frameCam(NDATA.center,NDATA.radius); renderHeadRoster(); };
  if(NDATA){go();} else {document.getElementById('ncaPanel').innerHTML='<h3>NCA+LLM</h3><div id="gsum">loading model internals…</div>';
    fetch('movie/nca_llm.json?v='+Date.now()).then(r=>r.json()).then(d=>{NDATA=d;go();});} }
function exitNca(){ ncaMode=false; document.getElementById('nca').classList.remove('on');
  document.getElementById('ncaPanel').style.display='none'; document.getElementById('moviePanel').style.display='block';
  if(ncaCtx) ncaCtx.visible=false; if(nSelMesh) nSelMesh.visible=false; build(cur); }
// ---- reveal toggle: jump between the adult skin (last mesh frame) and the anatomy reveal (last frame) ----
function _lastPhase(ph){ let idx=0; for(let i=0;i<DATA.frames.length;i++){ if(DATA.frames[i].phase===ph) idx=i; } return idx; }
function toggleReveal(){ if(!DATA) return; if(graysMode)exitGrays(); if(ncaMode)exitNca(); revealOn=!revealOn; playing=false; syncPlay();
  document.getElementById('reveal').classList.toggle('on',revealOn);
  cur=revealOn?_lastPhase('anatomy'):_lastPhase('mesh'); build(cur); }
document.getElementById('grays').onclick=()=>{ if(ncaMode)exitNca(); graysMode?exitGrays():enterGrays(); };
document.getElementById('nca').onclick=()=>{ if(graysMode)exitGrays(); ncaMode?exitNca():enterNca(); };
document.getElementById('reveal').onclick=toggleReveal;
// ============================ end extra tabs ============================
"""


def augment(path):
    p = Path(path); html = p.read_text(encoding="utf-8")
    reps = []
    # 1) CSS: append our rules just before the closing </style>
    reps.append(("  #key{margin-top:5px}#key span{margin-right:10px;white-space:nowrap;font-size:11px}\n</style>",
                 "  #key{margin-top:5px}#key span{margin-right:10px;white-space:nowrap;font-size:11px}\n" + CSS))
    # 2) wrap the movie panel + add the two tab panels
    reps.append(('<div id="panel"><h3>What is dialing — by layer</h3>',
                 '<div id="panel">\n<div id="moviePanel"><h3>What is dialing — by layer</h3>'))
    reps.append(('<div id="rows"></div></div>',
                 '<div id="rows"></div></div>\n<div id="graysPanel" style="display:none"></div>'
                 '\n<div id="ncaPanel" style="display:none"></div>\n</div>'))
    # 3) toolbar buttons
    reps.append(('  <button id="mode" class="on">anatomy</button>\n  <button id="rot">↻ auto-rotate</button>',
                 '  <button id="mode" class="on">anatomy</button>\n'
                 '  <button id="grays">🦴 Gray\'s</button>\n  <button id="nca">🧠 NCA+LLM</button>\n'
                 '  <button id="reveal">🔬 reveal</button>\n  <button id="rot">↻ auto-rotate</button>'))
    # 4) the big bottom-left plate box
    reps.append(('</div>\n<script type="importmap">',
                 '</div>\n<div id="gplateBox"></div>\n<script type="importmap">'))
    # 5) inject the tab JS right before the render loop
    reps.append(('function loop(t){ requestAnimationFrame(loop);', JS + '\nfunction loop(t){ requestAnimationFrame(loop);'))
    # 6) pause the movie clock while a tab / reveal is open
    reps.append(("if(DATA&&playing&&t-last>FRAME_MS)", "if(DATA&&!graysMode&&!ncaMode&&playing&&t-last>FRAME_MS)"))
    # 7) guard the movie controls so they don't fight a tab
    for a in ("playBtn.onclick=()=>{playing=!playing;syncPlay();};",
              "modeBtn.onclick=()=>{ mode=(mode==='anat')?'volt':'anat';",
              "slider.oninput=()=>{playing=false;syncPlay();cur=parseInt(slider.value);build(cur);};"):
        reps.append((a, a.replace("()=>{", "()=>{ if(graysMode||ncaMode)return; ", 1)))
    n_ok = 0
    for old, new in reps:
        if old in html:
            html = html.replace(old, new, 1); n_ok += 1
        else:
            print(f"  [viewer_tabs] anchor NOT found (skipped): {old[:60]}...")
    p.write_text(html, encoding="utf-8")
    print(f"  [viewer_tabs] augmented {p.name}: {n_ok}/{len(reps)} patches applied")


if __name__ == "__main__":
    augment("data/human_movie_viewer.html")
