"""_phase_bisect.py -- THE PHASE-BISECT HARNESS (cycle 82, 2026-09-06).

Cycle 81 left the build NONDETERMINISTIC run-to-run at the movie scale (identical code, two regens,
every mesh frame differs) while simulate@9k checksums identical and the antinode eigsh is exonerated.
This harness replays the movie's model path STAGE BY STAGE and checksums the state after every stage,
so two or more processes can be diffed to the FIRST diverging stage:

  A.  simulate(1 -> CLOUD_N)         -- checksum EVERY emitted frame (P raw + P sorted + F)  => if the
                                        drift is size-gated, the first differing frame names the born count
  A'. symmetrize / shape_limbs / flex / rep_idx  (the last cloud frame, the handoff source)
  C0. build_base(30000)              -- the scored body's cloud (B0, BF)
  Ck. per mesh frame k in --frames   -- mature_cloud, grow_limbs, scale, standing_register, tuck_limbs,
                                        fetal_curl, chest, populate_autopods, flesh_cloud, closed_surface,
                                        body_relief   (each frame starts from B0, so frames are independent)

Two hashes per array: RAW (rows in place) and SORTED (rows lexsorted) -- if SORTED agrees while RAW
differs, the stage only REORDERED cells (a kd-tree / set-order class), and the real value drift is a
downstream consumer of that order.

Run (three or more processes, ideally in parallel so thread scheduling is stressed):
    venv_win_new\\Scripts\\python.exe -m medic._phase_bisect run --tag r1 [--frames 33,34,35] [--no-a]
Env controls are taken from the LAUNCHER (they must be set before the interpreter starts):
    PYTHONHASHSEED=0                        -> kills str-hash (set/dict order) nondeterminism
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 -> kills threaded BLAS reductions
Compare:
    venv_win_new\\Scripts\\python.exe -m medic._phase_bisect compare r1 r2 r3
Cards -> data/organ_cascade/_bisect_<tag>.json (data/ is gitignored; quote the verdict into the commit).
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

OUT = Path("data/organ_cascade")


def _h(a, nd=8):
    a = np.asarray(a)
    if a.dtype.kind in "iub":
        return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12]
    a = np.ascontiguousarray(np.round(np.asarray(a, np.float64), nd))
    return hashlib.sha1(a.tobytes()).hexdigest()[:12]


def _hs(a, nd=8):
    """order-invariant hash: rows lexsorted (float) or values sorted (int)."""
    a = np.asarray(a)
    if a.dtype.kind in "iub":
        return hashlib.sha1(np.ascontiguousarray(np.sort(a.ravel())).tobytes()).hexdigest()[:12]
    a = np.round(np.asarray(a, np.float64), nd)
    if a.ndim == 1:
        a = a[:, None]
    idx = np.lexsort(a.T[::-1])
    return hashlib.sha1(np.ascontiguousarray(a[idx]).tobytes()).hexdigest()[:12]


class Ledger:
    def __init__(self, tag):
        self.tag = tag
        self.rows = []
        self.t0 = time.time()
        self.tl = self.t0

    def add(self, name, arr, extra=None):
        now = time.time()
        a = np.asarray(arr)
        row = dict(stage=name, shape=list(a.shape), raw=_h(a), sorted=_hs(a), t_s=round(now - self.tl, 1))
        if extra:
            row.update(extra)
        self.rows.append(row)
        self.tl = now
        print(f"  [{name:<28s}] shape {str(list(a.shape)):<16s} raw {row['raw']} sorted {row['sorted']}  ({row['t_s']}s)",
              flush=True)

    def save(self):
        OUT.mkdir(parents=True, exist_ok=True)
        env = {k: os.environ.get(k) for k in ("PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                              "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
        card = dict(tag=self.tag, pid=os.getpid(), env=env, total_s=round(time.time() - self.t0, 1),
                    stages=self.rows)
        p = OUT / f"_bisect_{self.tag}.json"
        json.dump(card, open(p, "w"), indent=1)
        print(f"saved {p}  ({card['total_s']}s)")
        return p


def run(tag, frames, do_a=True, do_c=True, cloud_n=None):
    from medic import human_movie as hm
    from medic.unified_embryo import simulate, _symmetrize
    L = Ledger(tag)
    print(f"phase-bisect run tag={tag} pid={os.getpid()} hashseed={os.environ.get('PYTHONHASHSEED')} "
          f"threads={os.environ.get('OMP_NUM_THREADS')}", flush=True)

    if do_a:
        n_end = int(cloud_n or hm.CLOUD_N)
        print(f"[A] simulate 1 -> {n_end}", flush=True)
        fr, _ = simulate(use_ecm=True, seed=0, n_start=1, n_end=n_end, limb_buds=True,
                         convergent_ext=1.0, limb_params=hm.LIMB_SEARCHED,
                         fate_params=hm.fate_map_for(n_end))
        for fi, f in enumerate(fr):
            born, P, F = int(f[0]), np.asarray(f[3]), np.asarray(f[5])
            L.add(f"A.f{fi:02d}.P born={born}", P, dict(born=born, F_raw=_h(F), F_sorted=_hs(F)))
        P, V, F = _symmetrize(fr[-1][3], fr[-1][4], fr[-1][5])
        L.add("A.sym.P", P, dict(F_raw=_h(F)))
        c = P.mean(0); c[2] = 0.0
        sc = 0.9 / hm._long_axis_len(P)
        Q = hm.flex(hm.shape_limbs((P - c) * sc, F, 1.0, hm.LIMB), 1.0)
        L.add("A.proc_last", Q)
        rng = np.random.default_rng(0)
        idx = hm._rep_idx(F, hm.N_R, rng)
        L.add("A.rep_idx", np.asarray(idx))
        del fr, P, V, F, Q

    if do_c:
        from medic.adult_persistence_audit import build_base
        from medic.skin_closed_surface import closed_surface, flesh_cloud
        from medic.fine_relief_head import body_relief
        print(f"[C0] build_base({max(hm.N_R, 30000)})", flush=True)
        B0, BF = build_base(max(hm.N_R, 30000))
        B0 = B0.astype(float)
        L.add("C0.build_base.B0", B0, dict(BF_raw=_h(BF), BF_sorted=_hs(BF), n=int(len(B0))))
        headf = [hm.FIDX[n] for n in ("Forebrain", "Eye", "Midbrain", "Hindbrain") if n in hm.FIDX]
        hmB = np.isin(BF, headf)
        if hmB.any() and B0[hmB, 0].mean() < np.median(B0[:, 0]):
            B0[:, 0] = -B0[:, 0]
        limbmask = (BF == hm.LIMB)
        MS = hm.MATURE_SEARCHED
        for k in frames:
            f = k / (hm.N_MESH - 1)
            t = 0.55 + 0.45 * f
            pre = f"C{k:02d}."
            print(f"[C{k}] f={f:.3f}", flush=True)
            Q = hm.mature_cloud(B0, BF, f, MS);                      L.add(pre + "mature_cloud", Q)
            Q = hm.grow_limbs(Q, limbmask, hm._limb_grow_model(t, MS["limb_ext"]),
                              hm._limb_grow_model(t, MS.get("leg_ext", MS["limb_ext"])),
                              pose=f, fate=BF);                       L.add(pre + "grow_limbs", Q)
            tgt = 0.9 + (3.2 - 0.9) * f ** 1.2
            Q = Q * (tgt / hm._long_axis_len(Q));                    L.add(pre + "scale", Q)
            Q = hm.standing_register(Q, BF, f);                      L.add(pre + "standing_register", Q)
            Q = hm.tuck_limbs(Q, BF, hm._curl_amt(t));               L.add(pre + "tuck_limbs", Q)
            Q = hm.fetal_curl(Q, hm._curl_amt(t), BF, sign=+1.0);    L.add(pre + "fetal_curl", Q)
            Q = Q - hm._chest(Q, BF);                                L.add(pre + "chest", Q)
            Q, hv, hf, fv, ff = hm.populate_autopods(Q, BF, frac=f)
            L.add(pre + "populate_autopods", Q, dict(hv=_h(hv) if hv is not None else None,
                                                     fv=_h(fv) if fv is not None else None))
            FC = flesh_cloud(Q, BF, autopods=(fv, hv));              L.add(pre + "flesh_cloud", FC)
            grid = int(np.clip(round(tgt / (3.2 / 210.0)), 126, 210))
            if k >= hm.N_MESH - 2:
                grid = 210
            sv, sf = closed_surface(FC, grid=grid, sigma=1.35, iso_frac=0.38, smooth_iters=12)
            L.add(pre + "closed_surface.V", sv, dict(F_raw=_h(np.asarray(sf)), grid=grid))
            sv2, _ = body_relief(sv, Q, BF, amp=f);                  L.add(pre + "body_relief.V", sv2)
    return L.save()


def compare(tags):
    cards = []
    for t in tags:
        p = OUT / f"_bisect_{t}.json"
        cards.append(json.load(open(p)))
    names = [r["stage"] for r in cards[0]["stages"]]
    print("env per run:", {c["tag"]: c["env"] for c in cards})
    first = None
    for i, nm in enumerate(names):
        rows = []
        for c in cards:
            r = c["stages"][i] if i < len(c["stages"]) else None
            if r is None or r["stage"] != nm:
                r = next((x for x in c["stages"] if x["stage"] == nm), None)
            rows.append(r)
        if any(r is None for r in rows):
            print(f"  {nm:<32s} MISSING in some run"); continue
        raw_ok = len({r["raw"] for r in rows}) == 1
        srt_ok = len({r["sorted"] for r in rows}) == 1
        extra_keys = [k for k in rows[0] if k not in ("stage", "shape", "raw", "sorted", "t_s")]
        ex_ok = all(len({json.dumps(r.get(k)) for r in rows}) == 1 for k in extra_keys)
        verdict = "IDENTICAL" if (raw_ok and srt_ok and ex_ok) else \
                  ("ORDER-ONLY" if (srt_ok and not raw_ok) else "DIFFERS")
        if not ex_ok and verdict == "IDENTICAL":
            verdict = "DIFFERS(extra)"
        flag = "" if verdict == "IDENTICAL" else "   <<<"
        print(f"  {nm:<32s} {verdict:<14s} raw {[r['raw'] for r in rows]} sorted {[r['sorted'] for r in rows]}{flag}")
        if first is None and verdict != "IDENTICAL":
            first = (nm, verdict)
    print()
    if first is None:
        print(f"VERDICT: all {len(names)} stages IDENTICAL across {len(cards)} runs "
              f"({len(cards) * (len(cards) - 1) // 2} pairs).")
    else:
        print(f"VERDICT: FIRST DIVERGENCE at [{first[0]}] ({first[1]}) across {len(cards)} runs.")


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--tag", required=True)
    r.add_argument("--frames", default="33,34,35")
    r.add_argument("--no-a", action="store_true")
    r.add_argument("--no-c", action="store_true")
    r.add_argument("--cloud-n", type=int, default=None)
    c = sub.add_parser("compare")
    c.add_argument("tags", nargs="+")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        frames = [int(x) for x in a.frames.split(",") if x.strip()]
        run(a.tag, frames, do_a=not a.no_a, do_c=not a.no_c, cloud_n=a.cloud_n)
    else:
        compare(a.tags)


if __name__ == "__main__":
    main()
