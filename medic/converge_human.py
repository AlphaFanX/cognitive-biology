"""
converge_human.py -- the UNATTENDED grand-process convergence loop.

Runs genome -> heads -> {atlas FORM, Physiome FUNCTION} for thousands of steps so the human converges on its
ground truths without a human in the loop. Each step perturbs the LGM read-out (the maturation knob vector),
re-matures the human, and scores it on two automated objectives:

  FORM     : the human-atlas silhouette + Vitruvian canon + shared-organ address, under a mechanism penalty
             (integrin continuity / lateral-inhibition bilaterality / CE aspect) so the search can never buy a
             better fit by breaking a mechanism -- reused verbatim from medic.atlas_relax_search.objective, whose
             target is the human atlas (HESTA/Gray's form + organ placement).
  FUNCTION : the electric-organ frame quality on the matured cloud -- each organ recovering a clean dominant
             axis and its subheads falling in order on its own gap-junction eigenframe (medic.electric_organs_head),
             the Physiome-grounded operator.

It is a (1+1) evolution strategy with an adaptive step size: propose, accept if the combined score improves, grow
the step on success and shrink it on failure. Every step is CHECKPOINTED, so a crash or a laptop sleep loses at
most one step; on restart it RESUMES from the checkpoint. On every improvement the best knobs are also written to
the cache the movie reads (data/organ_cascade/atlas_relax_search.json -> human_movie.MATURE_SEARCHED), so the
converged human flows straight into the render.

The base cloud is built ONCE (the knobs are maturation applied to it), so a step is a re-maturation + two scores,
a few seconds -- thousands of steps fit in a night. Designed to be launched in the background and left running.

Run (foreground test):  cd cognimed && venv_win_new/Scripts/python.exe -m medic.converge_human --steps 5 --ne 20000
Run (overnight):        venv_win_new/Scripts/python.exe -u -m medic.converge_human   (redirect to a log, background)
State: data/organ_cascade/converge_human_ckpt.json   Log: converge_human.log
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

from medic import atlas_relax_search as ARS
from medic import electric_organs_head as EO

CKPT = "data/organ_cascade/converge_human_ckpt.json"
CACHE = "data/organ_cascade/atlas_relax_search.json"   # the file human_movie.MATURE_SEARCHED reads
LOG = "converge_human.log"
W_FUNC = 0.8                    # weight of the Physiome/function term relative to the atlas/form term
FUNC_EVERY = 2                  # recompute the (costlier) electric-organ score every N steps; hold it between
SIG0, SIG_MIN, SIG_MAX = 0.18, 0.03, 0.45


def _log(msg):
    line = f"{msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def func_score(Q, F):
    """1 - electric-organ frame quality on the matured cloud (lower is better). Rewards each organ recovering a
    clean dominant axis and multi-subhead organs whose subheads fall in order on the organ's own eigenframe."""
    try:
        res = EO.build(Q, F)
    except Exception:
        return 1.0
    if not res:
        return 1.0
    corrs = [v["mode1_axis_corr"] for v in res.values()]
    multi = [v for v in res.values() if len(v["subheads"]) > 1]
    ordered = (sum(1 for v in multi if v["subheads_ordered"]) / len(multi)) if multi else 0.0
    return float(1.0 - (0.5 * np.mean(corrs) + 0.5 * ordered))


def _clip(k):
    return {kk: float(np.clip(v, *ARS.RANGES[kk])) if kk in ARS.RANGES else v for kk, v in k.items()}


def _perturb(k, sigma, rng):
    out = dict(k)
    for kk in ARS.RANGES:
        lo, hi = ARS.RANGES[kk]
        out[kk] = float(np.clip(k.get(kk, (lo + hi) / 2) + rng.normal() * sigma * (hi - lo), lo, hi))
    return out


def _save_ckpt(step, best, best_s, comps, sigma, hist):
    json.dump(dict(step=step, best_knobs=best, best_score=best_s, components=comps, sigma=sigma,
                   history=hist[-200:]), open(CKPT, "w"), indent=1)


def _save_cache(best, best_s, comps):
    json.dump(dict(best_knobs=best, best_score=round(best_s, 4), components=comps,
                   source="converge_human"), open(CACHE, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)     # effectively "until stopped"
    ap.add_argument("--ne", type=int, default=60000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _log(f"[converge] building base cloud at ne={args.ne} (once) ...")
    t0 = time.time()
    base, F = ARS.build_base(args.ne)
    tgt = ARS.atlas_target()
    base_aspect = ARS._metrics(ARS.mature_with(base, F, dict(ARS.MATURE_DEFAULTS), f=0.0), F)["aspect"]
    _log(f"[converge] base ready ({len(base)} cells, {time.time()-t0:.0f}s)")

    def form_score(k):
        return float(ARS.objective(base, F, k, tgt, base_aspect)[0])

    last_func = [None]

    def full_score(k, step):
        sf = form_score(k)
        if step % FUNC_EVERY == 0 or last_func[0] is None:
            Q = ARS.mature_with(base, F, k)
            last_func[0] = func_score(Q, F)
        su = last_func[0]
        return sf + W_FUNC * su, dict(form=round(sf, 4), func=round(su, 4))

    # resume, or start from the current cache best / defaults
    if os.path.exists(CKPT):
        st = json.load(open(CKPT))
        best, best_s, sigma, step0 = st["best_knobs"], st["best_score"], st.get("sigma", SIG0), st["step"]
        hist = st.get("history", [])
        _log(f"[converge] RESUMING from step {step0}, best_score {best_s:.4f}")
    else:
        if os.path.exists(CACHE):
            try:
                best = {**ARS.MATURE_DEFAULTS, **json.load(open(CACHE))["best_knobs"]}
            except Exception:
                best = dict(ARS.MATURE_DEFAULTS)
        else:
            best = dict(ARS.MATURE_DEFAULTS)
        best = _clip(best)
        best_s, comps = full_score(best, 0)
        sigma, step0, hist = SIG0, 0, []
        _save_ckpt(step0, best, best_s, comps, sigma, hist)
        _save_cache(best, best_s, comps)
        _log(f"[converge] start best_score {best_s:.4f}  {comps}")

    rng = np.random.default_rng(args.seed + step0)
    accepts = 0
    for i in range(step0 + 1, step0 + 1 + args.steps):
        try:
            cand = _perturb(best, sigma, rng)
            s, comps = full_score(cand, i)
            improved = s < best_s
            if improved:
                best, best_s, accepts = cand, s, accepts + 1
                sigma = min(SIG_MAX, sigma * 1.15)                 # 1/5-rule: grow on success
                _save_cache(best, best_s, comps)                   # feed the movie
                _log(f"[{i}] * accept score={best_s:.4f} {comps} sigma={sigma:.3f}")
            else:
                sigma = max(SIG_MIN, sigma * 0.97)                 # shrink on failure
            hist.append(dict(step=i, score=round(s, 4), best=round(best_s, 4), accepted=bool(improved)))
            _save_ckpt(i, best, best_s, comps, sigma, hist)
            if i % 25 == 0:
                _log(f"[{i}] best={best_s:.4f} sigma={sigma:.3f} accepts={accepts}/{i-step0}")
        except KeyboardInterrupt:
            _log(f"[converge] interrupted at step {i}; checkpoint saved."); break
        except Exception as e:
            _log(f"[{i}] ERROR {type(e).__name__}: {e} (candidate rejected, continuing)")
            sigma = max(SIG_MIN, sigma * 0.9)
            continue
    _log(f"[converge] stopped. best_score {best_s:.4f} best_knobs {best}")


if __name__ == "__main__":
    main()
