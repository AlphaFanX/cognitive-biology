"""
tuned_knobs.py -- load the per-head tuned knobs (medic.knob_tuning output) so each head USES them by default.

Mirrors LIMB_SEARCHED / FATE_SEARCHED / MATURE_SEARCHED: a head calls `tuned("<head>", {defaults})` and
gets its tuned values merged over the defaults, falling back cleanly to the defaults if the tuning JSON is
absent. No head imports here, so any head can import this without a cycle.
"""
from __future__ import annotations
import json
from pathlib import Path

_PATH = Path("data/organ_cascade/knob_tuning.json")


def _load():
    try:
        d = json.load(open(_PATH))
        return {t["head"]: t["after"] for t in d.get("tuned", [])}
    except Exception:
        return {}


TUNED = _load()


def tuned(head, default):
    """default merged with the head's tuned knobs (tuned wins). Returns a fresh dict."""
    return {**default, **TUNED.get(head, {})}
