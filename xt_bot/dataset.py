"""Dataset assembly with a strictly chronological 60/20/20 split and purging.

Purging: a labelled setup is kept in a split only if its whole outcome window (exit time)
ends before the next split starts, so no label peeks across a split boundary."""
import numpy as np
import pandas as pd

from .config import CFG
from .features import prepare, extract_events
from .labeling import label_events, RESOLVED


def events_for_symbol(df, symbol, cfg=CFG):
    x = prepare(df, cfg)
    ev = extract_events(x, symbol, cfg)
    return label_events(x, ev, cfg)


def split_bounds(ev, cfg=CFG):
    t0, t1 = float(ev["signal_time"].min()), float(ev["signal_time"].max())
    a, b, _ = cfg["split"]
    return t0, t0 + (t1 - t0) * a, t0 + (t1 - t0) * (a + b), t1


def assign_splits(ev, cfg=CFG):
    t0, c1, c2, t1 = split_bounds(ev, cfg)
    out = ev.copy()
    st = out["signal_time"].values
    out["split"] = np.where(st < c1, "train", np.where(st < c2, "val", "test"))
    limit = np.where(out["split"] == "train", c1, np.where(out["split"] == "val", c2, np.inf))
    resolved = out["label"].isin(RESOLVED).values
    leaks = resolved & (out["exit_time"].values >= limit)
    out["purged"] = leaks
    return out, {"start": t0, "train_end": c1, "val_end": c2, "end": t1}


def trade_set(ev, split):
    m = (ev["split"] == split) & ev["label"].isin(RESOLVED) & (~ev["purged"])
    return ev[m].reset_index(drop=True)


def events_multi_rr(df, symbol, rrs, cfg=CFG):
    """Download-once helper: the same candidates labelled for several reward:risk ratios."""
    from .features import with_rr
    x = prepare(df, cfg)
    base = extract_events(x, symbol, cfg)
    return {rr: label_events(x, with_rr(base, rr), cfg) for rr in rrs}
