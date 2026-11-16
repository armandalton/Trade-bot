"""'When should I sit at the chart?' — best 3-hour window of the day (UTC and Tehran time).

The window is chosen ONLY from validation-segment results of the model-selected signals; the untouched
test segment is then used as confirmation. If the window does not hold on the test segment it is reported
as not stable and no hour filter is applied to live signals."""
import numpy as np
import pandas as pd

from .backtest import signal_stats

TEHRAN_OFFSET_MIN = 210  # UTC+3:30 (Iran has no daylight saving time since 2022)


def _fmt(h_utc):
    loc = (h_utc * 60 + TEHRAN_OFFSET_MIN) % 1440
    return f"{h_utc:02d}:00", f"{loc // 60:02d}:{loc % 60:02d}"


def window_label(start):
    end = (start + 3) % 24
    su, st = _fmt(start)
    eu, et = _fmt(end)
    return f"{su}–{eu} UTC = {st}–{et} به وقت تهران"


def hour_of(ms):
    return (np.asarray(ms) // 3_600_000) % 24


def analyze(touch_ev, sel_by_split, min_val_n=30):
    """touch_ev: one row per touch (all splits). sel_by_split: {'train','val','test'} -> selected signals DataFrame."""
    t = touch_ev.copy()
    t["hour"] = hour_of(t["signal_time"].values)
    touches = t.groupby("hour").size().reindex(range(24), fill_value=0)
    atr = t.groupby("hour")["atr_bps"].mean().reindex(range(24))
    sel = {s: d.assign(hour=hour_of(d["signal_time"].values)) if len(d) else d.assign(hour=[]) for s, d in sel_by_split.items()}

    def stats(split, hours):
        d = sel[split]
        return signal_stats(d[d["hour"].isin(hours)] if len(d) else d)

    rows = []
    for h in range(24):
        hrs = [h % 24, (h + 1) % 24, (h + 2) % 24]
        row = {"start": h, "hours": hrs, "touches": int(touches[hrs].sum()), "touch_share": float(touches[hrs].sum() / max(1, touches.sum())),
               "atr_bps": float(np.nanmean(atr[hrs].values))}
        for sp in ("train", "val", "test"):
            st = stats(sp, hrs)
            row[sp] = st
            row[sp + "_total_r"] = float(st["mean_r"] * st["n"]) if st["n"] else 0.0
        rows.append(row)
    cands = [r for r in rows if r["val"]["n"] >= min_val_n]
    if cands:
        best = max(cands, key=lambda r: r["val_total_r"])
        basis = "performance"
    else:
        best = max(rows, key=lambda r: r["touches"])
        basis = "touches"
    stable = bool(basis == "performance" and best["val"]["mean_r"] is not None and best["val"]["mean_r"] > 0
                  and best["test"]["n"] >= 20 and best["test"]["mean_r"] is not None and best["test"]["mean_r"] > 0)
    ranked = sorted(rows, key=lambda r: (r["val_total_r"] if r["val"]["n"] >= min_val_n else -1e9), reverse=True)[:3]
    per_hour = []
    for h in range(24):
        v = [sel[sp][sel[sp]["hour"] == h] for sp in ("val", "test") if len(sel[sp])]
        d = pd.concat(v, ignore_index=True) if v else None
        per_hour.append({"hour": h, "touches": int(touches[h]), "atr_bps": None if np.isnan(atr[h]) else float(atr[h]),
                         "sig": signal_stats(d)})
    return {"best": best, "basis": basis, "stable": stable, "top3": ranked, "per_hour": per_hour,
            "label": window_label(best["start"])}
