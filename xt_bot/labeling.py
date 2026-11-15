"""Historical outcome labelling. FUTURE candles are used here and ONLY here (to create labels).

Label rules for a tradable setup (entry = close of reaction candle, SL/TP from features.py),
scanning candles r+1 .. r+horizon:
  valid_reversal : TP (2R) touched before SL
  weak_reversal  : neither SL nor TP within the horizon
  true_break     : SL touched first AND price closed on the wrong side of the SMA for >= 3
                   consecutive candles before/at the stop (sustained break)
  false_break    : SL touched first WITHOUT a sustained close beyond the SMA (spike / failed move)
  no_setup       : touch without a valid reaction candle, or setup rejected because costs > max_cost_r
If SL and TP are inside the same 1m candle the order is unknowable (1m is the finest data), so the
trade is conservatively booked as a stop-out and flagged `ambiguous`.
"""
import numpy as np
import pandas as pd

from .config import CFG, INTERVAL_MS

RESOLVED = ("valid_reversal", "weak_reversal", "true_break", "false_break")


def trade_net_r(side, entry, sl, tp, result, exit_close, distance, cfg=CFG):
    """Net result in R for one trade after fees and slippage (entry slip, adverse slip on stop/timeout exits)."""
    d = 1.0 if side == "LONG" else -1.0
    slip, fee = cfg["slippage_rate"], cfg["fee_rate"]
    entry_eff = entry * (1 + d * slip)
    if result == "tp":
        exit_eff = tp
    else:
        raw = sl if result == "sl" else exit_close
        exit_eff = raw * (1 - d * slip)
    gross = (exit_eff - entry_eff) * d
    fees = fee * (entry_eff + exit_eff)
    return (gross - fees) / distance


def _sustained(wrong, k=3):
    run = 0
    for w in wrong:
        run = run + 1 if w else 0
        if run >= k:
            return True
    return False


def label_events(x, ev, cfg=CFG):
    H = cfg["horizon"]
    n = len(x)
    hi, lo, cl = x["high"].values, x["low"].values, x["close"].values
    sm, tm, gap = x["sma"].values, x["time"].values, x["gap_bad"].values
    cols = {k: [] for k in ("label", "result", "exit_idx", "exit_time", "net_r", "ambiguous", "mfe_r", "mae_r")}
    for i in range(len(ev)):
        idx = int(ev["idx"].iat[i])
        if not bool(ev["tradable"].iat[i]):
            vals = ("no_setup", "none", -1, 0, np.nan, False, np.nan, np.nan)
        elif idx + H > n - 1:
            vals = ("pending", "none", -1, 0, np.nan, False, np.nan, np.nan)
        else:
            j0, j1 = idx + 1, idx + H
            if gap[j0:j1 + 1].sum() > 0:
                vals = ("gap", "none", -1, 0, np.nan, False, np.nan, np.nan)
            else:
                side = ev["side_name"].iat[i]
                entry, sl, tp = float(ev["entry"].iat[i]), float(ev["sl"].iat[i]), float(ev["tp1"].iat[i])
                dist = float(ev["distance"].iat[i])
                hs, ls = hi[j0:j1 + 1], lo[j0:j1 + 1]
                if side == "LONG":
                    sl_hit, tp_hit = ls <= sl, hs >= tp
                else:
                    sl_hit, tp_hit = hs >= sl, ls <= tp
                f_sl = int(np.argmax(sl_hit)) if sl_hit.any() else 10**9
                f_tp = int(np.argmax(tp_hit)) if tp_hit.any() else 10**9
                amb = False
                if f_sl == 10**9 and f_tp == 10**9:
                    result, e = "horizon", H - 1
                elif f_sl <= f_tp:
                    result, e, amb = "sl", f_sl, (f_sl == f_tp)
                else:
                    result, e = "tp", f_tp
                ej = j0 + e
                seg_h, seg_l = hi[j0:ej + 1], lo[j0:ej + 1]
                if side == "LONG":
                    mfe, mae = (seg_h.max() - entry) / dist, (entry - seg_l.min()) / dist
                    wrong = cl[j0:ej + 1] < sm[j0:ej + 1]
                else:
                    mfe, mae = (entry - seg_l.min()) / dist, (seg_h.max() - entry) / dist
                    wrong = cl[j0:ej + 1] > sm[j0:ej + 1]
                if result == "tp":
                    label = "valid_reversal"
                elif result == "horizon":
                    label = "weak_reversal"
                else:
                    label = "true_break" if _sustained(wrong) else "false_break"
                net = trade_net_r(side, entry, sl, tp, result, float(cl[ej]), dist, cfg)
                vals = (label, result, ej, int(tm[ej]) + INTERVAL_MS, net, amb, mfe, mae)
        for k, vv in zip(cols, vals):
            cols[k].append(vv)
    out = ev.copy()
    for k, vv in cols.items():
        out[k] = vv
    return out
