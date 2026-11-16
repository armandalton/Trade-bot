"""SMA64 touch detection and feature extraction.

Everything here is CAUSAL: each feature at row r only uses candles <= r (finite rolling
windows, no exponential smoothing, so results do not depend on how much history is loaded).

Event definition (all candles must be CLOSED):
  t = touch candle:   low <= SMA64 <= high, while the previous candle was entirely on one side.
  r = t + 1 = reaction candle. The signal is emitted at the CLOSE of r.
  Approach from above -> LONG candidate, from below -> SHORT candidate.
  For EVERY valid touch two candidate trades are built, both entered at the close of r:
    REVERSAL: trade with the approach direction s   (bounce off the SMA)
    BREAKOUT: trade against the approach direction  (continue through the SMA)
  The reaction candle is an input feature (and is reported as the `reaction` column), NOT a gate:
  the model learns which candidate is better in which situation (see model.select_best).
"""
import numpy as np
import pandas as pd

from .config import CFG, INTERVAL_MS

FEATURES = [
    "side", "sma_slope_atr", "pre_dist_atr", "approach_speed", "break_depth",
    "t_body_atr", "t_body_ratio", "t_rej_wick", "t_against_wick",
    "r_body_atr", "r_body_ratio", "r_rej_wick", "r_against_wick", "r_close_vs_sma",
    "rel_vol_t", "rel_vol_r", "vol_change", "bars_near_10", "closes_wrong_10",
    "cross_count_30", "short_trend", "atr_bps", "vol_regime", "cost_r",
    "sma_slope_long", "trend_60", "range_pos_60", "hour_sin", "hour_cos",
]


def prepare(df, cfg=CFG):
    x = df.reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close", "volume"):
        x[col] = x[col].astype(float)
    x["sma"] = x["close"].rolling(cfg["sma_period"]).mean()
    pc = x["close"].shift(1)
    tr = pd.concat([x["high"] - x["low"], (x["high"] - pc).abs(), (x["low"] - pc).abs()], axis=1).max(axis=1)
    x["atr"] = tr.rolling(cfg["atr_period"]).mean()
    bad = (x["time"].diff() != INTERVAL_MS).astype(float)
    if len(bad):
        bad.iloc[0] = 0.0
    x["gap_bad"] = bad
    x["gap_recent"] = bad.rolling(140).sum()
    return x


def _div(a, b):
    return a / b.replace(0, np.nan)


def _wicks(o, h, l, c):
    return h - np.maximum(o, c), np.minimum(o, c) - l


def extract_events(x, symbol, cfg=CFG):
    """Return one row per SMA touch (setup or not) with causal features, entry/SL/TP levels."""
    o, h, l, c, v, sma, atr = (x[k] for k in ("open", "high", "low", "close", "volume", "sma", "atr"))
    t_o, t_h, t_l, t_c, t_v = o.shift(1), h.shift(1), l.shift(1), c.shift(1), v.shift(1)
    t_sma, t_atr = sma.shift(1), atr.shift(1)

    touch = (t_l <= t_sma) & (t_h >= t_sma)
    from_above = touch & (l.shift(2) > sma.shift(2))
    from_below = touch & (h.shift(2) < sma.shift(2))
    cand = from_above | from_below
    s = pd.Series(np.where(from_above, 1.0, -1.0), index=x.index)
    long_ = s > 0

    f = pd.DataFrame(index=x.index)
    f["side"] = s
    f["sma_slope_atr"] = s * _div(sma.shift(1) - sma.shift(9), t_atr)
    f["pre_dist_atr"] = _div((c.shift(4) - sma.shift(4)).abs(), t_atr)
    f["approach_speed"] = s * _div(c.shift(5) - c.shift(2), t_atr)
    f["break_depth"] = s * _div(t_c - t_sma, t_atr)

    up_t, lo_t = _wicks(t_o, t_h, t_l, t_c)
    f["t_body_atr"] = _div((t_c - t_o).abs(), t_atr)
    f["t_body_ratio"] = _div((t_c - t_o).abs(), t_h - t_l)
    f["t_rej_wick"] = _div(lo_t.where(long_, up_t), t_atr)
    f["t_against_wick"] = _div(up_t.where(long_, lo_t), t_atr)

    up_r, lo_r = _wicks(o, h, l, c)
    f["r_body_atr"] = s * _div(c - o, t_atr)
    f["r_body_ratio"] = _div((c - o).abs(), h - l)
    f["r_rej_wick"] = _div(lo_r.where(long_, up_r), t_atr)
    f["r_against_wick"] = _div(up_r.where(long_, lo_r), t_atr)
    f["r_close_vs_sma"] = s * _div(c - sma, t_atr)

    vm = v.rolling(20).mean().shift(2)
    f["rel_vol_t"] = _div(t_v, vm)
    f["rel_vol_r"] = _div(v, vm)
    f["vol_change"] = _div(v, t_v)

    near = ((c - sma).abs() <= 0.5 * atr).astype(float).where(sma.notna() & atr.notna())
    f["bars_near_10"] = near.rolling(10).sum().shift(2)
    below10 = (c < sma).astype(float).where(sma.notna()).rolling(10).sum().shift(2)
    above10 = (c > sma).astype(float).where(sma.notna()).rolling(10).sum().shift(2)
    f["closes_wrong_10"] = below10.where(long_, above10)
    sign = np.sign(c - sma).where(sma.notna())
    f["cross_count_30"] = (sign != sign.shift(1)).astype(float).where(sign.notna() & sign.shift(1).notna()).rolling(30).sum().shift(2)
    f["short_trend"] = s * _div(c.shift(2) - c.shift(22), t_atr)
    f["atr_bps"] = t_atr / c.shift(1) * 1e4
    f["vol_regime"] = _div(t_atr, atr.rolling(100).mean().shift(2))
    f["sma_slope_long"] = s * _div(sma.shift(1) - sma.shift(31), t_atr)
    f["trend_60"] = s * _div(c.shift(2) - c.shift(62), t_atr)
    hi60, lo60 = h.rolling(60).max().shift(2), l.rolling(60).min().shift(2)
    f["range_pos_60"] = _div(c.shift(2) - lo60, hi60 - lo60)
    minutes = (x["time"] // 60_000) % 1440
    f["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    f["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)

    rev = ((s > 0) & (c > o) & (c > sma)) | ((s < 0) & (c < o) & (c < sma))
    brk = ((s > 0) & (c < o) & (c < sma)) | ((s < 0) & (c > o) & (c > sma))
    reaction = np.where(rev, "REVERSAL", np.where(brk, "BREAKOUT", "NEUTRAL"))
    entry = c
    ext_low, ext_high = np.minimum(t_l, l), np.maximum(t_h, h)
    buf = cfg["stop_buffer_atr"] * t_atr
    base_valid = cand & (x["gap_recent"] == 0)
    nonfeat = [k for k in FEATURES if k != "cost_r"]
    base_valid = base_valid & f[nonfeat].notna().all(axis=1)

    frames = []
    for kind, tdir in (("REVERSAL", s), ("BREAKOUT", -s)):
        tlong = tdir > 0
        sl_raw = (ext_low - buf).where(tlong, ext_high + buf)
        dist = (entry - sl_raw).where(tlong, sl_raw - entry)
        dist = np.maximum(dist, cfg["min_stop_atr"] * t_atr)
        dist = np.maximum(dist, cfg["min_stop_pct"] * entry)
        sl = (entry - dist).where(tlong, entry + dist)
        tp1 = (entry + cfg["rr"] * dist).where(tlong, entry - cfg["rr"] * dist)
        tp2 = (entry + 1.5 * cfg["rr"] * dist).where(tlong, entry - 1.5 * cfg["rr"] * dist)
        cost_r = (2 * cfg["fee_rate"] + 2 * cfg["slippage_rate"]) * entry / dist
        ev = f.copy()
        ev["cost_r"] = cost_r
        ev["symbol"] = symbol
        ev["idx"] = np.arange(len(x))
        ev["time"] = x["time"].values                        # open time of reaction candle
        ev["signal_time"] = x["time"].values + INTERVAL_MS   # close time of reaction candle
        ev["entry_time"] = ev["signal_time"]
        ev["side_name"] = np.where(tlong, "LONG", "SHORT")   # TRADE direction
        ev["kind"] = kind
        ev["approach_side"] = np.where(long_, "LONG", "SHORT")
        ev["reaction"] = reaction
        ev["entry"], ev["sl"], ev["tp1"], ev["tp2"], ev["distance"] = entry, sl, tp1, tp2, dist
        ev["sma"] = sma
        ev["setup"] = True
        ev["tradable"] = cost_r <= cfg["max_cost_r"]
        ev["no_setup_reason"] = np.where(ev["tradable"], "", "cost")
        valid = base_valid & ev[FEATURES].notna().all(axis=1) & dist.notna() & (dist > 0)
        frames.append(ev[valid])
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["idx", "kind"], kind="stable").reset_index(drop=True)


def with_rr(ev, rr):
    """Same candidates with the first target moved to `rr` x stop distance (second target = 1.5 x that)."""
    out = ev.copy()
    sign = np.where(out["side_name"] == "LONG", 1.0, -1.0)
    out["tp1"] = out["entry"] + sign * rr * out["distance"]
    out["tp2"] = out["entry"] + sign * 1.5 * rr * out["distance"]
    return out


def build_events(df, symbol, cfg=CFG):
    x = prepare(df, cfg)
    return extract_events(x, symbol, cfg)
