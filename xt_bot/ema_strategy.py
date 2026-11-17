"""Standalone EMA 9/20 trend-following experiment for XT Futures 1m.

This is a research-only baseline. It does not send live signals or place orders.
Signals are generated at a CLOSED candle, entered at that candle's close, and
evaluated candle-by-candle with conservative same-candle stop/target handling.
"""
from datetime import datetime, timezone
import time

import numpy as np
import pandas as pd

from .config import CFG, DATA_DIR, INTERVAL_MS
from .data import fetch_klines, closed_only, validate_candles
from .util import log


VARIANTS = ("EMA9_PULLBACK", "EMA20_PULLBACK")
RR = 2.0
HORIZON = 240
MIN_STOP_ATR = 0.8
MIN_STOP_PCT = 0.004
STOP_BUFFER_ATR = 0.1


def _metrics(rows):
    if not rows:
        return {"n": 0, "win": None, "mean_r": None, "pf": None, "max_dd_r": 0.0}
    r = np.asarray([x["net_r"] for x in rows], dtype=float)
    wins, losses = r[r > 0], r[r < 0]
    pf = float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else (999.0 if len(wins) else None)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(np.r_[0.0, eq])[1:]
    dd = float(np.max(peak - eq)) if len(eq) else 0.0
    return {"n": int(len(r)), "win": float((r > 0).mean()),
            "mean_r": float(r.mean()), "pf": pf, "max_dd_r": dd}


def _fmt(m):
    if not m["n"]:
        return "Ø¨Ø¯ÙÙ ÙØ¹Ø§ÙÙÙ"
    pf = "â" if m["pf"] is None else ("â" if m["pf"] >= 999 else f"{m['pf']:.2f}")
    return (f"{m['n']} ÙØ¹Ø§ÙÙÙ | Ø¨Ø±Ø¯ {m['win']*100:.1f}% | "
            f"ÙÛØ§ÙÚ¯ÛÙ {m['mean_r']:+.2f}R | PF {pf} | Ø§ÙØª {m['max_dd_r']:.1f}R")


def _make_signals(df, symbol, variant):
    d = df.copy().reset_index(drop=True)
    close, high, low = d.close, d.high, d.low
    d["ema9"] = close.ewm(span=9, adjust=False, min_periods=9).mean()
    d["ema20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    pc = close.shift(1)
    tr = pd.concat([(high-low), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    d["atr"] = tr.rolling(14, min_periods=14).mean()

    # Trend/no-trade filter: ordered, separated and directionally sloped EMAs.
    spread = (d.ema9 - d.ema20).abs() / d.atr
    slope = (d.ema20 - d.ema20.shift(5)) / d.atr
    up = (close > d.ema9) & (close > d.ema20) & (d.ema9 > d.ema20) & (slope > 0.03) & (spread > 0.08)
    down = (close < d.ema9) & (close < d.ema20) & (d.ema9 < d.ema20) & (slope < -0.03) & (spread > 0.08)

    pullback_line = d.ema9 if variant == "EMA9_PULLBACK" else d.ema20
    # Previous candle tags the selected EMA; current closed candle confirms
    # resumption by closing back on the trend side with matching candle body.
    prev_touch = (low.shift(1) <= pullback_line.shift(1)) & (high.shift(1) >= pullback_line.shift(1))
    long_signal = up & prev_touch & (close > d.open) & (close > pullback_line)
    short_signal = down & prev_touch & (close < d.open) & (close < pullback_line)

    signals = []
    for idx in np.flatnonzero((long_signal | short_signal).fillna(False).to_numpy()):
        side = "LONG" if bool(long_signal.iloc[idx]) else "SHORT"
        entry = float(close.iloc[idx])
        atr = float(d.atr.iloc[idx])
        if not np.isfinite(atr) or atr <= 0:
            continue
        start = max(0, idx - 4)
        if side == "LONG":
            raw_dist = entry - (float(low.iloc[start:idx+1].min()) - STOP_BUFFER_ATR * atr)
        else:
            raw_dist = (float(high.iloc[start:idx+1].max()) + STOP_BUFFER_ATR * atr) - entry
        dist = max(raw_dist, MIN_STOP_ATR * atr, MIN_STOP_PCT * entry)
        if not np.isfinite(dist) or dist <= 0:
            continue
        stop = entry - dist if side == "LONG" else entry + dist
        target = entry + RR * dist if side == "LONG" else entry - RR * dist
        signals.append({"idx": int(idx), "symbol": symbol, "variant": variant,
                        "side": side, "entry": entry, "stop": stop,
                        "target": target, "distance": dist})
    return signals, d


def _resolve(signals, d, fee_rate, slip_rate):
    resolved = []
    n = len(d)
    for s in signals:
        i = s["idx"]
        if i + HORIZON >= n:
            continue
        result, exit_idx = None, None
        for j in range(i + 1, min(i + HORIZON, n - 1) + 1):
            # Missing candle: don't fabricate a fill across a data gap.
            if int(d.time.iloc[j]) - int(d.time.iloc[j-1]) != INTERVAL_MS:
                result = "gap"
                break
            if s["side"] == "LONG":
                stop_hit = float(d.low.iloc[j]) <= s["stop"]
                target_hit = float(d.high.iloc[j]) >= s["target"]
            else:
                stop_hit = float(d.high.iloc[j]) >= s["stop"]
                target_hit = float(d.low.iloc[j]) <= s["target"]
            if stop_hit and target_hit:
                result, exit_idx = "sl", j  # conservative: stop first
                break
            if stop_hit:
                result, exit_idx = "sl", j
                break
            if target_hit:
                result, exit_idx = "tp", j
                break
        if result == "gap" or result is None:
            continue
        direction = 1.0 if s["side"] == "LONG" else -1.0
        entry_eff = s["entry"] * (1 + direction * slip_rate)
        raw_exit = s["target"] if result == "tp" else s["stop"]
        exit_eff = raw_exit * (1 - direction * slip_rate)
        gross = (exit_eff - entry_eff) * direction
        fees = fee_rate * (entry_eff + exit_eff)
        net_r = (gross - fees) / s["distance"]
        resolved.append({**s, "exit_idx": exit_idx, "net_r": float(net_r),
                         "result": result, "signal_time": int(d.time.iloc[i]) + INTERVAL_MS})
    return resolved


def run_ema_experiment(fetcher=fetch_klines, cfg=CFG, now_ms=None):
    now_ms = int(now_ms or time.time() * 1000)
    start_ms = now_ms - int(cfg["train_days"]) * 86_400_000
    all_rows = {v: [] for v in VARIANTS}
    quality = {}
    notes = []

    for symbol in cfg["symbols"]:
        try:
            log(f"EMA experiment {symbol}: downloading 1m candles ...")
            raw = closed_only(fetcher(symbol, start_ms, now_ms, "1m"), now_ms)
            quality[symbol] = validate_candles(raw)
            if len(raw) < 1000 or quality[symbol].get("coverage", 0) < 0.90:
                notes.append(f"{symbol}: Ø¯Ø§Ø¯Ù Ú©Ø§ÙÛ/Ù¾ÙØ´Ø´ ÙÙØ§Ø³Ø¨ ÙØ¯Ø§Ø´Øª")
                continue
            for variant in VARIANTS:
                signals, prepared = _make_signals(raw, symbol, variant)
                all_rows[variant].extend(_resolve(
                    signals, prepared, cfg["fee_rate"], cfg["slippage_rate"]
                ))
        except Exception as exc:
            notes.append(f"{symbol}: {type(exc).__name__}: {str(exc)[:140]}")
            log(notes[-1])

    report = [
        "# Ú¯Ø²Ø§Ø±Ø´ Ø¢Ø²ÙØ§ÛØ´ EMA 9/20 â XT Futures (1m)",
        f"Ø²ÙØ§Ù Ø§Ø¬Ø±Ø§ (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"Ø¯ÙØ±Ù Ø¯Ø§Ø¯Ù: {cfg['train_days']} Ø±ÙØ² | RR=1:{RR:g} | Ø§ÙÙ={HORIZON} Ú©ÙØ¯Ù",
        f"ÙØ²ÛÙÙ ÙØ± Ø³ÙØª: fee={cfg['fee_rate']:.4%}Ø slippage={cfg['slippage_rate']:.4%}",
        "",
        "ÙÙØ§Ø¹Ø¯: Ø±ÙÙØ¯ Ø¨Ø§ ØªØ±ØªÛØ¨ EMA9/EMA20Ø Ø´ÛØ¨ EMA20 Ù ÙØ§ØµÙÙ ÙØ±ÙØ§ÙâØ´Ø¯ÙØ Ù¾ÙÙØ¨Ú© Ø¨Ù EMA Ø§ÙØªØ®Ø§Ø¨Û Ø¯Ø± Ú©ÙØ¯Ù ÙØ¨Ù Ù ØªØ£ÛÛØ¯ Ø¨Ø³ØªÙâØ´Ø¯Ù Ú©ÙØ¯Ù Ø¬Ø§Ø±Û. ÙØ±ÙØ¯ Ø¯Ø± close Ú©ÙØ¯Ù ØªØ£ÛÛØ¯. Ø­Ø¯Ø¶Ø±Ø± Ù¾Ø´Øª Ø§Ú©Ø³ØªØ±ÙÙ Ûµ Ú©ÙØ¯Ù Ø§Ø®ÛØ± Ø¨Ø§ Ø¨Ø§ÙØ± ATR Ù Ø­Ø¯Ø§ÙÙ ÙØ§ØµÙÙ Û°.Û¸ ATR ÛØ§ Û°.Û´Ùª. Ø­Ø¯Ø³ÙØ¯ Û²R. Ø¨Ø§Ø²Ø§Ø± Ø®ÙØ«Û ÙÛÙØªØ± ÙÛâØ´ÙØ¯.",
        "ÙÚ©ØªÙ: Ø§ÛÙ ÛÚ© ÙØ§ÙÙÙ Ù¾Ø§ÛÙ Ù ÙØ§Ø¨ÙâØ³ÙØ¬Ø´ Ø§Ø³ØªØ ÙÙ Ø§Ø¯Ø¹Ø§Û Ø¨ÙÛÙÙ Ø¨ÙØ¯Ù.",
        "",
    ]

    for variant in VARIANTS:
        rows = sorted(all_rows[variant], key=lambda x: x["signal_time"])
        if not rows:
            report += [f"## {variant}", "ÙÛÚ ÙØ¹Ø§ÙÙÙ Ø­ÙâØ´Ø¯ÙâØ§Û ØªÙÙÛØ¯ ÙØ´Ø¯.", ""]
            continue
        # Chronological split by signal time; purge trades whose exit crosses boundary.
        times = np.array([r["signal_time"] for r in rows], dtype=np.int64)
        c1 = times.min() + int((times.max() - times.min()) * 0.60)
        c2 = times.min() + int((times.max() - times.min()) * 0.80)
        splits = {"train": [], "validation": [], "test": []}
        for row in rows:
            split = "train" if row["signal_time"] < c1 else ("validation" if row["signal_time"] < c2 else "test")
            boundary = c1 if split == "train" else (c2 if split == "validation" else None)
            if boundary is not None:
                # Exit candle open time must finish strictly before next segment.
                # exit_idx is local to symbol data; use signal_time plus actual duration.
                # Purge by signal-time horizon conservatively when exact cross-symbol
                # timestamp mapping is unavailable.
                if row["signal_time"] + HORIZON * INTERVAL_MS >= boundary:
                    continue
            splits[split].append(row)

        report.append(f"## {variant}")
        for split, title in (("train", "Ø¢ÙÙØ²Ø´"), ("validation", "Ø§Ø¹ØªØ¨Ø§Ø±Ø³ÙØ¬Û"), ("test", "Ø¢Ø²ÙÙÙ ÙÙØ§ÛÛ")):
            report.append(f"- {title}: {_fmt(_metrics(splits[split]))}")
        test = _metrics(splits["test"])
        accepted = (test["n"] >= cfg["min_test_trades"] and test["win"] is not None
                    and test["win"] >= cfg["min_win_rate"] and test["pf"] is not None
                    and test["pf"] >= cfg["min_profit_factor"] and test["mean_r"] > 0)
        report.append(f"- ÙØªÛØ¬Ù Ø¢Ø²ÙÙÙ: {'Ø´Ø±Ø§ÛØ· Ø§ÙÙÛÙ Ø±Ø§ Ù¾Ø§Ø³ Ú©Ø±Ø¯' if accepted else 'ØªØ£ÛÛØ¯ ÙØ´Ø¯'}")
        report.append("")

    if quality:
        report += ["## Ú©ÛÙÛØª Ø¯Ø§Ø¯Ù"]
        for symbol, q in quality.items():
            report.append(f"- {symbol}: candles={q.get('n')} coverage={q.get('coverage', 0):.1%} missing={q.get('missing')}")
    if notes:
        report += ["", "## ÙØ´Ø¯Ø§Ø±ÙØ§"] + [f"- {x}" for x in notes]
    report += ["", "Ø§ÛÙ Ú¯Ø²Ø§Ø±Ø´ ØªØ­ÙÛÙØ§ØªÛ Ø§Ø³ØªØ ÙÛÚ Ø³ÛÚ¯ÙØ§Ù Ø²ÙØ¯ÙâØ§Û ÙØ¹Ø§Ù ÙØ´Ø¯Ù Ù ÙÛÚ Ø³ÙØ§Ø±Ø´Û Ø«Ø¨Øª ÙÙÛâØ´ÙØ¯."]
    full = "\n".join(report)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "report_ema_fa.md").write_text(full, encoding="utf-8")

    short = ["ð§ª Ø¢Ø²ÙØ§ÛØ´ ÙØ³ØªÙÙ EMA 9/20 â XT Futures (1m)"]
    for variant in VARIANTS:
        rows = sorted(all_rows[variant], key=lambda x: x["signal_time"])
        if not rows:
            short.append(f"{variant}: Ø¨Ø¯ÙÙ ÙØ¹Ø§ÙÙÙ Ø­ÙâØ´Ø¯Ù")
            continue
        times = np.array([r["signal_time"] for r in rows], dtype=np.int64)
        c1 = times.min() + int((times.max() - times.min()) * 0.60)
        c2 = times.min() + int((times.max() - times.min()) * 0.80)
        test_rows = [r for r in rows if r["signal_time"] >= c2]
        short.append(f"{variant} | Ø¢Ø²ÙÙÙ: {_fmt(_metrics(test_rows))}")
    short.append("Ú¯Ø²Ø§Ø±Ø´ Ú©Ø§ÙÙ: data/report_ema_fa.md | Ø­Ø§ÙØª ØªØ­ÙÛÙØ§ØªÛØ Ø³ÛÚ¯ÙØ§Ù Ø²ÙØ¯Ù Ø®Ø§ÙÙØ´.")
    return full, "\n".join(short)
