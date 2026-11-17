"""Historical outcome labelling.

Future candles are used here and ONLY here (to create labels).

For each tradable event, scan candles in chronological order from idx + 1
through idx + horizon. The first decisive event wins:

* A data gap before a candle makes the outcome unknowable: label ``gap``.
* TP touched before SL: ``valid_reversal``.
* SL touched before TP: ``true_break`` or ``false_break`` according to
  sustained closes beyond the SMA.
* Neither touched within the horizon: ``weak_reversal``.

If TP and SL are both touched in the same 1m candle, their order is unknown.
The result is conservatively treated as an SL and flagged ``ambiguous``.

Important: a gap that occurs AFTER TP/SL has already been reached must not
invalidate the trade. This implementation checks gaps candle-by-candle.
"""

import numpy as np
import pandas as pd

from .config import CFG, INTERVAL_MS


RESOLVED = ("valid_reversal", "weak_reversal", "true_break", "false_break")


def trade_net_r(side, entry, sl, tp, result, exit_close, distance, cfg=CFG):
    """Net result in R after fees and slippage."""
    direction = 1.0 if side == "LONG" else -1.0
    slippage = cfg["slippage_rate"]
    fee = cfg["fee_rate"]

    entry_eff = entry * (1 + direction * slippage)
    if result == "tp":
        exit_eff = tp
    else:
        raw_exit = sl if result == "sl" else exit_close
        exit_eff = raw_exit * (1 - direction * slippage)

    gross = (exit_eff - entry_eff) * direction
    fees = fee * (entry_eff + exit_eff)
    return (gross - fees) / distance


def _sustained(wrong, k=3):
    """Return True if at least k consecutive closes are beyond the SMA."""
    run = 0
    for is_wrong in wrong:
        run = run + 1 if is_wrong else 0
        if run >= k:
            return True
    return False


def _empty_values(label, result="none"):
    """Values for events without a resolved trade outcome."""
    return (label, result, -1, 0, np.nan, False, np.nan, np.nan)


def label_events(x, ev, cfg=CFG):
    """Attach future outcome labels and net-R results to event rows."""
    horizon = int(cfg["horizon"])
    n = len(x)

    high = x["high"].to_numpy()
    low = x["low"].to_numpy()
    close = x["close"].to_numpy()
    sma = x["sma"].to_numpy()
    times = x["time"].to_numpy()
    gap_bad = x["gap_bad"].to_numpy()

    output_columns = (
        "label",
        "result",
        "exit_idx",
        "exit_time",
        "net_r",
        "ambiguous",
        "mfe_r",
        "mae_r",
    )
    columns = {name: [] for name in output_columns}

    for row in range(len(ev)):
        idx = int(ev["idx"].iat[row])

        if not bool(ev["tradable"].iat[row]):
            values = _empty_values("no_setup")
        elif idx + horizon > n - 1:
            values = _empty_values("pending")
        else:
            side = ev["side_name"].iat[row]
            entry = float(ev["entry"].iat[row])
            stop = float(ev["sl"].iat[row])
            target = float(ev["tp1"].iat[row])
            distance = float(ev["distance"].iat[row])

            result = None
            exit_idx = -1
            ambiguous = False

            # Scan one candle at a time. A gap only invalidates the setup if
            # it occurs before the first TP/SL event. Later gaps are irrelevant.
            for candle_idx in range(idx + 1, idx + horizon + 1):
                if bool(gap_bad[candle_idx]):
                    result = "gap"
                    break

                if side == "LONG":
                    stop_hit = low[candle_idx] <= stop
                    target_hit = high[candle_idx] >= target
                else:
                    stop_hit = high[candle_idx] >= stop
                    target_hit = low[candle_idx] <= target

                if stop_hit and target_hit:
                    # Intrabar order is unknowable on 1m OHLC data.
                    # Conservative convention: assume stop was hit first.
                    result = "sl"
                    exit_idx = candle_idx
                    ambiguous = True
                    break
                if stop_hit:
                    result = "sl"
                    exit_idx = candle_idx
                    break
                if target_hit:
                    result = "tp"
                    exit_idx = candle_idx
                    break

            if result == "gap":
                values = _empty_values("gap")
            else:
                if result is None:
                    result = "horizon"
                    exit_idx = idx + horizon

                segment_high = high[idx + 1 : exit_idx + 1]
                segment_low = low[idx + 1 : exit_idx + 1]
                segment_close = close[idx + 1 : exit_idx + 1]
                segment_sma = sma[idx + 1 : exit_idx + 1]

                if side == "LONG":
                    mfe = (float(np.max(segment_high)) - entry) / distance
                    mae = (entry - float(np.min(segment_low))) / distance
                    wrong_side = segment_close < segment_sma
                else:
                    mfe = (entry - float(np.min(segment_low))) / distance
                    mae = (float(np.max(segment_high)) - entry) / distance
                    wrong_side = segment_close > segment_sma

                if result == "tp":
                    label = "valid_reversal"
                elif result == "horizon":
                    label = "weak_reversal"
                else:
                    label = "true_break" if _sustained(wrong_side) else "false_break"

                net_r = trade_net_r(
                    side=side,
                    entry=entry,
                    sl=stop,
                    tp=target,
                    result=result,
                    exit_close=float(close[exit_idx]),
                    distance=distance,
                    cfg=cfg,
                )
                values = (
                    label,
                    result,
                    exit_idx,
                    int(times[exit_idx]) + INTERVAL_MS,
                    net_r,
                    ambiguous,
                    mfe,
                    mae,
                )

        for name, value in zip(output_columns, values):
            columns[name].append(value)

    out = ev.copy()
    for name, values in columns.items():
        out[name] = values
    return out
