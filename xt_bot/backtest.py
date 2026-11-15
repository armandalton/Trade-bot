"""Portfolio-level simulation with the same risk rules as live trading, and metrics."""
import numpy as np
import pandas as pd

from .config import CFG
from . import risk

MS_DAY = 86_400_000


def summarize(executed, initial):
    """executed: DataFrame with columns pnl, side_name, exit_time (any order)."""
    empty = {"trades": 0, "long": 0, "short": 0, "wins": 0, "losses": 0, "win_rate": None,
             "net_pnl": 0.0, "profit_factor": None, "max_dd_pct": 0.0, "avg_win": None, "avg_loss": None}
    if executed is None or len(executed) == 0:
        return empty
    e = executed.sort_values("exit_time")
    pnl = e["pnl"].values.astype(float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    eq = initial + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.r_[initial, eq])[1:]
    dd = float(((peak - eq) / peak).max())
    gl = abs(losses.sum())
    return {"trades": int(len(pnl)), "long": int((e["side_name"] == "LONG").sum()),
            "short": int((e["side_name"] == "SHORT").sum()), "wins": int(len(wins)), "losses": int(len(losses)),
            "win_rate": float(len(wins) / len(pnl)), "net_pnl": float(pnl.sum()),
            "profit_factor": float(wins.sum() / gl) if gl > 0 else (None if wins.sum() == 0 else 999.0),
            "max_dd_pct": dd,
            "avg_win": float(wins.mean()) if len(wins) else None,
            "avg_loss": float(losses.mean()) if len(losses) else None}


def simulate(trades, cfg=CFG):
    """trades: resolved setups (columns entry_time, exit_time, symbol, side_name, entry, distance, net_r).
    Applies: risk-based sizing, max concurrent positions, one position per symbol, daily loss lock,
    drawdown lock (permanent inside one simulation). Equity is realized-P&L equity."""
    initial = cfg["capital"]
    equity = peak = day_start = initial
    day = None
    open_, done, skipped = [], [], {}

    def realize(upto):
        nonlocal equity, peak
        for p in sorted([p for p in open_ if p["exit_time"] <= upto], key=lambda z: z["exit_time"]):
            open_.remove(p)
            equity += p["pnl"]
            peak = max(peak, equity)
            done.append(p)

    rows = trades.sort_values("entry_time") if len(trades) else trades
    halted = False
    for r in rows.itertuples(index=False):
        realize(r.entry_time)
        d = int(r.entry_time // MS_DAY)
        if d != day:
            day, day_start = d, equity
        if halted:
            skipped["drawdown"] = skipped.get("drawdown", 0) + 1
            continue
        reason = risk.check_limits(equity, peak, day_start, [p["symbol"] for p in open_], r.symbol, cfg)
        if reason:
            if reason == "drawdown":
                halted = True
            skipped[reason] = skipped.get(reason, 0) + 1
            continue
        qty, risk_usd, scale = risk.size_position(equity, r.entry, r.distance, cfg)
        open_.append({"symbol": r.symbol, "side_name": r.side_name, "entry_time": r.entry_time,
                      "exit_time": r.exit_time, "pnl": float(r.net_r) * risk_usd, "net_r": float(r.net_r),
                      "label": r.label, "risk_usd": risk_usd})
    realize(float("inf"))
    executed = pd.DataFrame(done)
    return executed, summarize(executed, initial), skipped
