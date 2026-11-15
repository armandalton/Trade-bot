"""Paper-trading engine: opens simulated positions from signals and resolves them on closed 1m candles."""
from datetime import datetime, timezone

from .config import INTERVAL_MS
from .labeling import trade_net_r
from .util import utc_iso


def new_state(cfg):
    return {"sent": {}, "open_positions": [], "closed_trades": [], "paper_equity": cfg["capital"],
            "peak_equity": cfg["capital"], "day": None, "day_start": cfg["capital"], "lock_notified": ""}


def roll_day(state, now_ms):
    today = datetime.fromtimestamp(now_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    if state.get("day") != today:
        state["day"], state["day_start"] = today, float(state["paper_equity"])


def update_positions(state, closed_by_symbol, cfg, notify):
    """Resolve open paper positions with candles that closed after the entry. SL wins if SL and TP share a candle."""
    remaining = []
    for pos in state["open_positions"]:
        df = closed_by_symbol.get(pos["symbol"])
        if df is None:
            remaining.append(pos)
            continue
        sub = df[df["time"] >= pos["entry_time"]].reset_index(drop=True)
        result, exit_close, n_seen = None, None, 0
        for row in sub.itertuples(index=False):
            n_seen += 1
            if pos["side"] == "LONG":
                hit_sl, hit_tp = row.low <= pos["sl"], row.high >= pos["tp1"]
            else:
                hit_sl, hit_tp = row.high >= pos["sl"], row.low <= pos["tp1"]
            if hit_sl or hit_tp:
                result = "sl" if hit_sl else "tp"
                exit_close = row.close
                break
            if n_seen >= cfg["horizon"]:
                result, exit_close = "horizon", row.close
                break
        if result is None:
            remaining.append(pos)
            continue
        net_r = trade_net_r(pos["side"], pos["entry"], pos["sl"], pos["tp1"], result, float(exit_close), pos["distance"], cfg)
        pnl = net_r * pos["risk_usd"]
        state["paper_equity"] = float(state["paper_equity"]) + pnl
        state["peak_equity"] = max(float(state["peak_equity"]), float(state["paper_equity"]))
        state["closed_trades"].append({"symbol": pos["symbol"], "side_name": pos["side"], "entry": pos["entry"],
                                       "result": result, "net_r": net_r, "pnl": pnl,
                                       "exit_time": int(sub["time"].iloc[n_seen - 1]) + INTERVAL_MS})
        state["closed_trades"] = state["closed_trades"][-1000:]
        names = {"tp": "هدف اول خورد ✅", "sl": "حد ضرر خورد ❌", "horizon": "خروج زمانی (افق تمام شد) ⏱"}
        notify(f"📒 بستن معامله Paper — {pos['symbol'].upper().replace('_', '/')} {pos['side']}\n"
               f"نتیجه: {names[result]}\nسود/زیان: {net_r:+.2f}R (${pnl:+.2f}) پس از کارمزد\n"
               f"موجودی Paper: ${state['paper_equity']:.2f}")
    state["open_positions"] = remaining
