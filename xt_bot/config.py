"""Central configuration. Every value can be overridden with an environment variable.
No secrets live here: the Telegram token/chat id are read from environment variables only."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

INTERVAL = "1m"
INTERVAL_MS = 60_000


def _num(name, default, cast=float):
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return cast(default)
    return cast(raw)


CFG = {
    "symbols": [s.strip().lower() for s in (os.getenv("SYMBOLS") or "btc_usdt,eth_usdt,sol_usdt,bnb_usdt,xrp_usdt").split(",") if s.strip()],
    # strategy
    "sma_period": 64,
    "atr_period": 14,
    "horizon": _num("HORIZON", 60, int),            # max candles a setup may stay open
    "rr": 2.0,                                       # first target (full exit in paper/backtest)
    "tp2_rr": 3.0,                                   # informational second target
    "stop_buffer_atr": 0.1,
    "min_stop_atr": _num("MIN_STOP_ATR", 0.8),
    "max_cost_r": _num("MAX_COST_R", 0.35),          # skip setups whose round-trip cost exceeds this many R
    # costs (verify against your XT fee tier!)
    "fee_rate": _num("FEE_RATE", 0.0006),            # per side
    "slippage_rate": _num("SLIPPAGE_RATE", 0.0003),  # per side
    # capital / risk
    "capital": _num("PAPER_CAPITAL", 2000),
    "risk_pct": _num("RISK_PCT", 0.005),
    "daily_loss_pct": _num("DAILY_LOSS_PCT", 0.015),
    "max_drawdown_pct": _num("MAX_DRAWDOWN_PCT", 0.06),
    "max_open": _num("MAX_OPEN", 3, int),
    "max_leverage": _num("MAX_LEVERAGE", 10),
    # training
    "train_days": _num("TRAIN_DAYS", 183, int),
    "split": (0.6, 0.2, 0.2),
    "min_train_setups": 300,
    "min_val_trades": 30,
    # acceptance gates (on the untouched test segment)
    "min_test_trades": 100,
    "min_win_rate": 0.50,
    "min_profit_factor": 1.4,
    "max_test_drawdown": 0.06,
    # live
    "live_candles": 400,
    "max_signal_age_min": _num("MAX_SIGNAL_AGE_MIN", 20, int),
    "max_drift_r": 0.5,
}
