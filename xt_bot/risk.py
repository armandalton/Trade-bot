"""Risk rules shared by the backtest and the live paper-trading engine."""


def size_position(equity, entry, distance, cfg):
    """Risk-based sizing: qty = risk$ / stop distance, capped by max leverage. Returns (qty, risk_usd, scale)."""
    risk_usd = equity * cfg["risk_pct"]
    qty = risk_usd / distance
    cap = equity * cfg["max_leverage"] / entry
    scale = 1.0 if qty <= cap else cap / qty
    return qty * scale, risk_usd * scale, scale


def global_lock(equity, peak, day_start, cfg):
    """Return 'drawdown' / 'daily_loss' when new signals must stop, else None."""
    cap = cfg["capital"]
    if peak - equity >= cfg["max_drawdown_pct"] * cap:
        return "drawdown"
    if day_start - equity >= cfg["daily_loss_pct"] * cap:
        return "daily_loss"
    return None


def check_limits(equity, peak, day_start, open_symbols, symbol, cfg):
    """None if a new position is allowed, otherwise the reason string."""
    lock = global_lock(equity, peak, day_start, cfg)
    if lock:
        return lock
    if len(open_symbols) >= cfg["max_open"]:
        return "max_open"
    if symbol in open_symbols:
        return "symbol_open"
    return None
