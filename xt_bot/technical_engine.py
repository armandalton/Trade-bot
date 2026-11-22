"""Deterministic OHLCV technical analysis. No AI/model/API calls."""
from __future__ import annotations
import math
import pandas as pd
import numpy as np

INTERVALS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
    "4h": 14_400_000, "1d": 86_400_000,
}

def _indicators(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x["ema20"] = x.close.ewm(span=20, adjust=False).mean()
    x["ema50"] = x.close.ewm(span=50, adjust=False).mean()
    x["ema200"] = x.close.ewm(span=200, adjust=False).mean()
    delta = x.close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    x["rsi"] = 100 - 100 / (1 + rs)
    tr = pd.concat([(x.high-x.low), (x.high-x.close.shift()).abs(),
                    (x.low-x.close.shift()).abs()], axis=1).max(axis=1)
    x["atr"] = tr.ewm(alpha=1/14, adjust=False).mean()
    x["vol_sma20"] = x.volume.rolling(20).mean()
    return x

def _swings(x: pd.DataFrame, left=3, right=3):
    highs, lows = [], []
    h, l = x.high.to_numpy(), x.low.to_numpy()
    for i in range(left, len(x)-right):
        if h[i] >= np.max(h[i-left:i+right+1]):
            highs.append((i, float(h[i])))
        if l[i] <= np.min(l[i-left:i+right+1]):
            lows.append((i, float(l[i])))
    return highs, lows

def analyze_ohlcv(df: pd.DataFrame, symbol="UNKNOWN", timeframe="1m", rr=2.0):
    """Analyze only fully closed candles supplied by caller."""
    if df is None or len(df) < 230:
        return {"direction":"WAIT", "reason":"حداقل ۲۳۰ کندل بسته‌شده برای تحلیل مطمئن لازم است."}
    x = _indicators(df).dropna().reset_index(drop=True)
    if len(x) < 210:
        return {"direction":"WAIT", "reason":"داده کافی پس از محاسبه اندیکاتورها باقی نماند."}
    last = x.iloc[-1]
    prev = x.iloc[-2]
    highs, lows = _swings(x.tail(100).reset_index(drop=True))
    recent_highs = [v for _,v in highs[-4:]]
    recent_lows = [v for _,v in lows[-4:]]
    if not recent_highs or not recent_lows:
        return {"direction":"WAIT", "reason":"سوئینگ معتبر برای تعیین حد ضرر پیدا نشد."}
    swing_high, swing_low = recent_highs[-1], recent_lows[-1]
    trend_up = last.close > last.ema50 and last.ema20 > last.ema50
    trend_down = last.close < last.ema50 and last.ema20 < last.ema50
    ema_long = last.close > last.ema200
    ema_short = last.close < last.ema200
    rsi = float(last.rsi)
    atr = float(last.atr)
    if not math.isfinite(atr) or atr <= 0:
        return {"direction":"WAIT", "reason":"ATR نامعتبر است."}
    vol_ok = (not np.isfinite(last.vol_sma20)) or last.volume >= 0.8 * last.vol_sma20
    # Confirm break of the latest confirmed swing or trend-aligned pullback/reclaim.
    bos_up = last.close > swing_high and prev.close <= swing_high
    bos_down = last.close < swing_low and prev.close >= swing_low
    reclaim_up = prev.close < prev.ema20 and last.close > last.ema20 and trend_up
    reclaim_down = prev.close > prev.ema20 and last.close < last.ema20 and trend_down
    long_score = sum([trend_up, ema_long, 48 <= rsi <= 68, bos_up or reclaim_up, vol_ok])
    short_score = sum([trend_down, ema_short, 32 <= rsi <= 52, bos_down or reclaim_down, vol_ok])
    direction = "WAIT"
    if long_score >= 4 and long_score > short_score:
        direction = "LONG"
    elif short_score >= 4 and short_score > long_score:
        direction = "SHORT"
    entry = float(last.close)
    if direction == "LONG":
        sl = min(swing_low, entry - 0.8*atr)
        risk = entry-sl
        if risk <= 0: direction = "WAIT"
        else: tp1, tp2 = entry + rr*risk, entry + 3*risk
    elif direction == "SHORT":
        sl = max(swing_high, entry + 0.8*atr)
        risk = sl-entry
        if risk <= 0: direction = "WAIT"
        else: tp1, tp2 = entry - rr*risk, entry - 3*risk
    if direction == "WAIT":
        return {
            "symbol":symbol, "timeframe":timeframe, "direction":"WAIT",
            "close":entry, "rsi":rsi, "atr":atr,
            "reason":"شرایط هم‌جهت و تأییدهای کافی برای ورود هم‌زمان برقرار نیست؛ برای جلوگیری از سیگنال ضعیف صبر می‌کنیم.",
            "scores":f"LONG {long_score}/5 | SHORT {short_score}/5",
            "trend":"صعودی" if trend_up else ("نزولی" if trend_down else "خنثی"),
        }
    # Reject pathological stops; preserve 1:R target.
    stop_pct = risk / entry
    if stop_pct > 0.025 or stop_pct < 0.001:
        return {"symbol":symbol,"timeframe":timeframe,"direction":"WAIT",
                "reason":f"فاصله حد ضرر نامناسب است ({stop_pct:.2%})؛ ورود رد شد."}
    return {
        "symbol":symbol, "timeframe":timeframe, "direction":direction,
        "entry":entry, "sl":sl, "tp1":tp1, "tp2":tp2, "rr":rr,
        "rsi":rsi, "atr":atr, "trend":"صعودی" if direction=="LONG" else "نزولی",
        "scores":f"LONG {long_score}/5 | SHORT {short_score}/5",
        "reason":"روند و EMA هم‌جهت‌اند و ساختار شکست/بازپس‌گیری به‌همراه RSI و حجم بررسی شده است.",
        "candle_time":int(last.time) if "time" in x.columns else None,
    }

def format_signal(s):
    if s.get("direction") not in ("LONG","SHORT"):
        return (f"🔎 تحلیل XT | {s.get('symbol','')}\n"
                f"تایم‌فریم: {s.get('timeframe','')}\n"
                f"نتیجه: WAIT\n{ s.get('reason','') }\n"
                f"امتیازها: {s.get('scores','—')}\n"
                f"روند: {s.get('trend','نامشخص')}")
    f=lambda v: f"{v:.8g}"
    return (f"📊 سیگنال آزمایشی XT\nنماد: {s['symbol'].upper()}\n"
            f"تایم‌فریم: {s['timeframe']}\nجهت: {s['direction']}\n"
            f"ورود: {f(s['entry'])}\nحد ضرر: {f(s['sl'])}\n"
            f"TP1 (1:{s['rr']:g}): {f(s['tp1'])}\nTP2 (1:3): {f(s['tp2'])}\n"
            f"RSI: {s['rsi']:.1f} | ATR: {f(s['atr'])}\n"
            f"امتیازها: {s['scores']}\nدلیل: {s['reason']}\n"
            "⚠️ سیگنال پژوهشی/دمو است؛ سفارش واقعی اجرا نمی‌شود و سود تضمین نیست.")
