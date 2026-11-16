"""Live monitoring: closed 1m candles only, model-gated signals, risk limits, dedupe, paper tracking."""
import json
import time

import numpy as np
import pandas as pd

from .config import CFG, DATA_DIR, INTERVAL_MS
from .data import fetch_klines, closed_only
from .features import build_events
from .model import load_bundle, predict_kinds, select_best
from .notifier import send
from .paper import new_state, roll_day, update_positions
from .util import log, read_json, write_json, utc_iso
from . import risk

STATE_FILE = DATA_DIR / "state.json"
META_FILE = DATA_DIR / "model_meta.json"
SIGNALS_FILE = DATA_DIR / "signals.jsonl"


def describe(row):
    """Human-readable (Persian) reasons + market state, built from the observable features."""
    brk = row["kind"] == "BREAKOUT"
    up_approach = row["approach_side"] == "LONG"   # LONG approach = price came down from above
    if brk:
        head = ("قیمت از بالا به SMA64 رسید و با کندل بسته‌شده زیر آن عبور کرد" if up_approach
                else "قیمت از پایین به SMA64 رسید و با کندل بسته‌شده بالای آن عبور کرد")
    else:
        head = ("قیمت از بالا به SMA64 رسید و کندل بعدی به‌سمت بالا برگشت" if up_approach
                else "قیمت از پایین به SMA64 رسید و کندل بعدی به‌سمت پایین برگشت")
    k = -1.0 if brk else 1.0
    sl = k * row["sma_slope_atr"]
    parts = [head, "شیب SMA همسو با معامله" if sl > 0.1 else ("شیب SMA خلاف معامله" if sl < -0.1 else "SMA تقریباً تخت"),
             f"کندل واکنشی {'قوی' if row['r_body_ratio'] >= 0.6 else 'متوسط'} (بدنه {row['r_body_ratio']*100:.0f}% دامنه)",
             f"حجم کندل واکنش {row['rel_vol_r']:.1f}× میانگین ۲۰ کندل"]
    trend = k * row["short_trend"]
    market = (f"نوسان ATR≈{row['atr_bps']:.1f}bp ({row['vol_regime']:.2f}× میانگین اخیر)؛ "
              f"روند کوتاه‌مدت {'همسو' if trend > 0.5 else ('خلاف' if trend < -0.5 else 'خنثی')} با معامله؛ "
              f"تعداد عبور از SMA در ۳۰ کندل: {row['cross_count_30']:.0f}")
    return "؛ ".join(parts), market


def format_signal(row, prob, calibrated, qty, risk_usd, now_ms):
    sym = row["symbol"].upper().replace("_", "/")
    reason, market = describe(row)
    prob_txt = (f"{prob*100:.0f}% (کالیبره‌شده روی داده اعتبارسنجی؛ تخمین مدل است نه Win Rate قطعی)" if calibrated
                else f"امتیاز مدل {prob:.2f} (کالیبره نشده؛ احتمال قطعی نیست)")
    name_en, name_fa = ("Reversal", "برگشت از SMA") if row["kind"] == "REVERSAL" else ("Breakout", "شکست SMA")
    rr_mult = abs(row["tp1"] - row["entry"]) / row["distance"]
    delay = max(0, (now_ms - int(row["signal_time"])) / 60000)
    return (f"📡 سیگنال آزمایشی (Paper) — سفارشی ثبت نشده\n"
            f"نماد: {sym}\nجهت: {row['side_name']}\nاستراتژی: SMA 64 {name_en} (1m) — {name_fa}\n"
            f"قیمت ورود: {row['entry']:.8g}\nحد ضرر: {row['sl']:.8g}\n"
            f"هدف اول ({rr_mult:g}R): {row['tp1']:.8g} — خروج کامل در Paper\n"
            f"هدف دوم ({rr_mult*1.5:g}R): {row['tp2']:.8g} — فقط اطلاعاتی\n"
            f"نسبت ریسک به بازده: 1:{rr_mult:g}\n"
            f"احتمال تخمینی موفقیت مدل: {prob_txt}\n"
            f"ریسک Paper: ${risk_usd:.2f} | حجم تقریبی: {qty:.6g}\n"
            f"دلیل ورود: {reason}\nوضعیت بازار: {market}\n"
            f"زمان سیگنال: {utc_iso(int(row['signal_time']))} (تأخیر ارسال ≈ {delay:.0f} دقیقه)\n"
            f"⚠️ قبل از هر اقدام، قیمت لحظه‌ای را با قیمت ورود مقایسه کنید. توصیه مالی نیست.")


def scan(now_ms=None, fetch=None, notify=None, cfg=CFG, state_file=STATE_FILE, meta_file=META_FILE,
         signals_file=SIGNALS_FILE, bundle=None):
    fetch = fetch or fetch_klines
    notify = notify or send
    meta = read_json(meta_file, {})
    if not meta.get("approved"):
        log("No approved model (run train; if rejected, no signals are sent by design).")
        return {"status": "no_approved_model", "sent": 0}
    cfg = {**cfg, **(meta.get("params") or {})}  # use exactly the parameters the model was trained with
    bundle = bundle or load_bundle()
    state = read_json(state_file, None) or new_state(cfg)
    for k, v in new_state(cfg).items():
        state.setdefault(k, v)
    now_ms = int(now_ms or time.time() * 1000)
    roll_day(state, now_ms)

    closed = {}
    for sym in cfg["symbols"]:
        try:
            df = fetch(sym, now_ms - (cfg["live_candles"] + 2) * INTERVAL_MS, now_ms, "1m")
            closed[sym] = closed_only(df, now_ms)
        except Exception as e:
            log(f"{sym}: candle fetch failed: {type(e).__name__}: {e}")

    update_positions(state, closed, cfg, notify)

    equity, peak = float(state["paper_equity"]), float(state["peak_equity"])
    lock = risk.global_lock(equity, peak, float(state["day_start"]), cfg)
    sent = 0
    if lock:
        tag = f"{lock}:{state['day']}"
        if state.get("lock_notified") != tag:
            names = {"daily_loss": "سقف ضرر روزانه ۱.۵٪ نقض شد", "drawdown": "سقف Drawdown ۶٪ نقض شد"}
            if notify(f"⛔ ارسال سیگنال جدید متوقف شد: {names[lock]}.\nموجودی Paper: ${equity:.2f} | اوج: ${peak:.2f}"):
                state["lock_notified"] = tag
        log(f"Risk lock active: {lock}")
        write_json(state_file, state)
        return {"status": f"locked_{lock}", "sent": 0}

    enabled = set(meta.get("enabled_symbols") or [])
    kinds_on = set(meta.get("enabled_kinds") or [])
    thrs = meta.get("thresholds") or {}
    age_ms = cfg["max_signal_age_min"] * 60_000
    for sym, df in closed.items():
        if sym not in enabled or len(df) < 200:
            continue
        try:
            ev = build_events(df, sym, cfg)
            ev = ev[ev["tradable"] & ev["kind"].isin(kinds_on) & (now_ms - ev["signal_time"] <= age_ms)].reset_index(drop=True)
            if ev.empty:
                continue
            ev = ev.assign(prob=predict_kinds(bundle, ev))
            ev = select_best(ev, {k: float(v) for k, v in thrs.items()})
            if meta.get("hour_filter") and len(ev):
                ev = ev[((ev["signal_time"] // 3_600_000) % 24).isin(meta.get("trade_hours_utc") or [])].reset_index(drop=True)
            probs = ev["prob"].values
            if ev.empty:
                continue
            last_close = float(df["close"].iloc[-1])
            for i, row in ev.iterrows():
                key = f"{sym}:{int(row['time'])}:{row['side_name']}:{row['kind']}"
                if key in state["sent"]:
                    continue
                open_syms = [p["symbol"] for p in state["open_positions"]]
                why = risk.check_limits(float(state["paper_equity"]), float(state["peak_equity"]),
                                        float(state["day_start"]), open_syms, sym, cfg)
                if why:
                    log(f"{key}: blocked by risk rule '{why}'")
                    continue
                move = (last_close - row["entry"]) / row["distance"] * (1 if row["side_name"] == "LONG" else -1)
                if abs(move) > cfg["max_drift_r"]:
                    log(f"{key}: expired (price moved {move:+.2f}R since signal)")
                    continue
                qty, risk_usd, scale = risk.size_position(float(state["paper_equity"]), row["entry"], row["distance"], cfg)
                msg = format_signal(row, float(probs[i]), bundle[row["kind"]]["calibrator"] is not None, qty, risk_usd, now_ms)
                if not notify(msg):
                    log(f"{key}: Telegram not delivered; not recorded as sent")
                    continue
                state["sent"][key] = now_ms
                state["open_positions"].append({
                    "symbol": sym, "side": row["side_name"], "entry": float(row["entry"]), "sl": float(row["sl"]),
                    "tp1": float(row["tp1"]), "tp2": float(row["tp2"]), "distance": float(row["distance"]),
                    "qty": qty, "risk_usd": risk_usd, "entry_time": int(row["signal_time"]), "key": key,
                    "prob": float(probs[i]), "kind": row["kind"]})
                with open(signals_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "prob": float(probs[i]), "entry": float(row["entry"]),
                                        "sl": float(row["sl"]), "tp1": float(row["tp1"]), "ts": now_ms}) + "\n")
                sent += 1
        except Exception as e:
            log(f"{sym}: scan failed: {type(e).__name__}: {e}")
    if len(state["sent"]) > 2000:
        state["sent"] = dict(sorted(state["sent"].items(), key=lambda kv: kv[1])[-1000:])
    write_json(state_file, state)
    return {"status": "ok", "sent": sent}


def live_loop(max_minutes=None):
    """Poll every minute (a few seconds after each candle closes). For long-running free hosts / manual Actions run."""
    import os
    max_minutes = max_minutes or int(os.getenv("LIVE_MAX_MINUTES") or 320)
    end = time.time() + max_minutes * 60
    while time.time() < end:
        try:
            scan()
        except Exception as e:
            log(f"scan error: {type(e).__name__}: {e}")
        nxt = (int(time.time() // 60) + 1) * 60 + 4
        time.sleep(max(1, min(nxt - time.time(), end - time.time())))
