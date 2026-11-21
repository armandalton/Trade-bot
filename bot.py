"""XT SMA64 + EMA 9/20 research bot. Usage: python bot.py check | train | scan | live | image-bot | selftest"""
import sys
import time
import unittest

from xt_bot.config import CFG
from xt_bot.util import log


def check():
    from xt_bot.data import fetch_klines, validate_candles, closed_only
    from xt_bot.notifier import send, configured
    now = int(time.time() * 1000)
    ok_all = True
    for sym in CFG["symbols"]:
        df = fetch_klines(sym, now - 1400 * 60_000, now, "1m")
        q = validate_candles(closed_only(df, now))
        age = (now - int(df["time"].iloc[-1])) / 60000
        log(f"{sym}: candles={q['n']} missing={q['missing']} sorted={q.get('sorted')} last_candle_age_min={age:.1f} coverage={q['coverage']:.3f}")
        ok_all &= bool(q["ok"]) and age < 10
    if not configured():
        log("Telegram secrets are missing (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        raise SystemExit(1)
    sent = send("✅ تست اتصال بات XT: داده یک‌دقیقه‌ای XT " + ("سالم دریافت شد." if ok_all else "دریافت شد ولی کیفیت داده مشکل دارد؛ لاگ را ببینید."))
    if not sent or not ok_all:
        raise SystemExit(1)
    log("Check complete.")


def train():
    from xt_bot.train import run_training
    from xt_bot.ema_strategy import run_ema_experiment
    from xt_bot.notifier import send
    from xt_bot.util import read_json
    from xt_bot.config import DATA_DIR

    state = read_json(DATA_DIR / "state.json", {})
    meta, results, text = run_training(state=state)
    log(f"SMA64 training finished. approved={meta['approved']} reasons={meta['reasons']}")
    if not send(text):
        log("SMA64 report was not delivered to Telegram (see data/report_fa.md).")

    log("Starting independent EMA 9/20 experiment ...")
    full, ema_text = run_ema_experiment()
    log("EMA 9/20 experiment completed (see data/report_ema_fa.md).")
    if not send(ema_text):
        log("EMA report was not delivered to Telegram (see data/report_ema_fa.md).")


def selftest():
    suite = unittest.defaultTestLoader.discover("tests")
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if res.wasSuccessful() else 1)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "check"
    if mode == "check":
        check()
    elif mode == "train":
        train()
    elif mode == "scan":
        from xt_bot.live import scan
        log(f"scan result: {scan()}")
    elif mode == "live":
        from xt_bot.live import live_loop
        live_loop()
    elif mode == "image-bot":
        from xt_bot.telegram_image import poll_once
        poll_once()
    elif mode == "selftest":
        selftest()
    else:
        raise SystemExit("Usage: python bot.py check|train|scan|live|image-bot|selftest")
