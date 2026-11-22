
"""Telegram webhook processing. No AI APIs."""
import json
import os
import time
from datetime import datetime, timezone

import requests

API = "https://api.telegram.org"
MAX_IMAGE_BYTES = 10 * 1024 * 1024


def telegram(method, payload=None, timeout=30):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN تنظیم نشده است.")

    r = requests.post(
        f"{API}/bot{token}/{method}",
        json=payload or {},
        timeout=timeout
    )
    r.raise_for_status()
    data = r.json()

    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed")

    return data.get("result")


def send(chat_id, message):
    telegram("sendMessage", {
        "chat_id": chat_id,
        "text": str(message)[:3900]
    })


def download_image(file_id):
    info = telegram("getFile", {"file_id": file_id})
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    r = requests.get(
        f"{API}/file/bot{token}/{info['file_path']}",
        timeout=35
    )
    r.raise_for_status()

    if len(r.content) > MAX_IMAGE_BYTES:
        raise ValueError("حجم تصویر بیشتر از ۱۰ مگابایت است.")

    return r.content


def market_command(chat_id, command):
    from xt_bot.data import (
        fetch_klines,
        closed_only,
        fetch_last_price
    )
    from xt_bot.technical_engine import (
        analyze_ohlcv,
        format_signal,
        INTERVALS
    )

    bits = command.strip().split()

    symbol = (
        bits[1] if len(bits) > 1 else "btc_usdt"
    ).lower().replace("/", "_")

    timeframe = (
        bits[2] if len(bits) > 2 else "1m"
    ).lower()

    if timeframe not in INTERVALS:
        send(
            chat_id,
            "تایم‌فریم معتبر: " + ", ".join(INTERVALS)
        )
        return

    step = INTERVALS[timeframe]

    # دریافت کندل‌های اخیر
    request_time = int(time.time() * 1000)

    df = fetch_klines(
        symbol,
        request_time - step * 320,
        request_time,
        timeframe
    )

    # فقط کندل‌های کاملاً بسته‌شده
    df = closed_only(df, request_time, step)

    if len(df) < 230:
        send(
            chat_id,
            f"⛔ تعداد کندل کافی نیست: {len(df)}"
        )
        return

    # بررسی تازگی آخرین کندل بسته‌شده
    last_candle = df.iloc[-1]
    candle_open_ms = int(last_candle["time"])
    candle_close_ms = candle_open_ms + step

    now_ms = int(time.time() * 1000)
    candle_delay_ms = now_ms - candle_close_ms

    if candle_delay_ms < 0 or candle_delay_ms > step * 2:
        send(
            chat_id,
            "⛔ داده کندل به‌روز نیست.\n"
            f"تأخیر نسبت به بسته‌شدن کندل: "
            f"{candle_delay_ms / 1000:.1f} ثانیه\n"
            "برای جلوگیری از سیگنال اشتباه، تحلیل متوقف شد."
        )
        return

    # دریافت قیمت آخرین معامله از تیکر XT
    ticker = fetch_last_price(symbol)

    current_price = float(ticker["price"])
    ticker_received_ms = int(ticker["received_at_ms"])

    # تحلیل تکنیکال همچنان بر اساس کندل بسته‌شده انجام می‌شود
    result = analyze_ohlcv(df, symbol, timeframe)

    candle_close_price = float(last_candle["close"])

    difference_pct = (
        (current_price - candle_close_price)
        / candle_close_price
    ) * 100

    candle_time = datetime.fromtimestamp(
        candle_open_ms / 1000,
        tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    ticker_time = datetime.fromtimestamp(
        ticker_received_ms / 1000,
        tz=timezone.utc
    ).strftime("%H:%M:%S UTC")

    message = (
        format_signal(result)
        + "\n\n"
        + "━━━━━━━━━━━━━━\n"
        + "🕒 وضعیت داده بازار\n"
        + f"آخرین کندل بسته‌شده: {candle_time}\n"
        + f"قیمت بسته‌شدن کندل: {candle_close_price}\n"
        + f"قیمت آخرین معامله XT: {current_price}\n"
        + f"اختلاف قیمت فعلی با کندل: {difference_pct:+.3f}%\n"
        + f"تأخیر کندل: {candle_delay_ms / 1000:.1f} ثانیه\n"
        + f"زمان دریافت تیکر توسط ربات: {ticker_time}\n"
        + "━━━━━━━━━━━━━━\n"
        + "⚠️ قیمت آخرین معامله، قیمت لحظه‌ای در زمان دریافت پاسخ API است؛ "
        + "قیمت اجرای سفارش را تضمین نمی‌کند."
    )

    send(chat_id, message)


def handle_message(message):
    allowed = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not allowed:
        raise RuntimeError("TELEGRAM_CHAT_ID تنظیم نشده است.")

    chat_id = str((message.get("chat") or {}).get("id", ""))

    if chat_id != allowed:
        print("Unauthorized chat ignored.")
        return

    txt = (message.get("text") or "").strip()

    try:
        if txt.startswith("/start"):
            send(
                chat_id,
                "سلام! ربات آماده است.\n"
                "تحلیل بازار:\n"
                "/market BTC_USDT 1m\n"
                "یا عکس چارت را ارسال کن.\n"
                "بدون Gemini و بدون اجرای سفارش واقعی."
            )

        elif txt.startswith("/market"):
            market_command(chat_id, txt)

        elif message.get("photo") or (
            message.get("document") or {}
        ).get("mime_type", "").startswith("image/"):

            photos = message.get("photo") or []
            document = message.get("document") or {}

            file_id = (
                max(
                    photos,
                    key=lambda p: p.get("file_size", 0)
                ).get("file_id")
                if photos else document.get("file_id")
            )

            if not file_id:
                send(chat_id, "فایل تصویر معتبر پیدا نشد.")
            else:
                from xt_bot.image_engine import analyze_chart_image

                result = analyze_chart_image(
                    download_image(file_id),
                    message.get("caption") or ""
                )
                send(chat_id, result)

        elif txt:
            send(
                chat_id,
                "برای تحلیل بازار:\n"
                "/market BTC_USDT 1m\n"
                "یا عکس چارت را ارسال کن."
            )

    except Exception as exc:
        send(
            chat_id,
            f"خطا: {type(exc).__name__}: {str(exc)[:300]}"
        )


def process_webhook_update():
    raw = os.getenv("TELEGRAM_UPDATE_JSON", "")

    if not raw:
        raise RuntimeError("TELEGRAM_UPDATE_JSON خالی است.")

    update = json.loads(raw)
    message = update.get("message") or update.get("edited_message")

    if message:
        handle_message(message)
    else:
        print("Unsupported update ignored.")
