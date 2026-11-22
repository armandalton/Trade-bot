
"""Telegram command-based XT market analysis and chart image bot."""
import json
import os
import time
from datetime import datetime, timezone

import requests

API = "https://api.telegram.org"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

SYMBOLS = (
    "btc_usdt",
    "eth_usdt",
    "sol_usdt",
    "bnb_usdt",
    "xrp_usdt",
    "ada_usdt",
)

COMMANDS = {
    "/btc": "btc_usdt",
    "/eth": "eth_usdt",
    "/sol": "sol_usdt",
    "/bnb": "bnb_usdt",
    "/xrp": "xrp_usdt",
    "/ada": "ada_usdt",
}


def telegram(method, payload=None, timeout=30):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN تنظیم نشده است.")

    response = requests.post(
        f"{API}/bot{token}/{method}",
        json=payload or {},
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed")

    return data.get("result")


def send(chat_id, message):
    telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": str(message)[:3900],
        },
    )


def download_image(file_id):
    info = telegram("getFile", {"file_id": file_id})
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    response = requests.get(
        f"{API}/file/bot{token}/{info['file_path']}",
        timeout=35,
    )
    response.raise_for_status()

    if len(response.content) > MAX_IMAGE_BYTES:
        raise ValueError("حجم تصویر بیشتر از ۱۰ مگابایت است.")

    return response.content


def create_journal_candidate(result):
    """Save a signal candidate through the authenticated Cloudflare Worker."""
    base = os.getenv(
        "WORKER_URL",
        "https://still-dust-02af.arman8256.workers.dev",
    ).rstrip("/")

    secret = os.getenv("WORKER_API_SECRET", "").strip()
    if not secret:
        raise RuntimeError(
            "WORKER_API_SECRET در GitHub Actions تنظیم نشده است."
        )

    payload = {
        "symbol": result["symbol"],
        "timeframe": result["timeframe"],
        "direction": result["direction"],
        "entry": result["entry"],
        "stop_loss": result["sl"],
        "take_profit_1": result["tp1"],
        "take_profit_2": result["tp2"],
        "candle_time_ms": result.get("candle_time"),
    }

    response = requests.post(
        base + "/candidate",
        json=payload,
        headers={"Authorization": f"Bearer {secret}"},
        timeout=20,
    )
    response.raise_for_status()

    data = response.json()
    candidate_id = data.get("id")

    if not data.get("ok") or not candidate_id:
        raise RuntimeError("Worker شناسه ثبت سیگنال را برنگرداند.")

    return str(candidate_id)


def market_command(chat_id, symbol):
    from xt_bot.data import (
        fetch_klines,
        closed_only,
        fetch_last_price,
    )
    from xt_bot.technical_engine import (
        analyze_ohlcv,
        format_signal,
        INTERVALS,
    )

    symbol = symbol.strip().lower().replace("/", "_")

    if symbol not in SYMBOLS:
        send(chat_id, "⛔ نماد انتخاب‌شده معتبر نیست. دستور /info را ببین.")
        return

    timeframe = "1m"
    step = INTERVALS[timeframe]

    request_time = int(time.time() * 1000)

    df = fetch_klines(
        symbol,
        request_time - step * 320,
        request_time,
        timeframe,
    )
    df = closed_only(df, request_time, step)

    if len(df) < 230:
        send(chat_id, f"⛔ تعداد کندل کافی نیست: {len(df)}")
        return

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
            "برای جلوگیری از سیگنال اشتباه، تحلیل متوقف شد.",
        )
        return

    ticker = fetch_last_price(symbol)

    current_price = float(ticker["price"])
    ticker_received_ms = int(ticker["received_at_ms"])

    result = analyze_ohlcv(df, symbol, timeframe)

    candle_close_price = float(last_candle["close"])
    difference_pct = (
        (current_price - candle_close_price) / candle_close_price
    ) * 100

    candle_time = datetime.fromtimestamp(
        candle_open_ms / 1000,
        tz=timezone.utc,
    ).strftime("%Y-%m-%d %H:%M:%S UTC")

    ticker_time = datetime.fromtimestamp(
        ticker_received_ms / 1000,
        tz=timezone.utc,
    ).strftime("%H:%M:%S UTC")

    message = (
        format_signal(result)
        + "\n\n━━━━━━━━━━━━━━\n"
        + "🕒 وضعیت داده بازار\n"
        + f"آخرین کندل بسته‌شده: {candle_time}\n"
        + f"قیمت بسته‌شدن کندل: {candle_close_price}\n"
        + f"قیمت آخرین معامله XT: {current_price}\n"
        + f"اختلاف قیمت فعلی با کندل: {difference_pct:+.3f}%\n"
        + f"تأخیر کندل: {candle_delay_ms / 1000:.1f} ثانیه\n"
        + f"زمان دریافت تیکر توسط ربات: {ticker_time}\n"
        + "━━━━━━━━━━━━━━\n"
        + "⚠️ قیمت آخرین معامله، قیمت لحظه‌ای در زمان دریافت "
        + "پاسخ API است؛ قیمت اجرای سفارش را تضمین نمی‌کند."
    )

    if result.get("direction") in ("LONG", "SHORT"):
        try:
            candidate_id = create_journal_candidate(result)

            message += (
                "\n\n📒 برای ثبت این سیگنال در ژورنال، "
                "دستور زیر را ارسال کن:\n"
                f"/register {candidate_id}"
                "\n\n⏳ شناسه سیگنال تا ۱۵ دقیقه معتبر است."
            )

        except Exception as exc:
            print(
                f"Journal candidate error: "
                f"{type(exc).__name__}: {exc}"
            )
            message += (
                "\n\n⚠️ ذخیره موقت سیگنال در ژورنال "
                "فعلاً در دسترس نیست."
            )

    send(chat_id, message)


def get_info():
    return (
        "📘 راهنمای ربات تحلیل XT\n\n"
        "🔎 تحلیل نمادها — تایم‌فریم ۱ دقیقه:\n"
        "/btc — بیت‌کوین\n"
        "/eth — اتریوم\n"
        "/sol — سولانا\n"
        "/bnb — بایننس‌کوین\n"
        "/xrp — ریپل\n"
        "/ada — کاردانو\n\n"
        "📒 ژورنال معاملاتی:\n"
        "/jr — گزارش معاملات ثبت‌شده\n"
        "/register شناسه — ثبت سیگنال در ژورنال\n\n"
        "ℹ️ راهنما:\n"
        "/start — شروع ربات\n"
        "/info — نمایش همین راهنما\n\n"
        "📊 تمام تحلیل‌های نمادها بر اساس "
        "کندل یک‌دقیقه‌ای بسته‌شده انجام می‌شوند.\n\n"
        "⚠️ این ربات صرفاً پژوهشی است و "
        "سفارش واقعی در صرافی ثبت نمی‌کند."
    )


def handle_message(message):
    allowed = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not allowed:
        raise RuntimeError("TELEGRAM_CHAT_ID تنظیم نشده است.")

    chat_id = str((message.get("chat") or {}).get("id", ""))

    if chat_id != allowed:
        print("Unauthorized chat ignored.")
        return

    txt = (message.get("text") or "").strip()
    parts = txt.split()

    command = parts[0].split("@")[0].lower() if parts else ""

    try:
        if command in ("/start", "/info", "/help"):
            send(chat_id, get_info())

        elif command in COMMANDS:
            market_command(chat_id, COMMANDS[command])

        elif command == "/market":
            symbol = parts[1] if len(parts) > 1 else "btc_usdt"
            market_command(chat_id, symbol)

        elif command == "/jr":
            send(
                chat_id,
                "در حال آماده‌سازی گزارش ژورنال...\n"
                "اگر گزارش نمایش داده نشد، کمی بعد دوباره تلاش کن.",
            )

        elif command == "/register":
            send(
                chat_id,
                "درخواست ثبت ژورنال باید توسط سرور ژورنال پردازش شود.",
            )

        elif (
            message.get("photo")
            or (
                (message.get("document") or {})
                .get("mime_type", "")
                .startswith("image/")
            )
        ):
            photos = message.get("photo") or []
            document = message.get("document") or {}

            file_id = (
                max(
                    photos,
                    key=lambda photo: photo.get("file_size", 0),
                ).get("file_id")
                if photos
                else document.get("file_id")
            )

            if not file_id:
                send(chat_id, "فایل تصویر معتبر پیدا نشد.")
                return

            from xt_bot.image_engine import analyze_chart_image

            result = analyze_chart_image(
                download_image(file_id),
                message.get("caption") or "",
            )

            send(chat_id, result)

        elif txt:
            send(
                chat_id,
                "دستور نامعتبر است.\n"
                "برای مشاهده فهرست دستورات /info را بفرست.",
            )

    except Exception as exc:
        print(
            f"Message processing error: "
            f"{type(exc).__name__}: {exc}"
        )
        try:
            send(
                chat_id,
                f"خطا: {type(exc).__name__}: {str(exc)[:250]}",
            )
        except Exception as send_exc:
            print(f"Telegram send error: {send_exc}")


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
