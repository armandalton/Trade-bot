"""Telegram webhook processing for XT signals and journal registration."""
import json
import os
import time
from datetime import datetime, timezone

import requests

API = "https://api.telegram.org"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
SYMBOLS = ("btc_usdt", "eth_usdt", "sol_usdt", "bnb_usdt", "xrp_usdt", "ada_usdt")


def telegram(method, payload=None, timeout=30):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN تنظیم نشده است.")
    r = requests.post(f"{API}/bot{token}/{method}", json=payload or {}, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    if not data.get("ok"):
        raise RuntimeError(f"Telegram {method} failed")
    return data.get("result")


def main_keyboard():
    return {
        "inline_keyboard": [
            [
                {"text": "BTC", "callback_data": "symbol:btc_usdt"},
                {"text": "ETH", "callback_data": "symbol:eth_usdt"},
                {"text": "SOL", "callback_data": "symbol:sol_usdt"},
            ],
            [
                {"text": "BNB", "callback_data": "symbol:bnb_usdt"},
                {"text": "XRP", "callback_data": "symbol:xrp_usdt"},
                {"text": "ADA", "callback_data": "symbol:ada_usdt"},
            ],
            [{"text": "📒 گزارش ژورنال", "callback_data": "report"}],
        ]
    }


def send(chat_id, message, reply_markup=None):
    payload = {"chat_id": chat_id, "text": str(message)[:3900]}
    if reply_markup:
        payload["reply_markup"] = reply_markup
    telegram("sendMessage", payload)


def answer_callback(callback_id, text=""):
    telegram("answerCallbackQuery", {
        "callback_query_id": callback_id,
        "text": str(text)[:180],
        "show_alert": False,
    })


def download_image(file_id):
    info = telegram("getFile", {"file_id": file_id})
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    r = requests.get(f"{API}/file/bot{token}/{info['file_path']}", timeout=35)
    r.raise_for_status()
    if len(r.content) > MAX_IMAGE_BYTES:
        raise ValueError("حجم تصویر بیشتر از ۱۰ مگابایت است.")
    return r.content


def create_journal_candidate(result):
    """Send the signal to Cloudflare D1 through the authenticated Worker endpoint."""
    base = os.getenv(
        "WORKER_URL",
        "https://still-dust-02af.arman8256.workers.dev"
    ).rstrip("/")
    secret = os.getenv("WORKER_API_SECRET", "").strip()
    if not secret:
        raise RuntimeError("WORKER_API_SECRET در GitHub Actions تنظیم نشده است.")

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
    from xt_bot.data import fetch_klines, closed_only, fetch_last_price
    from xt_bot.technical_engine import analyze_ohlcv, format_signal, INTERVALS

    symbol = symbol.strip().lower().replace("/", "_")
    if symbol not in SYMBOLS:
        send(chat_id, "نماد انتخاب‌شده معتبر نیست.", main_keyboard())
        return

    timeframe = "1m"
    step = INTERVALS[timeframe]
    request_time = int(time.time() * 1000)
    df = fetch_klines(symbol, request_time - step * 320, request_time, timeframe)
    df = closed_only(df, request_time, step)

    if len(df) < 230:
        send(chat_id, f"⛔ تعداد کندل کافی نیست: {len(df)}", main_keyboard())
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
            f"تأخیر نسبت به بسته‌شدن کندل: {candle_delay_ms / 1000:.1f} ثانیه\n"
            "برای جلوگیری از سیگنال اشتباه، تحلیل متوقف شد.",
            main_keyboard(),
        )
        return

    ticker = fetch_last_price(symbol)
    current_price = float(ticker["price"])
    ticker_received_ms = int(ticker["received_at_ms"])
    result = analyze_ohlcv(df, symbol, timeframe)
    candle_close_price = float(last_candle["close"])
    difference_pct = ((current_price - candle_close_price) / candle_close_price) * 100

    candle_time = datetime.fromtimestamp(
        candle_open_ms / 1000, tz=timezone.utc
    ).strftime("%Y-%m-%d %H:%M:%S UTC")
    ticker_time = datetime.fromtimestamp(
        ticker_received_ms / 1000, tz=timezone.utc
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
        + "⚠️ قیمت آخرین معامله، قیمت لحظه‌ای در زمان دریافت پاسخ API است؛ "
        + "قیمت اجرای سفارش را تضمین نمی‌کند."
    )

    if result.get("direction") in ("LONG", "SHORT"):
        try:
            candidate_id = create_journal_candidate(result)
            send(
                chat_id,
                message + "\n\nبرای ذخیره این سیگنال در ژورنال، دکمه زیر را بزن:",
                {"inline_keyboard": [[
                    {"text": "📒 ثبت در ژورنال", "callback_data": f"reg:{candidate_id}"}
                ], [
                    {"text": "🔄 انتخاب نماد دیگر", "callback_data": "menu"}
                ]]},
            )
        except Exception as exc:
            print(f"Journal candidate error: {type(exc).__name__}: {exc}")
            send(
                chat_id,
                message + "\n\n⚠️ ذخیره موقت سیگنال در ژورنال فعلاً در دسترس نیست.",
                main_keyboard(),
            )
    else:
        send(chat_id, message, main_keyboard())


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
        if txt.startswith("/start") or txt.startswith("/symbols"):
            send(
                chat_id,
                "ربات پژوهشی XT آماده است.\nیک نماد برای تحلیل یک‌دقیقه‌ای انتخاب کن:",
                main_keyboard(),
            )
        elif txt.startswith("/market"):
            bits = txt.split()
            symbol = bits[1] if len(bits) > 1 else "btc_usdt"
            market_command(chat_id, symbol)
        elif txt.startswith("/report"):
            send(chat_id, "برای گزارش ژورنال از دکمه «گزارش ژورنال» استفاده کن.", main_keyboard())
        elif message.get("photo") or (message.get("document") or {}).get("mime_type", "").startswith("image/"):
            photos = message.get("photo") or []
            document = message.get("document") or {}
            file_id = (
                max(photos, key=lambda p: p.get("file_size", 0)).get("file_id")
                if photos else document.get("file_id")
            )
            if not file_id:
                send(chat_id, "فایل تصویر معتبر پیدا نشد.")
            else:
                from xt_bot.image_engine import analyze_chart_image
                result = analyze_chart_image(download_image(file_id), message.get("caption") or "")
                send(chat_id, result, main_keyboard())
        elif txt:
            send(chat_id, "برای شروع /start را بزن یا یک نماد را از دکمه‌ها انتخاب کن.", main_keyboard())
    except Exception as exc:
        send(chat_id, f"خطا: {type(exc).__name__}: {str(exc)[:300]}", main_keyboard())


def handle_callback(callback):
    allowed = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    message = callback.get("message") or {}
    chat_id = str((message.get("chat") or {}).get("id", ""))
    callback_id = callback.get("id", "")
    if not allowed or chat_id != allowed:
        print("Unauthorized callback ignored.")
        if callback_id:
            answer_callback(callback_id, "دسترسی غیرمجاز")
        return

    data = str(callback.get("data") or "")
    try:
        if data.startswith("symbol:"):
            answer_callback(callback_id, "در حال دریافت داده و تحلیل…")
            market_command(chat_id, data.split(":", 1)[1])
        elif data in ("menu", "symbols"):
            answer_callback(callback_id)
            send(chat_id, "نماد موردنظر را انتخاب کن:", main_keyboard())
        else:
            # The Worker handles "reg:<id>" and "report" directly against D1.
            answer_callback(callback_id, "این عملیات توسط سرور ژورنال مدیریت می‌شود.")
    except Exception as exc:
        print(f"Callback processing: {type(exc).__name__}: {exc}")
        send(chat_id, f"خطا در پردازش دکمه: {str(exc)[:250]}", main_keyboard())


def process_webhook_update():
    raw = os.getenv("TELEGRAM_UPDATE_JSON", "")
    if not raw:
        raise RuntimeError("TELEGRAM_UPDATE_JSON خالی است.")
    update = json.loads(raw)
    if update.get("callback_query"):
        handle_callback(update["callback_query"])
        return
    message = update.get("message") or update.get("edited_message")
    if message:
        handle_message(message)
    else:
        print("Unsupported update ignored.")
