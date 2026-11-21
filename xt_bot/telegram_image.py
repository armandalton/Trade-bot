
"""Image-only Telegram chart analysis through Gemini Interactions API.
No live exchange data is requested and no orders are placed.
"""

import base64
import json
import os
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parent.parent
OFFSET_FILE = ROOT / "telegram_image_offset.json"

TELEGRAM_API = "https://api.telegram.org"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")

MAX_IMAGE_BYTES = 10 * 1024 * 1024

ALLOWED_MIME = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


ANALYSIS_PROMPT = """
تو یک تحلیلگر تکنیکال محتاط هستی. فقط همین تصویر پیوست‌شده را تحلیل کن.
مطلقاً از دیتای زنده، اینترنت، قیمت فعلی بازار، نمادهای دیگر یا اطلاعات خارج از تصویر استفاده نکن.
هیچ قیمتی را حدس نزن. اگر نماد، تایم‌فریم، محور قیمت، مقدار MA100 یا ساختار کندل‌ها خوانا نیست، صریحاً «قابل تشخیص نیست» بنویس.
اگر مقیاس قیمت روی تصویر خوانا نیست، Entry و SL و TP عددی نساز؛ فقط سطوح نسبی را توصیف کن و نتیجه را «NO TRADE / نیاز به تصویر واضح‌تر» قرار بده.
در صورت نبود MA100 روی تصویر، ادعا نکن قیمت بالاتر یا پایین‌تر از MA100 است.
الگوها را فقط وقتی تأیید کن که کندل‌ها به اندازه کافی واضح باشند. Engulfing باید بدنه کندل قبلی را بپوشاند؛ صرفاً سایه کافی نیست.
بررسی کن: جهت و شیب روند قابل مشاهده، جایگاه قیمت نسبت به MA100 در صورت وجود، Engulfing، Pullback، ساختار سقف و کف، حمایت/مقاومت قابل مشاهده، و حجم فقط اگر پنل حجم دیده می‌شود.
قانون جهت: اگر قیمت واضحاً بالای MA100 است فقط LONG را بررسی کن؛ اگر واضحاً زیر MA100 است فقط SHORT را بررسی کن؛ اگر MA100 نیست یا وضعیت مبهم است، نتیجه قطعی نده.
برای طرح معاملاتی، نسبت ریسک به بازده هدف 1:2 است. حد ضرر باید پشت سوئینگ واضح و قابل مشاهده باشد. حد سود فقط از روی فاصله ورود تا حد ضرر محاسبه شود. اگر این سطوح از تصویر قابل استخراج نیستند، عددسازی نکن.
فقط فارسی و با این قالب پاسخ بده:

📷 تحلیل تصویر (فقط بر اساس عکس)
نماد: ...
تایم‌فریم: ...
وضعیت MA100: ...
الگوی کندلی: ...
ساختار بازار: ...
جهت: LONG / SHORT / NO TRADE
ورود: ... یا قابل تعیین نیست
حد ضرر: ... یا قابل تعیین نیست
حد سود (RR 1:2): ... یا قابل تعیین نیست
کیفیت تصویر: خوب / متوسط / نامناسب
اطمینان تحلیلی: کم / متوسط / بالا
دلایل:
- ...
ریسک‌ها و اطلاعات نامشخص:
- ...
جمع‌بندی: اگر اطلاعات کافی نیست، واضح بگو منتظر تصویر واضح‌تر هستی. هیچ‌وقت برای راضی کردن کاربر سیگنال اجباری صادر نکن.
این تحلیل تضمین سود نیست و بات هیچ معامله‌ای اجرا نمی‌کند.
"""


def _telegram(method, payload=None, timeout=25):
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN secret is missing")

    response = requests.post(
        f"{TELEGRAM_API}/bot{token}/{method}",
        json=payload or {},
        timeout=timeout,
    )

    response.raise_for_status()

    body = response.json()

    if not body.get("ok"):
        raise RuntimeError(
            f"Telegram API returned ok=false for {method}"
        )

    return body.get("result")


def _send(chat_id, text):
    if len(text) > 3900:
        text = text[:3880] + "\n… (پاسخ کوتاه شد)"

    _telegram(
        "sendMessage",
        {
            "chat_id": chat_id,
            "text": text,
        },
        timeout=20,
    )


def _extract_image(message):
    photos = message.get("photo") or []

    if photos:
        largest = max(
            photos,
            key=lambda item: item.get("file_size", 0),
        )

        return largest.get("file_id"), "image/jpeg"

    document = message.get("document") or {}

    mime = document.get("mime_type", "")

    if mime in ALLOWED_MIME:
        return document.get("file_id"), mime

    return None, None


def _download_telegram_image(file_id):
    info = _telegram(
        "getFile",
        {"file_id": file_id},
    )

    path = info.get("file_path")

    if not path:
        raise RuntimeError(
            "Telegram did not return an image file path"
        )

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    response = requests.get(
        f"{TELEGRAM_API}/file/bot{token}/{path}",
        timeout=35,
        stream=True,
    )

    response.raise_for_status()

    content_length = int(
        response.headers.get("Content-Length", "0") or 0
    )

    if content_length > MAX_IMAGE_BYTES:
        raise ValueError(
            "حجم تصویر بیشتر از ۱۰ مگابایت است."
        )

    chunks = []
    total = 0

    for chunk in response.iter_content(64 * 1024):
        if not chunk:
            continue

        total += len(chunk)

        if total > MAX_IMAGE_BYTES:
            raise ValueError(
                "حجم تصویر بیشتر از ۱۰ مگابایت است."
            )

        chunks.append(chunk)

    data = b"".join(chunks)

    if not data:
        raise ValueError("فایل تصویر خالی است.")

    return data


def _gemini_analyze(image_bytes, mime_type, caption=""):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY secret is missing"
        )

    prompt = ANALYSIS_PROMPT

    if caption.strip():
        prompt += (
            "\nتوضیح همراه عکس از کاربر "
            "(فقط به‌عنوان زمینه، نه منبع قیمت): "
            + caption[:500]
        )

    image_base64 = base64.b64encode(
        image_bytes
    ).decode("ascii")

    # Gemini 3.8 Flash - Interactions API
    payload = {
        "model": GEMINI_MODEL,
        "store": False,
        "input": [
            {
                "type": "text",
                "text": prompt,
            },
            {
                "type": "image",
                "data": image_base64,
                "mime_type": mime_type,
            },
        ],
        "generation_config": {
            "temperature": 0.2,
            "max_output_tokens": 1300,
        },
    }

    response = requests.post(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        headers={
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
            "Api-Revision": "2026-05-20",
        },
        json=payload,
        timeout=120,
    )

    if response.status_code == 429:
        raise RuntimeError(
            "سهمیه رایگان Gemini فعلاً به سقف رسیده است؛ "
            "کمی بعد دوباره امتحان کن."
        )

    if response.status_code >= 400:
        try:
            error_body = response.json()

            detail = (
                error_body.get("error", {}).get("message")
                or "خطای سرویس Gemini"
            )

        except Exception:
            detail = "خطای سرویس Gemini"

        raise RuntimeError(
            f"Gemini API error {response.status_code}: "
            f"{str(detail)[:250]}"
        )

    try:
        body = response.json()

    except Exception:
        raise RuntimeError(
            "پاسخ دریافتی از Gemini قابل خواندن نیست."
        )

    if body.get("status") == "failed":
        error = body.get("error") or {}

        detail = (
            error.get("message")
            or error.get("status")
            or "تحلیل Gemini ناموفق بود."
        )

        raise RuntimeError(str(detail)[:250])

    # First try the convenience output_text field.
    answer = (body.get("output_text") or "").strip()

    # Fallback: extract text from model_output steps.
    if not answer:
        text_parts = []

        for step in body.get("steps") or []:

            if step.get("type") != "model_output":
                continue

            for part in step.get("content") or []:

                if (
                    part.get("type") == "text"
                    and part.get("text")
                ):
                    text_parts.append(part["text"])

        answer = "\n".join(text_parts).strip()

    if not answer:
        raise RuntimeError(
            "Gemini پاسخ متنی قابل استفاده برنگرداند."
        )

    return answer


def _read_offset():
    try:
        data = json.loads(
            OFFSET_FILE.read_text(encoding="utf-8")
        )

        return int(data.get("offset", 0))

    except Exception:
        return 0


def _write_offset(offset):
    OFFSET_FILE.write_text(
        json.dumps(
            {"offset": int(offset)},
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def poll_once():
    """Process queued Telegram updates once; called by scheduled GitHub Actions."""

    allowed_chat = os.getenv(
        "TELEGRAM_CHAT_ID",
        "",
    ).strip()

    if not allowed_chat:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID secret is missing"
        )

    offset = _read_offset()

    updates = _telegram(
        "getUpdates",
        {
            "offset": offset,
            "timeout": 0,
            "allowed_updates": ["message"],
        },
        timeout=20,
    ) or []

    handled = 0

    for update in updates:

        next_offset = (
            int(update.get("update_id", 0)) + 1
        )

        message = update.get("message") or {}

        chat_id = str(
            (message.get("chat") or {}).get("id", "")
        )

        # Only accept messages from the authorized chat.
        if chat_id != allowed_chat:

            _write_offset(next_offset)
            offset = next_offset

            continue

        text = (
            message.get("text") or ""
        ).strip()

        # Start command.
        if text.startswith("/start"):

            _send(
                chat_id,
                "سلام! عکس چارت را به‌صورت Photo یا فایل تصویر بفرست.\n"
                "من فقط بر اساس همین عکس تحلیل می‌کنم و از دیتای زنده صرافی استفاده نمی‌کنم.\n"
                "برای خوانایی بهتر، نماد، تایم‌فریم، محور قیمت و MA100 در تصویر مشخص باشند."
            )

        else:

            file_id, mime = _extract_image(message)

            if not file_id:

                if text:
                    _send(
                        chat_id,
                        "برای تحلیل تصویری، لطفاً اسکرین‌شات چارت را "
                        "به شکل عکس یا فایل PNG/JPG/WEBP بفرست."
                    )

            else:

                try:

                    _send(
                        chat_id,
                        "📥 تصویر دریافت شد؛ دارم فقط بر اساس همین عکس بررسی می‌کنم…"
                    )

                    image_bytes = _download_telegram_image(
                        file_id
                    )

                    result = _gemini_analyze(
                        image_bytes,
                        mime,
                        message.get("caption") or "",
                    )

                    _send(chat_id, result)

                except Exception as exc:

                    _send(
                        chat_id,
                        f"❌ تحلیل انجام نشد: {str(exc)[:700]}"
                    )

        _write_offset(next_offset)

        offset = next_offset

        handled += 1

    print(
        f"Image bot poll complete. "
        f"handled={handled} next_offset={offset}",
        flush=True,
    )

    return handled
