import os
import time
import requests

from .util import log


def configured():
    return bool(os.getenv("TELEGRAM_BOT_TOKEN", "").strip() and os.getenv("TELEGRAM_CHAT_ID", "").strip())


def _split(text, limit=3900):
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit and cur:
            parts.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur)
    return parts


def send(text):
    """Send a plain-text Telegram message (split if >4096 chars). Returns True only if all parts delivered.
    Errors are logged WITHOUT the request URL, so the bot token can never leak into logs."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    if not token or not chat:
        log("Telegram secrets missing; message not sent.")
        return False
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for part in _split(text):
        ok = False
        for attempt in range(3):
            try:
                r = requests.post(url, json={"chat_id": chat, "text": part, "disable_web_page_preview": True}, timeout=20)
                if r.status_code == 200:
                    ok = True
                    break
                log(f"Telegram HTTP {r.status_code}: {r.text[:120].replace(token, '***')}")
                if r.status_code == 429:
                    time.sleep(3)
            except requests.RequestException as e:
                log(f"Telegram network error: {type(e).__name__}")
            time.sleep(1 + attempt)
        if not ok:
            return False
        time.sleep(0.4)
    return True
