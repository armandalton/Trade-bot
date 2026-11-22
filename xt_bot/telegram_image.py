"""Telegram polling: image-only local CV analysis + /market command using XT OHLCV."""
import json, os, time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent.parent
OFFSET_FILE = ROOT / "telegram_image_offset.json"
API = "https://api.telegram.org"
MAX_IMAGE_BYTES = 10 * 1024 * 1024

def _telegram(method, payload=None, timeout=30):
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    if not token: raise RuntimeError("TELEGRAM_BOT_TOKEN تنظیم نشده است.")
    r=requests.post(f"{API}/bot{token}/{method}",json=payload or {},timeout=timeout)
    r.raise_for_status(); data=r.json()
    if not data.get("ok"): raise RuntimeError(f"Telegram {method} failed")
    return data.get("result")

def _send(chat_id, text):
    _telegram("sendMessage",{"chat_id":chat_id,"text":text[:3900]},timeout=25)

def _offset():
    try: return int(json.loads(OFFSET_FILE.read_text()).get("offset",0))
    except Exception: return 0

def _save_offset(v):
    OFFSET_FILE.write_text(json.dumps({"offset":int(v)},indent=2),encoding="utf-8")

def _download(file_id):
    info=_telegram("getFile",{"file_id":file_id})
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    r=requests.get(f"{API}/file/bot{token}/{info['file_path']}",timeout=35)
    r.raise_for_status()
    if len(r.content)>MAX_IMAGE_BYTES: raise ValueError("حجم تصویر بیشتر از ۱۰ مگابایت است.")
    return r.content

def _market_command(chat_id, text):
    from xt_bot.data import fetch_klines, closed_only
    from xt_bot.technical_engine import analyze_ohlcv, format_signal, INTERVALS
    bits=text.strip().split()
    symbol=(bits[1] if len(bits)>1 else "btc_usdt").lower().replace("/","_")
    tf=(bits[2] if len(bits)>2 else "1m").lower()
    if tf not in INTERVALS:
        _send(chat_id,"تایم‌فریم معتبر: "+", ".join(INTERVALS.keys())); return
    now=int(time.time()*1000); step=INTERVALS[tf]
    start=now-step*320
    df=fetch_klines(symbol,start,now,tf)
    df=closed_only(df,now,step)
    result=analyze_ohlcv(df,symbol,tf)
    _send(chat_id,format_signal(result))

def poll_once():
    allowed=os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not allowed: raise RuntimeError("TELEGRAM_CHAT_ID تنظیم نشده است.")
    offset=_offset()
    updates=_telegram("getUpdates",{"offset":offset,"timeout":0,"allowed_updates":["message"]}) or []
    handled=0
    for u in updates:
        nxt=int(u.get("update_id",0))+1
        m=u.get("message") or {}
        chat=str((m.get("chat") or {}).get("id",""))
        if chat!=allowed:
            _save_offset(nxt); offset=nxt; continue
        txt=(m.get("text") or "").strip()
        try:
            if txt.startswith("/start"):
                _send(chat,"سلام! دو قابلیت فعال است:\n۱) عکس چارت را بفرست تا فقط با پردازش محلی تصویر بررسی شود.\n۲) برای تحلیل بازار XT بنویس:\n/market BTC_USDT 1m\nتایم‌فریم‌ها: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 1d\nهیچ هوش مصنوعی/Gemini استفاده نمی‌شود و سفارش واقعی ثبت نمی‌شود.")
            elif txt.startswith("/market"):
                _market_command(chat,txt)
            elif m.get("photo") or (m.get("document") or {}).get("mime_type","").startswith("image/"):
                photo=m.get("photo") or []
                doc=m.get("document") or {}
                file_id=(max(photo,key=lambda z:z.get("file_size",0)).get("file_id") if photo else doc.get("file_id"))
                if not file_id: _send(chat,"فایل تصویر معتبر پیدا نشد.")
                else:
                    from xt_bot.image_engine import analyze_chart_image
                    _send(chat,analyze_chart_image(_download(file_id),m.get("caption") or ""))
            elif txt:
                _send(chat,"برای تحلیل بازار: /market BTC_USDT 1m\nیا عکس چارت را ارسال کن.")
        except Exception as e:
            _send(chat,f"❌ خطا: {type(e).__name__}: {str(e)[:400]}")
        _save_offset(nxt); offset=nxt; handled+=1
    print(f"Telegram poll complete handled={handled} offset={offset}",flush=True)
    return handled
