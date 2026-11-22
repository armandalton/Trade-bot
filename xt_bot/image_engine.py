"""Conservative, local-only chart screenshot inspection (OpenCV; no AI/OCR service).
Numeric trade levels are intentionally not inferred from pixels without calibrated price axis.
"""
from __future__ import annotations
import io
import numpy as np
from PIL import Image

def analyze_chart_image(image_bytes: bytes, caption: str = "") -> str:
    try:
        import cv2
        raw = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return "❌ تصویر قابل خواندن نیست. لطفاً اسکرین‌شات PNG یا JPG واضح بفرست."
        h, w = img.shape[:2]
        if w < 500 or h < 250:
            return "📷 رزولوشن تصویر برای استخراج کندل‌ها کافی نیست؛ تصویر واضح‌تر و بزرگ‌تر بفرست."
        # Restrict to central chart area, excluding most side panels.
        roi = img[int(h*.08):int(h*.88), int(w*.04):int(w*.94)]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Broad green/red masks; chart themes vary, so report extraction confidence.
        green = cv2.inRange(hsv, np.array([32,35,35]), np.array([100,255,255]))
        red1 = cv2.inRange(hsv, np.array([0,45,35]), np.array([12,255,255]))
        red2 = cv2.inRange(hsv, np.array([165,45,35]), np.array([180,255,255]))
        red = cv2.bitwise_or(red1, red2)
        kernel = np.ones((2,2), np.uint8)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, kernel)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, kernel)
        # Count columns containing colored candle pixels; this is a rough visibility metric.
        gcols = np.where((green > 0).sum(axis=0) >= 3)[0]
        rcols = np.where((red > 0).sum(axis=0) >= 3)[0]
        total_cols = len(set(gcols.tolist()+rcols.tolist()))
        green_px, red_px = int((green>0).sum()), int((red>0).sum())
        if total_cols < 8:
            return ("📷 تحلیل تصویر (پردازش محلی)\n"
                    "نتیجه: WAIT / تصویر برای تشخیص کندل‌ها کافی نیست.\n"
                    "کندل‌های رنگی به تعداد قابل اتکا استخراج نشدند. لطفاً نمودار را بزرگ‌تر بفرست؛ "
                    "از روی تصویر نامشخص Entry و SL و TP عددی نمی‌سازم.")
        bias = "خنثی/نامشخص"
        if green_px > red_px*1.25: bias = "در تصویر کندل‌های سبز غالب‌اند (این به‌تنهایی سیگنال LONG نیست)"
        elif red_px > green_px*1.25: bias = "در تصویر کندل‌های قرمز غالب‌اند (این به‌تنهایی سیگنال SHORT نیست)"
        return (f"📷 تحلیل تصویر (فقط همین عکس؛ بدون دیتای زنده)\n"
                f"ابعاد تصویر: {w}×{h}\n"
                f"ستون‌های دارای آثار کندلی: {total_cols}\n"
                f"برداشت رنگی: {bias}\n"
                "ساختار دقیق HH/HL و BOS/CHoCH، محور قیمت و آخرین کندل از این استخراج ساده با اطمینان کافی قابل تعیین نیست.\n"
                "جهت: WAIT\nورود: قابل تعیین نیست\nحد ضرر: قابل تعیین نیست\nحد سود: قابل تعیین نیست\n"
                "برای تحلیل عددی دقیق، نماد و تایم‌فریم را در متن عکس/کپشن بنویس و تصویر را طوری بفرست که محور قیمت و آخرین کندل واضح باشند.\n"
                "⚠️ این ابزار پردازش تصویر کلاسیک است، نه مدل هوش مصنوعی؛ هیچ معامله‌ای اجرا نمی‌کند.")
    except Exception as exc:
        return f"❌ پردازش محلی تصویر ناموفق بود: {type(exc).__name__}: {str(exc)[:180]}"
