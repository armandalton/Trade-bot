"""Persian Telegram/Markdown report."""
from .util import utc_iso
from .backtest import summarize
import pandas as pd

FA = {"valid_reversal": "برگشت معتبر", "weak_reversal": "برگشت ضعیف", "true_break": "شکست واقعی SMA",
      "false_break": "شکست کاذب", "no_setup": "بدون موقعیت معاملاتی"}
SEG = {"train": "آموزش", "val": "اعتبارسنجی", "test": "آزمون نهایی"}


def _pct(x):
    return "—" if x is None else f"{x*100:.1f}%"


def _pf(x):
    return "—" if x is None else f"{x:.2f}"


def seg_line(name, m):
    if not m or m["trades"] == 0:
        return f"• {name}: بدون معامله"
    return (f"• {name}: {m['trades']} معامله (Long {m['long']} / Short {m['short']}) | برنده {m['wins']} / بازنده {m['losses']} | "
            f"WinRate {_pct(m['win_rate'])} | خالص ${m['net_pnl']:+.2f} | PF {_pf(m['profit_factor'])} | MaxDD {m['max_dd_pct']*100:.1f}%")


def live_metrics(state):
    closed = (state or {}).get("closed_trades") or []
    return summarize(pd.DataFrame(closed), (state or {}).get("capital", 2000)) if closed else summarize(None, 2000)


def build_report(meta, results, counts, patterns, bounds, state=None):
    out = []
    ok = meta["approved"]
    head = "✅ مدل تأیید شد (برای Paper Trading)" if ok else "⛔ مدل تأیید نشد — سیگنال زنده ارسال نخواهد شد"
    out.append(f"📈 گزارش آموزش و بک‌تست SMA64 — XT Futures (1m)\n{head}")
    out.append(f"بازه داده: {utc_iso(bounds['start'])} تا {utc_iso(bounds['end'])}")
    out.append(f"تقسیم زمانی ۶۰/۲۰/۲۰ → آموزش تا {utc_iso(bounds['train_end'])} | اعتبارسنجی تا {utc_iso(bounds['val_end'])} | بقیه آزمون")
    out.append(f"هزینه‌ها: کارمزد هر طرف {meta['costs']['fee_rate']*100:.3f}% + اسلیپیج هر طرف {meta['costs']['slippage_rate']*100:.3f}%؛ "
               f"setupهایی با هزینه رفت‌وبرگشت بیش از {meta['costs']['max_cost_r']}R رد می‌شوند")
    if meta.get("threshold") is not None:
        cal = "کالیبره‌شده" if meta["calibrated"] else "کالیبره‌نشده"
        out.append(f"آستانه مدل (انتخاب‌شده فقط روی اعتبارسنجی): {meta['threshold']:.3f} | احتمال مدل {cal}"
                   + (f" | Brier آزمون: {meta['test_brier']:.3f}" if meta.get("test_brier") is not None else ""))
    if meta["reasons"]:
        out.append("دلایل رد/هشدار:\n" + "\n".join(f"- {r}" for r in meta["reasons"]))
    t = results["segments"]["test"]
    if t["trades"] < 100:
        out.append("⚠️ تعداد معاملات آزمون کمتر از ۱۰۰ است؛ نمونه آماری کافی نیست و نتیجه قابل‌اتکا نیست.")
    out.append("— توجه: «تعداد برخورد» با «تعداد معامله» یکی نیست؛ بیشتر برخوردها معامله نمی‌شوند.")
    for sym in sorted(counts):
        c = counts[sym]
        name = sym.upper().replace("_", "/")
        lines = [f"📊 {name}",
                 f"برخوردهای قیمت با SMA64 (کل بازه): {c['total']}  (Long {c['longs']} / Short {c['shorts']})",
                 f"• {FA['valid_reversal']}: {c['valid_reversal']} | {FA['weak_reversal']}: {c['weak_reversal']}",
                 f"• {FA['true_break']}: {c['true_break']} | {FA['false_break']}: {c['false_break']}",
                 f"• {FA['no_setup']}: {c['no_setup']} (رد شده به‌خاطر هزینه: {c['no_setup_cost']})",
                 f"• نامشخص/ناقص (افق ناتمام یا گپ داده): {c['pending_or_gap']} | سقف‌وحد در یک کندل (محافظه‌کارانه ضرر): {c['ambiguous']}",
                 "معاملات اجراشده با مدل (پس از کارمزد و اسلیپیج):"]
        for s in ("train", "val", "test"):
            lines.append(seg_line(SEG[s], results["by_symbol"][s].get(sym, {}).get("ALL")))
        for side in ("LONG", "SHORT"):
            lines.append(seg_line(f"آزمون {side}", results["by_symbol"]["test"].get(sym, {}).get(side)))
        best, worst = patterns.get(sym, ([], []))
        if best:
            lines.append("بهترین الگوها (آموزش+اعتبارسنجی، همه setupها):")
            lines += [f"  ✔ {p} | n={n} | موفقیت {w*100:.0f}% | میانگین {e:+.2f}R" for p, n, w, e in best]
            lines.append("بدترین الگوها:")
            lines += [f"  ✘ {p} | n={n} | موفقیت {w*100:.0f}% | میانگین {e:+.2f}R" for p, n, w, e in worst]
        else:
            lines.append("الگوها: نمونه برای رتبه‌بندی الگوها کافی نیست.")
        out.append("\n".join(lines))
    tot = ["🧮 مجموع پنج نماد (پورتفوی واحد با سقف ۳ موقعیت هم‌زمان و قفل‌های ریسک)"]
    for s in ("train", "val", "test"):
        tot.append(seg_line(SEG[s], results["segments"][s]))
    tot.append("مقایسه: همه setupها بدون فیلتر مدل (برای سنجش ارزش مدل):")
    for s in ("train", "val", "test"):
        tot.append(seg_line(SEG[s], results["baseline"][s]))
    sk = results["skipped"].get("test") or {}
    if sk:
        tot.append("معاملات ردشده توسط قوانین ریسک در آزمون: " + "، ".join(f"{k}={v}" for k, v in sk.items()))
    lm = live_metrics(state)
    tot.append(seg_line("معاملات زنده Paper تاکنون", lm))
    out.append("\n".join(tot))
    out.append("یادآوری: این نتایج بک‌تست است و تضمین سود نیست؛ احتمال مدل با Win Rate واقعی یکی نیست.")
    return "\n\n".join(out)
