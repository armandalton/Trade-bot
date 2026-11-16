"""Persian reports: a short, plain-language Telegram text and a full detailed Markdown file."""
import pandas as pd

from .util import utc_iso
from .backtest import summarize
from . import hours as hours_mod

KIND_FA = {"REVERSAL": "حالت برگشت", "BREAKOUT": "حالت شکست"}
KIND_ICON = {"REVERSAL": "🔁", "BREAKOUT": "💥"}
SEG = {"train": "آموزش", "val": "اعتبارسنجی", "test": "آزمون"}
LINE = "━━━━━━━━━━━━━━"
LABEL_FA = {"valid_reversal": "برگشت معتبر", "weak_reversal": "برگشت ضعیف", "true_break": "شکست واقعی SMA",
            "false_break": "شکست کاذب", "valid_breakout": "شکست موفق", "weak_breakout": "شکست ضعیف",
            "failed_breakout": "شکست ناموفق (برگشت به SMA)", "whipsaw": "نوسان بی‌جهت (ضرر)"}


def _pct(x):
    return "—" if x is None else f"{x*100:.0f}%"


def _pf(x):
    return "—" if x is None else f"{x:.2f}"


def sig_short(m):
    if not m or m["n"] == 0:
        return "بدون سیگنال"
    return f"{m['n']} سیگنال | برد {_pct(m['win_rate'])} | میانگین {m['mean_r']:+.2f}R | PF {_pf(m['pf'])}"


def seg_line(name, m):
    if not m or m["trades"] == 0:
        return f"• {name}: بدون معامله"
    return (f"• {name}: {m['trades']} معامله (Long {m['long']} / Short {m['short']}) | برنده {m['wins']} / بازنده {m['losses']} | "
            f"برد {_pct(m['win_rate'])} | خالص ${m['net_pnl']:+.2f} | PF {_pf(m['profit_factor'])} | MaxDD {m['max_dd_pct']*100:.1f}%")


def live_metrics(state):
    closed = (state or {}).get("closed_trades") or []
    return summarize(pd.DataFrame(closed), 2000) if closed else summarize(None, 2000)


def _window_text(h):
    b, lab = h["best"], h["label"]
    lines = []
    if h.get("source") == "rule":
        lines.append("چون هیچ مدلی فعال نبود، ساعت‌ها با قانون ساده سنجیده شد: معامله هم‌جهت با کندل واکنشی (بدون مدل).")
    if h["basis"] == "performance":
        v, t = b["val"], b["test"]
        lines.append(f"بهترین بازه ۳ ساعته (انتخاب روی اعتبارسنجی): {lab}")
        lines.append(f"در این بازه {b['touches']} برخورد با SMA ({b['touch_share']*100:.0f}% کل برخوردها) و میانگین نوسان {b['atr_bps']:.1f}bp بود")
        lines.append(f"اعتبارسنجی: {sig_short(v)}")
        lines.append(f"آزمون (تأیید روی داده جدید): {sig_short(t)}")
        if h["stable"]:
            lines.append("✅ در داده جدید هم مثبت ماند؛ سیگنال‌های زنده فقط در این بازه ارسال می‌شود.")
        else:
            lines.append("⚠️ در داده جدید تأیید نشد؛ این بازه پایدار نیست و فیلتر ساعت اعمال نمی‌شود. پای چارت نشستن در این ساعت‌ها هنوز اثبات نشده.")
    else:
        lines.append(f"سیگنال کافی برای رتبه‌بندی عملکرد ساعت‌ها نبود. پرتردد‌ترین بازه بر اساس تعداد برخورد: {lab}")
        lines.append(f"{b['touches']} برخورد ({b['touch_share']*100:.0f}% کل) | میانگین نوسان {b['atr_bps']:.1f}bp — این فقط یعنی بازار در این ساعت‌ها فعال‌تر است، نه سودده‌تر.")
    return lines


def build_report(meta, rep, state=None, cfg=None):
    R, hrs = rep["R"], rep["hours"]
    ok, bounds = meta["approved"], rep["bounds"]
    auto = R["auto"]
    t = ["📈 گزارش آموزش بات SMA64 — XT Futures (1m)", f"نتیجه: {'✅ تأیید شد' if ok else '⛔ تأیید نشد'}"]
    if ok:
        t.append(f"یعنی: مدل روی داده دیده‌نشده سودده بود و سیگنال Paper با ریسک به ریوارد 1:{meta['chosen_rr']:g} ارسال می‌شود.")
    else:
        t.append("یعنی: روی داده‌ای که مدل ندیده بود شرط‌های پذیرش برقرار نشد؛ پس سیگنال زنده ارسال نمی‌شود.")
        for r in (meta["reasons"] or [])[:3]:
            t.append(f"• {r}")
    t.append(f"داده: {utc_iso(bounds['start'])[:10]} تا {utc_iso(bounds['end'])[:10]} | آموزش ۶۰٪ ← اعتبارسنجی ۲۰٪ ← آزمون ۲۰٪ (به‌ترتیب زمان)")
    t += [LINE, "🕒 کدام ساعت‌ها پای چارت باشیم؟"] + _window_text(hrs)
    t += [LINE, "🧭 انتخاب خودکار بین برگشت و شکست", "روی هر برخورد، مدل هر دو حالت را می‌سنجد و فقط اگر یکی به آستانه برسد معامله می‌گیرد."]
    bk = auto["by_kind"]["test"]
    for k in ("REVERSAL", "BREAKOUT"):
        i = R["kinds"][k]
        state_txt = "فعال" if i["thr"] is not None else "غیرفعال"
        t.append(f"{KIND_ICON[k]} {KIND_FA[k]} ({state_txt}) — آزمون: {sig_short(bk[k])}")
        if i["thr"] is None and i["reasons"]:
            t.append(f"   چرا غیرفعال: {i['reasons'][0]}")
    t.append(f"جمع (RR 1:{meta['chosen_rr']:g}): {sig_short(auto['signals']['test'])}")
    if auto["break_even"] is not None:
        t.append(f"برای سودده بودن حداقل {auto['break_even']*100:.0f}% برد لازم است (هزینه‌ها لحاظ شده)")
    t += [LINE, "🎯 اگر ریسک به ریوارد دیگری می‌گرفتیم؟ (آزمون)"]
    for key in sorted(meta["rr_table"], key=float):
        r = meta["rr_table"][key]
        s = r["signals"]["test"]
        tag = "← انتخاب‌شده" if r["chosen"] else ("(فقط اطلاعاتی؛ قانون RR≥۲)" if not r["eligible"] else "")
        be = f" | حداقل برد برای سود {r['break_even']*100:.0f}%" if r["break_even"] else ""
        t.append(f"RR 1:{r['rr']:g} → {sig_short(s)}{be} {tag}".strip())
    t += [LINE, "🧠 حفظ‌کردن یا یادگیری؟ (درصد برد انتخاب‌های مدل)"]
    s = auto["signals"]
    t.append(f"آموزش {_pct(s['train']['win_rate'])} ← اعتبارسنجی {_pct(s['val']['win_rate'])} ← آزمون {_pct(s['test']['win_rate'])}")
    t.append("اگر آموزش خیلی بالاتر از آزمون باشد، مدل فقط حفظ کرده و قابل‌اعتماد نیست.")
    t += [LINE, "📊 هر نماد در آزمون (سیگنال | برد | میانگین R)"]
    for sym in rep["symbols"]:
        m = rep["sig_symbol"]["test"][sym]["ALL"]
        t.append(f"{sym.split('_')[0].upper()}: " + (f"{m['n']} | {_pct(m['win_rate'])} | {m['mean_r']:+.2f}" if m["n"] else "بدون سیگنال"))
    t += [LINE, "💼 پورتفوی (ریسک ۰.۵٪، حداکثر ۳ موقعیت، قفل ضرر روزانه ۱.۵٪)"]
    for sp in ("train", "val", "test"):
        t.append(seg_line(SEG[sp], auto["segments"][sp]))
    if auto["segments"]["test"]["trades"] < 100:
        t.append(f"⚠️ فقط {auto['segments']['test']['trades']} معامله پورتفوی در آزمون؛ نمونه کافی نیست.")
    lv = live_metrics(state)
    if lv["trades"]:
        t.append(seg_line("معاملات زنده Paper", lv))
    t += [LINE, "راهنما: برد = معامله‌ای که پس از کارمزد سودده بسته شد. R = ریسک هر معامله (فاصله حد ضرر)؛ +1R یعنی سودی برابر ریسک. PF = جمع سودها ÷ جمع ضررها (هدف ما ≥ 1.4). این‌ها نتایج بک‌تست‌اند، نه تضمین سود. جزئیات کامل: data/report_fa.md در GitHub."]
    short = "\n".join(t)

    # ---------------------------------------------------------------- full
    f = ["# گزارش کامل آموزش و بک‌تست SMA64", "", short, "", "---", "## جزئیات"]
    f.append(f"- بازه: {utc_iso(bounds['start'])} تا {utc_iso(bounds['end'])}؛ آموزش تا {utc_iso(bounds['train_end'])}؛ اعتبارسنجی تا {utc_iso(bounds['val_end'])}")
    f.append(f"- هزینه هر طرف: کارمزد {meta['costs']['fee_rate']*100:.3f}% + اسلیپیج {meta['costs']['slippage_rate']*100:.3f}%؛ سقف هزینه مجاز {meta['costs']['max_cost_r']}R")
    f.append(f"- پارامترها: {meta['params']}")
    f.append(f"- آستانه‌های مدل (فقط روی اعتبارسنجی): {meta['thresholds']}")
    if meta["reasons"]:
        f += ["- همه دلایل رد/هشدار:"] + [f"  - {r}" for r in meta["reasons"]]
    f += ["", "### ساعت‌ها (UTC | تهران)", "ردیف: ساعت | برخورد | نوسان | سیگنال‌های اعتبارسنجی+آزمون"]
    for h in hrs["per_hour"]:
        u, tz = hours_mod._fmt(h["hour"])
        atr_txt = "—" if h["atr_bps"] is None else "%.1fbp" % h["atr_bps"]
        f.append(f"- {u} | {tz}: برخورد {h['touches']} | ATR {atr_txt} | {sig_short(h['sig'])}")
    f.append("سه بازه برتر روی اعتبارسنجی:")
    for r in hrs["top3"]:
        f.append(f"- {hours_mod.window_label(r['start'])}: اعتبارسنجی {sig_short(r['val'])} | آزمون {sig_short(r['test'])}")
    for k in ("REVERSAL", "BREAKOUT"):
        i = R["kinds"][k]
        f += ["", f"### {KIND_ICON[k]} {KIND_FA[k]}", f"- تعداد کاندیدها: آموزش {i['n_setups']['train']} | اعتبارسنجی {i['n_setups']['val']} | آزمون {i['n_setups']['test']}"]
        if i["brier"] is not None:
            f.append(f"- Brier آزمون: {i['brier']:.3f} (احتمال مدل {'کالیبره‌شده' if i['calibrated'] else 'کالیبره‌نشده'})")
        f.append("انتخاب این مدل به‌تنهایی:")
        f += [f"- {SEG[s]}: {sig_short(i['signals'][s]['model'])}" for s in ("train", "val", "test")]
        f.append("همه کاندیدها بدون مدل (اگر روی هر برخورد این حالت را می‌گرفتیم):")
        f += [f"- {SEG[s]}: {sig_short(i['signals'][s]['all'])}" for s in ("train", "val", "test")]
        b, w = rep["patterns"][k]["ALL"]
        if b:
            f.append("بهترین الگوها (آموزش+اعتبارسنجی):")
            f += [f"- ✔ {p} | n={n} | برد {w_*100:.0f}% | میانگین {e:+.2f}R" for p, n, w_, e in b]
            f.append("بدترین الگوها:")
            f += [f"- ✘ {p} | n={n} | برد {w_*100:.0f}% | میانگین {e:+.2f}R" for p, n, w_, e in w]
    f += ["", "### جمع خودکار (ترکیب دو حالت) — هر نماد و جهت"]
    for sym in rep["symbols"]:
        f.append(f"**{sym.upper().replace('_', '/')}**")
        for s in ("train", "val", "test"):
            ss = rep["sig_symbol"][s][sym]
            f.append(f"- {SEG[s]}: {sig_short(ss['ALL'])}")
            f.append(f"  - Long: {sig_short(ss['LONG'])}")
            f.append(f"  - Short: {sig_short(ss['SHORT'])}")
    f += ["", "### پورتفوی (قوانین ریسک)"] + [seg_line(SEG[s], auto["segments"][s]) for s in ("train", "val", "test")]
    if auto["skipped"]["test"]:
        f.append("ردشده توسط قوانین ریسک در آزمون: " + "، ".join(f"{a}={b}" for a, b in auto["skipped"]["test"].items()))
    f += ["", "### شمارش برخوردها (تعداد برخورد با تعداد معامله یکی نیست)"]
    for sym in rep["symbols"]:
        c = rep["counts"][sym]
        f.append(f"**{sym.upper().replace('_', '/')}**: کل برخوردها {c['total']} (Long {c['longs']} / Short {c['shorts']}) | کندل واکنشی: برگشتی {c['react_reversal']}، شکستی {c['react_breakout']}، خنثی {c['react_neutral']} | رد شده به‌خاطر هزینه {c['cost_rejected']}")
        f.append("  اگر روی هر برخورد معامله می‌گرفتیم: " + " | ".join(f"{LABEL_FA[k]} {c[k]}" for k in LABEL_FA) + f" | ناتمام/گپ {c['pending_or_gap']} | سقف‌وحد هم‌کندل (ضرر) {c['ambiguous']}")
    f += ["", "نمونه‌های سطح سیگنال همبسته‌اند (برخوردهای پشت‌سرهم)؛ نتایج را تقریبی بدانید. احتمال مدل با Win Rate واقعی یکی نیست."]
    return short, "\n".join(f)
