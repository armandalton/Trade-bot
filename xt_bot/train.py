"""Training, honest evaluation and acceptance decision."""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import CFG, DATA_DIR, INTERVAL_MS
from .data import fetch_klines, closed_only, validate_candles
from .dataset import events_for_symbol, assign_splits, trade_set
from .labeling import RESOLVED
from .model import train_model, predict_proba, pick_threshold, save_bundle, MODEL_PATH
from .backtest import simulate, summarize
from .util import log, write_json, read_json
from . import report

META_FILE = DATA_DIR / "model_meta.json"
LABELS = ("valid_reversal", "weak_reversal", "true_break", "false_break", "no_setup")


def pattern_tag(r):
    slope = "شیب همسو" if r["sma_slope_atr"] > 0.1 else ("شیب خلاف" if r["sma_slope_atr"] < -0.1 else "SMA تخت")
    vol = "حجم کم" if r["rel_vol_r"] < 0.8 else ("حجم زیاد" if r["rel_vol_r"] > 1.5 else "حجم عادی")
    rx = "واکنش قوی" if r["r_body_ratio"] >= 0.6 else ("واکنش ضعیف" if r["r_body_ratio"] < 0.4 else "واکنش متوسط")
    return f"{slope} + {vol} + {rx}"


def pattern_stats(df, min_n=15):
    if len(df) == 0:
        return [], []
    d = df.copy()
    d["pattern"] = d.apply(pattern_tag, axis=1)
    g = d.groupby("pattern").agg(n=("net_r", "size"), exp_r=("net_r", "mean"),
                                 win=("label", lambda s: float((s == "valid_reversal").mean())))
    g = g[g["n"] >= min_n].sort_values("exp_r", ascending=False)
    rows = [(i, int(r.n), float(r.win), float(r.exp_r)) for i, r in g.iterrows()]
    return rows[:2], rows[::-1][:2]


def build_all(fetcher=fetch_klines, cfg=CFG, now_ms=None):
    now_ms = int(now_ms or time.time() * 1000)
    start = now_ms - cfg["train_days"] * 86_400_000
    frames, notes, quality = [], [], {}
    for sym in cfg["symbols"]:
        try:
            log(f"{sym}: downloading {cfg['train_days']} days of 1m candles ...")
            df = closed_only(fetcher(sym, start, now_ms, "1m"), now_ms)
            q = validate_candles(df)
            quality[sym] = q
            log(f"{sym}: {q}")
            if q["coverage"] < 0.90:
                notes.append(f"{sym}: پوشش داده فقط {q['coverage']*100:.0f}% بود؛ نماد کنار گذاشته شد")
                continue
            frames.append(events_for_symbol(df, sym, cfg))
        except Exception as e:
            notes.append(f"{sym}: دریافت/پردازش داده ناموفق ({type(e).__name__}: {str(e)[:120]})")
            log(notes[-1])
    return frames, notes, quality


def run_training(fetcher=fetch_klines, cfg=CFG, now_ms=None, save=True, meta_file=META_FILE, state=None):
    frames, notes, quality = build_all(fetcher, cfg, now_ms)
    if not frames:
        raise RuntimeError("هیچ نمادی داده قابل استفاده نداشت: " + " | ".join(notes))
    ev = pd.concat(frames, ignore_index=True)
    ev, bounds = assign_splits(ev, cfg)
    sets = {s: trade_set(ev, s) for s in ("train", "val", "test")}
    reasons = list(notes)
    approved, bundle, thr_info = False, None, None

    if len(sets["train"]) < cfg["min_train_setups"] or len(sets["val"]) < 100:
        reasons.append(f"داده کافی نیست: setup آموزشی={len(sets['train'])}، اعتبارسنجی={len(sets['val'])}")
    else:
        bundle = train_model(sets["train"], sets["val"])
        for s in sets:
            sets[s] = sets[s].assign(prob=predict_proba(bundle, sets[s]))
        thr_info = pick_threshold(sets["val"]["prob"].values, sets["val"]["net_r"].values, cfg["min_val_trades"])
        if thr_info is None:
            reasons.append("روی اعتبارسنجی هیچ آستانه‌ای حداقل ۳۰ معامله نداد")

    results = {"segments": {}, "baseline": {}, "by_symbol": {}, "skipped": {}}
    thr = thr_info[0] if thr_info else None
    for s, d in sets.items():
        base_exec, base_m, _ = simulate(d, cfg)
        results["baseline"][s] = base_m
        if thr is not None:
            sel = d[d["prob"] >= thr]
            exe, m, sk = simulate(sel, cfg)
        else:
            exe, m, sk = pd.DataFrame(), summarize(None, cfg["capital"]), {}
        results["segments"][s] = m
        results["skipped"][s] = sk
        results["by_symbol"][s] = {}
        for sym in sorted(ev["symbol"].unique()):
            for side in ("ALL", "LONG", "SHORT"):
                e = exe
                if len(e):
                    e = e[e["symbol"] == sym] if "symbol" in e else e
                    if side != "ALL":
                        e = e[e["side_name"] == side]
                results["by_symbol"][s].setdefault(sym, {})[side] = summarize(e, cfg["capital"])

    test_m = results["segments"]["test"]
    if thr is not None:
        if test_m["trades"] < cfg["min_test_trades"]:
            reasons.append(f"معاملات آزمون {test_m['trades']} است (< {cfg['min_test_trades']}): نمونه آماری کافی نیست")
        if test_m["win_rate"] is None or test_m["win_rate"] < cfg["min_win_rate"]:
            wr = "—" if test_m["win_rate"] is None else f"{test_m['win_rate']*100:.1f}%"
            reasons.append(f"Win Rate آزمون {wr} (< {cfg['min_win_rate']*100:.0f}%)")
        if test_m["profit_factor"] is None or test_m["profit_factor"] < cfg["min_profit_factor"]:
            pf = "—" if test_m["profit_factor"] is None else f"{test_m['profit_factor']:.2f}"
            reasons.append(f"Profit Factor آزمون {pf} (< {cfg['min_profit_factor']})")
        if test_m["max_dd_pct"] > cfg["max_test_drawdown"]:
            reasons.append(f"Drawdown آزمون {test_m['max_dd_pct']*100:.1f}% (> {cfg['max_test_drawdown']*100:.0f}%)")
        if test_m["net_pnl"] <= 0:
            reasons.append("سود خالص آزمون پس از کارمزد منفی یا صفر است")
        approved = not [r for r in reasons if r not in notes]

    enabled = []
    if approved:
        for sym, d in results["by_symbol"]["test"].items():
            m = d["ALL"]
            if m["trades"] >= 10 and m["net_pnl"] > 0 and (m["profit_factor"] or 0) >= 1.0:
                enabled.append(sym)
        if not enabled:
            approved = False
            reasons.append("هیچ نمادی به‌تنهایی روی آزمون مثبت نبود")

    counts = {}
    for sym, g in ev.groupby("symbol"):
        c = {k: int((g["label"] == k).sum()) for k in LABELS}
        c["total"] = int(len(g))
        c["no_setup_cost"] = int(((g["label"] == "no_setup") & (g["no_setup_reason"] == "cost")).sum())
        c["pending_or_gap"] = int(g["label"].isin(("pending", "gap")).sum())
        c["ambiguous"] = int(g["ambiguous"].sum())
        c["longs"] = int((g["side_name"] == "LONG").sum())
        c["shorts"] = int((g["side_name"] == "SHORT").sum())
        counts[sym] = c

    trainval = pd.concat([sets["train"], sets["val"]], ignore_index=True)
    patterns = {sym: pattern_stats(trainval[trainval["symbol"] == sym]) for sym in counts}
    patterns["ALL"] = pattern_stats(trainval, min_n=30)

    brier = None
    if bundle is not None and len(sets["test"]):
        p = sets["test"]["prob"].values
        y = (sets["test"]["label"] == "valid_reversal").astype(int).values
        brier = float(np.mean((p - y) ** 2))
    meta = {"created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "approved": bool(approved),
            "threshold": thr, "enabled_symbols": enabled, "reasons": reasons, "calibrated": bool(bundle and bundle["calibrator"] is not None),
            "bounds_ms": bounds, "counts": counts, "segments": results["segments"], "baseline": results["baseline"],
            "test_brier": brier, "data_quality": quality, "validation_threshold_info": thr_info,
            "setup_sizes": {k: int(len(v)) for k, v in sets.items()},
            "purged": int(ev["purged"].sum()), "costs": {"fee_rate": cfg["fee_rate"], "slippage_rate": cfg["slippage_rate"],
                                                          "max_cost_r": cfg["max_cost_r"]}}
    if save:
        if bundle is not None:
            save_bundle(bundle)
        write_json(meta_file, meta)
    text = report.build_report(meta, results, counts, patterns, bounds, state)
    if save:
        (DATA_DIR / "report_fa.md").write_text(text, encoding="utf-8")
    return meta, results, text
