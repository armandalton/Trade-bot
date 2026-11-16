"""Training, honest evaluation and acceptance decision.

For every touch of SMA64 two candidate trades exist (REVERSAL / BREAKOUT). One model per kind learns
P(target before stop); the better candidate is selected per touch (model.select_best). The whole procedure
is repeated for several reward:risk ratios (what-if table); the live ratio is chosen on VALIDATION data only
and must obey the project rule RR >= 2. Hours of the day are analysed the same way (validation picks, test confirms)."""
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import CFG, DATA_DIR
from .data import fetch_klines, closed_only, validate_candles
from .dataset import assign_splits, trade_set
from .features import prepare, extract_events, with_rr
from .labeling import label_events
from .model import train_model, predict_proba, pick_threshold, save_bundle, select_best
from .backtest import simulate, summarize, signal_stats
from .util import log, write_json
from . import report, hours as hours_mod

META_FILE = DATA_DIR / "model_meta.json"
KINDS = ("REVERSAL", "BREAKOUT")
SPLITS = ("train", "val", "test")
RR_GRID = (1.0, 1.5, 2.0, 3.0)
MIN_RR_RULE = 2.0
ALL_LABELS = ("valid_reversal", "weak_reversal", "true_break", "false_break",
              "valid_breakout", "weak_breakout", "failed_breakout", "whipsaw")


def pattern_tag(r):
    k = 1.0 if r["kind"] == "REVERSAL" else -1.0   # "aligned" = aligned with the TRADE direction
    sl = k * r["sma_slope_atr"]
    slope = "شیب همسو" if sl > 0.1 else ("شیب خلاف" if sl < -0.1 else "SMA تخت")
    vol = "حجم کم" if r["rel_vol_r"] < 0.8 else ("حجم زیاد" if r["rel_vol_r"] > 1.5 else "حجم عادی")
    rx = "کندل قوی" if r["r_body_ratio"] >= 0.6 else ("کندل ضعیف" if r["r_body_ratio"] < 0.4 else "کندل متوسط")
    return f"{slope} + {vol} + {rx}"


def pattern_stats(df, min_n=15):
    if len(df) == 0:
        return [], []
    d = df.copy()
    d["pattern"] = d.apply(pattern_tag, axis=1)
    d["win"] = (d["net_r"] > 0).astype(float)
    g = d.groupby("pattern").agg(n=("net_r", "size"), exp_r=("net_r", "mean"), win=("win", "mean"))
    g = g[g["n"] >= min_n].sort_values("exp_r", ascending=False)
    rows = [(i, int(r.n), float(r.win), float(r.exp_r)) for i, r in g.iterrows()]
    return rows[:2], rows[::-1][:2]


def build_all(fetcher=fetch_klines, cfg=CFG, now_ms=None, rrs=RR_GRID):
    """Download once per symbol; label the same candidates for every reward:risk ratio."""
    now_ms = int(now_ms or time.time() * 1000)
    start = now_ms - cfg["train_days"] * 86_400_000
    frames, notes, quality = {rr: [] for rr in rrs}, [], {}
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
            x = prepare(df, cfg)
            base = extract_events(x, sym, cfg)
            for rr in rrs:
                frames[rr].append(label_events(x, with_rr(base, rr), cfg))
        except Exception as e:
            notes.append(f"{sym}: دریافت/پردازش داده ناموفق ({type(e).__name__}: {str(e)[:120]})")
            log(notes[-1])
    return frames, notes, quality


def _sig_by_symbol(sel, symbols):
    out = {}
    for sym in symbols:
        ss = sel[sel["symbol"] == sym] if len(sel) else sel
        out[sym] = {"ALL": signal_stats(ss),
                    "LONG": signal_stats(ss[ss["side_name"] == "LONG"] if len(ss) else ss),
                    "SHORT": signal_stats(ss[ss["side_name"] == "SHORT"] if len(ss) else ss)}
    return out


def _gate_reasons(sig, port, skipped, cfg):
    r = []
    if sig["n"] < cfg["min_test_trades"]:
        r.append(f"فقط {sig['n']} سیگنال در آزمون (حداقل {cfg['min_test_trades']} لازم است)")
    if sig["win_rate"] is None or sig["win_rate"] < cfg["min_win_rate"]:
        r.append("Win Rate آزمون " + ("—" if sig["win_rate"] is None else f"{sig['win_rate']*100:.1f}%") + f" (کمتر از {cfg['min_win_rate']*100:.0f}%)")
    if sig["pf"] is None or sig["pf"] < cfg["min_profit_factor"]:
        r.append("Profit Factor آزمون " + ("—" if sig["pf"] is None else f"{sig['pf']:.2f}") + f" (کمتر از {cfg['min_profit_factor']})")
    if sig["mean_r"] is None or sig["mean_r"] <= 0:
        r.append("میانگین نتیجه هر معامله پس از کارمزد منفی یا صفر است")
    if port["max_dd_pct"] > cfg["max_test_drawdown"] or skipped.get("dd_breaches"):
        r.append(f"Drawdown پورتفوی آزمون {port['max_dd_pct']*100:.1f}% (سقف {cfg['max_test_drawdown']*100:.0f}%)")
    return r


def evaluate_rr(ev, cfg, rr):
    ev, bounds = assign_splits(ev, cfg)
    sets = {s: trade_set(ev, s) for s in SPLITS}
    for s in SPLITS:
        sets[s] = sets[s].assign(prob=0.0)
    res = {"rr": rr, "bounds": bounds, "ev": ev, "sets": sets, "bundles": {}, "thresholds": {}, "kinds": {}, "kind_reasons": []}
    for kind in KINDS:
        masks = {s: (sets[s]["kind"] == kind).values for s in SPLITS}
        ks = {s: sets[s][masks[s]] for s in SPLITS}
        info = {"n_setups": {s: int(len(ks[s])) for s in SPLITS}, "thr": None, "reasons": [], "signals": {}, "brier": None,
                "calibrated": False}
        res["bundles"][kind] = None
        if len(ks["train"]) < cfg["min_train_setups"] or len(ks["val"]) < 100:
            info["reasons"].append(f"داده کافی نیست (آموزش {len(ks['train'])}، اعتبارسنجی {len(ks['val'])})")
        else:
            bundle = train_model(ks["train"], ks["val"])
            res["bundles"][kind] = bundle
            info["calibrated"] = bundle["calibrator"] is not None
            for s in SPLITS:
                arr = sets[s]["prob"].values.copy()
                arr[masks[s]] = predict_proba(bundle, ks[s])
                sets[s]["prob"] = arr
            vk = sets["val"][masks["val"]]
            ti = pick_threshold(vk["prob"].values, vk["net_r"].values, cfg["min_val_trades"])
            if ti is None:
                info["reasons"].append(f"روی اعتبارسنجی هیچ آستانه‌ای حداقل {cfg['min_val_trades']} سیگنال نداد")
            elif ti[1] <= 0:
                info["reasons"].append(f"روی اعتبارسنجی هم میانگین نتیجه منفی بود ({ti[1]:+.2f}R)؛ این حالت غیرفعال شد")
            else:
                info["thr"] = ti[0]
                res["thresholds"][kind] = ti[0]
            tk = sets["test"][masks["test"]]
            if len(tk):
                y = (tk["result"] == "tp").astype(int).values
                info["brier"] = float(np.mean((tk["prob"].values - y) ** 2))
        for s in SPLITS:
            k_all = sets[s][masks[s]]
            k_sel = k_all[k_all["prob"] >= info["thr"]] if info["thr"] is not None else k_all.iloc[0:0]
            info["signals"][s] = {"model": signal_stats(k_sel), "all": signal_stats(k_all)}
        res["kinds"][kind] = info
        res["kind_reasons"] += [f"[{report.KIND_FA[kind]}] {r}" for r in info["reasons"]]

    auto = {"segments": {}, "skipped": {}, "signals": {}, "by_kind": {}}
    sel = {}
    for s in SPLITS:
        sel[s] = select_best(sets[s], res["thresholds"])
        _, m, sk = simulate(sel[s], cfg)
        auto["segments"][s], auto["skipped"][s], auto["signals"][s] = m, sk, signal_stats(sel[s])
        auto["by_kind"][s] = {k: signal_stats(sel[s][sel[s]["kind"] == k] if len(sel[s]) else sel[s]) for k in KINDS}
    auto["all_setups"] = {s: signal_stats(sets[s]) for s in SPLITS}
    res["auto"], res["sel"] = auto, sel
    ts = auto["signals"]["test"]
    c = float(sel["test"]["cost_r"].mean()) if len(sel["test"]) else None
    auto["cost_r"], auto["break_even"] = c, (None if c is None else (1 + c) / (rr + 1))
    if res["thresholds"]:
        res["gate_reasons"] = _gate_reasons(ts, auto["segments"]["test"], auto["skipped"]["test"], cfg)
    else:
        res["gate_reasons"] = ["هیچ‌کدام از دو حالت (برگشت/شکست) روی اعتبارسنجی آستانه سودده نداشت"]
    res["gates_ok"] = bool(res["thresholds"]) and not res["gate_reasons"]
    return res


def run_training(fetcher=fetch_klines, cfg=CFG, now_ms=None, save=True, meta_file=META_FILE, state=None, rrs=RR_GRID):
    frames, notes, quality = build_all(fetcher, cfg, now_ms, rrs)
    if not frames[rrs[0]]:
        raise RuntimeError("هیچ نمادی داده قابل استفاده نداشت: " + " | ".join(notes))
    results = {}
    for rr in rrs:
        log(f"evaluating RR={rr} ...")
        results[rr] = evaluate_rr(pd.concat(frames[rr], ignore_index=True), cfg, rr)

    # live RR: chosen on VALIDATION only, restricted by the project rule RR >= 2
    elig = [rr for rr in rrs if rr >= MIN_RR_RULE and results[rr]["thresholds"]
            and results[rr]["auto"]["signals"]["val"]["n"] >= 50 and results[rr]["auto"]["signals"]["val"]["mean_r"] is not None]
    chosen_rr = max(elig, key=lambda r: results[r]["auto"]["signals"]["val"]["mean_r"]) if elig else (MIN_RR_RULE if MIN_RR_RULE in results else rrs[0])
    R = results[chosen_rr]
    ev, bounds = R["ev"], R["bounds"]
    symbols = sorted(ev["symbol"].unique())
    reasons = list(notes) + list(R["kind_reasons"]) + list(R["gate_reasons"] if R["thresholds"] else [])
    if not R["thresholds"]:
        reasons.append(R["gate_reasons"][0])

    enabled_symbols = []
    if R["gates_ok"]:
        for sym in symbols:
            d = R["sel"]["test"]
            st = signal_stats(d[d["symbol"] == sym] if len(d) else d)
            if st["n"] >= 20 and st["mean_r"] is not None and st["mean_r"] > 0:
                enabled_symbols.append(sym)
        if not enabled_symbols:
            reasons.append("هیچ نمادی به‌تنهایی روی آزمون مثبت نبود")
    approved = bool(R["gates_ok"] and enabled_symbols)

    touches = ev[ev["kind"] == "REVERSAL"]
    if R["thresholds"]:
        hrs = hours_mod.analyze(touches, R["sel"])
        hrs["source"] = "model"
    else:  # no active model: judge hours with the simple rule "trade in the direction the reaction candle shows"
        rule_sel = {s: R["sets"][s][R["sets"][s]["kind"] == R["sets"][s]["reaction"]] for s in SPLITS}
        hrs = hours_mod.analyze(touches, rule_sel)
        hrs["source"] = "rule"
    hour_filter = bool(approved and hrs["stable"])

    counts = {}
    for sym, g in ev.groupby("symbol"):
        t = g[g["kind"] == "REVERSAL"]
        c = {k: int((g["label"] == k).sum()) for k in ALL_LABELS}
        c.update({"total": int(len(t)), "react_reversal": int((t["reaction"] == "REVERSAL").sum()),
                  "react_breakout": int((t["reaction"] == "BREAKOUT").sum()), "react_neutral": int((t["reaction"] == "NEUTRAL").sum()),
                  "cost_rejected": int(((g["no_setup_reason"] == "cost") & (g["kind"] == "REVERSAL")).sum()),
                  "pending_or_gap": int(t["label"].isin(("pending", "gap")).sum()), "ambiguous": int(g["ambiguous"].sum()),
                  "longs": int((t["approach_side"] == "LONG").sum()), "shorts": int((t["approach_side"] == "SHORT").sum())})
        counts[sym] = c
    trainval = pd.concat([R["sets"]["train"], R["sets"]["val"]], ignore_index=True)
    patterns = {k: {"ALL": pattern_stats(trainval[trainval["kind"] == k], min_n=30)} for k in KINDS}
    sig_symbol = {s: _sig_by_symbol(R["sel"][s], symbols) for s in SPLITS}

    rr_table = {}
    for rr, r in results.items():
        rr_table[str(rr)] = {"rr": rr, "thresholds": r["thresholds"], "gates_ok": r["gates_ok"], "gate_reasons": r["gate_reasons"],
                             "signals": r["auto"]["signals"], "segments": r["auto"]["segments"], "break_even": r["auto"]["break_even"],
                             "eligible": rr >= MIN_RR_RULE, "chosen": rr == chosen_rr}
    params = {k: cfg[k] for k in ("horizon", "min_stop_pct", "min_stop_atr", "max_cost_r", "stop_buffer_atr", "fee_rate", "slippage_rate")}
    params["rr"] = chosen_rr
    meta = {"params": params, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "approved": approved,
            "chosen_rr": chosen_rr, "thresholds": R["thresholds"], "enabled_kinds": sorted(R["thresholds"]),
            "enabled_symbols": enabled_symbols, "hour_filter": hour_filter,
            "trade_hours_utc": hrs["best"]["hours"] if hour_filter else [],
            "best_window": {"label": hrs["label"], "basis": hrs["basis"], "stable": hrs["stable"], "hours_utc": hrs["best"]["hours"]},
            "reasons": reasons, "bounds_ms": bounds, "counts": counts, "data_quality": quality, "rr_table": rr_table,
            "kinds": {k: R["kinds"][k] for k in KINDS}, "auto": R["auto"], "purged": int(ev["purged"].sum()),
            "costs": {"fee_rate": cfg["fee_rate"], "slippage_rate": cfg["slippage_rate"], "max_cost_r": cfg["max_cost_r"]}}
    rep = {"R": R, "hours": hrs, "counts": counts, "patterns": patterns, "sig_symbol": sig_symbol, "bounds": bounds,
           "results": results, "symbols": symbols}
    if save:
        save_bundle(R["bundles"])
        write_json(meta_file, meta)
    short, full = report.build_report(meta, rep, state, cfg)
    if save:
        (DATA_DIR / "report_fa.md").write_text(full, encoding="utf-8")
    return meta, rep, short
