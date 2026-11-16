"""Offline tests. They use SYNTHETIC data and mocked network/Telegram: they prove the logic,
not that XT or Telegram are reachable (that is what `python bot.py check` is for)."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from xt_bot.config import CFG, INTERVAL_MS
from xt_bot import data, features, labeling, dataset, risk, backtest, live, paper, report
from xt_bot.model import train_model, predict_proba, pick_threshold

TCFG = dict(CFG, max_cost_r=50.0)   # synthetic prices are far more volatile than real costs allow
T0 = 1_700_000_000_000 - (1_700_000_000_000 % 60_000)


def synth(n=3000, seed=1, start=T0, price=100.0, vol=0.0008):
    rng = np.random.default_rng(seed)
    close = price * np.exp(np.cumsum(rng.normal(0, vol, n)))
    open_ = np.r_[price, close[:-1]]
    spread = np.abs(rng.normal(0, vol * 0.6, n)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    return pd.DataFrame({"time": start + np.arange(n) * INTERVAL_MS, "open": open_, "high": high, "low": low,
                         "close": close, "volume": rng.uniform(50, 150, n)})


def fake_rows(df):
    return [{"t": int(r.time), "o": str(r.open), "h": str(r.high), "l": str(r.low), "c": str(r.close),
             "a": str(r.volume), "v": "1"} for r in df.itertuples()]


class TestData(unittest.TestCase):
    def test_parse_order_dupes_seconds(self):
        df = synth(10)
        rows = fake_rows(df)[::-1] + fake_rows(df)[:3]
        for r in rows:
            r["t"] = r["t"] // 1000  # seconds
        out = data.parse_klines(rows)
        self.assertEqual(len(out), 10)
        self.assertTrue(out["time"].is_monotonic_increasing)
        self.assertEqual(int(out["time"].iloc[0]), int(df["time"].iloc[0]))

    def test_pagination_with_fake_session(self):
        df = synth(4000)

        class S:
            calls = 0
            def get(self, url, params, timeout):
                S.calls += 1
                sub = df[(df.time >= params["startTime"]) & (df.time <= params["endTime"])]
                class R:
                    status_code = 200
                    def raise_for_status(self): pass
                    def json(self_): return {"returnCode": 0, "result": fake_rows(sub)}
                return R()
        out = data.fetch_klines("btc_usdt", int(df.time.iloc[0]), int(df.time.iloc[-1]), session=S(), pause=0)
        self.assertEqual(len(out), 4000)
        self.assertGreater(S.calls, 2)
        q = data.validate_candles(out)
        self.assertTrue(q["ok"])
        self.assertEqual(q["missing"], 0)

    def test_closed_only_drops_forming_candle(self):
        df = synth(5)
        now = int(df.time.iloc[-1]) + 30_000  # last candle still forming
        self.assertEqual(len(data.closed_only(df, now)), 4)
        self.assertEqual(len(data.closed_only(df, now + 30_000)), 5)


class TestFeatures(unittest.TestCase):
    def test_sma64_matches_manual(self):
        df = synth(300)
        x = features.prepare(df)
        i = 200
        self.assertAlmostEqual(x["sma"].iloc[i], df["close"].iloc[i - 63:i + 1].mean(), places=10)
        self.assertTrue(np.isnan(x["sma"].iloc[62]) and not np.isnan(x["sma"].iloc[63]))

    def test_touch_direction_and_reaction(self):
        # flat at 100 for 150 candles, drift up above SMA, dip to touch it, bullish reaction -> LONG setup
        n = 200
        close = np.full(n, 100.0)
        close[150:] = 101.0
        o = close.copy(); h = close + 0.05; l = close - 0.05
        df = pd.DataFrame({"time": T0 + np.arange(n) * INTERVAL_MS, "open": o, "high": h, "low": l, "close": close,
                           "volume": np.full(n, 100.0)})
        x = features.prepare(df)
        sma = x["sma"].iloc[180]
        # touch candle 180 reaches SMA from above, reaction candle 181 is bullish and closes above SMA
        df.loc[180, ["open", "high", "low", "close"]] = [101.0, 101.05, sma - 0.02, 100.9]
        df.loc[181, ["open", "high", "low", "close"]] = [100.9, 101.6, 100.88, 101.5]
        ev = features.build_events(df, "t")
        both = ev[ev["idx"] == 181]
        self.assertEqual(sorted(both["kind"]), ["BREAKOUT", "REVERSAL"])      # two candidates per touch
        self.assertEqual(set(both["reaction"]), {"REVERSAL"})                 # the reaction candle bounced
        row = both[both["kind"] == "REVERSAL"]
        self.assertEqual(row["side_name"].iat[0], "LONG")
        self.assertEqual(both[both["kind"] == "BREAKOUT"]["side_name"].iat[0], "SHORT")
        self.assertTrue(bool(row["setup"].iat[0]))
        self.assertGreater(row["entry"].iat[0], row["sl"].iat[0])
        self.assertAlmostEqual(row["tp1"].iat[0] - row["entry"].iat[0], 2 * row["distance"].iat[0], places=9)

    def test_breakout_kind_trades_against_approach(self):
        n = 200
        close = np.full(n, 100.0); close[150:] = 101.0
        df = pd.DataFrame({"time": T0 + np.arange(n) * INTERVAL_MS, "open": close.copy(), "high": close + 0.05,
                           "low": close - 0.05, "close": close, "volume": np.full(n, 100.0)})
        sma = features.prepare(df)["sma"].iloc[180]
        # price comes from above, touches SMA, then the reaction candle CLOSES BELOW it -> breakout SHORT
        df.loc[180, ["open", "high", "low", "close"]] = [101.0, 101.05, sma - 0.02, 100.9]
        df.loc[181, ["open", "high", "low", "close"]] = [100.9, 100.92, 100.2, 100.3]
        ev = features.build_events(df, "t")
        self.assertEqual(set(ev[ev["idx"] == 181]["reaction"]), {"BREAKOUT"})
        row = ev[(ev["idx"] == 181) & (ev["kind"] == "BREAKOUT")].iloc[0]
        self.assertEqual(row["kind"], "BREAKOUT")
        self.assertEqual(row["side_name"], "SHORT")
        self.assertEqual(row["approach_side"], "LONG")
        self.assertGreater(row["sl"], row["entry"])
        self.assertLess(row["tp1"], row["entry"])
        self.assertAlmostEqual(row["entry"] - row["tp1"], 2 * row["distance"], places=9)

    def test_no_lookahead_features_identical_on_truncated_data(self):
        df = synth(2500, seed=5)
        full = features.build_events(df, "t")
        self.assertGreater(len(full), 30)
        for k in (900, 1500, 2100, 2499):
            part = features.build_events(df.iloc[:k + 1], "t")
            cut = full[full["idx"] <= k].reset_index(drop=True)
            self.assertEqual(len(part), len(cut))
            cols = features.FEATURES + ["entry", "sl", "tp1", "distance"]
            np.testing.assert_allclose(part[cols].values.astype(float), cut[cols].values.astype(float), rtol=1e-9, atol=1e-12)

    def test_future_change_does_not_change_past_features(self):
        df = synth(1800, seed=9)
        a = features.build_events(df, "t")
        df2 = df.copy()
        df2.loc[1500:, ["open", "high", "low", "close"]] *= 1.07  # alter the future only
        b = features.build_events(df2, "t")
        cols = features.FEATURES
        aa, bb = a[a["idx"] < 1500], b[b["idx"] < 1500]
        np.testing.assert_allclose(aa[cols].values.astype(float), bb[cols].values.astype(float))


def _event(side="LONG", entry=100.0, dist=1.0, idx=10, kind="REVERSAL"):
    sl = entry - dist if side == "LONG" else entry + dist
    tp = entry + 2 * dist if side == "LONG" else entry - 2 * dist
    return pd.DataFrame([{"idx": idx, "tradable": True, "kind": kind, "side_name": side, "entry": entry, "sl": sl, "tp1": tp,
                          "distance": dist}])


def _path(highs, lows, closes=None, n_pre=11, sma=100.0):
    n = n_pre + len(highs)
    h = np.r_[np.full(n_pre, 100.0), highs]; l = np.r_[np.full(n_pre, 100.0), lows]
    c = np.r_[np.full(n_pre, 100.0), closes if closes is not None else np.full(len(highs), 100.0)]
    return pd.DataFrame({"time": T0 + np.arange(n) * INTERVAL_MS, "open": c, "high": h, "low": l, "close": c,
                         "volume": 1.0, "sma": sma, "gap_bad": 0.0})


class TestLabeling(unittest.TestCase):
    cfg = dict(CFG, horizon=5)

    def run_one(self, highs, lows, closes=None, side="LONG"):
        x = _path(highs, lows, closes)
        return labeling.label_events(x, _event(side), self.cfg).iloc[0]

    def test_tp_first(self):
        r = self.run_one([100.5, 102.5, 100, 100, 100], [99.5, 100.2, 100, 100, 100])
        self.assertEqual(r["label"], "valid_reversal"); self.assertEqual(r["result"], "tp")

    def test_sl_first(self):
        r = self.run_one([100.2, 100.1, 103, 100, 100], [99.5, 98.5, 100, 100, 100])
        self.assertIn(r["label"], ("true_break", "false_break")); self.assertEqual(r["result"], "sl")
        self.assertLess(r["net_r"], -1.0)  # costs make a stop worse than -1R

    def test_same_candle_is_conservative_stop(self):
        r = self.run_one([102.5, 100, 100, 100, 100], [98.5, 100, 100, 100, 100])
        self.assertEqual(r["result"], "sl"); self.assertTrue(bool(r["ambiguous"]))

    def test_horizon_is_weak(self):
        r = self.run_one([100.5] * 5, [99.5] * 5)
        self.assertEqual(r["label"], "weak_reversal")

    def test_true_vs_false_break(self):
        closes = [99.6, 99.5, 99.4, 99.3, 99.2]  # below SMA(100) for >=3 closes then stop
        t = self.run_one([100.0, 100, 100, 100, 100], [99.5, 99.4, 99.3, 98.5, 98.4], closes)
        self.assertEqual(t["label"], "true_break")
        f = self.run_one([100.0, 100, 100, 100, 100], [98.5, 99.9, 99.9, 99.9, 99.9], [100.2] * 5)
        self.assertEqual(f["label"], "false_break")

    def test_short_side(self):
        r = self.run_one([100.5, 100.2, 100, 100, 100], [99.5, 97.5, 100, 100, 100], side="SHORT")
        self.assertEqual(r["label"], "valid_reversal")

    def test_breakout_label_names(self):
        ev = _event("LONG"); ev["kind"] = "BREAKOUT"
        tp = labeling.label_events(_path([100.5, 102.5, 100, 100, 100], [99.5, 100.2, 100, 100, 100]), ev, self.cfg).iloc[0]
        self.assertEqual(tp["label"], "valid_breakout")
        wk = labeling.label_events(_path([100.5] * 5, [99.5] * 5), ev, self.cfg).iloc[0]
        self.assertEqual(wk["label"], "weak_breakout")
        fl = labeling.label_events(_path([100.0] * 5, [99.5, 99.4, 99.3, 98.5, 98.4], [99.6, 99.5, 99.4, 99.3, 99.2]), ev, self.cfg).iloc[0]
        self.assertEqual(fl["label"], "failed_breakout")
        wh = labeling.label_events(_path([100.0] * 5, [98.5, 99.9, 99.9, 99.9, 99.9], [100.2] * 5), ev, self.cfg).iloc[0]
        self.assertEqual(wh["label"], "whipsaw")

    def test_pending_when_future_missing(self):
        x = _path([100.5, 100.5], [99.5, 99.5])
        r = labeling.label_events(x, _event(), self.cfg).iloc[0]
        self.assertEqual(r["label"], "pending")


class TestSplitAndModel(unittest.TestCase):
    def build(self):
        frames = [dataset.events_for_symbol(synth(6000, seed=s), f"s{s}", TCFG) for s in (1, 2, 3)]
        return pd.concat(frames, ignore_index=True)

    def test_chronological_split_and_purge(self):
        ev, b = dataset.assign_splits(self.build())
        tr, va, te = (dataset.trade_set(ev, s) for s in ("train", "val", "test"))
        self.assertTrue(len(tr) and len(va) and len(te))
        self.assertLess(tr["signal_time"].max(), b["train_end"])
        self.assertGreaterEqual(va["signal_time"].min(), b["train_end"])
        self.assertLess(va["signal_time"].max(), b["val_end"])
        self.assertGreaterEqual(te["signal_time"].min(), b["val_end"])
        self.assertLess(tr["exit_time"].max(), b["train_end"])   # purged: no outcome crosses the boundary
        self.assertLess(va["exit_time"].max(), b["val_end"])
        frac = (b["train_end"] - b["start"]) / (b["end"] - b["start"])
        self.assertAlmostEqual(frac, 0.6, places=6)

    def test_model_trains_and_probabilities_valid(self):
        ev, _ = dataset.assign_splits(self.build())
        tr, va = dataset.trade_set(ev, "train"), dataset.trade_set(ev, "val")
        bundle = train_model(tr, va)
        p = predict_proba(bundle, va)
        self.assertTrue(((p >= 0) & (p <= 1)).all())
        self.assertIsNone(pick_threshold(np.array([0.9, 0.9]), np.array([1.0, 1.0]), 30))


class TestRiskAndBacktest(unittest.TestCase):
    def test_sizing_risks_half_percent(self):
        qty, risk_usd, scale = risk.size_position(2000, 100.0, 2.0, CFG)
        self.assertAlmostEqual(risk_usd, 10.0); self.assertAlmostEqual(qty * 2.0, 10.0); self.assertEqual(scale, 1.0)
        qty2, risk2, scale2 = risk.size_position(2000, 100.0, 0.01, CFG)  # tiny stop -> leverage cap applies
        self.assertLessEqual(qty2 * 100.0, 2000 * CFG["max_leverage"] + 1e-6); self.assertLess(scale2, 1.0)

    def test_locks(self):
        self.assertIsNone(risk.global_lock(2000, 2000, 2000, CFG))
        self.assertEqual(risk.global_lock(1969, 2000, 2000, CFG), "daily_loss")   # -31 > 30
        self.assertEqual(risk.global_lock(1879, 2000, 1900, CFG), "drawdown")      # -121 > 120

    def _trades(self, rows):
        return pd.DataFrame(rows, columns=["symbol", "side_name", "entry_time", "exit_time", "entry", "distance", "net_r", "label"])

    def test_max_three_concurrent_and_one_per_symbol(self):
        rows = [(s, "LONG", 1000 + i, 10**9, 100.0, 1.0, 2.0, "valid_reversal") for i, s in enumerate(["a", "b", "c", "d", "e"])]
        rows.append(("a", "SHORT", 1100, 10**9, 100.0, 1.0, 2.0, "valid_reversal"))
        ex, m, sk = backtest.simulate(self._trades(rows))
        self.assertEqual(m["trades"], 3)
        self.assertEqual(sk.get("max_open"), 3)

    def test_symbol_not_doubled(self):
        rows = [("a", "LONG", 1000, 5000, 100.0, 1.0, 2.0, "valid_reversal"), ("a", "LONG", 2000, 6000, 100.0, 1.0, 2.0, "valid_reversal")]
        ex, m, sk = backtest.simulate(self._trades(rows))
        self.assertEqual(m["trades"], 1); self.assertEqual(sk.get("symbol_open"), 1)

    def test_daily_loss_halts_new_trades(self):
        day = 86_400_000
        rows = [("a" + str(i), "LONG", day * 5 + i * 1000, day * 5 + i * 1000 + 500, 100.0, 1.0, -1.0, "false_break") for i in range(10)]
        ex, m, sk = backtest.simulate(self._trades(rows))
        self.assertLess(m["trades"], 10)       # 0.5% x 3 losses = 1.5% -> lock
        self.assertGreater(sk.get("daily_loss", 0), 0)

    def test_metrics(self):
        ex = pd.DataFrame({"pnl": [20.0, -10.0, 20.0, -10.0], "side_name": ["LONG", "SHORT", "LONG", "LONG"], "exit_time": [1, 2, 3, 4]})
        m = backtest.summarize(ex, 2000)
        self.assertEqual((m["wins"], m["losses"], m["long"], m["short"]), (2, 2, 3, 1))
        self.assertAlmostEqual(m["profit_factor"], 2.0); self.assertAlmostEqual(m["net_pnl"], 20.0)


class TestLiveDedupeAndTelegram(unittest.TestCase):
    def make_env(self):
        tmp = Path(tempfile.mkdtemp())
        # find a synthetic series whose last closed candle yields a tradable setup
        for seed in range(1, 400):
            df = synth(900, seed=seed, vol=0.003)
            ev = features.build_events(df.iloc[:-1], "btc_usdt", TCFG)
            ev = ev[ev["tradable"]]
            if len(ev) and ev["idx"].iloc[-1] >= len(df) - 3:
                last = int(ev["signal_time"].iloc[-1])
                recent = ev[ev["signal_time"] >= last - CFG["max_signal_age_min"] * 60_000]
                if recent["idx"].nunique() == 1:       # exactly one touch inside the freshness window
                    return tmp, df, ev.iloc[-1]
        self.skipTest("no synthetic setup found")

    def test_formatting_has_all_fields(self):
        tmp, df, row = self.make_env()
        msg = live.format_signal(row, 0.61, True, 1.5, 10.0, int(row["signal_time"]) + 60000)
        for key in ("نماد:", "جهت:", "SMA 64 Reversal", "قیمت ورود", "حد ضرر", "هدف اول", "هدف دوم",
                    "نسبت ریسک به بازده", "احتمال تخمینی موفقیت مدل", "دلیل ورود", "وضعیت بازار", "زمان سیگنال"):
            self.assertIn(key, msg)
        self.assertNotIn("Win Rate قطعی را", msg)
        un = live.format_signal(row, 0.61, False, 1.5, 10.0, int(row["signal_time"]))
        self.assertIn("کالیبره نشده", un)

    def test_no_duplicate_signals_and_paper_tracking(self):
        tmp, df, row = self.make_env()
        now = int(row["signal_time"]) + 5_000
        cfg = dict(TCFG, symbols=["btc_usdt"])
        from xt_bot.util import write_json
        write_json(tmp / "meta.json", {"approved": True, "thresholds": {"REVERSAL": 0.0, "BREAKOUT": 0.0}, "enabled_kinds": ["REVERSAL", "BREAKOUT"], "enabled_symbols": ["btc_usdt"]})
        sent = []
        fetch = lambda sym, a, b, itv="1m": df[(df.time >= a) & (df.time <= b)].reset_index(drop=True)
        ds = dataset.assign_splits(pd.concat([dataset.events_for_symbol(synth(6000, seed=s), "x", TCFG) for s in (1, 2)], ignore_index=True))[0]
        b1 = train_model(dataset.trade_set(ds, "train"), dataset.trade_set(ds, "val")); bundle = {"REVERSAL": b1, "BREAKOUT": b1}
        kw = dict(now_ms=now, fetch=fetch, notify=lambda m: sent.append(m) or True, cfg=cfg, state_file=tmp / "s.json",
                  meta_file=tmp / "meta.json", signals_file=tmp / "sig.jsonl", bundle=bundle)
        r1 = live.scan(**kw)
        r2 = live.scan(**kw)
        self.assertEqual(r1["sent"], 1)
        self.assertEqual(r2["sent"], 0)
        self.assertEqual(len([m for m in sent if "سیگنال آزمایشی" in m]), 1)
        from xt_bot.util import read_json
        st = read_json(tmp / "s.json", {})
        self.assertEqual(len(st["open_positions"]), 1)

    def test_unapproved_model_sends_nothing(self):
        tmp = Path(tempfile.mkdtemp())
        from xt_bot.util import write_json
        write_json(tmp / "meta.json", {"approved": False})
        sent = []
        r = live.scan(notify=lambda m: sent.append(m) or True, meta_file=tmp / "meta.json", state_file=tmp / "s.json")
        self.assertEqual(r["status"], "no_approved_model"); self.assertEqual(sent, [])

    def test_undelivered_telegram_is_not_recorded_as_sent(self):
        tmp, df, row = self.make_env()
        now = int(row["signal_time"]) + 5_000
        cfg = dict(TCFG, symbols=["btc_usdt"])
        from xt_bot.util import write_json, read_json
        write_json(tmp / "meta.json", {"approved": True, "thresholds": {"REVERSAL": 0.0, "BREAKOUT": 0.0}, "enabled_kinds": ["REVERSAL", "BREAKOUT"], "enabled_symbols": ["btc_usdt"]})
        ds = dataset.assign_splits(pd.concat([dataset.events_for_symbol(synth(6000, seed=s), "x", TCFG) for s in (1, 2)], ignore_index=True))[0]
        b1 = train_model(dataset.trade_set(ds, "train"), dataset.trade_set(ds, "val")); bundle = {"REVERSAL": b1, "BREAKOUT": b1}
        fetch = lambda sym, a, b, itv="1m": df[(df.time >= a) & (df.time <= b)].reset_index(drop=True)
        r = live.scan(now_ms=now, fetch=fetch, notify=lambda m: False, cfg=cfg, state_file=tmp / "s.json",
                      meta_file=tmp / "meta.json", signals_file=tmp / "sig.jsonl", bundle=bundle)
        self.assertEqual(r["sent"], 0)
        self.assertEqual(read_json(tmp / "s.json", {})["sent"], {})

    def test_paper_position_resolution_sl_first(self):
        cfg = CFG
        state = paper.new_state(cfg)
        t = T0
        state["open_positions"].append({"symbol": "x", "side": "LONG", "entry": 100.0, "sl": 99.0, "tp1": 102.0, "tp2": 103.0,
                                        "distance": 1.0, "qty": 10.0, "risk_usd": 10.0, "entry_time": t})
        df = pd.DataFrame({"time": [t], "open": [100.0], "high": [102.5], "low": [98.5], "close": [100.0], "volume": [1.0]})
        msgs = []
        paper.update_positions(state, {"x": df}, cfg, msgs.append)
        self.assertEqual(state["open_positions"], [])
        self.assertLess(state["closed_trades"][0]["pnl"], -10.0 + 1e-9)  # stop first, plus costs
        self.assertEqual(state["closed_trades"][0]["result"], "sl")


class TestEndToEnd(unittest.TestCase):
    def test_training_pipeline_on_random_walk_is_not_approved(self):
        from xt_bot import train as tr
        cfg = dict(CFG, symbols=["btc_usdt", "eth_usdt"], train_days=30, max_cost_r=5.0)
        now = T0 + 40_000 * INTERVAL_MS
        series = {"btc_usdt": synth(40_000, seed=11, vol=0.0015), "eth_usdt": synth(40_000, seed=12, vol=0.0015)}
        fetch = lambda sym, a, b, itv="1m": series[sym][(series[sym].time >= a) & (series[sym].time <= b)].reset_index(drop=True)
        tmp = Path(tempfile.mkdtemp())
        import xt_bot.train as T, xt_bot.model as M
        meta, results, text = T.run_training(fetcher=fetch, cfg=cfg, now_ms=now, save=False, meta_file=tmp / "m.json")
        self.assertFalse(meta["approved"])      # a random walk must never be approved
        self.assertTrue(meta["reasons"])
        for k in ("حالت برگشت", "حالت شکست", "حفظ‌کردن یا یادگیری", "BTC", "ETH", "RR 1:1", "RR 1:3", "کدام ساعت‌ها", "پورتفوی"):
            self.assertIn(k, text)
        self.assertLess(len(text), 7000)
        self.assertGreaterEqual(meta["chosen_rr"], 2.0)   # project rule: live RR >= 2
        self.assertEqual(set(meta["rr_table"]), {"1.0", "1.5", "2.0", "3.0"})


if __name__ == "__main__":
    unittest.main()


class TestSelectionHoursRR(unittest.TestCase):
    def test_with_rr_moves_targets(self):
        ev = features.build_events(synth(3000, seed=4), "t", TCFG)
        e3 = features.with_rr(ev, 3.0)
        sign = np.where(e3["side_name"] == "LONG", 1.0, -1.0)
        np.testing.assert_allclose((e3["tp1"] - e3["entry"]) * sign, 3.0 * e3["distance"])
        np.testing.assert_allclose(e3["sl"].values, ev["sl"].values)   # stop never changes with RR

    def test_select_best_one_candidate_per_touch(self):
        from xt_bot.model import select_best
        df = pd.DataFrame({"symbol": ["a"] * 4, "idx": [5, 5, 9, 9], "kind": ["REVERSAL", "BREAKOUT"] * 2,
                           "prob": [0.4, 0.7, 0.6, 0.3], "signal_time": [1, 1, 2, 2]})
        out = select_best(df, {"REVERSAL": 0.5, "BREAKOUT": 0.5})
        self.assertEqual(len(out), 2)
        self.assertEqual(list(out["kind"]), ["BREAKOUT", "REVERSAL"])
        out2 = select_best(df, {"REVERSAL": 0.65})          # BREAKOUT has no threshold -> disabled
        self.assertEqual(len(out2), 0)

    def _sel(self, hour_good, n_per=40, good_r=0.6, bad_r=-0.5):
        rows = []
        for h in range(24):
            for i in range(n_per):
                rows.append({"signal_time": (h * 3600 + i) * 1000 + 86_400_000 * 3, "net_r": good_r if h in hour_good else bad_r,
                             "side_name": "LONG", "symbol": "a"})
        return pd.DataFrame(rows)

    def test_best_window_and_tehran_time(self):
        from xt_bot import hours
        good = {10, 11, 12}
        touches = pd.DataFrame({"signal_time": [h * 3_600_000 for h in range(24) for _ in range(5)], "atr_bps": 4.0})
        sel = {"train": self._sel(good), "val": self._sel(good), "test": self._sel(good)}
        r = hours.analyze(touches, sel)
        self.assertEqual(r["best"]["start"], 10)
        self.assertTrue(r["stable"])
        self.assertIn("10:00–13:00 UTC = 13:30–16:30", r["label"])
        # window good on validation but bad on test -> not stable
        sel2 = {"train": self._sel(good), "val": self._sel(good), "test": self._sel(good, good_r=-0.5)}
        self.assertFalse(hours.analyze(touches, sel2)["stable"])
        # too few validation signals -> falls back to touch counts and says so
        sel3 = {"train": self._sel(good), "val": self._sel(good, n_per=1), "test": self._sel(good)}
        self.assertEqual(hours.analyze(touches, sel3)["basis"], "touches")


class TestDrawdownReset(unittest.TestCase):
    def test_drawdown_breach_is_counted_not_permanent(self):
        day = 86_400_000
        rows = []
        for d in range(1, 12):  # one winner-less day after another: breaches, then restart next day
            for i in range(3):
                t = day * d + i * 1000
                rows.append((f"s{i}", "LONG", t, t + 500, 100.0, 1.0, -1.0, "false_break"))
        df = pd.DataFrame(rows, columns=["symbol", "side_name", "entry_time", "exit_time", "entry", "distance", "net_r", "label"])
        ex, m, sk = backtest.simulate(df)
        self.assertGreater(m["trades"], 12)               # trading continued after a breach
        self.assertGreater(sk.get("dd_breaches", 0), 0)   # but breaches are visible

    def test_signal_stats(self):
        df = pd.DataFrame({"net_r": [1.8, -1.1, 1.8, -1.1], "side_name": ["LONG", "SHORT", "LONG", "LONG"]})
        s = backtest.signal_stats(df)
        self.assertEqual((s["n"], s["long"], s["short"]), (4, 3, 1))
        self.assertAlmostEqual(s["win_rate"], 0.5); self.assertAlmostEqual(s["pf"], 3.6 / 2.2)
