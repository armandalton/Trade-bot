import os, sys, json, time, math, hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta
import requests
import pandas as pd
import numpy as np

BASE = "https://fapi.xt.com"
KLINE = BASE + "/future/market/v1/public/q/kline"
CONFIG = {
    "symbols": [s.strip().lower() for s in os.getenv("SYMBOLS", "btc_usdt,eth_usdt,sol_usdt,bnb_usdt,xrp_usdt").split(",") if s.strip()],
    "capital": float(os.getenv("PAPER_CAPITAL", "2000")),
    "risk_pct": float(os.getenv("RISK_PCT", "0.005")),
    "daily_loss_pct": float(os.getenv("DAILY_LOSS_PCT", "0.015")),
    "max_drawdown_pct": float(os.getenv("MAX_DRAWDOWN_PCT", "0.06")),
    "fee_rate": float(os.getenv("FEE_RATE", "0.0006")),
    "slippage_rate": float(os.getenv("SLIPPAGE_RATE", "0.0005")),
    "min_trades": int(os.getenv("MIN_TRADES", "30")),
}
DATA = Path("data")
DATA.mkdir(exist_ok=True)
STATE_FILE = DATA / "state.json"
MODEL_FILE = DATA / "model.json"
HISTORY_FILE = DATA / "signals.jsonl"

def log(msg):
    print(f"[{datetime.now(timezone.utc).isoformat()}] {msg}", flush=True)

def read_json(path, default):
    try:
        return json.loads(path.read_text())
    except Exception:
        return default

def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))

def xt_get_klines(symbol, interval, days=185):
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400 * 1000
    rows, cursor = [], start_ms
    # XT documents max limit 1500. Paginate by time windows and deduplicate.
    step_ms = {"15m": 15*60*1000, "1h": 60*60*1000}[interval] * 1490
    session = requests.Session()
    while cursor < end_ms:
        window_end = min(end_ms, cursor + step_ms)
        params = {"symbol": symbol, "interval": interval, "startTime": cursor,
                  "endTime": window_end, "limit": 1500}
        response = session.get(KLINE, params=params, timeout=25,
                               headers={"Content-Type": "application/x-www-form-urlencoded"})
        response.raise_for_status()
        payload = response.json()
        if payload.get("returnCode") not in (0, "0", None):
            raise RuntimeError(f"XT API {symbol} {interval}: {payload}")
        batch = payload.get("result") or []
        if isinstance(batch, dict):
            batch = batch.get("items") or batch.get("list") or []
        rows.extend(batch)
        cursor = window_end + 1
        time.sleep(0.08)
    if not rows:
        raise RuntimeError(f"No candle data returned for {symbol} {interval}. Check XT symbol/API availability.")
    df = pd.DataFrame(rows)
    # Official XT kline response fields: t,o,h,l,c,a,v; t is candle timestamp.
    required = {"t", "o", "h", "l", "c"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"Unexpected XT kline schema for {symbol}: {list(df.columns)}")
    df = df.rename(columns={"t":"time", "o":"open", "h":"high", "l":"low", "c":"close", "a":"volume", "v":"turnover"})
    for col in ["time","open","high","low","close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["time","open","high","low","close"]).drop_duplicates("time").sort_values("time")
    # Normalize timestamps to milliseconds.
    if df["time"].median() < 10_000_000_000:
        df["time"] *= 1000
    df = df[(df.time >= start_ms) & (df.time <= end_ms)].reset_index(drop=True)
    if len(df) < 300:
        raise RuntimeError(f"Too few {interval} candles for {symbol}: {len(df)}")
    return df

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    return 100 - 100/(1 + up/dn.replace(0, np.nan))

def atr(df, n=14):
    prev = df.close.shift()
    tr = pd.concat([(df.high-df.low), (df.high-prev).abs(), (df.low-prev).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def add_indicators(df):
    x = df.copy()
    x["ema20"] = ema(x.close, 20)
    x["ema50"] = ema(x.close, 50)
    x["ema200"] = ema(x.close, 200)
    x["rsi"] = rsi(x.close)
    x["atr"] = atr(x)
    return x.dropna().reset_index(drop=True)

def make_signals(df15, df1h, params):
    a, b = add_indicators(df15), add_indicators(df1h)
    # XT candle timestamps are candle-open times. Shift 1h bars to their close
    # time so a 15m candle never sees an unfinished/future hourly candle.
    trend = b[["time","close","ema50","ema200"]].copy()
    trend["time"] = trend["time"] + 60*60*1000
    trend["up"] = trend.ema50 > trend.ema200
    trend["down"] = trend.ema50 < trend.ema200
    x = pd.merge_asof(a.sort_values("time"), trend.sort_values("time"), on="time",
                      direction="backward", suffixes=("", "_1h"))
    # Signals only on closed candles; conservative pullback + reclaim trigger.
    x["long"] = (x.up & (x.close > x.ema20) &
                 (x.low.shift(1) <= x.ema20.shift(1) * (1 + params["pullback"])) &
                 (x.rsi >= params["rsi_long"]) & (x.rsi <= 68))
    x["short"] = (x.down & (x.close < x.ema20) &
                  (x.high.shift(1) >= x.ema20.shift(1) * (1 - params["pullback"])) &
                  (x.rsi <= params["rsi_short"]) & (x.rsi >= 32))
    return x

def backtest(df15, df1h, params):
    x = make_signals(df15, df1h, params)
    trades, pos = [], None
    fee = CONFIG["fee_rate"]
    slip = CONFIG["slippage_rate"]
    # One position at a time per symbol. Stops/targets checked using candle extremes.
    for i in range(1, len(x)):
        row = x.iloc[i]
        if pos:
            hit_sl = row.low <= pos["sl"] if pos["side"] == "LONG" else row.high >= pos["sl"]
            hit_tp = row.high >= pos["tp"] if pos["side"] == "LONG" else row.low <= pos["tp"]
            # If both touched in one candle, assume stop first (conservative).
            if hit_sl or hit_tp:
                exit_price = pos["sl"] if hit_sl else pos["tp"]
                gross = (exit_price-pos["entry"])/pos["entry"] if pos["side"]=="LONG" else (pos["entry"]-exit_price)/pos["entry"]
                net = gross - 2*fee - 2*slip
                trades.append(net / (pos["risk_distance"]/pos["entry"]))
                pos = None
        if pos is None:
            side = "LONG" if bool(row.long) else ("SHORT" if bool(row.short) else None)
            if not side or not np.isfinite(row.atr) or row.atr <= 0:
                continue
            entry = float(row.close) * (1 + slip if side=="LONG" else 1-slip)
            distance = max(float(row.atr) * params["atr_mult"], entry*0.002)
            sl = entry-distance if side=="LONG" else entry+distance
            tp = entry+2*distance if side=="LONG" else entry-2*distance
            pos = {"side":side,"entry":entry,"sl":sl,"tp":tp,"risk_distance":distance}
    if pos:
        last = float(x.iloc[-1].close)
        gross = (last-pos["entry"])/pos["entry"] if pos["side"]=="LONG" else (pos["entry"]-last)/pos["entry"]
        trades.append((gross-2*fee-2*slip)/(pos["risk_distance"]/pos["entry"]))
    if not trades:
        return {"trades":0,"win_rate":0,"profit_factor":0,"expectancy_r":0,"max_drawdown_r":0}
    arr = np.array(trades, dtype=float)
    gains, losses = arr[arr>0].sum(), abs(arr[arr<0].sum())
    equity = np.cumsum(arr)
    peak = np.maximum.accumulate(np.r_[0,equity])[1:]
    dd = peak-equity
    return {"trades":int(len(arr)), "win_rate":float((arr>0).mean()),
            "profit_factor":float(gains/losses) if losses else 99.0,
            "expectancy_r":float(arr.mean()), "max_drawdown_r":float(dd.max())}

CANDIDATES = [
    {"pullback":p, "rsi_long":rl, "rsi_short":rs, "atr_mult":am}
    for p in (0.001,0.0025,0.004)
    for rl,rs in ((48,52),(50,50),(52,48))
    for am in (1.5,2.0,2.5)
]

def split_df(df, start, end):
    return df[(df.time >= start) & (df.time < end)].reset_index(drop=True)

def train():
    now = int(time.time()*1000)
    six_months = now - 185*86400*1000
    results = []
    for symbol in CONFIG["symbols"]:
        log(f"Downloading {symbol} historical 15m and 1h candles...")
        d15 = xt_get_klines(symbol,"15m",185)
        d1h = xt_get_klines(symbol,"1h",185)
        # Chronological 60/20/20 split. Optimize only on train; choose using validation;
        # final test is reported but not used to choose a candidate.
        t1 = six_months + int((now-six_months)*0.60)
        t2 = six_months + int((now-six_months)*0.80)
        train15, val15, test15 = split_df(d15,six_months,t1), split_df(d15,t1,t2), split_df(d15,t2,now+1)
        train1, val1, test1 = split_df(d1h,six_months,t1), split_df(d1h,t1,t2), split_df(d1h,t2,now+1)
        ranked = []
        for p in CANDIDATES:
            tr = backtest(train15,train1,p)
            va = backtest(val15,val1,p)
            if tr["trades"] >= 10 and va["trades"] >= 5:
                score = va["expectancy_r"] - 0.15*va["max_drawdown_r"]
                ranked.append((score,p,tr,va))
        if not ranked:
            results.append({"symbol":symbol,"approved":False,"reason":"Insufficient trades for robust validation"})
            continue
        ranked.sort(key=lambda z:z[0],reverse=True)
        score,p,tr,va = ranked[0]
        te = backtest(test15,test1,p)
        approved = (tr["expectancy_r"]>0 and va["expectancy_r"]>0 and
                    va["profit_factor"]>1.05 and va["max_drawdown_r"]<12 and
                    te["trades"]>=5 and te["expectancy_r"]>0 and te["profit_factor"]>1.0)
        results.append({"symbol":symbol,"approved":bool(approved),"params":p,
                        "train":tr,"validation":va,"test":te,"validation_score":score})
        log(f"{symbol}: approved={approved}, validation={va}, test={te}")
    good = [r for r in results if r.get("approved")]
    model = {"created_utc":datetime.now(timezone.utc).isoformat(),
             "data_source":"XT USDT-M public futures klines",
             "method":"chronological 60/20/20; train-select, validation-rank, untouched test gate",
             "approved":bool(good), "symbols":results}
    write_json(MODEL_FILE,model)
    log(f"Training finished. Approved symbols: {len(good)}/{len(results)}")
    return model

def telegram(text):
    token=os.getenv("TELEGRAM_BOT_TOKEN","").strip()
    chat=os.getenv("TELEGRAM_CHAT_ID","").strip()
    if not token or not chat:
        log("Telegram secrets missing; message not sent.")
        return False
    url=f"https://api.telegram.org/bot{token}/sendMessage"
    r=requests.post(url,json={"chat_id":chat,"text":text,"disable_web_page_preview":True},timeout=20)
    r.raise_for_status()
    return True

def scan():
    model=read_json(MODEL_FILE,{})
    if not model.get("approved"):
        log("No approved model. Run train first; no signals will be sent.")
        return
    state=read_json(STATE_FILE,{"sent":{},"open_positions":[],"paper_equity":CONFIG["capital"],
                                "peak_equity":CONFIG["capital"],"day":None,"day_start":CONFIG["capital"]})
    state.setdefault("sent", {})
    state.setdefault("open_positions", [])
    state.setdefault("peak_equity", CONFIG["capital"])
    today=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    if state.get("day") != today:
        state["day"],state["day_start"]=today,float(state.get("paper_equity",CONFIG["capital"]))

    # Resolve paper positions against newly closed candles. If SL and TP touch in
    # the same candle, assume SL first (conservative). This is a simulation only.
    approved = {x["symbol"]: x for x in model["symbols"] if x.get("approved")}
    for symbol in list(approved):
        try:
            d15=xt_get_klines(symbol,"15m",12)
            closed=d15.iloc[:-1]  # ignore currently forming candle
            remaining=[]
            for pos in state["open_positions"]:
                if pos["symbol"] != symbol:
                    remaining.append(pos)
                    continue
                finished=False
                for _, candle in closed[closed.time > pos["entry_time"]].iterrows():
                    hit_sl = candle.low <= pos["sl"] if pos["side"]=="LONG" else candle.high >= pos["sl"]
                    hit_tp = candle.high >= pos["tp"] if pos["side"]=="LONG" else candle.low <= pos["tp"]
                    if hit_sl or hit_tp:
                        exit_price = pos["sl"] if hit_sl else pos["tp"]
                        raw = ((exit_price-pos["entry"])*pos["qty"] if pos["side"]=="LONG"
                               else (pos["entry"]-exit_price)*pos["qty"])
                        costs = pos["qty"]*pos["entry"]*(2*CONFIG["fee_rate"]+2*CONFIG["slippage_rate"])
                        pnl = raw-costs
                        state["paper_equity"] = float(state["paper_equity"])+pnl
                        state["peak_equity"] = max(float(state["peak_equity"]),float(state["paper_equity"]))
                        state.setdefault("closed_trades", []).append({"symbol":symbol,"side":pos["side"],
                            "entry":pos["entry"],"exit":exit_price,"pnl":pnl,"closed_at":datetime.now(timezone.utc).isoformat()})
                        telegram(f"PAPER CLOSE — {symbol.upper()} {pos['side']}\\nExit: {exit_price:.8g}\\nP/L: ${pnl:.2f}\\nPaper equity: ${state['paper_equity']:.2f}")
                        finished=True
                        break
                if not finished:
                    remaining.append(pos)
            state["open_positions"]=remaining
        except Exception as e:
            log(f"{symbol} paper-position update failed: {e}")

    equity=float(state.get("paper_equity",CONFIG["capital"]))
    daily_loss=max(0,float(state["day_start"])-equity)
    drawdown=max(0,float(state["peak_equity"])-equity)
    if daily_loss >= CONFIG["capital"]*CONFIG["daily_loss_pct"] or drawdown >= CONFIG["capital"]*CONFIG["max_drawdown_pct"]:
        log("Paper risk lock active. No new signals.")
        write_json(STATE_FILE,state)
        return

    for symbol,item in approved.items():
        try:
            # One open paper position per symbol at a time.
            if any(p["symbol"]==symbol for p in state["open_positions"]):
                continue
            d15=xt_get_klines(symbol,"15m",12)
            d1h=xt_get_klines(symbol,"1h",30)
            x=make_signals(d15,d1h,item["params"])
            row=x.iloc[-2]  # last fully closed candle
            side="LONG" if bool(row.long) else ("SHORT" if bool(row.short) else None)
            if not side: continue
            key=f"{symbol}:{int(row.time)}:{side}"
            if key in state["sent"]: continue
            entry=float(row.close)
            distance=max(float(row.atr)*item["params"]["atr_mult"],entry*0.002)
            sl=entry-distance if side=="LONG" else entry+distance
            tp=entry+2*distance if side=="LONG" else entry-2*distance
            risk_usd=CONFIG["capital"]*CONFIG["risk_pct"]
            qty=risk_usd/distance
            msg=(f"XT Adaptive — PAPER SIGNAL\\n{symbol.upper()} | {side}\\n"
                 f"Entry: {entry:.8g}\\nStop: {sl:.8g}\\nTarget (2R): {tp:.8g}\\n"
                 f"Paper risk: ${risk_usd:.2f} | Qty approx: {qty:.6g}\\n"
                 f"15m entry / 1h trend\\nNot financial advice. No order was placed.")
            delivered = telegram(msg)
            if not delivered:
                log(f"Telegram not configured; signal not recorded as sent: {key}")
                continue
            state["sent"][key]=datetime.now(timezone.utc).isoformat()
            state["open_positions"].append({"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,
                                             "qty":qty,"entry_time":float(row.time),"key":key})
            if len(state["sent"])>1000:
                state["sent"]=dict(list(state["sent"].items())[-500:])
            with HISTORY_FILE.open("a") as f:
                f.write(json.dumps({"key":key,"symbol":symbol,"side":side,"entry":entry,"sl":sl,"tp":tp,
                                    "risk_usd":risk_usd,"time":state["sent"][key]})+"\\n")
            log(f"Paper signal processed: {key}")
        except Exception as e:
            log(f"{symbol} scan failed: {e}")
    write_json(STATE_FILE,state)

def check():
    log("Checking XT public futures API...")
    df=xt_get_klines("btc_usdt","15m",7)
    log(f"XT connection OK. BTC 15m candles received: {len(df)}; latest close={df.iloc[-1].close}")
    telegram("XT Adaptive Bot: connection test successful. Public XT futures market data is reachable.")
    log("Check complete.")

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "check"
    if mode=="check": check()
    elif mode=="train": train()
    elif mode=="scan": scan()
    else: raise SystemExit("Usage: python bot.py check|train|scan")
