"""XT public futures kline download (no API key needed)."""
import time
import numpy as np
import pandas as pd
import requests

from .config import INTERVAL_MS
from .util import log

BASE = "https://fapi.xt.com"
KLINE = BASE + "/future/market/v1/public/q/kline"
MAX_PER_REQUEST = 1400  # XT documents a 1500 limit; stay below it


def _request(session, symbol, interval, start_ms, end_ms, attempts=5):
    params = {"symbol": symbol, "interval": interval, "startTime": int(start_ms),
              "endTime": int(end_ms), "limit": 1500}
    last = None
    for k in range(attempts):
        try:
            r = session.get(KLINE, params=params, timeout=25)
            if r.status_code in (418, 429) or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}")
            r.raise_for_status()
            payload = r.json()
            if payload.get("returnCode") not in (0, "0", None):
                raise RuntimeError(f"XT API error: {payload.get('msgInfo') or payload.get('returnCode')}")
            batch = payload.get("result") or []
            if isinstance(batch, dict):
                batch = batch.get("items") or batch.get("list") or []
            return batch
        except Exception as e:  # network / parse / API errors -> retry with backoff
            last = e
            time.sleep(min(30, 1.5 ** k))
    raise RuntimeError(f"XT kline request failed for {symbol}: {type(last).__name__}: {last}")


def parse_klines(rows):
    if not rows:
        raise RuntimeError("No candle data returned (check XT symbol name / API availability).")
    df = pd.DataFrame(rows)
    need = {"t", "o", "h", "l", "c"}
    if not need.issubset(df.columns):
        raise RuntimeError(f"Unexpected XT kline schema: {list(df.columns)}")
    vol_col = "a" if "a" in df.columns else ("v" if "v" in df.columns else None)
    out = pd.DataFrame({
        "time": pd.to_numeric(df["t"], errors="coerce"),
        "open": pd.to_numeric(df["o"], errors="coerce"),
        "high": pd.to_numeric(df["h"], errors="coerce"),
        "low": pd.to_numeric(df["l"], errors="coerce"),
        "close": pd.to_numeric(df["c"], errors="coerce"),
        "volume": pd.to_numeric(df[vol_col], errors="coerce") if vol_col else 0.0,
    })
    out = out.dropna(subset=["time", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0.0)
    if len(out) and out["time"].median() < 10_000_000_000:  # seconds -> ms
        out["time"] = out["time"] * 1000
    out["time"] = out["time"].astype("int64")
    return out.drop_duplicates("time").sort_values("time").reset_index(drop=True)


def fetch_klines(symbol, start_ms, end_ms, interval="1m", session=None, pause=0.08):
    session = session or requests.Session()
    step = INTERVAL_MS * MAX_PER_REQUEST
    rows, cursor, windows = [], int(start_ms), 0
    while cursor < end_ms:
        win_end = min(int(end_ms), cursor + step - 1)
        rows.extend(_request(session, symbol, interval, cursor, win_end))
        cursor = win_end + 1
        windows += 1
        if windows % 40 == 0:
            log(f"{symbol}: {windows} windows, {len(rows)} candles so far")
        time.sleep(pause)
    df = parse_klines(rows)
    return df[(df.time >= start_ms) & (df.time <= end_ms)].reset_index(drop=True)


def closed_only(df, now_ms, interval_ms=INTERVAL_MS):
    """Keep only fully closed candles (XT timestamps are candle open times)."""
    return df[df["time"] + interval_ms <= now_ms].reset_index(drop=True)


def validate_candles(df, interval_ms=INTERVAL_MS):
    """Sanity report: ordering, duplicates, gaps, coverage."""
    if len(df) < 2:
        return {"n": len(df), "ok": False, "missing": None, "max_gap_candles": None, "coverage": 0.0}
    t = df["time"].values
    diffs = np.diff(t)
    expected = int((t[-1] - t[0]) // interval_ms) + 1
    return {"n": int(len(df)), "sorted": bool((diffs > 0).all()),
            "missing": int(expected - len(df)),
            "max_gap_candles": int(diffs.max() // interval_ms),
            "coverage": float(len(df) / expected),
            "ok": bool((diffs > 0).all() and len(df) / expected >= 0.97)}
    
def fetch_last_price(symbol, session=None):
    """Fetch latest traded futures price from XT public ticker."""
    session = session or requests.Session()
    url = BASE + "/future/market/v1/public/q/tickers"

    response = session.get(url, timeout=15)
    response.raise_for_status()

    payload = response.json()

    if payload.get("returnCode") not in (0, "0", None):
        raise RuntimeError(
            f"XT ticker API error: {payload.get('msgInfo')}"
        )

    tickers = payload.get("result") or []

    if isinstance(tickers, dict):
        tickers = tickers.get("items") or []

    wanted = symbol.strip().lower()

    for item in tickers:
        if str(item.get("s", "")).lower() == wanted:
            price = float(item["c"])

            if not np.isfinite(price) or price <= 0:
                raise RuntimeError("Invalid XT last price")

            return {
                "symbol": wanted,
                "price": price,
                "received_at_ms": int(time.time() * 1000),
            }

    raise RuntimeError(
        f"Ticker not found for symbol: {wanted}"
    )
