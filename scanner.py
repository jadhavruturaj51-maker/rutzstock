#!/usr/bin/env python3
"""
F&O 9:50 Scanner - independent version (runs on GitHub Actions, free).

Two scanners, evaluated every trading day at 9:50 IST on all NSE F&O stocks:

  Scanner A "RJ Scanner"     - RJ's original 12 criteria, unchanged.
  Scanner B "Muse Scanner" - V2-Quality rules + RSI extreme exclusion.

Data source: Yahoo Finance (free) via yfinance - the same source the
backtests were validated on. Criteria 10 (OI buildup) and 11 (no event
today) cannot be checked from this free source, so PASSED is reported
out of the 10 checkable criteria, shown as x/12 like the backtests.

Definitions (identical to the validated backtest code):
  - "9:50 price" = close of the 09:45 five-minute candle.
  - Volume window = candles 09:20-09:45 vs same window, prev 10 trading days.
  - VWAP from candles 09:15-09:45 (typical price x volume).
  - Opening range = first 15 minutes (candles 09:15, 09:20, 09:25).
  - EMA10/EMA20 and RSI(14) Wilder on 15m candles, value of the candle
    labelled 09:30 (the last completed 15m candle before 9:50).
  - Market side: Nifty 9:50 price vs previous close -> LONG or SHORT.
  - Sector strength = mean move of the sector's F&O stocks at 9:50.

Usage:
  python scanner.py                      live run for today (waits for the
                                         09:45 candle; exits quietly on
                                         market holidays)
  python scanner.py --date 2026-11-13    evaluate a past trading day
  python scanner.py --date 2026-11-13 --backfill
                                         same, but record it in
                                         docs/results.json as a backtest day
  python scanner.py --date 2026-11-13 --dry-run
                                         print only, write nothing
"""
import argparse, csv, io, json, os, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

IST = ZoneInfo("Asia/Kolkata")
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "docs", "results.json")
SYMBOLS_FILE = os.path.join(HERE, "fo_symbols.txt")
SECTORS_FILE = os.path.join(HERE, "sectors.json")

WIN = {"09:20", "09:25", "09:30", "09:35", "09:40", "09:45"}   # volume window
EARLY = {"09:15", "09:20", "09:25", "09:30", "09:35", "09:40", "09:45"}
ORNG = {"09:15", "09:20", "09:25"}                              # opening range


# ---------------------------------------------------------------- universe
def load_universe():
    """F&O symbol list: fresh from NSE when reachable, else the bundled copy."""
    syms = []
    try:
        r = requests.get("https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
        rows = list(csv.reader(io.StringIO(r.text)))
        skip = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "NIFTYFPI",
                "NIFTYNXT50", "Symbol"}
        for row in rows:
            if len(row) < 2:
                continue
            s = row[1].strip()
            if s and s not in skip and s not in syms and row[0].strip().upper() != "UNDERLYING":
                syms.append(s)
        syms = [s for s in syms if any(ch.isalpha() for ch in s)]
    except Exception as e:
        print("NSE list fetch failed, using bundled list:", e, flush=True)
    if len(syms) < 100:
        syms = [l.strip() for l in open(SYMBOLS_FILE) if l.strip()]
    return syms


def load_sectors():
    if os.path.exists(SECTORS_FILE):
        return json.load(open(SECTORS_FILE))
    return {}


# ---------------------------------------------------------------- download
def _split(df, tickers):
    out = {}
    if df is None or len(df) == 0:
        return out
    for t in tickers:
        try:
            sub = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            sub = sub.dropna(how="all")
            if len(sub):
                out[t] = sub
        except Exception:
            pass
    return out


def download_all(tickers, interval, period, batch=40):
    data = {}
    for i in range(0, len(tickers), batch):
        chunk = tickers[i:i + batch]
        try:
            df = yf.download(chunk, period=period, interval=interval,
                             group_by="ticker", auto_adjust=False,
                             threads=True, progress=False)
            data.update(_split(df, chunk))
        except Exception as e:
            print("batch error", interval, i, e, flush=True)
        time.sleep(1)
    return data


def _flat(df):
    """Single-ticker yfinance frames can carry (field, ticker) columns."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def norm_daily(df):
    d = _flat(df).copy()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("Asia/Kolkata").tz_localize(None)
    d.index = idx.normalize()
    return d


def norm_intra(df):
    d = _flat(df).copy()
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is None:
        idx = idx.tz_localize("Asia/Kolkata")
    else:
        idx = idx.tz_convert("Asia/Kolkata")
    d.index = idx
    return d


# ---------------------------------------------------------------- per-symbol stats
def day_stats(intr_day):
    """Stats for one symbol on one date from its 5m frame of that day."""
    m = intr_day.index.strftime("%H:%M")
    c945 = intr_day[m == "09:45"]
    if not len(c945):
        return None
    early = intr_day[np.isin(m, list(EARLY))]
    tp = (early["High"] + early["Low"] + early["Close"]) / 3
    vsum = float(early["Volume"].sum())
    vwap = float((tp * early["Volume"]).sum() / vsum) if vsum > 0 else float(tp.mean())
    orng = intr_day[np.isin(m, list(ORNG))]
    return {
        "p950": float(c945["Close"].iloc[0]),
        "open": float(intr_day["Open"].iloc[0]),
        "vol_win": float(intr_day[np.isin(m, list(WIN))]["Volume"].sum()),
        "vwap": vwap,
        "orh": float(orng["High"].max()), "orl": float(orng["Low"].min()),
    }


def ema_rsi_at(intr, ds):
    """EMA10/EMA20 and Wilder RSI(14) on 15m candles, at the 09:30 candle."""
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    r = intr.resample("15min").agg(agg).dropna(subset=["Close"])
    if not len(r):
        return None, None, None
    r["EMA10"] = r["Close"].ewm(span=10, adjust=False).mean()
    r["EMA20"] = r["Close"].ewm(span=20, adjust=False).mean()
    delta = r["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    r["RSI"] = 100 - 100 / (1 + rs)
    r.loc[avg_loss == 0, "RSI"] = 100.0
    etime = pd.Timestamp(f"{ds} 09:30:00", tz="Asia/Kolkata")
    if etime not in r.index:
        return None, None, None
    row = r.loc[etime]
    rsi = float(row["RSI"]) if pd.notna(row["RSI"]) else None
    return float(row["EMA10"]), float(row["EMA20"]), rsi


# ---------------------------------------------------------------- evaluation
def evaluate(date, daily, m5, nifty_d, nifty_5, sectors, syms):
    """Run both scanners for one trading date. Returns a day dict or None
    when the date is not a trading day / data is missing."""
    ds = date.strftime("%Y-%m-%d")
    ref = daily.get("RELIANCE.NS")
    if ref is None:
        return None
    tdays = sorted(ref.index)
    prev_list = [x for x in tdays if x < date]
    if len(prev_list) < 10:
        return None
    prev = prev_list[-1]
    prev10 = prev_list[-10:]

    # Nifty direction at 9:50
    nday = nifty_5[nifty_5.index.strftime("%Y-%m-%d") == ds]
    nc = nday[nday.index.strftime("%H:%M") == "09:45"]
    if not len(nc) or prev not in nifty_d.index:
        return None
    n950 = float(nc["Close"].iloc[0])
    nmove = (n950 / float(nifty_d.loc[prev, "Close"]) - 1) * 100
    side = "LONG" if nmove > 0 else "SHORT"
    sgn = 1 if side == "LONG" else -1
    long_side = sgn > 0

    monday = date - pd.Timedelta(days=date.weekday())
    pw_start = monday - pd.Timedelta(days=7)
    pw_end = monday - pd.Timedelta(days=3)

    # first pass: prices, so sector means can be computed
    base = {}
    for sym in syms:
        t = sym + ".NS"
        if t not in daily or t not in m5:
            continue
        d = daily[t]
        past = d[d.index < date]
        if not len(past):
            continue
        prev_row_date = past.index[-1]
        intr = m5[t]
        iday = intr[intr.index.strftime("%Y-%m-%d") == ds]
        if not len(iday):
            continue
        st = day_stats(iday)
        if st is None:
            continue
        base[sym] = {"d": d, "past": past, "prev_row_date": prev_row_date,
                     "intr": intr, "st": st}
    if len(base) < 50:
        return None

    moves = {}
    for sym, b in base.items():
        prev_close = float(b["past"].loc[b["prev_row_date"], "Close"])
        b["prev_close"] = prev_close
        moves[sym] = (b["st"]["p950"] / prev_close - 1) * 100
    sec_moves = {}
    for sym, mv in moves.items():
        sec_moves.setdefault(sectors.get(sym, "Unknown"), []).append(mv)
    sec_mean = {k: float(np.mean(v)) for k, v in sec_moves.items()}
    ranked_desc = sorted(sec_mean.items(), key=lambda x: -x[1])
    top5 = {k for k, _ in ranked_desc[:5]}
    bottom5 = {k for k, _ in ranked_desc[-5:]}

    rows = []
    for sym, b in base.items():
        d, past, st = b["d"], b["past"], b["st"]
        p950, prev_close = st["p950"], b["prev_close"]
        prev_row = past.loc[b["prev_row_date"]]
        # volume ratio vs prev 10 trading days (same window)
        vols_prev = []
        for p in prev10:
            pds = p.strftime("%Y-%m-%d")
            pd_day = b["intr"][b["intr"].index.strftime("%Y-%m-%d") == pds]
            if len(pd_day):
                pst = day_stats(pd_day)
                if pst is not None:
                    vols_prev.append(pst["vol_win"])
        vol_ratio = (st["vol_win"] / float(np.mean(vols_prev))
                     if len(vols_prev) >= 5 and np.mean(vols_prev) > 0 else 0.0)
        # previous week high/low
        pw = past[(past.index >= pw_start) & (past.index <= pw_end)]
        if not len(pw):
            continue
        pwh = float(pw["High"].max()); pwl = float(pw["Low"].min())
        pdh = float(prev_row["High"]); pdl = float(prev_row["Low"])
        e10, e20, rsi = ema_rsi_at(b["intr"], ds)
        sec = sectors.get(sym, "Unknown")
        vwap = st["vwap"]
        openp = st["open"]
        gap = (openp / prev_close - 1) * 100
        move = (p950 / prev_close - 1) * 100
        vwap_dist = sgn * (p950 / vwap - 1) * 100
        sec_strength = sgn * sec_mean.get(sec, 0.0)

        fails = []
        if vol_ratio < 2:
            fails.append("volume")
        if long_side and p950 <= pwh:
            fails.append("pwh")
        if (not long_side) and p950 >= pwl:
            fails.append("pwl")
        if long_side and p950 <= pdh:
            fails.append("pdh")
        if (not long_side) and p950 >= pdl:
            fails.append("pdl")
        if e10 is None or e20 is None:
            fails.append("ema")
        elif long_side and (p950 <= e10 or p950 <= e20):
            fails.append("ema")
        elif (not long_side) and (p950 >= e10 or p950 >= e20):
            fails.append("ema")
        if long_side and sec not in top5:
            fails.append("sector")
        if (not long_side) and sec not in bottom5:
            fails.append("sector")
        if long_side and p950 <= vwap:
            fails.append("vwap")
        if (not long_side) and p950 >= vwap:
            fails.append("vwap")
        if long_side and p950 <= st["orh"]:
            fails.append("orange")
        if (not long_side) and p950 >= st["orl"]:
            fails.append("orange")
        if abs(gap) >= 3:
            fails.append("gap")
        if long_side and not (0 < move < 2.5):
            fails.append("move")
        if (not long_side) and not (-2.5 < move < 0):
            fails.append("move")

        # same-day close for scoring (known for past days; for live runs the
        # daily row holds the last price and is corrected on the next run)
        close = None
        if date in d.index:
            close = float(d.loc[date, "Close"])
        rows.append({
            "symbol": sym, "side": side, "entry": round(p950, 2),
            "fails": sorted(set(fails)), "vol_ratio": round(float(vol_ratio), 3),
            "vwap_dist": round(float(vwap_dist), 3),
            "sector_strength": round(float(sec_strength), 3),
            "sector": sec, "rsi": (round(float(rsi), 2) if rsi is not None else None),
            "close": close,
        })

    # ---- Scanner A: RJ's original criteria, 0 fails among the 10 checkable.
    a_rows = [r for r in rows if len(r["fails"]) == 0]
    a_rows.sort(key=lambda r: (-r["sector_strength"], -r["vwap_dist"]))
    a_picks = [{"symbol": r["symbol"], "entry": r["entry"], "passed": 10}
               for r in a_rows[:3]]

    # ---- Scanner B: V2-Quality pool -> quality gate -> cap 3 -> RSI filter.
    pool = []
    for r in rows:
        f = set(r["fails"]) - {"pwh", "pwl", "sector"}
        if r["vol_ratio"] >= 1.0:
            f.discard("volume")
        else:
            f.add("volume")
        if len(f) == 0 and r["vwap_dist"] >= 0.3:
            pool.append(r)
    if pool:
        ss = pd.Series([r["sector_strength"] for r in pool])
        vd = pd.Series([r["vwap_dist"] for r in pool])
        q_ss = ss.rank(pct=True).tolist()
        q_vd = vd.rank(pct=True).tolist()
        for r, a, b_ in zip(pool, q_ss, q_vd):
            r["q"] = a + b_
    gated = [r for r in pool
             if r["sector_strength"] >= 1.0 and r["vwap_dist"] >= 0.5]
    gated.sort(key=lambda r: -r["q"])
    b_picks = []
    for r in gated[:3]:
        if r["rsi"] is not None:
            if long_side and r["rsi"] > 80:
                continue
            if (not long_side) and r["rsi"] < 20:
                continue
        b_picks.append({"symbol": r["symbol"], "entry": r["entry"], "passed": 10})

    return {"date": ds, "side": side, "nifty_move_pct": round(float(nmove), 2),
            "rj": {"picks": a_picks, "no_pick": len(a_picks) == 0},
            "claude": {"picks": b_picks, "no_pick": len(b_picks) == 0},
            "_rows": rows}


# ---------------------------------------------------------------- results store
def load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {"updated_at": None, "days": []}


def scoreboard(days):
    out = {}
    for key in ("rj", "claude"):
        picks = [p for d in days for p in d[key]["picks"]]
        scored = [p for p in picks if p.get("return_pct") is not None]
        out[key] = {
            "days": len(days),
            "picks": len(picks),
            "scored": len(scored),
            "positive": sum(1 for p in scored if p["return_pct"] > 0),
            "pct_positive": (round(100 * sum(1 for p in scored if p["return_pct"] > 0) / len(scored), 1)
                             if scored else None),
            "avg_return": (round(float(np.mean([p["return_pct"] for p in scored])), 2)
                           if scored else None),
        }
    return out


def save_results(store):
    store["days"].sort(key=lambda d: d["date"], reverse=True)
    store["scoreboard"] = scoreboard(store["days"])
    store["updated_at"] = datetime.now(IST).isoformat(timespec="seconds")
    with open(RESULTS, "w") as f:
        json.dump(store, f, indent=1)


def score_pending(store, daily):
    """Fill return_pct for picks from earlier days using daily closes."""
    for d in store["days"]:
        sgn = 1 if d["side"] == "LONG" else -1
        date = pd.Timestamp(d["date"])
        for key in ("rj", "claude"):
            for p in d[key]["picks"]:
                if p.get("return_pct") is not None:
                    continue
                t = p["symbol"] + ".NS"
                if t in daily and date in daily[t].index:
                    close = float(daily[t].loc[date, "Close"])
                    p["close"] = round(close, 2)
                    p["return_pct"] = round(sgn * (close / p["entry"] - 1) * 100, 2)


def record_day(store, day, source, rows_by_symbol):
    """Insert/replace a day entry, scoring its picks when closes are known."""
    sgn = 1 if day["side"] == "LONG" else -1
    entry = {"date": day["date"], "source": source, "side": day["side"],
             "nifty_move_pct": day["nifty_move_pct"],
             "rj": day["rj"], "claude": day["claude"]}
    for key in ("rj", "claude"):
        for p in entry[key]["picks"]:
            r = rows_by_symbol.get(p["symbol"])
            if r and r.get("close"):
                p["close"] = round(r["close"], 2)
                p["return_pct"] = round(sgn * (r["close"] / p["entry"] - 1) * 100, 2)
            else:
                p["return_pct"] = None
    store["days"] = [d for d in store["days"] if d["date"] != entry["date"]]
    store["days"].append(entry)
    return entry


# ---------------------------------------------------------------- main
def wait_for_candle(today_ds):
    """Live mode: poll until today's 09:45 candle exists (data lag buffer),
    or give up at 10:25 IST (holiday / no data)."""
    deadline = datetime.now(IST).replace(hour=10, minute=25, second=0, microsecond=0)
    while True:
        try:
            df = yf.download(["RELIANCE.NS"], period="5d", interval="5m",
                             auto_adjust=False, progress=False)
            if len(df):
                d = norm_intra(df)
                day = d[d.index.strftime("%Y-%m-%d") == today_ds]
                if len(day) and "09:45" in set(day.index.strftime("%H:%M")):
                    return True
        except Exception as e:
            print("poll error", e, flush=True)
        if datetime.now(IST) >= deadline:
            return False
        print("waiting for 09:45 candle...", flush=True)
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", help="trading date YYYY-MM-DD (default: today IST)")
    ap.add_argument("--backfill", action="store_true",
                    help="with --date: record the day as a backtest day")
    ap.add_argument("--dry-run", action="store_true", help="print only, write nothing")
    args = ap.parse_args()

    live = args.date is None
    if live:
        now = datetime.now(IST)
        date = pd.Timestamp(now.date())
        if now.weekday() >= 5:
            print("Weekend - market closed, nothing to do.")
            return
        if not wait_for_candle(date.strftime("%Y-%m-%d")):
            print("No 09:45 candle by 10:25 IST - market holiday or data outage. Skipping.")
            return
    else:
        date = pd.Timestamp(args.date)

    syms = load_universe()
    sectors = load_sectors()
    tickers = [s + ".NS" for s in syms]
    print("Universe:", len(syms), "symbols; date:", date.strftime("%Y-%m-%d"), flush=True)

    daily_raw = download_all(tickers, "1d", "6mo")
    m5_raw = download_all(tickers, "5m", "60d")
    daily = {t: norm_daily(df) for t, df in daily_raw.items()}
    m5 = {t: norm_intra(df) for t, df in m5_raw.items()}
    nd = yf.download("^NSEI", period="6mo", interval="1d", auto_adjust=False, progress=False)
    n5 = yf.download("^NSEI", period="60d", interval="5m", auto_adjust=False, progress=False)
    if not len(nd) or not len(n5):
        print("Nifty data unavailable - aborting.")
        return
    nifty_d = norm_daily(nd)
    nifty_5 = norm_intra(n5)
    print("Data ready: daily", len(daily), "intraday", len(m5), flush=True)

    store = load_results()
    if not args.dry_run:
        score_pending(store, daily)

    day = evaluate(date, daily, m5, nifty_d, nifty_5, sectors, syms)
    if day is None:
        print("Not a trading day (or insufficient data) for", date.strftime("%Y-%m-%d"))
        if not args.dry_run:
            save_results(store)   # still persist any scoring updates
        return

    rows_by_symbol = {r["symbol"]: r for r in day.pop("_rows")}
    print(f"{day['date']} side={day['side']} nifty={day['nifty_move_pct']:+.2f}%")
    print("RJ Scanner picks:    ", [p["symbol"] for p in day["rj"]["picks"]] or "no pick")
    print("Claude Scanner picks:", [p["symbol"] for p in day["claude"]["picks"]] or "no pick")

    if not args.dry_run:
        source = "live" if live else ("backtest" if args.backfill else "live")
        # A live run must never overwrite a recorded live day with partial
        # same-day closes; scoring is refreshed by score_pending next run.
        record_day(store, day, source, rows_by_symbol)
        if live:
            # same-day closes are not final yet - clear today's returns so
            # they are scored properly by the next run
            for d in store["days"]:
                if d["date"] == day["date"]:
                    for key in ("rj", "claude"):
                        for p in d[key]["picks"]:
                            p.pop("close", None)
                            p["return_pct"] = None
        save_results(store)
        print("Saved to", RESULTS)


if __name__ == "__main__":
    main()
