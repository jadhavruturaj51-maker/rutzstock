#!/usr/bin/env python3
"""
F&O 9:50 Scanner - independent version (runs on GitHub Actions, free).

Three scanners, evaluated every trading day at 9:50 IST on all NSE F&O stocks:

  Scanner A "RJ Scanner"     - RJ's original 12 criteria, unchanged.
  Scanner B "Muse Scanner" - V2-Quality rules + RSI extreme exclusion.
  Scanner C "One-Direction"  - Scanner B's final picks, filtered by two
                               clean-trend gates (no replacement picks):
                               efficiency ratio of the 09:15-09:45 morning
                               move >= 0.5, and the signed move by 9:50
                               within 0.8%-2.0%.

Data source: Yahoo Finance (free) via yfinance - the same source the
backtests were validated on. Criterion 11 (no event today) cannot be
Criterion 10 (OI buildup) is checked on LIVE runs only, from
Moneycontrol's live OI change for the near-month future (finalist
stocks only). Criterion 11 (no event today) is checked on LIVE runs
from 2026-11-19, via Moneycontrol board-meeting pages plus Yahoo's
calendar (earnings and ex-dividend dates); split/bonus ex-dates and
just-declared interim dividends can still be missed. Anything that
cannot be fetched stays unchecked. PASSED is shown as x/12 - the
maximum is 12/12 when both live checks pass, 11/12 with OI only,
otherwise 10/12, as in the backtests.

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
import argparse, csv, io, json, os, re, sys, time
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
MC_CODES_FILE = os.path.join(HERE, "mc_codes.json")

WIN = {"09:20", "09:25", "09:30", "09:35", "09:40", "09:45"}   # volume window
EARLY = {"09:15", "09:20", "09:25", "09:30", "09:35", "09:40", "09:45"}
ORNG = {"09:15", "09:20", "09:25"}                              # opening range


# ---------------------------------------------------------------- OI (live)
_MC_CODES = None
_MC_SESSION = None


def _mc_codes():
    global _MC_CODES
    if _MC_CODES is None:
        try:
            _MC_CODES = json.load(open(MC_CODES_FILE))
        except Exception:
            _MC_CODES = {}
    return _MC_CODES


def _mc_session():
    global _MC_SESSION
    if _MC_SESSION is None:
        _MC_SESSION = requests.Session()
        _MC_SESSION.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"})
    return _MC_SESSION


_MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
           "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def fetch_event_today(symbol, ds):
    """Is there a same-day event (criterion 11) for this symbol on ds
    (YYYY-MM-DD)? Sources: Moneycontrol's per-stock Board Meetings page
    (results / fund raising / dividend meetings) and Yahoo's calendar
    (earnings date, ex-dividend date). Returns True (event), False
    (sources checked, no event) or None (nothing could be fetched).
    Known gap: ex-dates for splits/bonus and just-declared interim
    dividends can be missing or stale in these free sources."""
    target = ds
    found = False
    checked = False
    info = _mc_codes().get(symbol)
    if info:
        S = _mc_session()
        for slug in info.get("slugs", []):
            url = ("https://www.moneycontrol.com/company-facts/%s/board-meetings/%s/"
                   % (slug, info["code"]))
            try:
                r = S.get(url, timeout=20)
                if r.status_code != 200:
                    continue
                text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
                i = text.find("Meeting Date")
                if i < 0:
                    continue
                block = text[i:i + 4000]
                for dnum, mon, yr in re.findall(r"(\d{1,2}) ([A-Za-z]{3}), (\d{4})", block):
                    if _MONTHS.get(mon) is None:
                        continue
                    dstr = "%04d-%02d-%02d" % (int(yr), _MONTHS[mon], int(dnum))
                    checked = True
                    if dstr == target:
                        found = True
                break
            except Exception:
                continue
    try:
        cal = yf.Ticker(symbol + ".NS").calendar
        if cal:
            checked = True
            for key in ("Earnings Date", "Ex-Dividend Date"):
                v = cal.get(key)
                vals = v if isinstance(v, (list, tuple)) else [v]
                for x in vals:
                    if x is not None and str(x)[:10] == target:
                        found = True
    except Exception:
        pass
    if found:
        return True
    return False if checked else None


def fetch_oi_chg_pct(symbol):
    """Live OI change % of the symbol's near-month future, from
    Moneycontrol (free, no login). Positive = OI buildup today.
    Returns None when the value cannot be fetched or parsed."""
    info = _mc_codes().get(symbol)
    if not info:
        return None
    _MC_SESSION_USED = _mc_session()
    for slug in info.get("slugs", []):
        url = ("https://www.moneycontrol.com/india/fnoquote/%s/%s"
               % (slug, info["code"]))
        for _try in range(2):
            try:
                r = _MC_SESSION_USED.get(url, timeout=20)
                if r.status_code != 200:
                    break
                text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", r.text))
                m = re.search(r"Open Int\. Chg % (-?[\d.]+)", text)
                if m:
                    return float(m.group(1))
                break
            except Exception:
                time.sleep(1)
    return None


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
    # Kaufman efficiency ratio of the morning move: |C - O| / sum |changes|
    # over [open of the 09:15 candle, closes of the 09:15..09:45 candles].
    c915 = intr_day[m == "09:15"]
    er = None
    if len(c915) and len(early):
        pts = [float(c915["Open"].iloc[0])] + [float(x) for x in early["Close"]]
        path = sum(abs(pts[i + 1] - pts[i]) for i in range(len(pts) - 1))
        if path > 0:
            er = abs(pts[-1] - pts[0]) / path
    return {
        "p950": float(c945["Close"].iloc[0]),
        "open": float(intr_day["Open"].iloc[0]),
        "vol_win": float(intr_day[np.isin(m, list(WIN))]["Volume"].sum()),
        "vwap": vwap,
        "orh": float(orng["High"].max()), "orl": float(orng["Low"].min()),
        "er": er,
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
def evaluate(date, daily, m5, nifty_d, nifty_5, sectors, syms, live_oi=False,
             live_events=False):
    """Run all three scanners for one trading date. Returns a day dict or
    None when the date is not a trading day / data is missing."""
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
            "close": close, "move_raw": round(float(move), 2),
            "move_signed": round(float(sgn * move), 3),
            "er": (round(float(st["er"]), 4) if st.get("er") is not None else None),
            "orh": st["orh"], "orl": st["orl"],
        })

    # ---- live OI state (criterion 10); only fetched on live runs, and
    # only for finalist / displayed rows (bounded fetch budget per run).
    oi_cache = {}
    oi_budget = [25]

    def oi_state(sym, fetch=False):
        if not live_oi:
            return "na", None
        if sym in oi_cache:
            v = oi_cache[sym]
        elif fetch and oi_budget[0] > 0:
            oi_budget[0] -= 1
            v = fetch_oi_chg_pct(sym)
            oi_cache[sym] = v
        else:
            return "na", None
        if v is None:
            return "na", None
        return ("pass" if v > 0 else "fail"), v

    ev_cache = {}
    ev_budget = [25]

    def ev_state(sym, fetch=False):
        if not live_events:
            return "na"
        if sym in ev_cache:
            return ev_cache[sym]
        if fetch and ev_budget[0] > 0:
            ev_budget[0] -= 1
            v = fetch_event_today(sym, ds)
            st = "na" if v is None else ("yes" if v else "no")
            ev_cache[sym] = st
            return st
        return "na"

    def live_passed(base, sym):
        p = base
        st, _ = oi_state(sym)
        if st == "pass":
            p += 1
        if ev_state(sym) == "no":
            p += 1
        return p

    def with_oi(d, sym):
        st, v = oi_state(sym)
        d["oi"] = st
        if v is not None:
            d["oi_chg_pct"] = round(float(v), 2)
        if live_events:
            d["event"] = ev_state(sym)
        return d

    # ---- display helpers (entry / SL / T1 reference levels, failed names)
    def levels(r):
        entry = r["entry"]
        if long_side:
            sl = round(float(r["orl"]), 2)
            if entry - sl <= 0:
                sl = round(entry * 0.98, 2)
            return sl, round(entry + 2 * (entry - sl), 2)
        sl = round(float(r["orh"]), 2)
        if sl - entry <= 0:
            sl = round(entry * 1.02, 2)
        return sl, round(entry - 2 * (sl - entry), 2)

    def disp(r, passed, failed_names=None):
        sl, t1 = levels(r)
        d = {"symbol": r["symbol"], "entry": r["entry"], "passed": passed,
             "move_pct": r["move_raw"], "sl": sl, "t1": t1}
        if failed_names is not None:
            d["failed"] = failed_names
        return d

    A_NAMES = {"volume": "volume 2x", "pwh": "previous week high",
               "pwl": "previous week low", "pdh": "previous day high",
               "pdl": "previous day low", "ema": "EMA 10/20",
               "sector": "sector top 5", "vwap": "VWAP",
               "orange": "opening range", "gap": "gap under 3%",
               "move": "move under 2.5%"}

    # ---- Scanner A: RJ's original criteria, 0 fails among the 10 checkable.
    a_rows = [r for r in rows if len(r["fails"]) == 0]
    a_rows.sort(key=lambda r: (-r["sector_strength"], -r["vwap_dist"]))
    a_pick_rows = []
    for r in a_rows:
        if len(a_pick_rows) >= 3:
            break
        _st, _ = oi_state(r["symbol"], fetch=True)
        if _st == "fail":
            continue
        if ev_state(r["symbol"], fetch=True) == "yes":
            continue
        a_pick_rows.append(r)
    a_pick_syms = {r["symbol"] for r in a_pick_rows}
    a_picks = []
    for r in a_pick_rows:
        d = disp(r, live_passed(10, r["symbol"]))
        if live_oi:
            d = with_oi(d, r["symbol"])
        a_picks.append(d)
    a_near_pool = [r for r in rows if r["symbol"] not in a_pick_syms]
    a_near_pool.sort(key=lambda r: (len(r["fails"]), -r["sector_strength"]))
    a_near = []
    for r in a_near_pool[:3]:
        _st, _ = oi_state(r["symbol"], fetch=True)
        _ev = ev_state(r["symbol"], fetch=True)
        _names = [A_NAMES.get(f, f) for f in r["fails"]]
        if _st == "fail":
            _names.append("OI buildup")
        if _ev == "yes":
            _names.append("event today")
        _total = 10 + (1 if _st != "na" else 0) + (1 if _ev != "na" else 0)
        d = disp(r, _total - len(_names), _names)
        if live_oi:
            d = with_oi(d, r["symbol"])
        a_near.append(d)

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
    b_pick_rows = []
    for r in gated:
        if len(b_pick_rows) >= 3:
            break
        if r["rsi"] is not None:
            if long_side and r["rsi"] > 80:
                continue
            if (not long_side) and r["rsi"] < 20:
                continue
        _st, _ = oi_state(r["symbol"], fetch=True)
        if _st == "fail":
            continue
        if ev_state(r["symbol"], fetch=True) == "yes":
            continue
        b_pick_rows.append(r)
    b_pick_syms = {r["symbol"] for r in b_pick_rows}

    # Scanner B judged by its own 10 checkable checks (market direction is
    # the 10th and is the same for every stock that day).
    def b_checks(r):
        ch = {
            "volume 1x": r["vol_ratio"] >= 1.0,
            "previous day high" if long_side else "previous day low":
                "pdh" not in r["fails"] and "pdl" not in r["fails"],
            "EMA 10/20": "ema" not in r["fails"],
            "opening range": "orange" not in r["fails"],
            "gap under 3%": "gap" not in r["fails"],
            "move under 2.5%": "move" not in r["fails"],
            "VWAP distance 0.5%": r["vwap_dist"] >= 0.5,
            "sector strength 1%": r["sector_strength"] >= 1.0,
            "RSI limit": (r["rsi"] is None or
                          (r["rsi"] <= 80 if long_side else r["rsi"] >= 20)),
        }
        if live_oi:
            _st, _ = oi_state(r["symbol"])
            ch["OI buildup"] = (True if _st == "pass" else
                                False if _st == "fail" else None)
        if live_events:
            _ev = ev_state(r["symbol"])
            ch["no event today"] = (True if _ev == "no" else
                                    False if _ev == "yes" else None)
        return ch

    def b_disp(r):
        ch = b_checks(r)
        passed = 1 + sum(1 for v in ch.values() if v)
        failed = [k for k, v in ch.items() if v is False]
        d = disp(r, passed, failed)
        if live_oi:
            d = with_oi(d, r["symbol"])
        return d

    b_picks = []
    for r in b_pick_rows:
        d = b_disp(r)
        d.pop("failed", None)
        d["passed"] = live_passed(10, r["symbol"])
        b_picks.append(d)
    pool_syms = {r["symbol"] for r in pool}
    b_near_rows = [r for r in pool if r["symbol"] not in b_pick_syms]
    b_near_rows.sort(key=lambda r: -r.get("q", 0))
    b_near_chosen = list(b_near_rows[:3])
    if len(b_near_chosen) < 3:
        rest = [r for r in rows
                if r["symbol"] not in b_pick_syms and r["symbol"] not in pool_syms]
        rest.sort(key=lambda r: (-sum(1 for v in b_checks(r).values() if v),
                                 -r["sector_strength"]))
        for r in rest:
            if len(b_near_chosen) >= 3:
                break
            b_near_chosen.append(r)
    for r in b_near_chosen:
        oi_state(r["symbol"], fetch=True)
        ev_state(r["symbol"], fetch=True)
    b_near = [b_disp(r) for r in b_near_chosen]

    # ---- Scanner C: Scanner B's final picks + two clean-trend gates,
    # no replacement picks: (a) efficiency ratio of the 09:15-09:45 morning
    # move >= 0.5; (b) signed move by 9:50 within 0.8%-2.0% inclusive.
    def c_checks(r):
        ch = {}
        for k, v in b_checks(r).items():
            if k == "move under 2.5%":
                ch["move band 0.8-2.0%"] = 0.8 <= r["move_signed"] <= 2.0
            else:
                ch[k] = v
        ch["efficiency ratio 0.5+"] = r.get("er") is not None and r["er"] >= 0.5
        return ch

    def c_disp(r):
        # failed names from c_checks (accurate), passed count on the
        # Scanner B convention (the 10 checkable checks), like b_disp.
        ch = b_checks(r)
        passed = 1 + sum(1 for v in ch.values() if v)
        failed = [k for k, v in c_checks(r).items() if v is False]
        d = disp(r, passed, failed)
        if live_oi:
            d = with_oi(d, r["symbol"])
        return d

    c_pick_rows = [r for r in b_pick_rows
                   if r.get("er") is not None and r["er"] >= 0.5
                   and 0.8 <= r["move_signed"] <= 2.0]
    c_pick_syms = {r["symbol"] for r in c_pick_rows}
    c_picks = []
    for r in c_pick_rows:
        d = c_disp(r)
        d.pop("failed", None)
        d["passed"] = live_passed(10, r["symbol"])
        c_picks.append(d)
    c_near_rows = [r for r in pool if r["symbol"] not in c_pick_syms]
    c_near_rows.sort(key=lambda r: (-sum(1 for v in c_checks(r).values() if v),
                                    -r.get("q", 0)))
    c_near_chosen = list(c_near_rows[:3])
    if len(c_near_chosen) < 3:
        rest = [r for r in rows
                if r["symbol"] not in c_pick_syms and r["symbol"] not in pool_syms]
        rest.sort(key=lambda r: (-sum(1 for v in c_checks(r).values() if v),
                                 -r["sector_strength"]))
        for r in rest:
            if len(c_near_chosen) >= 3:
                break
            c_near_chosen.append(r)
    for r in c_near_chosen:
        oi_state(r["symbol"], fetch=True)
        ev_state(r["symbol"], fetch=True)
    c_near = [c_disp(r) for r in c_near_chosen]

    return {"date": ds, "side": side, "nifty_move_pct": round(float(nmove), 2),
            "rj": {"picks": a_picks, "near": a_near, "no_pick": len(a_picks) == 0},
            "claude": {"picks": b_picks, "near": b_near, "no_pick": len(b_picks) == 0},
            "trend": {"picks": c_picks, "near": c_near, "no_pick": len(c_picks) == 0},
            "_rows": rows}


# ---------------------------------------------------------------- results store
def load_results():
    if os.path.exists(RESULTS):
        return json.load(open(RESULTS))
    return {"updated_at": None, "days": []}


def scoreboard(days):
    out = {}
    for key in ("rj", "claude", "trend"):
        picks = [p for d in days for p in (d.get(key) or {}).get("picks", [])]
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
        for key in ("rj", "claude", "trend"):
            for p in (d.get(key) or {}).get("picks", []):
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
             "rj": day["rj"], "claude": day["claude"], "trend": day["trend"]}
    for key in ("rj", "claude", "trend"):
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
    if live and any(d["date"] == date.strftime("%Y-%m-%d")
                    and d.get("source") == "live" for d in store["days"]):
        # An earlier trigger of today's scan already recorded the day
        # (backup schedule entries) - do not re-evaluate or overwrite it.
        if not args.dry_run:
            save_results(store)
        print("Today already recorded by an earlier live run - nothing to do.")
        return

    # Events check (criterion 11) starts Day 4, 2026-11-19 (RJ's call).
    day = evaluate(date, daily, m5, nifty_d, nifty_5, sectors, syms,
                   live_oi=live,
                   live_events=live and date >= pd.Timestamp("2026-11-19"))
    if day is None:
        print("Not a trading day (or insufficient data) for", date.strftime("%Y-%m-%d"))
        if not args.dry_run:
            save_results(store)   # still persist any scoring updates
        return

    rows_by_symbol = {r["symbol"]: r for r in day.pop("_rows")}
    print(f"{day['date']} side={day['side']} nifty={day['nifty_move_pct']:+.2f}%")
    print("RJ Scanner picks:    ", [p["symbol"] for p in day["rj"]["picks"]] or "no pick")
    print("Claude Scanner picks:", [p["symbol"] for p in day["claude"]["picks"]] or "no pick")
    print("One-Direction picks: ", [p["symbol"] for p in day["trend"]["picks"]] or "no pick")

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
                    for key in ("rj", "claude", "trend"):
                        for p in (d.get(key) or {}).get("picks", []):
                            p.pop("close", None)
                            p["return_pct"] = None
        save_results(store)
        print("Saved to", RESULTS)


if __name__ == "__main__":
    main()
