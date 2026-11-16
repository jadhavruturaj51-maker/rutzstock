# Rutzstock — F&O 9:50 Scanner (independent)

A fully independent stock scanner that lives in this GitHub repository.
It does **not** depend on Muse or any paid service:

- **GitHub Actions** runs `scanner.py` every trading day at **9:50 am IST**
  (Mon–Fri) on GitHub's free runners.
- The scanner checks all **NSE Futures & Options stocks** against three
  criteria sets and writes the result to `docs/results.json`.
- **GitHub Pages** serves `docs/index.html`, which renders the picks,
  the running scoreboard and the full history.

## The three scanners

- **Scanner A — "RJ Scanner":** the original 12 criteria, unchanged
  (volume ≥ 2x, previous-week + previous-day breakouts, EMA10/EMA20,
  sector top-5, VWAP, opening range, gap < 3%, move < 2.5% by 9:50).
- **Scanner B — "Muse Scanner":** V2-Quality rules (volume ≥ 1.0x, no
  previous-week/sector hard gates, VWAP distance ≥ 0.3% pool, quality gate
  sector strength ≥ 1.0% + VWAP distance ≥ 0.5%) plus the RSI extreme
  exclusion (LONG: 15m RSI ≤ 80, SHORT: RSI ≥ 20).
- **Scanner C — "One-Direction":** Scanner B's final picks, filtered by
  two clean-trend gates with no replacement picks: efficiency ratio of
  the 9:15–9:45 morning move ≥ 0.5 (a clean, straight move, not zigzag),
  and the signed move by 9:50 within 0.8%–2.0%.

Both: market direction decides the side, max 3 picks/day, ranked by sector
strength. A no-pick day is a valid result.

## Honest limits

- Data comes from **Yahoo Finance (free)** — the same source the backtests
  were validated on. Criteria 10 (OI buildup) and 11 (no event today)
  cannot be checked from free data, so PASSED is reported out of the 10
  checkable criteria, shown as `x/12`, exactly as in the backtests.
- GitHub's scheduler can start a run a few minutes late; `scanner.py`
  waits for the 09:45 candle itself (until 10:25 IST) before scanning.
- On market holidays the run detects the missing candle and skips.
- This is a personal research tool, not investment advice.

## One-time setup (after creating the repo)

1. Push all these files to a **public** GitHub repository (public repos get
   free GitHub Pages and generous free Actions minutes).
2. Enable Pages: **Settings → Pages → Build and deployment → Deploy from a
   branch → Branch: `main`, Folder: `/docs`** → Save.
3. (Optional, to test immediately) **Actions → Daily 9:50 F&O scan → Run
   workflow**. Note: a manual run outside market hours will wait for a
   candle that never comes and skip — use a backfill instead (below).

Your site will be at `https://<your-username>.github.io/<repo-name>/`.

## Backfill / test a past day locally

```bash
pip install -r requirements.txt
python scanner.py --date 2026-11-13 --dry-run     # print only
python scanner.py --date 2026-11-13 --backfill   # also record it (marked "backtest")
```

## How results are stored

`docs/results.json` holds every scanned day (all three scanners' picks, 9:50
entry prices, and returns to that day's close, filled in by the next run)
plus the running scoreboard. The daily commit from the workflow also keeps
the schedule active.
