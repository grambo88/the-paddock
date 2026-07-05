"""
test_scrape.py
==============
Collects N days of Saratoga race data to validate format
before running a full season scrape.

Writes to data/saratoga-test/ and saratoga-test.db
so it never touches production data.

Usage:
  python test_scrape.py                      # 5 days from meet start
  python test_scrape.py --days 3            # fewer days
  python test_scrape.py --start 2025-08-01  # specific start date
  python test_scrape.py --year 2024         # 2024 season test
"""

import argparse
import logging
import random
import time

import config as _cfg

# ── Redirect to test dirs before importing anything that reads config ──────────
_cfg.DATA_DIR     = _cfg.BASE_DIR / "data" / "saratoga-test"
_cfg.DB_PATH      = _cfg.BASE_DIR / "saratoga-test.db"
_cfg.PARQUET_PATH = _cfg.BASE_DIR / "training-test.parquet"

from scraper  import (build_driver, scrape_day, _page_exists,
                       _polite_sleep, _iter_dates)
from database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_DAYS = 2
DEFAULT_YEAR = 2025


# ── Summary printers ───────────────────────────────────────────────────────────

def print_race_summary(all_race_days: list) -> None:
    sep = "═" * 76
    print(f"\n{sep}")
    print("  SARATOGA TEST SCRAPE — DATA SUMMARY")
    print(sep)

    total_races = total_horses = total_results = total_exotics = 0

    for rd in all_race_days:
        print(f"\n  📅  {rd['date']}  ({len(rd['races'])} races)")
        print(f"  {'─'*70}")

        for race in rd["races"]:
            entries = [e for e in race["entries"] if not e["scratched"]]
            results = race["results"]
            exotics = race["exotic_payouts"]
            times   = race["race_times"]

            purse_str = f"${race.get('purse') or 0:,}"
            print(
                f"    Race {race['race_num']:>2}  "
                f"{str(race.get('distance_furlongs') or '?')+'f':>6}  "
                f"{str(race.get('surface') or '?'):>10}  "
                f"{str(race.get('race_type') or '?'):>25}  "
                f"Purse: {purse_str:>10}  "
                f"Field: {len(entries):>2}  "
                f"Results:{'✓' if results else '✗'}  "
                f"Exotics:{len(exotics):>2}  "
                f"Times:{'✓' if times else '✗'}"
            )

            for e in entries[:3]:
                jock   = e.get("jockey") or "?"
                trn    = e.get("trainer") or "?"
                ml     = e.get("morning_line_odds") or "?"
                mld    = e.get("morning_line_decimal")
                sf     = e.get("hrn_speed_figure")
                mld_str = f"{mld:.2f}" if mld is not None else "?"
                sf_str  = str(sf) if sf is not None else "?"
                print(
                    f"         PP{e.get('post_position','?'):>2}  "
                    f"{str(e.get('horse_name','?')):<24}  "
                    f"J: {jock:<22}  T: {trn:<22}  "
                    f"ML: {ml:>5} ({mld_str})  SpFig: {sf_str}"
                )
            if len(entries) > 3:
                print(f"         ... {len(entries)-3} more horses")

            winner = next((r for r in results if r.get("finish_position") == 1), None)
            if winner:
                print(
                    f"         🏆  {winner['horse_name']:<24}  "
                    f"Win: ${winner.get('win_payout') or '-'}  "
                    f"Pl: ${winner.get('place_payout') or '-'}  "
                    f"Sh: ${winner.get('show_payout') or '-'}  "
                    f"SpFig: {winner.get('hrn_speed_figure_post') or '?'}"
                )

            if exotics:
                ex   = exotics[0]
                pool = f"Pool: ${ex.get('total_pool'):,}" if ex.get("total_pool") else ""
                print(
                    f"         🎰  {ex['bet_type']}: {ex.get('combination','?')}  "
                    f"Pays: ${ex.get('payout','?')}  {pool}"
                )

            if times:
                fracs    = [times.get(f"fraction_{i}") for i in range(1, 4)]
                frac_str = "  ".join(f"{f:.2f}s" for f in fracs if f)
                final    = times.get("final_time")
                if final:
                    print(f"         ⏱   Fractions: {frac_str}  Final: {final:.2f}s")

            total_races   += 1
            total_horses  += len(entries)
            total_results += len(results)
            total_exotics += len(exotics)

    print(f"\n{sep}")
    print(
        f"  TOTALS  Days:{len(all_race_days)}  Races:{total_races}  "
        f"Horses:{total_horses}  Results:{total_results}  Exotics:{total_exotics}"
    )
    print(sep)


def print_db_stats(db: Database) -> None:
    print("\n── SQLite DB ─────────────────────────────────────")
    for k, v in db.summary().items():
        print(f"   {k:<18} {v:>6,} rows")

    try:
        df = db.export_ml_dataset()
        if not df.empty:
            print(f"\n── ML dataset: {len(df):,} rows × {len(df.columns)} columns")
            print(f"   Columns:\n   {list(df.columns)}")
            print(f"\n── Sample row (first horse, first race):")
            print(df.iloc[0].to_string())
            db.export_parquet()
            print(f"\n── Parquet written: {_cfg.PARQUET_PATH}")
    except Exception as e:
        log.warning("ML export preview failed: %s", e)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(year: int = DEFAULT_YEAR, target_days: int = DEFAULT_DAYS,
         start: str = None) -> None:

    log.info("Test scrape | year=%d | days=%d | output=%s | db=%s",
             year, target_days, _cfg.DATA_DIR, _cfg.DB_PATH)

    db     = Database()
    driver = build_driver()

    collected = []
    checked   = 0
    no_data   = 0

    try:
        for race_date in _iter_dates(year, start):
            if len(collected) >= target_days:
                break

            url = f"{_cfg.BASE_URL}/{race_date}"
            checked += 1

            if not _page_exists(url):
                no_data += 1
                time.sleep(random.uniform(0.5, 1.2))
                continue

            log.info("▶ [%d/%d] %s", len(collected)+1, target_days, race_date)

            try:
                driver.get(url)
                time.sleep(random.uniform(2.0, 4.0))

                # TEMP DEBUG — remove after
                from selenium.webdriver.common.by import By
                tables = driver.find_elements(By.XPATH, "//table[@class='table table-sm table-hrn table-entries']")
                if tables:
                    rows = tables[0].find_elements(By.TAG_NAME, "tr")
                    for tr in rows[:3]:
                        cells = tr.find_elements(By.TAG_NAME, "td")
                        for i, cell in enumerate(cells):
                            print(f"  cell[{i}] text='{cell.text}' inner='{cell.get_attribute('innerHTML')[:100]}'")
                        print("---")
                # END DEBUG

                race_day = scrape_day(driver, race_date)

                if race_day.get("races"):
                    db.insert_day(race_day)
                    collected.append(race_day)
                    log.info("✓ %s — %d races", race_date, len(race_day["races"]))
                else:
                    no_data += 1
                    log.info("  No races: %s", race_date)

            except Exception as exc:
                log.error("✗ %s: %s", race_date, exc, exc_info=True)

            if len(collected) < target_days:
                _polite_sleep()

    finally:
        driver.quit()
        log.info("Checked %d dates | Found %d race days | No data: %d",
                 checked, len(collected), no_data)

    if collected:
        print_race_summary(collected)
        print_db_stats(db)
    else:
        log.warning("No race data collected.")
        log.warning("Check MEET_WINDOWS in config.py is correct for %d.", year)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saratoga test scrape")
    parser.add_argument("--year",  type=int, default=DEFAULT_YEAR)
    parser.add_argument("--days",  type=int, default=DEFAULT_DAYS)
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    args = parser.parse_args()
    main(args.year, args.days, args.start)
