"""
backfill.py
===========
Scrape and insert a specific race day (or single race) that was missed.
Inserts into the existing saratoga.db and updates training.parquet.

Usage:
  # Scrape an entire missed day
  python backfill.py --date 2025-08-15

  # Scrape a specific race on a day (useful if one race was missed)
  python backfill.py --date 2025-08-15 --race 7

  # Force re-scrape even if already in DB
  python backfill.py --date 2025-08-15 --force
"""

import argparse
import logging
import time
import random

from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from scraper  import build_driver, scrape_day, _is_complete, _page_exists
from database import Database
from config   import BASE_URL, DATA_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_TABLE_XPATH = "//table[@class='table table-hrn table-payouts']"
RESULTS_WAIT        = 25   # seconds to wait for results table to render


def backfill_date(race_date: str, race_num: int = None, force: bool = False) -> None:
    url = f"{BASE_URL}/{race_date}"

    if _is_complete(race_date) and not force:
        log.info("%s already scraped — use --force to re-scrape", race_date)
        return

    if not _page_exists(url):
        log.error("No page found for %s — check the date is correct", race_date)
        return

    log.info("Loading %s ...", race_date)
    db     = Database()
    driver = build_driver()

    try:
        driver.get(url)

        # Wait specifically for results table — slower to render than entries
        log.info("Waiting up to %ds for results table ...", RESULTS_WAIT)
        try:
            WebDriverWait(driver, RESULTS_WAIT).until(
                EC.presence_of_element_located((By.XPATH, RESULTS_TABLE_XPATH))
            )
            log.info("Results table found — scraping ...")
        except Exception:
            log.warning("Results table did not appear — entries only page")

        # Extra buffer after table appears
        time.sleep(3.0)

        race_day = scrape_day(driver, race_date)

        if not race_day.get("races"):
            log.warning("No races found on %s", race_date)
            return

        if race_num is not None:
            races = [r for r in race_day["races"] if r["race_num"] == race_num]
            if not races:
                log.error("Race %d not found on %s — available: %s",
                          race_num, race_date,
                          [r["race_num"] for r in race_day["races"]])
                return
            race_day["races"] = races
            log.info("Filtered to race %d only", race_num)

        n = db.insert_day(race_day)
        log.info("Inserted %d race(s) from %s into DB", n, race_date)

        db.export_parquet()
        log.info("Parquet updated")

        for race in race_day["races"]:
            log.info("  race%d: %d horses | %d results | %d exotics",
                     race["race_num"],
                     len([e for e in race["entries"] if not e["scratched"]]),
                     len(race["results"]),
                     len(race["exotic_payouts"]))

    except Exception as exc:
        log.error("Error backfilling %s: %s", race_date, exc, exc_info=True)
    finally:
        driver.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill a missed race day or race")
    parser.add_argument("--date",  required=True, help="Date to scrape e.g. 2025-08-15")
    parser.add_argument("--race",  type=int,      help="Specific race number (optional)")
    parser.add_argument("--force", action="store_true",
                        help="Re-scrape even if date already exists in DB")
    args = parser.parse_args()
    backfill_date(args.date, args.race, args.force)