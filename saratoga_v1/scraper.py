"""
scraper.py
==========
Full-season scraper for Saratoga race data from horseracingnation.com.

Captures per race:
  Race level  : date, race#, post time, distance, surface, race type,
                purse, conditions, available bet types
  Entry level : program#, post position, horse, sire, trainer, jockey,
                morning line odds (raw + decimal), HRN speed figure, scratched
  Results     : finish position, win/place/show payouts, post-race speed fig
  Exotic pools: bet type, combination, $2 payout, total pool
  Race times  : fractional splits + final time (in seconds)

After each successful race day:
  → CSV files  written to data/saratoga/<date>/race<N>/
  → SQLite DB  updated via database.py
  → Parquet    re-exported (optional, set EXPORT_PARQUET=True)

Usage:
  python scraper.py                              # scrape SEASONS from config.py
  python scraper.py --year 2025                  # single year
  python scraper.py --year 2025 --start 2025-08-01  # resume from date
"""

import argparse
import csv
import logging
import random
import re
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from config import (
    BASE_URL, DATA_DIR,
    DELAY_MIN, DELAY_MAX, DISTANCE_FURLONGS,
    HEAD_DELAY_MIN, HEAD_DELAY_MAX,
    LOG_PATH, MEET_WINDOWS, PAGE_WAIT_SECONDS,
    RACE_TYPE_ENCODE, SCRAPER_HEADLESS,
    SEASONS, SURFACE_ENCODE, XPATH,
)
from database import Database

# ── Logging — console + file ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

EXPORT_PARQUET = True   # set False to skip parquet re-export after each day


# ── Driver ─────────────────────────────────────────────────────────────────────

def build_driver() -> webdriver.Chrome:
    opts = Options()
    opts.add_argument("--ignore-certificate-errors")
    opts.add_argument("--ignore-ssl-errors")
    opts.add_argument("--start-maximized")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_experimental_option("excludeSwitches", ["enable-logging", "enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    if SCRAPER_HEADLESS:
        opts.add_argument("--headless=new")
    service = Service(ChromeDriverManager().install())
    driver  = webdriver.Chrome(service=service, options=opts)
    driver.execute_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )
    return driver


# ── Parse helpers ──────────────────────────────────────────────────────────────

def _strip_currency(text: str) -> str:
    return text.replace("$", "").replace(",", "").strip()

def _parse_float(val) -> float:
    """Convert string to float; return None on failure or empty/dash."""
    s = str(val).strip()
    if s in ("", "-", "None", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None

def _odds_to_decimal(raw: str) -> float:
    """'8/1' → 8.0  |  '5/2' → 2.5  |  '4' → 4.0  |  else → None"""
    s = raw.strip()
    if "/" in s:
        try:
            n, d = s.split("/")
            return round(float(n) / float(d), 4)
        except (ValueError, ZeroDivisionError):
            return None
    return _parse_float(s)

def _parse_speed_figure(text: str) -> int:
    """Extract integer from parentheses: 'Noble Thought (65)' → 65"""
    m = re.search(r'\((-?\d+)\)', text)
    return int(m.group(1)) if m else None

def _strip_speed_figure(text: str) -> str:
    """Remove trailing '(65)' from horse name."""
    return re.sub(r'\s*\(-?\d+\)\s*$', '', text).strip()

def _parse_distance(raw: str) -> float:
    for key in sorted(DISTANCE_FURLONGS, key=len, reverse=True):
        if raw.strip().startswith(key):
            return DISTANCE_FURLONGS[key]
    log.warning("Unknown distance in '%s'", raw)
    return None

def _encode_surface(raw: str) -> tuple:
    for label, code in SURFACE_ENCODE.items():
        if label in raw:
            return label, code
    return None, None

def _encode_race_type(raw: str) -> tuple:
    for label in sorted(RACE_TYPE_ENCODE, key=len, reverse=True):
        if label in raw:
            return label, RACE_TYPE_ENCODE[label]
    return None, None

def _parse_time_to_seconds(raw: str) -> float:
    """':22.51' → 22.51  |  '1:13.53' → 73.53"""
    raw = raw.strip().lstrip(":")
    try:
        if ":" in raw:
            m, s = raw.split(":")
            return round(float(m) * 60 + float(s), 2)
        return round(float(raw), 2)
    except ValueError:
        return None

def _parse_trainer_jockey(cell_text: str) -> tuple:
    """
    HRN puts Trainer and Jockey in one cell separated by newline.
    Returns (trainer, jockey). If only one name present, assumes jockey.
    """
    parts = [p.strip() for p in cell_text.split("\n") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    if len(parts) == 1:
        return None, parts[0]
    return None, None

def _parse_program(raw: str) -> int:
    """Parse '1', '1A', '2B' → integer (strips letter suffixes)."""
    digits = re.sub(r'[^0-9]', '', raw.strip())
    return int(digits) if digits else None


# ── File I/O ───────────────────────────────────────────────────────────────────

def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

def _save_race(race_dir: Path, race: dict) -> None:
    """Write all CSVs for a single race to race_dir."""

    # info.csv
    _write_csv(race_dir / "info.csv", [
        ["date","race_num","post_time","distance_furlongs","surface",
         "surface_code","race_type","race_type_code","purse",
         "conditions","field_size","available_bets"],
        [race["date"], race["race_num"], race.get("post_time"),
         race.get("distance_furlongs"), race.get("surface"),
         race.get("surface_code"), race.get("race_type"),
         race.get("race_type_code"), race.get("purse"),
         race.get("conditions"), race.get("field_size"),
         race.get("available_bets")],
    ])

    # entries.csv
    rows = [["program_num","post_position","horse_name","sire","trainer",
             "jockey","morning_line_odds","morning_line_decimal",
             "hrn_speed_figure","scratched"]]
    for e in race.get("entries", []):
        rows.append([
            e.get("program_num"), e.get("post_position"),
            e.get("horse_name"), e.get("sire"),
            e.get("trainer"), e.get("jockey"),
            e.get("morning_line_odds"), e.get("morning_line_decimal"),
            e.get("hrn_speed_figure"), int(e.get("scratched", False)),
        ])
    _write_csv(race_dir / "entries.csv", rows)

    # results.csv
    rows = [["horse_name","finish_position","win_payout",
             "place_payout","show_payout","hrn_speed_figure_post"]]
    for r in race.get("results", []):
        rows.append([
            r.get("horse_name"), r.get("finish_position"),
            r.get("win_payout"), r.get("place_payout"),
            r.get("show_payout"), r.get("hrn_speed_figure_post"),
        ])
    _write_csv(race_dir / "results.csv", rows)

    # exotics.csv
    rows = [["bet_type","combination","payout","total_pool"]]
    for ex in race.get("exotic_payouts", []):
        rows.append([ex.get("bet_type"), ex.get("combination"),
                     ex.get("payout"), ex.get("total_pool")])
    _write_csv(race_dir / "exotics.csv", rows)

    # times.csv
    times = race.get("race_times")
    if times:
        _write_csv(race_dir / "times.csv", [
            ["fraction_1","fraction_2","fraction_3","final_time"],
            [times.get("fraction_1"), times.get("fraction_2"),
             times.get("fraction_3"), times.get("final_time")],
        ])


# ── Discovery helpers ──────────────────────────────────────────────────────────

def _page_exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=10, allow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code == 200
    except requests.RequestException:
        return False

def _is_complete(race_date: str) -> bool:
    """Return True if this date already has at least one complete race scraped."""
    date_dir  = DATA_DIR / race_date
    if not date_dir.exists():
        return False
    race_dirs = sorted([d for d in date_dir.iterdir() if d.is_dir()])
    if not race_dirs:
        return False
    first = race_dirs[0]
    return all((first / f).exists()
               for f in ["entries.csv", "results.csv", "info.csv"])

def _iter_dates(year: int, start_override: str = None):
    """
    Yield dates within the Saratoga meet window for the given year.
    Falls back to full year if year not in MEET_WINDOWS.
    """
    if year in MEET_WINDOWS:
        meet_start, meet_end = MEET_WINDOWS[year]
        start = date.fromisoformat(meet_start)
        end   = date.fromisoformat(meet_end)
    else:
        log.warning("No meet window defined for %d — checking full year", year)
        start = date(year, 1, 1)
        end   = date(year, 12, 31)

    if start_override:
        override = date.fromisoformat(start_override)
        if override > start:
            start = override

    while start <= end:
        yield start.isoformat()
        start += timedelta(days=1)

def _polite_sleep(min_s: float = DELAY_MIN, max_s: float = DELAY_MAX) -> None:
    time.sleep(random.uniform(min_s, max_s))


# ── Core scrape ────────────────────────────────────────────────────────────────

def scrape_day(driver: webdriver.Chrome, race_date: str) -> dict:
    """
    Scrape all races for race_date (YYYY-MM-DD).
    Returns: { "date": ..., "races": [ { race dict }, ... ] }
    Returns empty dict if no race tables found.
    """
    try:
        WebDriverWait(driver, PAGE_WAIT_SECONDS).until(
            EC.presence_of_element_located((By.XPATH, XPATH["entries_table"]))
        )
    except Exception:
        log.info("  No race tables found — not a race day")
        return {}

    # ── Page-level elements ───────────────────────────────────────────────────
    entry_tables   = driver.find_elements(By.XPATH, XPATH["entries_table"])
    payout_tables  = driver.find_elements(By.XPATH, XPATH["payout_table"])
    info_els       = driver.find_elements(By.XPATH, XPATH["race_distance"])
    purse_els      = driver.find_elements(By.XPATH, XPATH["race_purse"])
    conditions_els = driver.find_elements(By.XPATH, XPATH["race_conditions"])
    post_time_els  = driver.find_elements(By.XPATH, XPATH["post_time"])
    exotic_tables  = driver.find_elements(By.XPATH, XPATH["exotic_table"])
    scratched_els  = set(driver.find_elements(By.XPATH, XPATH["scratched"]))

    if not entry_tables:
        return {}

    races = []

    for race_num, entry_table in enumerate(entry_tables, start=1):

        # ── Race-level info ───────────────────────────────────────────────────
        race_header = info_els[race_num-1].text.strip()  if race_num <= len(info_els)  else ""
        purse_raw   = purse_els[race_num-1].text.strip() if race_num <= len(purse_els) else ""
        purse_raw   = _strip_currency(purse_raw.replace("Purse:", ""))

        surface_label, surface_code     = _encode_surface(race_header)
        race_type_label, race_type_code = _encode_race_type(race_header)
        distance_furlongs               = _parse_distance(race_header)

        post_time  = post_time_els[race_num-1].text.strip()  if race_num <= len(post_time_els)  else None
        conditions = conditions_els[race_num-1].text.strip() if race_num <= len(conditions_els) else None

        available_bets = None
        try:
            wager_els = driver.find_elements(By.XPATH, XPATH["wager_types"])
            if race_num <= len(wager_els):
                available_bets = wager_els[race_num-1].text.strip() or None
        except Exception:
            pass

        # ── Entries ───────────────────────────────────────────────────────────
        # Column layout: [0] #  [1] PP  [2] Horse/Sire  [3] Trainer/Jockey  [4] ML
        entries = []
        for tr in entry_table.find_elements(By.TAG_NAME, "tr"):
            cells = tr.find_elements(By.TAG_NAME, "td")
            if len(cells) < 4:
                continue

            scratched    = tr in scratched_els
            horse_cell   = cells[2].text.strip()
            trainer_cell = cells[3].text.strip()
            ml_raw       = cells[4].text.strip() if len(cells) > 4 else ""

            horse_parts = [p.strip() for p in horse_cell.split("\n") if p.strip()]
            horse_name  = horse_parts[0] if horse_parts else ""
            speed_fig   = _parse_speed_figure(horse_name)
            horse_name  = _strip_speed_figure(horse_name)
            sire        = horse_parts[1] if len(horse_parts) > 1 else None

            trainer, jockey = _parse_trainer_jockey(trainer_cell)

            ml_clean   = ml_raw.strip()
            ml_decimal = _odds_to_decimal(ml_clean) if ml_clean and ml_clean != "-" else None

            entries.append({
                "program_num":          _parse_program(cells[0].text.strip()),
                "post_position":        int(cells[1].text.strip()) if cells[1].text.strip().isdigit() else None,
                "horse_name":           horse_name,
                "sire":                 sire,
                "trainer":              trainer,
                "jockey":               jockey,
                "morning_line_odds":    ml_clean or None,
                "morning_line_decimal": ml_decimal,
                "hrn_speed_figure":     speed_fig,
                "scratched":            scratched,
            })

        field_size = sum(1 for e in entries if not e["scratched"])

        # ── Results ───────────────────────────────────────────────────────────
        # Column layout: [0] badge  [1] Runner (Speed)  [2] Win  [3] Place  [4] Show
        results = []
        if race_num <= len(payout_tables):
            position = 1
            for tr in payout_tables[race_num-1].find_elements(By.TAG_NAME, "tr"):
                cells = tr.find_elements(By.TAG_NAME, "td")
                if len(cells) < 3:
                    continue

                runner_cell = ""
                if len(cells) > 1:
                    c0 = cells[0].text.strip()
                    c1 = cells[1].text.strip()
                    runner_cell = c1 if (len(c0) <= 2 and c0.replace("-","").isdigit()) else c0

                speed_post   = _parse_speed_figure(runner_cell)
                horse_name_r = _strip_speed_figure(runner_cell)

                offset    = 2 if (len(cells) > 1 and len(cells[0].text.strip()) <= 2) else 1
                win_pay   = _parse_float(_strip_currency(cells[offset].text))   if len(cells) > offset   else None
                place_pay = _parse_float(_strip_currency(cells[offset+1].text)) if len(cells) > offset+1 else None
                show_pay  = _parse_float(_strip_currency(cells[offset+2].text)) if len(cells) > offset+2 else None

                if horse_name_r:
                    results.append({
                        "horse_name":            horse_name_r,
                        "finish_position":       position,
                        "win_payout":            win_pay,
                        "place_payout":          place_pay,
                        "show_payout":           show_pay,
                        "hrn_speed_figure_post": speed_post,
                    })
                    position += 1

        # ── Exotic payouts ────────────────────────────────────────────────────
        exotic_payouts = []
        if race_num <= len(exotic_tables):
            for tr in exotic_tables[race_num-1].find_elements(By.TAG_NAME, "tr"):
                cells = tr.find_elements(By.TAG_NAME, "td")
                if len(cells) < 3:
                    continue
                bet_type    = cells[0].text.strip()
                combination = cells[1].text.strip() or None
                payout      = _parse_float(_strip_currency(cells[2].text))
                total_pool  = _parse_float(_strip_currency(cells[3].text)) if len(cells) > 3 else None
                if bet_type:
                    exotic_payouts.append({
                        "bet_type":    bet_type,
                        "combination": combination,
                        "payout":      payout,
                        "total_pool":  total_pool,
                    })

        # ── Race times ────────────────────────────────────────────────────────
        race_times = None
        try:
            frac_els = driver.find_elements(By.XPATH, XPATH["race_fractions"])
            if race_num <= len(frac_els):
                tokens = re.findall(r'[\d]*:[\d]+\.[\d]+', frac_els[race_num-1].text)
                secs   = [_parse_time_to_seconds(t) for t in tokens]
                while len(secs) < 4:
                    secs.append(None)
                race_times = {
                    "fraction_1": secs[0],
                    "fraction_2": secs[1],
                    "fraction_3": secs[2],
                    "final_time": secs[3],
                }
        except Exception:
            pass

        # ── Assemble + save ───────────────────────────────────────────────────
        race = {
            "date":              race_date,
            "track":             "Saratoga",
            "race_num":          race_num,
            "post_time":         post_time,
            "distance_furlongs": distance_furlongs,
            "surface":           surface_label,
            "surface_code":      surface_code,
            "race_type":         race_type_label,
            "race_type_code":    race_type_code,
            "purse":             int(purse_raw) if purse_raw.isdigit() else None,
            "conditions":        conditions,
            "field_size":        field_size,
            "available_bets":    available_bets,
            "entries":           entries,
            "results":           results,
            "exotic_payouts":    exotic_payouts,
            "race_times":        race_times,
        }
        _save_race(DATA_DIR / race_date / f"race{race_num}", race)

        log.info("  race%d: %d horses | %d results | %d exotics | times: %s",
                 race_num, field_size, len(results), len(exotic_payouts),
                 "✓" if race_times else "✗")

        races.append(race)

    return {"date": race_date, "races": races}


# ── Main ──────────────────────────────────────────────────────────────────────

def main(years: list = None, start_override: str = None) -> None:
    years  = years or SEASONS
    db     = Database()
    driver = build_driver()

    total_days = total_races = already_done = no_data = 0

    try:
        for year in years:
            dates = list(_iter_dates(year, start_override))
            log.info("═══ Season %d — checking %d meet dates ═══", year, len(dates))

            for race_date in dates:
                url = f"{BASE_URL}/{race_date}"

                if _is_complete(race_date):
                    already_done += 1
                    log.debug("Already done: %s", race_date)
                    continue

                if not _page_exists(url):
                    no_data += 1
                    _polite_sleep(HEAD_DELAY_MIN, HEAD_DELAY_MAX)
                    continue

                log.info("▶ %s", race_date)
                try:
                    driver.get(url)
                    _polite_sleep(2.0, 4.0)

                    race_day = scrape_day(driver, race_date)

                    if race_day.get("races"):
                        n = db.insert_day(race_day)
                        total_days  += 1
                        total_races += n
                        log.info("✓ %s — %d races written to DB", race_date, n)

                        if EXPORT_PARQUET:
                            try:
                                db.export_parquet()
                            except Exception as exc:
                                log.warning("Parquet export failed: %s", exc)
                    else:
                        no_data += 1

                except Exception as exc:
                    log.error("✗ Error on %s: %s", race_date, exc, exc_info=True)

                _polite_sleep()

    finally:
        driver.quit()
        log.info(
            "Finished | Days: %d | Races: %d | Already done: %d | No data: %d",
            total_days, total_races, already_done, no_data,
        )
        log.info("DB: %s", db.summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Saratoga season scraper")
    parser.add_argument("--year",  type=int, help="Single year e.g. 2025")
    parser.add_argument("--start", type=str, help="Resume from date e.g. 2025-08-01")
    args  = parser.parse_args()
    main([args.year] if args.year else SEASONS, args.start)
