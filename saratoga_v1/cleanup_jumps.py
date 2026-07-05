"""
cleanup_jumps.py
================
Removes all hurdle / steeplechase / jump races from:
  1. SQLite database (saratoga.db)
  2. CSV files on disk (data/saratoga/)
  3. Re-exports a clean training.parquet

A race is identified as a jump race if:
  - race_type_code >= 20  (Hurdle=20, Steeplechase=21, Hunt=22)
  - OR the race header contains 'hurdle', 'steeplechase', or 'hunt'
    (catches any that weren't encoded due to missing race type)

Run once after scraping is complete:
  python cleanup_jumps.py

Run with --dry-run to preview what would be removed without deleting:
  python cleanup_jumps.py --dry-run
"""

import argparse
import logging
import shutil
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from config import DATA_DIR, DB_PATH, PARQUET_PATH, JUMP_RACE_CODES
from database import Database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Keywords that identify jump races in the race_type or conditions fields
JUMP_KEYWORDS = {"hurdle", "steeplechase", "hunt", "jump"}


@contextmanager
def _conn(path: Path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def find_jump_races(dry_run: bool = False) -> list:
    """
    Return list of (race_id, date, race_num, race_type) for all jump races in DB.
    """
    jump_races = []
    with _conn(DB_PATH) as con:
        rows = con.execute(
            """
            SELECT race_id, date, race_num, race_type, race_type_code
            FROM races
            WHERE race_type_code >= 20
               OR LOWER(COALESCE(race_type, '')) LIKE '%hurdle%'
               OR LOWER(COALESCE(race_type, '')) LIKE '%steeplechase%'
               OR LOWER(COALESCE(race_type, '')) LIKE '%hunt%'
            ORDER BY date, race_num
            """
        ).fetchall()
        for row in rows:
            jump_races.append(dict(row))

    log.info("Found %d jump races in database", len(jump_races))
    for r in jump_races:
        log.info("  %s race%d — %s (code %s)",
                 r["date"], r["race_num"], r["race_type"], r["race_type_code"])

    return jump_races


def remove_from_db(jump_races: list, dry_run: bool = False) -> None:
    """Delete all jump races and their related rows from the database."""
    if not jump_races:
        log.info("No jump races to remove from DB")
        return

    race_ids = [r["race_id"] for r in jump_races]
    placeholders = ",".join("?" * len(race_ids))

    if dry_run:
        log.info("[DRY RUN] Would delete %d races from DB (ids: %s)",
                 len(race_ids), race_ids[:10])
        return

    with _conn(DB_PATH) as con:
        # Delete in dependency order
        for table in ["race_times", "exotic_payouts", "results", "entries"]:
            n = con.execute(
                f"DELETE FROM {table} WHERE race_id IN ({placeholders})", race_ids
            ).rowcount
            log.info("  Deleted %d rows from %s", n, table)

        n = con.execute(
            f"DELETE FROM races WHERE race_id IN ({placeholders})", race_ids
        ).rowcount
        log.info("  Deleted %d rows from races", n)

    log.info("DB cleanup complete")


def remove_from_disk(jump_races: list, dry_run: bool = False) -> None:
    """Delete CSV directories for all jump races."""
    if not jump_races:
        log.info("No jump race directories to remove")
        return

    removed = 0
    missing = 0

    for r in jump_races:
        race_dir = DATA_DIR / r["date"] / f"race{r['race_num']}"
        if race_dir.exists():
            if dry_run:
                log.info("[DRY RUN] Would delete %s", race_dir)
            else:
                shutil.rmtree(race_dir)
                log.debug("Deleted %s", race_dir)
            removed += 1
        else:
            missing += 1

    if dry_run:
        log.info("[DRY RUN] Would delete %d race directories (%d not found)",
                 removed, missing)
    else:
        log.info("Deleted %d race directories (%d already missing)", removed, missing)

    # Clean up any empty date directories left behind
    if not dry_run:
        cleaned = 0
        for date_dir in sorted(DATA_DIR.iterdir()):
            if date_dir.is_dir() and not any(date_dir.iterdir()):
                date_dir.rmdir()
                cleaned += 1
        if cleaned:
            log.info("Removed %d empty date directories", cleaned)


def reexport_parquet() -> None:
    """Re-export clean parquet after jump races are removed."""
    log.info("Re-exporting training.parquet without jump races...")
    db = Database()
    db.export_parquet()


def main(dry_run: bool = False) -> None:
    if dry_run:
        log.info("═══ DRY RUN — no changes will be made ═══")
    else:
        log.info("═══ Removing all jump / hurdle races ═══")

    # Step 1: find
    jump_races = find_jump_races()

    if not jump_races:
        log.info("No jump races found — dataset is already clean.")
        return

    # Step 2: remove from DB
    remove_from_db(jump_races, dry_run)

    # Step 3: remove from disk
    remove_from_disk(jump_races, dry_run)

    # Step 4: re-export parquet
    if not dry_run:
        reexport_parquet()
        log.info("Done — dataset is clean.")
    else:
        log.info("[DRY RUN] Complete — no changes made.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove jump/hurdle races from dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be removed without deleting")
    args = parser.parse_args()
    main(args.dry_run)
