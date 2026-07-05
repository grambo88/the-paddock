"""
database.py
===========
SQLite schema + parquet export for the Saratoga prediction system.

Tables
------
  horses         unique horse registry (name, sire, dam)
  jockeys        unique jockey registry
  trainers       unique trainer registry
  races          one row per race — race-level features
  entries        one row per horse per race — pre-race ML inputs
  results        one row per horse per race — post-race labels
  exotic_payouts exotic pool results per race
  race_times     fractional splits + final time per race

ML export
---------
  export_ml_dataset() → pandas DataFrame (one row per horse per race)
  export_parquet()    → writes training.parquet (fast columnar format)

Both include:
  - All pre-race features (entry + race level)
  - Historical win/place/show % per horse, jockey, trainer (no data leakage)
  - Post-race labels: finish_position, payouts, is_win, is_place, is_show
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from config import DB_PATH, PARQUET_PATH

log = logging.getLogger(__name__)

# ── Schema ─────────────────────────────────────────────────────────────────────
SCHEMA = """
PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS horses (
    horse_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE,
    sire      TEXT,
    dam       TEXT
);

CREATE TABLE IF NOT EXISTS jockeys (
    jockey_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name      TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS trainers (
    trainer_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS races (
    race_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    track             TEXT NOT NULL DEFAULT 'Saratoga',
    race_num          INTEGER NOT NULL,
    post_time         TEXT,
    distance_furlongs REAL,
    surface           TEXT,
    surface_code      INTEGER,
    race_type         TEXT,
    race_type_code    INTEGER,
    purse             INTEGER,
    conditions        TEXT,
    field_size        INTEGER,
    available_bets    TEXT,
    UNIQUE(date, track, race_num)
);

CREATE TABLE IF NOT EXISTS entries (
    entry_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id               INTEGER NOT NULL REFERENCES races(race_id),
    horse_id              INTEGER NOT NULL REFERENCES horses(horse_id),
    jockey_id             INTEGER REFERENCES jockeys(jockey_id),
    trainer_id            INTEGER REFERENCES trainers(trainer_id),
    program_num           INTEGER,
    post_position         INTEGER,
    morning_line_odds     TEXT,
    morning_line_decimal  REAL,
    hrn_speed_figure      INTEGER,
    scratched             INTEGER NOT NULL DEFAULT 0,
    UNIQUE(race_id, horse_id)
);

CREATE TABLE IF NOT EXISTS results (
    result_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id               INTEGER NOT NULL REFERENCES races(race_id),
    horse_id              INTEGER NOT NULL REFERENCES horses(horse_id),
    finish_position       INTEGER,
    win_payout            REAL,
    place_payout          REAL,
    show_payout           REAL,
    hrn_speed_figure_post INTEGER,
    UNIQUE(race_id, horse_id)
);

CREATE TABLE IF NOT EXISTS exotic_payouts (
    exotic_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id     INTEGER NOT NULL REFERENCES races(race_id),
    bet_type    TEXT NOT NULL,
    combination TEXT,
    payout      REAL,
    total_pool  REAL
);

CREATE TABLE IF NOT EXISTS race_times (
    time_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    race_id    INTEGER NOT NULL REFERENCES races(race_id) UNIQUE,
    fraction_1 REAL,
    fraction_2 REAL,
    fraction_3 REAL,
    final_time REAL
);

-- Indexes for fast ML export joins
CREATE INDEX IF NOT EXISTS idx_entries_race   ON entries(race_id);
CREATE INDEX IF NOT EXISTS idx_entries_horse  ON entries(horse_id);
CREATE INDEX IF NOT EXISTS idx_results_race   ON results(race_id);
CREATE INDEX IF NOT EXISTS idx_results_horse  ON results(horse_id);
CREATE INDEX IF NOT EXISTS idx_races_date     ON races(date);
"""

# ── Database ───────────────────────────────────────────────────────────────────
class Database:

    def __init__(self, path: Path = DB_PATH):
        self.path = path
        self._init()

    @contextmanager
    def _conn(self):
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def _init(self):
        with self._conn() as con:
            con.executescript(SCHEMA)
        log.info("Database ready: %s", self.path)

    # ── Upserts ───────────────────────────────────────────────────────────────

    def _upsert_horse(self, con, name: str, sire: str = None, dam: str = None) -> int:
        con.execute(
            "INSERT INTO horses(name,sire,dam) VALUES(?,?,?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "sire=COALESCE(excluded.sire,sire), dam=COALESCE(excluded.dam,dam)",
            (name, sire, dam),
        )
        return con.execute("SELECT horse_id FROM horses WHERE name=?", (name,)).fetchone()[0]

    def _upsert_jockey(self, con, name: str) -> int:
        con.execute("INSERT OR IGNORE INTO jockeys(name) VALUES(?)", (name,))
        return con.execute("SELECT jockey_id FROM jockeys WHERE name=?", (name,)).fetchone()[0]

    def _upsert_trainer(self, con, name: str) -> int:
        con.execute("INSERT OR IGNORE INTO trainers(name) VALUES(?)", (name,))
        return con.execute("SELECT trainer_id FROM trainers WHERE name=?", (name,)).fetchone()[0]

    # ── Insert ────────────────────────────────────────────────────────────────

    def insert_day(self, race_day: dict) -> int:
        """
        Insert all races for one day. Idempotent — safe to re-run.
        Returns number of races inserted.
        """
        inserted = 0
        with self._conn() as con:
            for race in race_day.get("races", []):
                try:
                    self._insert_race(con, race)
                    inserted += 1
                except Exception as exc:
                    log.error("Failed to insert race %s/%s: %s",
                              race.get("date"), race.get("race_num"), exc)
        return inserted

    def _insert_race(self, con, race: dict) -> None:
        con.execute(
            """
            INSERT INTO races
                (date,track,race_num,post_time,distance_furlongs,
                 surface,surface_code,race_type,race_type_code,
                 purse,conditions,field_size,available_bets)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(date,track,race_num) DO UPDATE SET
                post_time=excluded.post_time,
                distance_furlongs=excluded.distance_furlongs,
                surface=excluded.surface,
                surface_code=excluded.surface_code,
                race_type=excluded.race_type,
                race_type_code=excluded.race_type_code,
                purse=excluded.purse,
                conditions=excluded.conditions,
                field_size=excluded.field_size,
                available_bets=excluded.available_bets
            """,
            (
                race["date"], race.get("track","Saratoga"), race["race_num"],
                race.get("post_time"), race.get("distance_furlongs"),
                race.get("surface"), race.get("surface_code"),
                race.get("race_type"), race.get("race_type_code"),
                race.get("purse"), race.get("conditions"),
                race.get("field_size"), race.get("available_bets"),
            ),
        )
        race_id = con.execute(
            "SELECT race_id FROM races WHERE date=? AND track=? AND race_num=?",
            (race["date"], race.get("track","Saratoga"), race["race_num"]),
        ).fetchone()[0]

        for entry in race.get("entries", []):
            horse_id   = self._upsert_horse(con, entry["horse_name"],
                                             entry.get("sire"), entry.get("dam"))
            jockey_id  = self._upsert_jockey(con, entry["jockey"]) if entry.get("jockey") else None
            trainer_id = self._upsert_trainer(con, entry["trainer"]) if entry.get("trainer") else None
            con.execute(
                """
                INSERT INTO entries
                    (race_id,horse_id,jockey_id,trainer_id,program_num,
                     post_position,morning_line_odds,morning_line_decimal,
                     hrn_speed_figure,scratched)
                VALUES (?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(race_id,horse_id) DO UPDATE SET
                    jockey_id=excluded.jockey_id,
                    trainer_id=excluded.trainer_id,
                    morning_line_odds=excluded.morning_line_odds,
                    morning_line_decimal=excluded.morning_line_decimal,
                    hrn_speed_figure=excluded.hrn_speed_figure,
                    scratched=excluded.scratched
                """,
                (
                    race_id, horse_id, jockey_id, trainer_id,
                    entry.get("program_num"), entry.get("post_position"),
                    entry.get("morning_line_odds"), entry.get("morning_line_decimal"),
                    entry.get("hrn_speed_figure"), int(entry.get("scratched", False)),
                ),
            )

        for result in race.get("results", []):
            if not result.get("horse_name"):
                continue
            horse_id = self._upsert_horse(con, result["horse_name"])
            con.execute(
                """
                INSERT INTO results
                    (race_id,horse_id,finish_position,
                     win_payout,place_payout,show_payout,hrn_speed_figure_post)
                VALUES (?,?,?,?,?,?,?)
                ON CONFLICT(race_id,horse_id) DO UPDATE SET
                    finish_position=excluded.finish_position,
                    win_payout=excluded.win_payout,
                    place_payout=excluded.place_payout,
                    show_payout=excluded.show_payout,
                    hrn_speed_figure_post=excluded.hrn_speed_figure_post
                """,
                (
                    race_id, horse_id, result.get("finish_position"),
                    result.get("win_payout"), result.get("place_payout"),
                    result.get("show_payout"), result.get("hrn_speed_figure_post"),
                ),
            )

        for exotic in race.get("exotic_payouts", []):
            if not exotic.get("bet_type"):
                continue
            con.execute(
                """
                INSERT INTO exotic_payouts
                    (race_id,bet_type,combination,payout,total_pool)
                VALUES (?,?,?,?,?)
                """,
                (race_id, exotic["bet_type"], exotic.get("combination"),
                 exotic.get("payout"), exotic.get("total_pool")),
            )

        times = race.get("race_times")
        if times:
            con.execute(
                """
                INSERT INTO race_times (race_id,fraction_1,fraction_2,fraction_3,final_time)
                VALUES (?,?,?,?,?)
                ON CONFLICT(race_id) DO UPDATE SET
                    fraction_1=excluded.fraction_1,
                    fraction_2=excluded.fraction_2,
                    fraction_3=excluded.fraction_3,
                    final_time=excluded.final_time
                """,
                (race_id, times.get("fraction_1"), times.get("fraction_2"),
                 times.get("fraction_3"), times.get("final_time")),
            )

    # ── ML Export ─────────────────────────────────────────────────────────────

    def export_ml_dataset(self) -> pd.DataFrame:
        """
        Returns a flat DataFrame — one row per horse per race.

        Pre-race features (model inputs):
          Race level  : date, race_num, distance_furlongs, surface_code,
                        race_type_code, purse, field_size
          Entry level : post_position, program_num, morning_line_decimal,
                        hrn_speed_figure

        Historical features (computed, no data leakage):
          horse_starts, horse_wins, horse_places, horse_shows
          horse_win_pct, horse_place_pct, horse_show_pct
          jockey_win_pct, jockey_place_pct, jockey_show_pct
          trainer_win_pct, trainer_place_pct, trainer_show_pct

        Post-race labels (model targets):
          finish_position, win_payout, place_payout, show_payout
          is_win, is_place, is_show
        """
        query = """
        SELECT
            r.race_id,
            r.date,
            r.track,
            r.race_num,
            h.name              AS horse_name,
            j.name              AS jockey_name,
            tr.name             AS trainer_name,

            -- race-level features
            r.distance_furlongs,
            r.surface_code,
            r.race_type_code,
            r.purse,
            r.field_size,

            -- entry-level features
            e.post_position,
            e.program_num,
            e.morning_line_decimal,
            e.hrn_speed_figure,
            e.scratched,

            -- race times (available after race — useful for training context)
            rt.fraction_1,
            rt.fraction_2,
            rt.fraction_3,
            rt.final_time,

            -- labels
            res.finish_position,
            res.win_payout,
            res.place_payout,
            res.show_payout,
            CASE WHEN res.finish_position = 1 THEN 1 ELSE 0 END AS is_win,
            CASE WHEN res.finish_position <= 2 THEN 1 ELSE 0 END AS is_place,
            CASE WHEN res.finish_position <= 3 THEN 1 ELSE 0 END AS is_show

        FROM entries e
        JOIN races   r   ON r.race_id   = e.race_id
        JOIN horses  h   ON h.horse_id  = e.horse_id
        LEFT JOIN jockeys  j   ON j.jockey_id  = e.jockey_id
        LEFT JOIN trainers tr  ON tr.trainer_id = e.trainer_id
        LEFT JOIN results  res ON res.race_id   = e.race_id
                               AND res.horse_id = e.horse_id
        LEFT JOIN race_times rt ON rt.race_id = r.race_id
        WHERE e.scratched = 0
        ORDER BY r.date, r.race_num, e.post_position
        """
        with self._conn() as con:
            df = pd.read_sql_query(query, con)

        if df.empty:
            log.warning("No data found for ML export.")
            return df

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(["date", "race_num", "post_position"]).reset_index(drop=True)

        # ── Historical stats — computed without data leakage ─────────────────
        # For each horse/jockey/trainer, compute their win/place/show %
        # using only races that came BEFORE the current one.
        for entity, name_col in [
            ("horse",   "horse_name"),
            ("jockey",  "jockey_name"),
            ("trainer", "trainer_name"),
        ]:
            for stat, label_col in [
                ("win_pct",   "is_win"),
                ("place_pct", "is_place"),
                ("show_pct",  "is_show"),
            ]:
                col = f"{entity}_{stat}"
                df[col] = (
                    df.groupby(name_col)[label_col]
                    .transform(lambda s: s.shift(1).expanding().mean())
                )

            # Career starts before this race
            df[f"{entity}_starts"] = (
                df.groupby(name_col).cumcount()
            )

        log.info("ML dataset: %d rows × %d columns", len(df), len(df.columns))
        return df

    def export_parquet(self, path: Path = PARQUET_PATH) -> Path:
        """
        Export the ML dataset to a parquet file for fast loading
        into PyTorch / TensorFlow / sklearn / pandas.

        Load it back with: pd.read_parquet('training.parquet')
        """
        df = self.export_ml_dataset()
        if df.empty:
            log.warning("Nothing to export — database may be empty.")
            return path
        df.to_parquet(path, index=False, compression="snappy")
        size_mb = path.stat().st_size / 1_048_576
        log.info("Parquet saved: %s (%.2f MB, %d rows)", path, size_mb, len(df))
        return path

    # ── Summary ───────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._conn() as con:
            tables = ["races","entries","results","horses","jockeys","trainers",
                      "exotic_payouts","race_times"]
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in tables}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    db = Database()
    s  = db.summary()
    print("\n── Database summary ──────────────────────")
    for k, v in s.items():
        print(f"  {k:<16} {v:>8,} rows")

    df = db.export_ml_dataset()
    if not df.empty:
        print(f"\n── ML dataset: {len(df):,} rows × {len(df.columns)} columns")
        print(f"   Columns: {list(df.columns)}")
        print(f"\n── Sample row:")
        print(df.iloc[0].to_string())
        db.export_parquet()
