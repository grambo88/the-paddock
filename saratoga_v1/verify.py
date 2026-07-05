import sqlite3
from config import DB_PATH

con = sqlite3.connect(DB_PATH)

# Check race counts per day
print("── Races per day ──────────────────────")
rows = con.execute("""
    SELECT date, COUNT(*) as races,
           SUM(field_size) as horses
    FROM races
    GROUP BY date
    ORDER BY date
""").fetchall()
for r in rows:
    print(f"  {r[0]}  {r[1]:>3} races  {r[2]:>4} horses")

# Check for races missing results
print("\n── Races with no results ──────────────")
rows = con.execute("""
    SELECT r.date, r.race_num, r.race_type, r.field_size
    FROM races r
    LEFT JOIN results res ON res.race_id = r.race_id
    WHERE res.race_id IS NULL
    ORDER BY r.date, r.race_num
""").fetchall()
if rows:
    for r in rows:
        print(f"  {r[0]} race{r[1]} — {r[2]} ({r[3]} horses)")
else:
    print("  None — all races have results ✓")

# Check for races missing times
print("\n── Races with no times ────────────────")
rows = con.execute("""
    SELECT r.date, r.race_num
    FROM races r
    LEFT JOIN race_times rt ON rt.race_id = r.race_id
    WHERE rt.race_id IS NULL
    ORDER BY r.date, r.race_num
""").fetchall()
if rows:
    for r in rows:
        print(f"  {r[0]} race{r[1]}")
else:
    print("  None — all races have times ✓")

# Check expected vs actual race days
known_dates = [
    "2025-07-10","2025-07-11","2025-07-12","2025-07-13",
    "2025-07-16","2025-07-17","2025-07-18","2025-07-19","2025-07-20",
    "2025-07-23","2025-07-24","2025-07-25","2025-07-26","2025-07-27",
    "2025-07-30","2025-07-31","2025-08-01","2025-08-02","2025-08-03",
    "2025-08-06","2025-08-07","2025-08-08","2025-08-09","2025-08-10",
    "2025-08-13","2025-08-14","2025-08-15","2025-08-16","2025-08-17",
    "2025-08-20","2025-08-21","2025-08-22","2025-08-23","2025-08-24",
    "2025-08-27","2025-08-28","2025-08-29","2025-08-30","2025-08-31",
    "2025-09-01",
]
scraped = [r[0] for r in con.execute("SELECT DISTINCT date FROM races ORDER BY date").fetchall()]
missing = [d for d in known_dates if d not in scraped]
print(f"\n── Missing race days ({len(missing)}/40) ────────")
if missing:
    for d in missing:
        print(f"  {d}")
else:
    print("  None — all 40 days present ✓")

con.close()