"""
config.py
=========
All settings for the Saratoga horse racing prediction system.
This is the only file you should need to edit for basic configuration.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data" / "saratoga"
DB_PATH       = BASE_DIR / "saratoga.db"
PARQUET_PATH  = BASE_DIR / "training.parquet"
LOG_PATH      = BASE_DIR / "scraper.log"

# ── Scraper ────────────────────────────────────────────────────────────────────
TRACK_SLUG        = "saratoga"
BASE_URL          = f"https://entries.horseracingnation.com/entries-results/{TRACK_SLUG}"
SEASONS           = [2025]

# Exact race dates for each season — avoids checking dark days
# and ensures no race days are missed
RACE_DATES = {
    2025: [
        "2025-07-10", "2025-07-11", "2025-07-12", "2025-07-13",
        "2025-07-16", "2025-07-17", "2025-07-18", "2025-07-19", "2025-07-20",
        "2025-07-23", "2025-07-24", "2025-07-25", "2025-07-26", "2025-07-27",
        "2025-07-30", "2025-07-31", "2025-08-01", "2025-08-02", "2025-08-03",
        "2025-08-06", "2025-08-07", "2025-08-08", "2025-08-09", "2025-08-10",
        "2025-08-13", "2025-08-14", "2025-08-15", "2025-08-16", "2025-08-17",
        "2025-08-20", "2025-08-21", "2025-08-22", "2025-08-23", "2025-08-24",
        "2025-08-27", "2025-08-28", "2025-08-29", "2025-08-30", "2025-08-31",
        "2025-09-01",
    ],
}
SCRAPER_HEADLESS  = True    # headless = faster, no visual rendering
DELAY_MIN         = 2.0     # seconds — min pause between page loads
DELAY_MAX         = 5.0     # seconds — max pause between page loads
HEAD_DELAY_MIN    = 0.3     # pause after a HEAD miss
HEAD_DELAY_MAX    = 0.8
PAGE_WAIT_SECONDS = 10      # headless loads faster, 10s is sufficient

# Saratoga meet typically runs late July through early September.
# These windows keep the scraper focused — only dates in range are checked.
# If a year's data turns out to be missing, the HEAD check will skip it cleanly.
MEET_WINDOWS = {
    2018: ("2018-07-20", "2018-09-03"),
    2019: ("2019-07-11", "2019-09-02"),
    2020: ("2020-07-16", "2020-09-07"),  # COVID — shortened/modified meet
    2021: ("2021-07-15", "2021-09-06"),
    2022: ("2022-07-14", "2022-09-05"),
    2023: ("2023-07-13", "2023-09-04"),
    2024: ("2024-07-10", "2024-09-05"),
    2025: ("2025-07-09", "2025-09-01"),
}

# ── XPaths — update here if the site changes its HTML structure ────────────────
XPATH = {
    "entries_table":  "//table[@class='table table-sm table-hrn table-entries']",
    "payout_table":   "//table[@class='table table-hrn table-payouts']",
    "race_distance":  "//*[@class='col-lg-auto flex-grow-1 race-distance']",
    "race_purse":     "//*[@class='col-lg-auto race-purse']",
    "race_conditions":"//*[contains(@class,'race-conditions') or contains(@class,'race-restriction')]",
    "post_time":      "//*[contains(@class,'race-post-time') or contains(@class,'post-time')]",
    "wager_types":    "//div[contains(@class,'race-wager-types')]",
    "scratched":      "//*[@class='scratched']",
    "race_fractions": "//div[contains(@class,'race-fractions') or contains(text(),'Fractions')]",
    "exotic_table":   "//table[contains(@class,'table-exotic') or contains(@class,'table-pools') or contains(@class,'table-wagers')]",
}

# ── Track surface encoding ─────────────────────────────────────────────────────
SURFACE_ENCODE = {
    "Dirt":       0,
    "Turf":       1,
    "Inner Turf": 2,
    "Wet Dirt":   3,
    "Yielding":   4,
}

# ── Race type encoding ─────────────────────────────────────────────────────────
RACE_TYPE_ENCODE = {
    # Flat races
    "Maiden Special Weight":       0,
    "Maiden Claiming":             1,
    "Maiden":                      2,
    "Claiming":                    3,
    "Allowance Optional Claiming": 4,
    "Allowance":                   5,
    "Starter Allowance":           6,
    "Starter Claiming":            7,
    "Stakes":                      8,
    "Handicap":                    9,
    "Graded Stakes":               10,
    # Jump races — flagged separately so they can be filtered from flat ML models
    "Hurdle":                      20,
    "Steeplechase":                21,
    "Hunt":                        22,
}

# Jump race codes — used to filter hurdle/steeplechase from flat race training data
JUMP_RACE_CODES = {20, 21, 22}

# ── Distance → furlongs ────────────────────────────────────────────────────────
DISTANCE_FURLONGS = {
    # Flat distances
    "4F":       4.0,
    "4 1/2F":   4.5,
    "5F":       5.0,
    "5 1/2F":   5.5,
    "6F":       6.0,
    "6 1/2F":   6.5,
    "7F":       7.0,
    "1M":       8.0,
    "1 1/16M":  8.5,
    "1 1/8M":   9.0,
    "1 3/16M":  9.5,
    "1 1/4M":  10.0,
    "1 3/8M":  11.0,
    "1 1/2M":  12.0,
    "1 5/8M":  13.0,
    "1 3/4M":  14.0,
    # Hurdle / steeplechase distances
    "2M":       16.0,
    "2 1/16M":  16.5,
    "2 1/8M":   17.0,
    "2 3/16M":  17.5,
    "2 1/4M":   18.0,
    "2 3/8M":   19.0,
    "2 1/2M":   20.0,
    "2 5/8M":   21.0,
    "2 3/4M":   22.0,
    "3M":       24.0,
}

# ── Exotic bet types to capture ────────────────────────────────────────────────
EXOTIC_BET_TYPES = [
    "Exacta", "Quinella", "Trifecta", "Superfecta",
    "Daily Double", "Pick 3", "Pick 4", "Pick 5", "Pick 6",
]
