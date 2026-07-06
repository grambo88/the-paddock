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
SEASONS           = [2025, 2024, 2023]

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
    2024: [
        "2024-07-11", "2024-07-12", "2024-07-13", "2024-07-14",
        "2024-07-17", "2024-07-18", "2024-07-19", "2024-07-20", "2024-07-21",
        "2024-07-24", "2024-07-25", "2024-07-26", "2024-07-27", "2024-07-28",
        "2024-07-31", "2024-08-01", "2024-08-02", "2024-08-03", "2024-08-04",
        "2024-08-07", "2024-08-08", "2024-08-09", "2024-08-10", "2024-08-11",
        "2024-08-14", "2024-08-15", "2024-08-16", "2024-08-17", "2024-08-18",
        "2024-08-21", "2024-08-22", "2024-08-23", "2024-08-24", "2024-08-25",
        "2024-08-28", "2024-08-29", "2024-08-30", "2024-08-31",
        "2024-09-01", "2024-09-02",
    ],
    2023: [
        "2023-07-13", "2023-07-14", "2023-07-15", "2023-07-16",
        "2023-07-19", "2023-07-20", "2023-07-21", "2023-07-22", "2023-07-23",
        "2023-07-26", "2023-07-27", "2023-07-28", "2023-07-29", "2023-07-30",
        "2023-08-02", "2023-08-03", "2023-08-04", "2023-08-05", "2023-08-06",
        "2023-08-09", "2023-08-10", "2023-08-11", "2023-08-12", "2023-08-13",
        "2023-08-16", "2023-08-17", "2023-08-18", "2023-08-19", "2023-08-20",
        "2023-08-23", "2023-08-24", "2023-08-25", "2023-08-26", "2023-08-27",
        "2023-08-30", "2023-08-31", "2023-09-01", "2023-09-02", "2023-09-03",
        "2023-09-04",
    ],
    2022: [
        "2022-07-14", "2022-07-15", "2022-07-16", "2022-07-17",
        "2022-07-20", "2022-07-21", "2022-07-22", "2022-07-23", "2022-07-24",
        "2022-07-27", "2022-07-28", "2022-07-29", "2022-07-30", "2022-07-31",
        "2022-08-03", "2022-08-04", "2022-08-05", "2022-08-06", "2022-08-07",
        "2022-08-10", "2022-08-11", "2022-08-12", "2022-08-13", "2022-08-14",
        "2022-08-17", "2022-08-18", "2022-08-19", "2022-08-20", "2022-08-21",
        "2022-08-24", "2022-08-25", "2022-08-26", "2022-08-27", "2022-08-28",
        "2022-08-31", "2022-09-01", "2022-09-02", "2022-09-03", "2022-09-04",
        "2022-09-05",
    ],
    2021: [
        "2021-07-15", "2021-07-16", "2021-07-17", "2021-07-18",
        "2021-07-21", "2021-07-22", "2021-07-23", "2021-07-24", "2021-07-25",
        "2021-07-28", "2021-07-29", "2021-07-30", "2021-07-31", "2021-08-01",
        "2021-08-04", "2021-08-05", "2021-08-06", "2021-08-07", "2021-08-08",
        "2021-08-11", "2021-08-12", "2021-08-13", "2021-08-14", "2021-08-15",
        "2021-08-18", "2021-08-19", "2021-08-20", "2021-08-21", "2021-08-22",
        "2021-08-25", "2021-08-26", "2021-08-27", "2021-08-28", "2021-08-29",
        "2021-09-01", "2021-09-02", "2021-09-03", "2021-09-04", "2021-09-05",
        "2021-09-06",
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
