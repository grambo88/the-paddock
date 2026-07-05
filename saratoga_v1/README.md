# grambo88
# Saratoga Horse Racing — ML Dataset Pipeline

Scrapes Saratoga race results from horseracingnation.com and builds a
neural-network-ready dataset for Win / Place / Show prediction.

## Project structure

```
saratoga/
├── config.py         ← all settings (edit here first)
├── scraper.py        ← Selenium scraper → CSV + SQLite + Parquet
├── database.py       ← SQLite schema + ML export
├── test_scrape.py    ← collect 5 days to validate before full run
├── requirements.txt
└── data/
    └── saratoga/
        └── YYYY-MM-DD/
            └── race<N>/
                ├── entries.csv    pre-race features
                ├── results.csv    post-race labels
                ├── new_info.csv   race-level info
                ├── exotics.csv    exotic pool payouts
                └── times.csv      fractional + final times
```

## Quickstart

```bash
pip install -r requirements.txt

# Validate first — collect 5 race days
python test_scrape.py

# If data looks good, run the full season
python scraper.py --year 2025
python scraper.py --year 2024

# Check DB and export parquet
python database.py
```

## Resuming a scrape

```bash
# Resume 2025 from a specific date
python scraper.py --year 2025 --start 2025-08-15
```

## Loading the training data in Python

```python
import pandas as pd

df = pd.read_parquet("training.parquet")
print(df.shape)          # (rows, columns)
print(df.columns)        # all features + labels

# Pre-race feature columns (model inputs)
feature_cols = [
    "distance_furlongs", "surface_code", "race_type_code",
    "purse", "field_size", "post_position", "morning_line_decimal",
    "hrn_speed_figure", "horse_win_pct", "horse_place_pct",
    "horse_show_pct", "jockey_win_pct", "jockey_place_pct",
    "jockey_show_pct", "trainer_win_pct", "trainer_place_pct",
    "trainer_show_pct", "horse_starts",
]

# Label columns (model targets)
label_cols = ["is_win", "is_place", "is_show"]

X = df[feature_cols].fillna(0).values
y = df[label_cols].values
```

## Meet windows (config.py)

The scraper only checks dates within the Saratoga meet window
to avoid unnecessary HEAD requests during the off-season.
Update `MEET_WINDOWS` in `config.py` if the meet dates shift.

| Year | Window |
|------|--------|
| 2024 | Jul 10 – Sep 5 |
| 2025 | Jul 9 – Sep 1 |

## Adding more years

```python
# config.py
SEASONS = [2023, 2024, 2025]
MEET_WINDOWS = {
    2023: ("2023-07-13", "2023-09-04"),
    2024: ("2024-07-10", "2024-09-05"),
    2025: ("2025-07-09", "2025-09-01"),
}
```
