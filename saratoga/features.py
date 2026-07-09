"""
features.py
===========
Loads training.parquet and builds race-grouped tensors ready for the
Transformer model. Each race becomes one sample — a matrix of shape
(MAX_HORSES, N_FEATURES) with a corresponding label matrix
(MAX_HORSES, 3) for [is_win, is_place, is_show].

Usage:
    from features import load_races, FEATURE_COLS, MAX_HORSES
    races, labels, masks = load_races()
"""

import numpy as np
import pandas as pd
from pathlib import Path
from config import PARQUET_PATH

# ── Feature columns fed to the model ──────────────────────────────────────────
# These are ALL pre-race features — nothing that leaks post-race info
FEATURE_COLS = [
    # Entry level
    "post_position",
    "morning_line_decimal",
    "hrn_speed_figure",
    # Race level
    "distance_furlongs",
    "surface_code",
    "race_type_code",
    "purse",
    "field_size",
    # Historical — horse
    "horse_win_pct",
    "horse_place_pct",
    "horse_show_pct",
    "horse_starts",
    # Historical — jockey
    "jockey_win_pct",
    "jockey_place_pct",
    "jockey_show_pct",
    "jockey_starts",
    # Historical — trainer
    "trainer_win_pct",
    "trainer_place_pct",
    "trainer_show_pct",
    "trainer_starts",
]

TARGET_COLS  = ["is_win", "is_place", "is_show"]
MAX_HORSES   = 16    # pad/truncate all races to this field size
N_FEATURES   = len(FEATURE_COLS)


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numeric features to [0, 1] range.
    Uses fixed reasonable bounds so inference-time normalization
    is consistent with training.
    """
    bounds = {
        "post_position":        (1,   20),
        "morning_line_decimal": (0,   50),
        "hrn_speed_figure":     (0,  130),
        "distance_furlongs":    (4,   14),
        "surface_code":         (0,    4),
        "race_type_code":       (0,   10),
        "purse":                (0, 500000),
        "field_size":           (1,   20),
        "horse_win_pct":        (0,    1),
        "horse_place_pct":      (0,    1),
        "horse_show_pct":       (0,    1),
        "horse_starts":         (0,  200),
        "jockey_win_pct":       (0,    1),
        "jockey_place_pct":     (0,    1),
        "jockey_show_pct":      (0,    1),
        "jockey_starts":        (0, 2000),
        "trainer_win_pct":      (0,    1),
        "trainer_place_pct":    (0,    1),
        "trainer_show_pct":     (0,    1),
        "trainer_starts":       (0, 2000),
    }
    out = df.copy()
    for col, (lo, hi) in bounds.items():
        if col in out.columns:
            out[col] = (out[col].fillna(0).clip(lo, hi) - lo) / max(hi - lo, 1e-8)
    return out


def load_races(path: Path = PARQUET_PATH):
    """
    Load and preprocess the parquet file into race-grouped tensors.

    Returns
    -------
    X     : np.ndarray  shape (N_races, MAX_HORSES, N_FEATURES)  float32
    y     : np.ndarray  shape (N_races, MAX_HORSES, 3)            float32
    masks : np.ndarray  shape (N_races, MAX_HORSES)               bool
              True = real horse, False = padding slot
    meta  : list of dicts  [{date, race_num, horse_names}, ...]
    """
    df = pd.read_parquet(path)

    # Drop scratched horses — they have no result and confuse the model
    df = df[df["scratched"] == 0].copy()

    # Drop races with no label data (future races / missing results)
    df = df.dropna(subset=["is_win"])

    # Normalize features
    df = _normalize(df)

    # Fill remaining NaNs with 0 (first-time starters have no history)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    df[TARGET_COLS]  = df[TARGET_COLS].fillna(0)

    # Group by race
    groups = df.groupby(["date", "race_num"], sort=True)

    X_list, y_list, mask_list, meta_list = [], [], [], []

    for (race_date, race_num), group in groups:
        group = group.reset_index(drop=True)
        n     = len(group)

        # Feature matrix for this race
        X_race = group[FEATURE_COLS].values.astype(np.float32)
        y_race = group[TARGET_COLS].values.astype(np.float32)

        # Pad or truncate to MAX_HORSES
        if n >= MAX_HORSES:
            X_race = X_race[:MAX_HORSES]
            y_race = y_race[:MAX_HORSES]
            mask   = np.ones(MAX_HORSES, dtype=bool)
        else:
            pad    = MAX_HORSES - n
            X_race = np.vstack([X_race, np.zeros((pad, N_FEATURES), dtype=np.float32)])
            y_race = np.vstack([y_race, np.zeros((pad, 3),          dtype=np.float32)])
            mask   = np.array([True] * n + [False] * pad)

        X_list.append(X_race)
        y_list.append(y_race)
        mask_list.append(mask)
        meta_list.append({
            "date":        str(race_date),
            "race_num":    int(race_num),
            "horse_names": group["horse_name"].tolist(),
        })

    X     = np.stack(X_list)
    y     = np.stack(y_list)
    masks = np.stack(mask_list)

    print(f"Loaded {len(X_list)} races | "
          f"X: {X.shape} | y: {y.shape} | "
          f"avg field size: {masks.sum(axis=1).mean():.1f}")

    return X, y, masks, meta_list


if __name__ == "__main__":
    X, y, masks, meta = load_races()
    print(f"\nSample race: {meta[0]}")
    print(f"X[0] (first horse features):\n{X[0][0]}")
    print(f"y[0] (first horse labels):   {y[0][0]}")
