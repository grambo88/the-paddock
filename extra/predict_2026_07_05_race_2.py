"""
predict_race2_2025_07_05.py
===========================
Predicts Win / Place / Show probabilities for:
  Saratoga Race #2 — July 5, 2025
  6F, Dirt, Maiden Special Weight
  Open | 3 Year Olds And Up
  Purse: $100,000

Run:
  python predict_race2_2025_07_05.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

# ── Race info (same for all horses) ───────────────────────────────────────────
RACE_INFO = {
    "distance_furlongs": 6.0,
    "surface_code":      0,      # Dirt
    "race_type_code":    0,      # Maiden Special Weight
    "purse":             100000,
    "field_size":        6,
}

# ── Entries (from screenshot) ──────────────────────────────────────────────────
# Historical stats (win_pct etc.) are unknown for new horses — set to 0
# The model handles this — 0 = no history, same as first-time starters in training
ENTRIES = [
    {
        "horse_name":           "Recurring Revenue",
        "post_position":        1,
        "morning_line_odds":    "7/2",
        "morning_line_decimal": 3.5,
        "hrn_speed_figure":     58,
        "trainer":              "Chad C. Brown",
        "jockey":               "Manuel Franco",
        # historical — unknown for today's race
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
    {
        "horse_name":           "Isthereanormalwife",
        "post_position":        2,
        "morning_line_odds":    "6/1",
        "morning_line_decimal": 6.0,
        "hrn_speed_figure":     0,    # no figure shown
        "trainer":              "David G. Donk",
        "jockey":               "John R. Velazquez",
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
    {
        "horse_name":           "Happy Go More",
        "post_position":        3,
        "morning_line_odds":    "3/1",
        "morning_line_decimal": 3.0,
        "hrn_speed_figure":     0,
        "trainer":              "Jena M. Antonucci",
        "jockey":               "Javier Castellano",
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
    {
        "horse_name":           "Onebigbeautifulbill",
        "post_position":        4,
        "morning_line_odds":    "5/2",
        "morning_line_decimal": 2.5,
        "hrn_speed_figure":     0,
        "trainer":              "Brad H. Cox",
        "jockey":               "Flavien Prat",
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
    {
        "horse_name":           "Bye for Now",
        "post_position":        5,
        "morning_line_odds":    "7/2",
        "morning_line_decimal": 3.5,
        "hrn_speed_figure":     68,
        "trainer":              "Raymond Handal",
        "jockey":               "Dylan Davis",
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
    {
        "horse_name":           "Music in Motion",
        "post_position":        6,
        "morning_line_odds":    "6/1",
        "morning_line_decimal": 6.0,
        "hrn_speed_figure":     106,
        "trainer":              "Linda Rice",
        "jockey":               "Jose L. Ortiz",
        "horse_win_pct":   0, "horse_place_pct":  0, "horse_show_pct":  0, "horse_starts":  0,
        "jockey_win_pct":  0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct":0, "trainer_show_pct":0, "trainer_starts": 0,
    },
]


def predict():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found — run: python train.py")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model      = build_model("cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']} "
          f"(val loss: {checkpoint['val_loss']:.4f})\n")

    # Build feature dataframe
    rows = [{**RACE_INFO, **e} for e in ENTRIES]
    df   = pd.DataFrame(rows)
    df   = _normalize(df)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    n      = len(df)
    X_race = df[FEATURE_COLS].values.astype(np.float32)

    # Pad to MAX_HORSES
    pad    = MAX_HORSES - n
    X_pad  = np.vstack([X_race, np.zeros((pad, len(FEATURE_COLS)), dtype=np.float32)])
    mask   = np.array([True] * n + [False] * pad)

    X_t    = torch.tensor(X_pad[np.newaxis], dtype=torch.float32)
    mask_t = torch.tensor(mask[np.newaxis],  dtype=torch.bool)

    with torch.no_grad():
        probs = model(X_t, mask_t)[0].numpy()  # (MAX_HORSES, 3)

    # Print results
    print("=" * 65)
    print(f"  SARATOGA RACE #2 — July 5, 2025")
    print(f"  6F Dirt | Maiden Special Weight | Purse: $100,000")
    print("=" * 65)
    print(f"  {'#':<3} {'Horse':<24} {'ML':>5}  {'Win%':>6} {'Place%':>7} {'Show%':>7}")
    print("  " + "─" * 60)

    results = []
    for i, entry in enumerate(ENTRIES):
        results.append({
            "pp":        entry["post_position"],
            "horse":     entry["horse_name"],
            "ml":        entry["morning_line_odds"],
            "win_pct":   probs[i, 0],
            "place_pct": probs[i, 1],
            "show_pct":  probs[i, 2],
        })

    # Sort by win probability
    results.sort(key=lambda x: x["win_pct"], reverse=True)

    for r in results:
        print(f"  {r['pp']:<3} {r['horse']:<24} {r['ml']:>5}  "
              f"{r['win_pct']:>6.1%} {r['place_pct']:>7.1%} {r['show_pct']:>7.1%}")

    print("  " + "─" * 60)
    print(f"\n  🏆 Top pick: #{results[0]['pp']} {results[0]['horse']}")
    print(f"  🥈 Place:    #{results[1]['pp']} {results[1]['horse']}")
    print(f"  🥉 Show:     #{results[2]['pp']} {results[2]['horse']}")
    print()


if __name__ == "__main__":
    predict()