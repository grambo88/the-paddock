"""
predict_race5_2025_07_05.py
===========================
Predicts Win / Place / Show probabilities for:
  Saratoga Race #5 — July 5, 2025
  1 1/16M, Turf, $55,000 Maiden Claiming
  Open | 3 Year Olds And Up
  Purse: $62,000

Run:
  python predict_race5_2025_07_05.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

RACE_INFO = {
    "distance_furlongs": 8.5,
    "surface_code":      1,      # Turf
    "race_type_code":    1,      # Maiden Claiming
    "purse":             62000,
    "field_size":        9,
}

ENTRIES = [
    {
        "horse_name": "Big Braciole", "post_position": 1,
        "morning_line_odds": "15/1", "morning_line_decimal": 15.0,
        "hrn_speed_figure": 91, "trainer": "Wayne Potts", "jockey": "Javier Castellano",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Dynadee", "post_position": 2,
        "morning_line_odds": "5/1", "morning_line_decimal": 5.0,
        "hrn_speed_figure": 95, "trainer": "Michael Dini", "jockey": "Jose Lezcano",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Dixie Hex", "post_position": 3,
        "morning_line_odds": "20/1", "morning_line_decimal": 20.0,
        "hrn_speed_figure": 80, "trainer": "Raymond Handal", "jockey": "Dylan Davis",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Harrier", "post_position": 4,
        "morning_line_odds": "10/1", "morning_line_decimal": 10.0,
        "hrn_speed_figure": 87, "trainer": "Archibald J. Kingsley, Jr.", "jockey": "Taylor Kingsley",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Languid", "post_position": 5,
        "morning_line_odds": "8/1", "morning_line_decimal": 8.0,
        "hrn_speed_figure": 91, "trainer": "Richard E. Dutrow, Jr.", "jockey": "John R. Velazquez",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Crowned Moment (GB)", "post_position": 6,
        "morning_line_odds": "6/1", "morning_line_decimal": 6.0,
        "hrn_speed_figure": 0, "trainer": "Miguel Clement", "jockey": "Manuel Franco",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Inherent Promise", "post_position": 7,
        "morning_line_odds": "7/2", "morning_line_decimal": 3.5,
        "hrn_speed_figure": 87, "trainer": "Lisa L. Lewis", "jockey": "Tyler Gaffalione",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Capricious Outcome", "post_position": 8,
        "morning_line_odds": "8/5", "morning_line_decimal": 1.6,
        "hrn_speed_figure": 86, "trainer": "Chad C. Brown", "jockey": "Flavien Prat",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "No Filter", "post_position": 9,
        "morning_line_odds": "20/1", "morning_line_decimal": 20.0,
        "hrn_speed_figure": 70, "trainer": "Chris J. Englehart", "jockey": "Ruben Silvera",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
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

    rows  = [{**RACE_INFO, **e} for e in ENTRIES]
    df    = pd.DataFrame(rows)
    df    = _normalize(df)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)

    n      = len(df)
    X_race = df[FEATURE_COLS].values.astype(np.float32)
    pad    = MAX_HORSES - n
    X_pad  = np.vstack([X_race, np.zeros((pad, len(FEATURE_COLS)), dtype=np.float32)])
    mask   = np.array([True] * n + [False] * pad)

    X_t    = torch.tensor(X_pad[np.newaxis], dtype=torch.float32)
    mask_t = torch.tensor(mask[np.newaxis],  dtype=torch.bool)

    with torch.no_grad():
        probs = model(X_t, mask_t)[0].numpy()

    print("=" * 65)
    print(f"  SARATOGA RACE #5 — July 5, 2025")
    print(f"  1 1/16M Turf | Maiden Claiming | 3YO+ | Purse: $62,000")
    print("=" * 65)
    print(f"  {'PP':<4} {'Horse':<24} {'ML':>5}  {'SpFig':>6} {'Win%':>6} {'Place%':>7} {'Show%':>7}")
    print("  " + "─" * 62)

    results = []
    for i, entry in enumerate(ENTRIES):
        results.append({
            "pp":        entry["post_position"],
            "horse":     entry["horse_name"],
            "ml":        entry["morning_line_odds"],
            "sf":        entry["hrn_speed_figure"],
            "win_pct":   probs[i, 0],
            "place_pct": probs[i, 1],
            "show_pct":  probs[i, 2],
        })

    results.sort(key=lambda x: x["win_pct"], reverse=True)

    for r in results:
        print(f"  {r['pp']:<4} {r['horse']:<24} {r['ml']:>5}  "
              f"{r['sf']:>6} {r['win_pct']:>6.1%} {r['place_pct']:>7.1%} {r['show_pct']:>7.1%}")

    print("  " + "─" * 62)
    print(f"\n  🏆 Win:   #{results[0]['pp']} {results[0]['horse']}  ({results[0]['win_pct']:.1%})")
    print(f"  🥈 Place: #{results[1]['pp']} {results[1]['horse']}  ({results[1]['place_pct']:.1%})")
    print(f"  🥉 Show:  #{results[2]['pp']} {results[2]['horse']}  ({results[2]['show_pct']:.1%})")
    print()


if __name__ == "__main__":
    predict()