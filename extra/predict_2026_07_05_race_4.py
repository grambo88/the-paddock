"""
predict_race4_2025_07_05.py
===========================
Predicts Win / Place / Show probabilities for:
  Saratoga Race #4 — July 5, 2025
  6 1/2F, Dirt, Maiden Special Weight
  Open | 3 Year Olds And Up
  Purse: $115,000

Run:
  python predict_race4_2025_07_05.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

RACE_INFO = {
    "distance_furlongs": 6.5,
    "surface_code":      0,      # Dirt
    "race_type_code":    0,      # Maiden Special Weight
    "purse":             115000,
    "field_size":        7,
}

ENTRIES = [
    {
        "horse_name": "Spherical", "post_position": 1,
        "morning_line_odds": "8/1", "morning_line_decimal": 8.0,
        "hrn_speed_figure": 79, "trainer": "Chad Summers", "jockey": "Dylan Davis",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Sidearm", "post_position": 2,
        "morning_line_odds": "5/1", "morning_line_decimal": 5.0,
        "hrn_speed_figure": 91, "trainer": "Edward R. Barker", "jockey": "Manuel Franco",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Commitment Fund", "post_position": 3,
        "morning_line_odds": "5/2", "morning_line_decimal": 2.5,
        "hrn_speed_figure": 0, "trainer": "Chad C. Brown", "jockey": "Flavien Prat",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Implacable", "post_position": 4,
        "morning_line_odds": "10/1", "morning_line_decimal": 10.0,
        "hrn_speed_figure": 78, "trainer": "Chad Summers", "jockey": "Tyler Gaffalione",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Neigh Baby", "post_position": 5,
        "morning_line_odds": "12/1", "morning_line_decimal": 12.0,
        "hrn_speed_figure": 93, "trainer": "Jeremiah C. Englehart", "jockey": "Jaime Rodriguez",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Party Animal", "post_position": 6,
        "morning_line_odds": "7/2", "morning_line_decimal": 3.5,
        "hrn_speed_figure": 101, "trainer": "Jena M. Antonucci", "jockey": "Edgard J. Zayas",
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    {
        "horse_name": "Holy Seven", "post_position": 7,
        "morning_line_odds": "2/1", "morning_line_decimal": 2.0,
        "hrn_speed_figure": 80, "trainer": "Steven M. Asmussen", "jockey": "Jose L. Ortiz",
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
    print(f"  SARATOGA RACE #4 — July 5, 2025")
    print(f"  6 1/2F Dirt | Maiden Special Weight | 3YO+ | Purse: $115,000")
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