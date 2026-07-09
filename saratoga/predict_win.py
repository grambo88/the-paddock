"""
predict.py
==========
Reusable Win / Place / Show predictor for any Saratoga race.
Edit the RACE_INFO and ENTRIES sections below and run.

Usage:
  python predict.py

Output shows:
  - Clear predicted WINNER with confidence level
  - Full field ranked by win probability
  - Exacta and Trifecta suggestions
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

# ── EDIT THIS SECTION FOR EACH RACE ───────────────────────────────────────────

RACE_LABEL = "Saratoga Race #X — YYYY-MM-DD"

RACE_INFO = {
    "distance_furlongs": 8.0,    # 6F=6.0, 6.5F=6.5, 1M=8.0, 1 1/16M=8.5, 1 1/8M=9.0
    "surface_code":      0,       # 0=Dirt, 1=Turf, 2=Inner Turf
    "race_type_code":    0,       # 0=MSW, 1=MClaim, 2=Maiden, 3=Claim, 4=AOC, 5=Allow, 8=Stakes
    "purse":             100000,
    "field_size":        8,
}

# One dict per horse — copy/paste and fill in from the race card
ENTRIES = [
    {
        "horse_name": "Horse Name", "post_position": 1,
        "morning_line_odds": "5/1", "morning_line_decimal": 5.0,
        "hrn_speed_figure": 90,
        "horse_win_pct": 0, "horse_place_pct": 0, "horse_show_pct": 0, "horse_starts": 0,
        "jockey_win_pct": 0, "jockey_place_pct": 0, "jockey_show_pct": 0, "jockey_starts": 0,
        "trainer_win_pct": 0, "trainer_place_pct": 0, "trainer_show_pct": 0, "trainer_starts": 0,
    },
    # ... add more horses ...
]

# ── END EDIT SECTION ──────────────────────────────────────────────────────────


def predict(entries=None, race_info=None, label=None):
    entries   = entries   or ENTRIES
    race_info = race_info or RACE_INFO
    label     = label     or RACE_LABEL

    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found — run: python train.py")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model      = build_model("cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows = [{**race_info, **e} for e in entries]
    df   = pd.DataFrame(rows)
    df   = _normalize(df)
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

    results = []
    for i, entry in enumerate(entries):
        results.append({
            "pp":        entry["post_position"],
            "horse":     entry["horse_name"],
            "ml":        entry.get("morning_line_odds", "?"),
            "sf":        entry.get("hrn_speed_figure", 0),
            "win_pct":   probs[i, 0],
            "place_pct": probs[i, 1],
            "show_pct":  probs[i, 2],
        })

    results.sort(key=lambda x: x["win_pct"], reverse=True)

    winner = results[0]
    second = results[1]
    third  = results[2]

    # Confidence = how much stronger is #1 vs #2
    confidence = winner["win_pct"] / (second["win_pct"] + 1e-8)
    if confidence > 1.8:
        conf_label = "HIGH"
    elif confidence > 1.3:
        conf_label = "MEDIUM"
    else:
        conf_label = "LOW — consider using top 2"

    print(f"\n{'=' * 66}")
    print(f"  {label}")
    print(f"  Model: epoch {checkpoint['epoch']} | val loss: {checkpoint['val_loss']:.4f}")
    print(f"{'=' * 66}")

    # ── WINNER CALL ───────────────────────────────────────────────────────
    print(f"\n  🏆  PREDICTED WINNER")
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  #{winner['pp']} {winner['horse']:<24} ML: {winner['ml']:>5}        ║")
    print(f"  ║  Win probability: {winner['win_pct']:.1%}   Confidence: {conf_label:<24}║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")

    # ── FULL FIELD ────────────────────────────────────────────────────────
    print(f"\n  {'':>2} {'PP':<4} {'Horse':<24} {'ML':>5}  {'SpFig':>6} "
          f"{'Win%':>6} {'Place%':>7} {'Show%':>7}")
    print(f"  {'─' * 66}")

    for rank, r in enumerate(results):
        marker = "★" if rank == 0 else ("→" if rank == 1 else " ")
        print(f"  {marker} {r['pp']:<4} {r['horse']:<24} {r['ml']:>5}  "
              f"{r['sf']:>6} {r['win_pct']:>6.1%} {r['place_pct']:>7.1%} {r['show_pct']:>7.1%}")

    print(f"  {'─' * 66}")

    # ── EXOTICS ───────────────────────────────────────────────────────────
    print(f"\n  ── SUGGESTED BETS ──────────────────────────────────────────")
    print(f"  Win      : #{winner['pp']} {winner['horse']}")
    print(f"  Exacta   : #{winner['pp']}-#{second['pp']}  "
          f"({winner['horse']} over {second['horse']})")
    print(f"  Trifecta : #{winner['pp']}-#{second['pp']}-#{third['pp']}  "
          f"({winner['horse']} / {second['horse']} / {third['horse']})")
    print()

    return results


if __name__ == "__main__":
    predict()