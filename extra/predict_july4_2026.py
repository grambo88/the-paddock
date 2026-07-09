"""
predict_july4_2026.py
=====================
Predicts Win / Place / Show for all flat races on July 4, 2026 at Saratoga.
Races 1-4, 6, 10-11 are flat. Race 5 is Sanford Stakes. Races 7-9 are stakes.
Skip hurdle/steeplechase races — model was not trained on them.

Run:
  python predict_july4_2026.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

def run_race(label, race_info, entries):
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model found — run: python train.py")
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    model      = build_model("cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    rows  = [{**race_info, **e} for e in entries]
    df    = pd.DataFrame(rows)
    df    = _normalize(df)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
    n      = len(df)
    X_race = df[FEATURE_COLS].values.astype(np.float32)
    pad    = MAX_HORSES - n
    X_pad  = np.vstack([X_race, np.zeros((pad, len(FEATURE_COLS)), dtype=np.float32)])
    mask   = np.array([True]*n + [False]*pad)
    X_t    = torch.tensor(X_pad[np.newaxis], dtype=torch.float32)
    mask_t = torch.tensor(mask[np.newaxis],  dtype=torch.bool)

    with torch.no_grad():
        probs = model(X_t, mask_t)[0].numpy()

    results = []
    for i, e in enumerate(entries):
        results.append({
            "pp":        e["post_position"],
            "horse":     e["horse_name"],
            "ml":        e.get("morning_line_odds","?"),
            "sf":        e.get("hrn_speed_figure",0),
            "win_pct":   probs[i,0],
            "place_pct": probs[i,1],
            "show_pct":  probs[i,2],
        })
    results.sort(key=lambda x: x["win_pct"], reverse=True)

    winner = results[0]
    second = results[1]
    third  = results[2]
    conf   = winner["win_pct"] / (second["win_pct"] + 1e-8)
    conf_label = "HIGH" if conf > 1.8 else ("MEDIUM" if conf > 1.3 else "LOW — consider top 2")

    print(f"\n{'='*66}")
    print(f"  {label}")
    print(f"{'='*66}")
    print(f"  🏆  PREDICTED WINNER")
    print(f"  ╔══════════════════════════════════════════════════════════╗")
    print(f"  ║  #{winner['pp']} {winner['horse']:<24} ML: {winner['ml']:>5}       ║")
    print(f"  ║  Win: {winner['win_pct']:.1%}   Confidence: {conf_label:<30}║")
    print(f"  ╚══════════════════════════════════════════════════════════╝")
    print(f"\n  {'':>2} {'PP':<4} {'Horse':<24} {'ML':>5}  {'SpFig':>6} {'Win%':>6} {'Pl%':>6} {'Sh%':>6}")
    print(f"  {'─'*64}")
    for rank, r in enumerate(results):
        m = "★" if rank==0 else ("→" if rank==1 else " ")
        print(f"  {m} {r['pp']:<4} {r['horse']:<24} {r['ml']:>5}  {r['sf']:>6} "
              f"{r['win_pct']:>6.1%} {r['place_pct']:>6.1%} {r['show_pct']:>6.1%}")
    print(f"  {'─'*64}")
    print(f"\n  Exacta   : #{winner['pp']}-#{second['pp']}  ({winner['horse']} / {second['horse']})")
    print(f"  Trifecta : #{winner['pp']}-#{second['pp']}-#{third['pp']}")

BLANK = {"horse_win_pct":0,"horse_place_pct":0,"horse_show_pct":0,"horse_starts":0,
         "jockey_win_pct":0,"jockey_place_pct":0,"jockey_show_pct":0,"jockey_starts":0,
         "trainer_win_pct":0,"trainer_place_pct":0,"trainer_show_pct":0,"trainer_starts":0}

# ── RACE 1 ────────────────────────────────────────────────────────────────────
r1_info = {"distance_furlongs":6.5,"surface_code":0,"race_type_code":4,"purse":120000,"field_size":5}
r1 = [
    {**BLANK,"horse_name":"Feminism",        "post_position":1,"morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Irresistible",    "post_position":2,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":94},
    {**BLANK,"horse_name":"Lovely Christina","post_position":3,"morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":96},
    {**BLANK,"horse_name":"Mashallah",       "post_position":4,"morning_line_odds":"3/5", "morning_line_decimal":0.6,  "hrn_speed_figure":105},
    {**BLANK,"horse_name":"Lightscape",      "post_position":5,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":87},
]

# ── RACE 2 ────────────────────────────────────────────────────────────────────
r2_info = {"distance_furlongs":8.0,"surface_code":2,"race_type_code":5,"purse":120000,"field_size":8}
r2 = [
    {**BLANK,"horse_name":"Ori",              "post_position":1,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":92},
    {**BLANK,"horse_name":"Accent (GB)",      "post_position":2,"morning_line_odds":"8/5", "morning_line_decimal":1.6,  "hrn_speed_figure":109},
    {**BLANK,"horse_name":"Vekoma View",      "post_position":3,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":91},
    {**BLANK,"horse_name":"Eponine (IRE)",    "post_position":4,"morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":105},
    {**BLANK,"horse_name":"Make You Mine",    "post_position":5,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"New Rose",         "post_position":6,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":99},
    {**BLANK,"horse_name":"Bourbon Betty",    "post_position":7,"morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":92},
    {**BLANK,"horse_name":"Special Wood (FR)","post_position":8,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":96},
]

# ── RACE 3 ────────────────────────────────────────────────────────────────────
r3_info = {"distance_furlongs":8.5,"surface_code":1,"race_type_code":0,"purse":115000,"field_size":10}
r3 = [
    {**BLANK,"horse_name":"Saint Tropez",          "post_position":1, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Fango Creek",            "post_position":2, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Madeline's Agenda",      "post_position":3, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Romala",                 "post_position":4, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Secretly Delighted",     "post_position":5, "morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":96},
    {**BLANK,"horse_name":"Shelzawa (FR)",           "post_position":6, "morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":96},
    {**BLANK,"horse_name":"Home Wrecker",            "post_position":7, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":97},
    {**BLANK,"horse_name":"Nonconsecutivetrms",      "post_position":8, "morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":88},
    {**BLANK,"horse_name":"River Empress",           "post_position":9, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Bourbon Milk Punch",      "post_position":10,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":82},
]

# ── RACE 4 ────────────────────────────────────────────────────────────────────
r4_info = {"distance_furlongs":6.5,"surface_code":0,"race_type_code":5,"purse":105000,"field_size":8}
r4 = [
    {**BLANK,"horse_name":"Lightning Strike",  "post_position":1,"morning_line_odds":"9/5", "morning_line_decimal":1.8,  "hrn_speed_figure":70},
    {**BLANK,"horse_name":"Graceful Rose",     "post_position":2,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Princess Wadadli",  "post_position":3,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":92},
    {**BLANK,"horse_name":"Angel Gift",        "post_position":4,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":81},
    {**BLANK,"horse_name":"Queens Cat",        "post_position":5,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":68},
    {**BLANK,"horse_name":"Baseball Lady",     "post_position":6,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":105},
    {**BLANK,"horse_name":"Britain",           "post_position":7,"morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":71},
    {**BLANK,"horse_name":"Grace and Grit",    "post_position":8,"morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":92},
]

# ── RACE 5 — Sanford Stakes 2YO ───────────────────────────────────────────────
r5_info = {"distance_furlongs":6.0,"surface_code":0,"race_type_code":8,"purse":225000,"field_size":9}
r5 = [
    {**BLANK,"horse_name":"Waggley",           "post_position":1,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":60},
    {**BLANK,"horse_name":"Booked",            "post_position":2,"morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":94},
    {**BLANK,"horse_name":"Goodbye to Romance","post_position":3,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":70},
    {**BLANK,"horse_name":"Pocket Listing",    "post_position":4,"morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":79},
    {**BLANK,"horse_name":"Vissino",           "post_position":5,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":56},
    {**BLANK,"horse_name":"Jack's Golden Goal","post_position":6,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":94},
    {**BLANK,"horse_name":"Ashcroft Lane",     "post_position":7,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":90},
    {**BLANK,"horse_name":"Regent's Park",     "post_position":8,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":83},
    {**BLANK,"horse_name":"Rasasi",            "post_position":9,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":62},
]

# ── RACE 6 — 5.5F Turf Allowance ─────────────────────────────────────────────
r6_info = {"distance_furlongs":5.5,"surface_code":1,"race_type_code":5,"purse":105000,"field_size":12}
r6 = [
    {**BLANK,"horse_name":"Punto Forty",       "post_position":1, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Cristobal",         "post_position":2, "morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Diamond Child",     "post_position":3, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":53},
    {**BLANK,"horse_name":"Truman's Commander","post_position":4, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Rhyton",            "post_position":5, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":76},
    {**BLANK,"horse_name":"Stormy Birthday",   "post_position":6, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":86},
    {**BLANK,"horse_name":"Mozambique",        "post_position":7, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":111},
    {**BLANK,"horse_name":"Guilty",            "post_position":8, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":77},
    {**BLANK,"horse_name":"Three Thirteen",    "post_position":9, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":75},
    {**BLANK,"horse_name":"Diliello",          "post_position":10,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":83},
    {**BLANK,"horse_name":"Van Vollenhoven",   "post_position":11,"morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":88},
    {**BLANK,"horse_name":"King Puck",         "post_position":12,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":89},
]

# ── RACE 7 — Belmont Oaks Grade 1 ────────────────────────────────────────────
r7_info = {"distance_furlongs":9.0,"surface_code":2,"race_type_code":8,"purse":600000,"field_size":9}
r7 = [
    {**BLANK,"horse_name":"Just Aloof",            "post_position":1, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Time to Dream",          "post_position":2, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":107},
    {**BLANK,"horse_name":"Kensington Lane (IRE)",  "post_position":3, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Faithful Departed",      "post_position":4, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Storm's Wake",           "post_position":5, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":111},
    {**BLANK,"horse_name":"Fitz Right",             "post_position":6, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":109},
    {**BLANK,"horse_name":"Carmensita (ARG)",        "post_position":7, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":102},
    {**BLANK,"horse_name":"Abashiri (GB)",           "post_position":8, "morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Ultimate Love",           "post_position":9, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":111},
    {**BLANK,"horse_name":"Imaginationthelady",      "post_position":10,"morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":113},
]

# ── RACE 8 — Suburban Stakes Grade 1 ─────────────────────────────────────────
r8_info = {"distance_furlongs":10.0,"surface_code":0,"race_type_code":8,"purse":500000,"field_size":11}
r8 = [
    {**BLANK,"horse_name":"Classicist",    "post_position":1, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":111},
    {**BLANK,"horse_name":"Forged Steel",  "post_position":2, "morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":130},
    {**BLANK,"horse_name":"Yo Daddy",      "post_position":3, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":108},
    {**BLANK,"horse_name":"Parchment Party","post_position":4,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":125},
    {**BLANK,"horse_name":"Tiztastic",     "post_position":5, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Phileas Fogg",  "post_position":6, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":115},
    {**BLANK,"horse_name":"Antiquarian",   "post_position":7, "morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":120},
    {**BLANK,"horse_name":"Hit Show",      "post_position":8, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":110},
    {**BLANK,"horse_name":"Stars and Stripes","post_position":9,"morning_line_odds":"6/1","morning_line_decimal":6.0,  "hrn_speed_figure":128},
    {**BLANK,"horse_name":"Original Sin",  "post_position":10,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Obstacle (BRZ)","post_position":11,"morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":0},
]

# ── RACE 9 — Belmont Derby Grade 1 ───────────────────────────────────────────
r9_info = {"distance_furlongs":9.0,"surface_code":1,"race_type_code":8,"purse":750000,"field_size":9}
r9 = [
    {**BLANK,"horse_name":"Blackmail",       "post_position":1, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":112},
    {**BLANK,"horse_name":"Bottas",          "post_position":2, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":121},
    {**BLANK,"horse_name":"Remember Mamba",  "post_position":3, "morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Third Coast",     "post_position":4, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":111},
    {**BLANK,"horse_name":"Turf Star",       "post_position":5, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":101},
    {**BLANK,"horse_name":"Pacific Avenue",  "post_position":6, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Tiernanogue",     "post_position":7, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":104},
    {**BLANK,"horse_name":"West End Kid",    "post_position":8, "morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Title Role (GB)", "post_position":9, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Touch of Fire",   "post_position":10,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":108},
]

# ── RACE 10 ───────────────────────────────────────────────────────────────────
r10_info = {"distance_furlongs":8.0,"surface_code":0,"race_type_code":4,"purse":130000,"field_size":10}
r10 = [
    {**BLANK,"horse_name":"Bank Frenzy",      "post_position":1, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":112},
    {**BLANK,"horse_name":"Tarantino",        "post_position":2, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":128},
    {**BLANK,"horse_name":"Full Screen",      "post_position":3, "morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Bourbon Day",      "post_position":4, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":115},
    {**BLANK,"horse_name":"Reasoned Analysis","post_position":5, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":117},
    {**BLANK,"horse_name":"Tuscan Sky",       "post_position":6, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":117},
    {**BLANK,"horse_name":"Capital Idea",     "post_position":7, "morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":110},
    {**BLANK,"horse_name":"Bramito",          "post_position":8, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Warp Nine",        "post_position":9, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":107},
    {**BLANK,"horse_name":"Flood Zone",       "post_position":10,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":120},
]

# ── RACE 11 ───────────────────────────────────────────────────────────────────
r11_info = {"distance_furlongs":8.5,"surface_code":2,"race_type_code":0,"purse":100000,"field_size":11}
r11 = [
    {**BLANK,"horse_name":"Morning Prayer",  "post_position":1, "morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":94},
    {**BLANK,"horse_name":"Coach of the Year","post_position":2,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":79},
    {**BLANK,"horse_name":"Lyn's Legacy",    "post_position":3, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":0},
    {**BLANK,"horse_name":"No Ordinary Love","post_position":4, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":76},
    {**BLANK,"horse_name":"Chaumet",         "post_position":5, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":78},
    {**BLANK,"horse_name":"Nobody Knows",    "post_position":6, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":0},
    {**BLANK,"horse_name":"Probable Choice", "post_position":7, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":85},
    {**BLANK,"horse_name":"Zap That Ghost",  "post_position":8, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Tank Girl",       "post_position":9, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":50},
    {**BLANK,"horse_name":"Force of Mischief","post_position":10,"morning_line_odds":"15/1","morning_line_decimal":15.0,"hrn_speed_figure":88},
    {**BLANK,"horse_name":"Silly Season",    "post_position":11,"morning_line_odds":"2/1", "morning_line_decimal":2.0,  "hrn_speed_figure":80},
]

if __name__ == "__main__":
    print("\n" + "█"*66)
    print("  SARATOGA — SATURDAY JULY 4, 2026 — ALL RACES")
    print("█"*66)

    run_race("RACE 1 | 6.5F Dirt AOC | Fillies 3YO | $120,000",            r1_info, r1)
    run_race("RACE 2 | 1M Inner Turf Allowance | F&M 3YO+ | $120,000",     r2_info, r2)
    run_race("RACE 3 | 1 1/16M Turf MSW | F&M 3YO+ | $115,000",           r3_info, r3)
    run_race("RACE 4 | 6.5F Dirt Allowance | F&M 3YO+ | $105,000",        r4_info, r4)
    run_race("RACE 5 | 6F Dirt Sanford Stakes | 2YO | $225,000",           r5_info, r5)
    run_race("RACE 6 | 5.5F Turf Allowance | 3YO+ | $105,000",            r6_info, r6)
    run_race("RACE 7 | 1 1/8M Inner Turf Belmont Oaks G1 | $600,000",     r7_info, r7)
    run_race("RACE 8 | 1 1/4M Dirt Suburban Stakes G1 | $500,000",         r8_info, r8)
    run_race("RACE 9 | 1 1/8M Turf Belmont Derby G1 | $750,000",           r9_info, r9)
    run_race("RACE 10 | 1M Dirt AOC | 4YO+ | $130,000",                   r10_info, r10)
    run_race("RACE 11 | 1 1/16M Inner Turf MSW | F&M 3YO+ | $100,000",    r11_info, r11)
