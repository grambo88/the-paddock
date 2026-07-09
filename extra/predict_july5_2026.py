"""
predict_july5_2026.py
=====================
Predicts Win / Place / Show for all flat races on July 5, 2026 at Saratoga.
Actual results are included as comments so you can compare after running.

Race 1 is a Hurdle — skipped (model not trained on jump races).

Run:
  python predict_july5_2026.py
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from features import FEATURE_COLS, MAX_HORSES, _normalize
from model import build_model

MODEL_PATH = Path("model_best.pt")

def run_race(label, race_info, entries, actual=None):
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

    if actual:
        hit = "✅" if actual["winner_pp"] == winner["pp"] else "❌"
        print(f"\n  ACTUAL RESULT {hit}: Winner #{actual['winner_pp']} {actual['winner']}  "
              f"Win: ${actual['win_pay']}  Place: ${actual['place_pay']}  Show: ${actual['show_pay']}")

BLANK = {"horse_win_pct":0,"horse_place_pct":0,"horse_show_pct":0,"horse_starts":0,
         "jockey_win_pct":0,"jockey_place_pct":0,"jockey_show_pct":0,"jockey_starts":0,
         "trainer_win_pct":0,"trainer_place_pct":0,"trainer_show_pct":0,"trainer_starts":0}

# ── RACE 2 — 6F Dirt MSW ──────────────────────────────────────────────────────
r2_info = {"distance_furlongs":6.0,"surface_code":0,"race_type_code":0,"purse":100000,"field_size":6}
r2 = [
    {**BLANK,"horse_name":"Recurring Revenue",  "post_position":1,"morning_line_odds":"7/2","morning_line_decimal":3.5, "hrn_speed_figure":56},
    {**BLANK,"horse_name":"Isthereanormalwife",  "post_position":2,"morning_line_odds":"6/1","morning_line_decimal":6.0, "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Happy Go More",        "post_position":3,"morning_line_odds":"3/1","morning_line_decimal":3.0, "hrn_speed_figure":65},
    {**BLANK,"horse_name":"Onebigbeautfulbill",   "post_position":4,"morning_line_odds":"5/2","morning_line_decimal":2.5, "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Bye for Now",          "post_position":5,"morning_line_odds":"7/2","morning_line_decimal":3.5, "hrn_speed_figure":64},
    {**BLANK,"horse_name":"Music in Motion",      "post_position":6,"morning_line_odds":"6/1","morning_line_decimal":6.0, "hrn_speed_figure":88},
]
r2_actual = {"winner_pp":4,"winner":"Onebigbeautfulbill","win_pay":7.08,"place_pay":3.98,"show_pay":2.68}

# ── RACE 3 — 1M Dirt $20K Claiming F&M ───────────────────────────────────────
r3_info = {"distance_furlongs":8.0,"surface_code":0,"race_type_code":3,"purse":52000,"field_size":7}
r3 = [
    {**BLANK,"horse_name":"Pistol Liz Ablazen", "post_position":1,"morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":41},
    {**BLANK,"horse_name":"Pens Street",         "post_position":2,"morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Kyle's Mom",          "post_position":3,"morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Always Angels",       "post_position":4,"morning_line_odds":"2/1", "morning_line_decimal":2.0,  "hrn_speed_figure":77},
    {**BLANK,"horse_name":"Shezanarcticqueen",   "post_position":5,"morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":78},
    {**BLANK,"horse_name":"Princess Becca",      "post_position":6,"morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Moon Gate",           "post_position":7,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":92},
]
r3_actual = {"winner_pp":6,"winner":"Princess Becca","win_pay":7.28,"place_pay":4.32,"show_pay":3.06}

# ── RACE 4 — 6.5F Dirt MSW ────────────────────────────────────────────────────
r4_info = {"distance_furlongs":6.5,"surface_code":0,"race_type_code":0,"purse":115000,"field_size":7}
r4 = [
    {**BLANK,"horse_name":"Spherical",       "post_position":1,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":93},
    {**BLANK,"horse_name":"Sidearm",         "post_position":2,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":87},
    {**BLANK,"horse_name":"Commitment Fund", "post_position":3,"morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":96},
    {**BLANK,"horse_name":"Implacable",      "post_position":4,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":78},
    {**BLANK,"horse_name":"Neigh Baby",      "post_position":5,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":97},
    {**BLANK,"horse_name":"Party Animal",    "post_position":6,"morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":92},
    {**BLANK,"horse_name":"Holy Seven",      "post_position":7,"morning_line_odds":"2/1", "morning_line_decimal":2.0,  "hrn_speed_figure":89},
]
r4_actual = {"winner_pp":5,"winner":"Neigh Baby","win_pay":31.72,"place_pay":11.98,"show_pay":6.54}

# ── RACE 5 — 1 1/16M Turf Maiden Claiming ─────────────────────────────────────
r5_info = {"distance_furlongs":8.5,"surface_code":1,"race_type_code":1,"purse":62000,"field_size":9}
r5 = [
    {**BLANK,"horse_name":"Big Braciole",       "post_position":1,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":91},
    {**BLANK,"horse_name":"Dynadee",             "post_position":2,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":92},
    {**BLANK,"horse_name":"Dixie Hex",           "post_position":3,"morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":90},
    {**BLANK,"horse_name":"Harrier",             "post_position":4,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Languid",             "post_position":5,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":104},
    {**BLANK,"horse_name":"Crowned Moment (GB)", "post_position":6,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":89},
    {**BLANK,"horse_name":"Inherent Promise",    "post_position":7,"morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":81},
    {**BLANK,"horse_name":"Capricious Outcome",  "post_position":8,"morning_line_odds":"8/5", "morning_line_decimal":1.6,  "hrn_speed_figure":105},
    {**BLANK,"horse_name":"No Filter",           "post_position":9,"morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":83},
]
r5_actual = {"winner_pp":8,"winner":"Capricious Outcome","win_pay":3.26,"place_pay":2.48,"show_pay":2.30}

# ── RACE 6 — 1M Inner Turf Kelso Stakes ───────────────────────────────────────
r6_info = {"distance_furlongs":8.0,"surface_code":2,"race_type_code":8,"purse":225000,"field_size":11}
r6 = [
    {**BLANK,"horse_name":"Zulu Kingdom (IRE)",  "post_position":1, "morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Maycocks Bay",         "post_position":2, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":125},
    {**BLANK,"horse_name":"My Boy Prince",        "post_position":3, "morning_line_odds":"9/2", "morning_line_decimal":4.5,  "hrn_speed_figure":107},
    {**BLANK,"horse_name":"Neat",                 "post_position":4, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":104},
    {**BLANK,"horse_name":"Pass the Hat",         "post_position":5, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":110},
    {**BLANK,"horse_name":"Cruise the Nile",      "post_position":6, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":107},
    {**BLANK,"horse_name":"Itsallcomintogetha",   "post_position":7, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":108},
    {**BLANK,"horse_name":"Mi Bago",              "post_position":8, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Cosmic Year (GB)",     "post_position":9, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":107},
    {**BLANK,"horse_name":"Tiz Dashing",          "post_position":10,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":108},
    {**BLANK,"horse_name":"Capitol Hill",         "post_position":11,"morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":109},
]
r6_actual = {"winner_pp":8,"winner":"Mi Bago (dead heat)","win_pay":8.40,"place_pay":8.80,"show_pay":6.22}

# ── RACE 7 — 6F Dirt MSW Fillies & Mares ─────────────────────────────────────
r7_info = {"distance_furlongs":6.0,"surface_code":0,"race_type_code":0,"purse":100000,"field_size":9}
r7 = [
    {**BLANK,"horse_name":"Midnight Honor",  "post_position":1,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":63},
    {**BLANK,"horse_name":"Luckbeourlady",   "post_position":2,"morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":63},
    {**BLANK,"horse_name":"Liberty's Secret","post_position":3,"morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":97},
    {**BLANK,"horse_name":"Lively Pal",      "post_position":4,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":62},
    {**BLANK,"horse_name":"Carmen Amalia",   "post_position":5,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":88},
    {**BLANK,"horse_name":"Garden of Grace", "post_position":6,"morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":63},
    {**BLANK,"horse_name":"Run On States",   "post_position":7,"morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":56},
    {**BLANK,"horse_name":"Liberty's Advance","post_position":8,"morning_line_odds":"5/2","morning_line_decimal":2.5,  "hrn_speed_figure":84},
    {**BLANK,"horse_name":"Run Flat",        "post_position":9,"morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":62},
]
r7_actual = {"winner_pp":3,"winner":"Liberty's Secret","win_pay":7.02,"place_pay":4.80,"show_pay":3.44}

# ── RACE 8 — 5.5F Turf Harvey Pack Stakes ─────────────────────────────────────
r8_info = {"distance_furlongs":5.5,"surface_code":1,"race_type_code":8,"purse":200000,"field_size":10}
r8 = [
    {**BLANK,"horse_name":"Bring Theband Home","post_position":1, "morning_line_odds":"7/2", "morning_line_decimal":3.5,  "hrn_speed_figure":80},
    {**BLANK,"horse_name":"Boss Sully",         "post_position":2, "morning_line_odds":"5/1", "morning_line_decimal":5.0,  "hrn_speed_figure":75},
    {**BLANK,"horse_name":"Possiblemente",      "post_position":3, "morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":100},
    {**BLANK,"horse_name":"Twenty Six Black",   "post_position":4, "morning_line_odds":"3/1", "morning_line_decimal":3.0,  "hrn_speed_figure":108},
    {**BLANK,"horse_name":"Jean Valjean",       "post_position":5, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":101},
    {**BLANK,"horse_name":"Coppola",            "post_position":6, "morning_line_odds":"30/1","morning_line_decimal":30.0, "hrn_speed_figure":100},
    {**BLANK,"horse_name":"Chasing Liberty",    "post_position":7, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":97},
    {**BLANK,"horse_name":"Outlaw Kid",         "post_position":8, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":102},
    {**BLANK,"horse_name":"Full Disclosure",    "post_position":9, "morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":76},
    {**BLANK,"horse_name":"We're in Trouble",   "post_position":10,"morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":97},
]
r8_actual = {"winner_pp":4,"winner":"Twenty Six Black","win_pay":5.30,"place_pay":2.86,"show_pay":2.32}

# ── RACE 9 — 1.5M Inner Turf AOC ──────────────────────────────────────────────
r9_info = {"distance_furlongs":12.0,"surface_code":2,"race_type_code":4,"purse":120000,"field_size":12}
r9 = [
    {**BLANK,"horse_name":"Miztertonic",       "post_position":1, "morning_line_odds":"4/1", "morning_line_decimal":4.0,  "hrn_speed_figure":105},
    {**BLANK,"horse_name":"Complex Agenda",    "post_position":2, "morning_line_odds":"12/1","morning_line_decimal":12.0, "hrn_speed_figure":116},
    {**BLANK,"horse_name":"Bettrluckythangood","post_position":3, "morning_line_odds":"6/1", "morning_line_decimal":6.0,  "hrn_speed_figure":118},
    {**BLANK,"horse_name":"Fort Thomas",       "post_position":4, "morning_line_odds":"8/1", "morning_line_decimal":8.0,  "hrn_speed_figure":106},
    {**BLANK,"horse_name":"Alakan",            "post_position":5, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":112},
    {**BLANK,"horse_name":"Blue Pill",         "post_position":6, "morning_line_odds":"50/1","morning_line_decimal":50.0, "hrn_speed_figure":103},
    {**BLANK,"horse_name":"Presider",          "post_position":7, "morning_line_odds":"20/1","morning_line_decimal":20.0, "hrn_speed_figure":58},
    {**BLANK,"horse_name":"Offlee Naughty",    "post_position":8, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":113},
    {**BLANK,"horse_name":"Versailles Road",   "post_position":9, "morning_line_odds":"10/1","morning_line_decimal":10.0, "hrn_speed_figure":116},
    {**BLANK,"horse_name":"Noble Dynasty",     "post_position":10,"morning_line_odds":"5/2", "morning_line_decimal":2.5,  "hrn_speed_figure":114},
    {**BLANK,"horse_name":"Greystone",         "post_position":11,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":118},
    {**BLANK,"horse_name":"Noble Factor",      "post_position":12,"morning_line_odds":"15/1","morning_line_decimal":15.0, "hrn_speed_figure":118},
]
r9_actual = {"winner_pp":3,"winner":"Bettrluckythangood","win_pay":16.82,"place_pay":8.94,"show_pay":5.82}

if __name__ == "__main__":
    print("\n" + "█"*66)
    print("  SARATOGA — SUNDAY JULY 5, 2026 — ALL RACES")
    print("  (Actual results shown for comparison)")
    print("█"*66)
    print("\n  Race 1 — SKIPPED (Hurdle race, model not trained on jump races)")
    print("  Actual winner: McTigue (IRE) at $13.24\n")

    run_race("RACE 2 | 6F Dirt MSW | 3YO+ | $100,000",                       r2_info, r2, r2_actual)
    run_race("RACE 3 | 1M Dirt $20K Claiming | F&M 4YO+ | $52,000",          r3_info, r3, r3_actual)
    run_race("RACE 4 | 6.5F Dirt MSW | 3YO+ | $115,000",                     r4_info, r4, r4_actual)
    run_race("RACE 5 | 1 1/16M Turf Maiden Claiming | 3YO+ | $62,000",       r5_info, r5, r5_actual)
    run_race("RACE 6 | 1M Inner Turf Kelso Stakes | 4YO+ | $225,000",        r6_info, r6, r6_actual)
    run_race("RACE 7 | 6F Dirt MSW | Fillies & Mares 3YO+ | $100,000",       r7_info, r7, r7_actual)
    run_race("RACE 8 | 5.5F Turf Harvey Pack Stakes | 4YO+ | $200,000",      r8_info, r8, r8_actual)
    run_race("RACE 9 | 1.5M Inner Turf AOC | 3YO+ | $120,000",               r9_info, r9, r9_actual)

    print("\n" + "═"*66)
    print("  ACTUAL RESULTS SUMMARY — JULY 5, 2026")
    print("═"*66)
    print("  R1  McTigue (IRE)         9/2   $13.24  HURDLE — skipped")
    print("  R2  Onebigbeautfulbill    5/2   $7.08")
    print("  R3  Princess Becca        4/1   $7.28")
    print("  R4  Neigh Baby            12/1  $31.72  UPSET")
    print("  R5  Capricious Outcome    8/5   $3.26")
    print("  R6  Mi Bago (dead heat)   12/1  $8.40")
    print("  R7  Liberty's Secret      5/1   $7.02")
    print("  R8  Twenty Six Black      3/1   $5.30")
    print("  R9  Bettrluckythangood    6/1   $16.82")
    print("═"*66)
