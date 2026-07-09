"""
insert_2024_08_09_race9.py
==========================
Manually inserts Saratoga Race 9 from 2024-08-09.
Run once:
  python insert_2024_08_09_race9.py
"""

import logging
from database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

race_day = {
    "date": "2024-08-09",
    "races": [
        {
            "date":              "2024-08-09",
            "track":             "Saratoga",
            "race_num":          9,
            "post_time":         "5:42 PM",
            "distance_furlongs": 8.5,
            "surface":           "Turf",
            "surface_code":      1,
            "race_type":         "Maiden Claiming",
            "race_type_code":    1,
            "purse":             55000,
            "conditions":        "Fillies & Mares | 3 Year Olds And Up",
            "field_size":        13,
            "available_bets":    "Exacta ($1), Trifecta (.50), Super (.10)",

            "entries": [
                {"program_num":1,  "post_position":9,  "horse_name":"True Myth",              "sire":"Mendelssohn",      "trainer":"John P. Terranova II",    "jockey":None,                    "morning_line_odds":"9/2",  "morning_line_decimal":4.5,   "hrn_speed_figure":96,  "scratched":False},
                {"program_num":1,  "post_position":16, "horse_name":"Thedreamcontinues",       "sire":"Always Dreaming",  "trainer":None,                      "jockey":None,                    "morning_line_odds":"9/2",  "morning_line_decimal":4.5,   "hrn_speed_figure":74,  "scratched":True},
                {"program_num":2,  "post_position":1,  "horse_name":"Luna Love",               "sire":"Vino Rosso",       "trainer":None,                      "jockey":None,                    "morning_line_odds":"12/1", "morning_line_decimal":12.0,  "hrn_speed_figure":96,  "scratched":False},
                {"program_num":3,  "post_position":2,  "horse_name":"Lion's Miracle Mo",       "sire":"Mo Town",          "trainer":None,                      "jockey":None,                    "morning_line_odds":"30/1", "morning_line_decimal":30.0,  "hrn_speed_figure":-1,  "scratched":False},
                {"program_num":4,  "post_position":3,  "horse_name":"The Taco Lady",           "sire":"Enticed",          "trainer":"Michael J. Maker",        "jockey":"Kendrick Carmouche",    "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":98,  "scratched":False},
                {"program_num":5,  "post_position":4,  "horse_name":"Looking for Ginny",       "sire":"Lookin At Lucky",  "trainer":"Kenneth G. McPeek",       "jockey":None,                    "morning_line_odds":"12/1", "morning_line_decimal":12.0,  "hrn_speed_figure":88,  "scratched":False},
                {"program_num":6,  "post_position":5,  "horse_name":"Classic Cara",            "sire":"Mendelssohn",      "trainer":"Mitchell E. Friedman",    "jockey":"Javier Castellano",     "morning_line_odds":"8/1",  "morning_line_decimal":8.0,   "hrn_speed_figure":85,  "scratched":False},
                {"program_num":7,  "post_position":6,  "horse_name":"Don't Touch Me (IRE)",    "sire":"Camelot (GB)",     "trainer":None,                      "jockey":None,                    "morning_line_odds":"20/1", "morning_line_decimal":20.0,  "hrn_speed_figure":64,  "scratched":False},
                {"program_num":8,  "post_position":7,  "horse_name":"American Kestrel",        "sire":"Medaglia d'Oro",   "trainer":"Jacob Palacios Molina",   "jockey":None,                    "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":68,  "scratched":False},
                {"program_num":9,  "post_position":8,  "horse_name":"Coquito",                 "sire":"Connect",          "trainer":None,                      "jockey":None,                    "morning_line_odds":"8/1",  "morning_line_decimal":8.0,   "hrn_speed_figure":98,  "scratched":False},
                {"program_num":10, "post_position":10, "horse_name":"Fantasy Performer",       "sire":"Oscar Performance","trainer":None,                      "jockey":None,                    "morning_line_odds":"5/1",  "morning_line_decimal":5.0,   "hrn_speed_figure":110, "scratched":False},
                {"program_num":11, "post_position":11, "horse_name":"Blind Speed",             "sire":"Uncle Mo",         "trainer":"Luis Angel Batista",      "jockey":"Ramon A. Vazquez",      "morning_line_odds":"20/1", "morning_line_decimal":20.0,  "hrn_speed_figure":62,  "scratched":False},
                {"program_num":12, "post_position":12, "horse_name":"Purrfect Girl",           "sire":"Kitten's Joy",     "trainer":"Anthony W. Dutrow",       "jockey":None,                    "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":82,  "scratched":False},
                {"program_num":13, "post_position":13, "horse_name":"Pay the Bills (FR)",      "sire":"Wootton Bassett (GB)", "trainer":None,                  "jockey":None,                    "morning_line_odds":"7/2",  "morning_line_decimal":3.5,   "hrn_speed_figure":77,  "scratched":True},
                {"program_num":14, "post_position":14, "horse_name":"Summer Festival",         "sire":"Summer Front",     "trainer":"Domenick L. Schettino",   "jockey":None,                    "morning_line_odds":"15/1", "morning_line_decimal":15.0,  "hrn_speed_figure":38,  "scratched":True},
                {"program_num":15, "post_position":15, "horse_name":"Maturity Date",           "sire":"Upstart",          "trainer":"Linda Rice",              "jockey":None,                    "morning_line_odds":"4/5",  "morning_line_decimal":0.8,   "hrn_speed_figure":98,  "scratched":False},
            ],

            "results": [
                {"horse_name":"Maturity Date",      "finish_position":1, "win_payout":3.60,  "place_payout":2.40,  "show_payout":2.10,  "hrn_speed_figure_post":97},
                {"horse_name":"The Taco Lady",      "finish_position":2, "win_payout":None,  "place_payout":3.90,  "show_payout":3.00,  "hrn_speed_figure_post":92},
                {"horse_name":"Looking for Ginny",  "finish_position":3, "win_payout":None,  "place_payout":None,  "show_payout":3.10,  "hrn_speed_figure_post":90},
                {"horse_name":"Coquito",            "finish_position":4, "win_payout":None,  "place_payout":None,  "show_payout":None,  "hrn_speed_figure_post":89},
            ],

            "exotic_payouts": [
                {"bet_type":"Pick 3",        "combination":"5-4-15",       "payout":77.50,   "total_pool":177857.00},
                {"bet_type":"Pick 4",        "combination":"2-5-4-15",     "payout":421.00,  "total_pool":324166.00},
                {"bet_type":"Pick 5",        "combination":"10-2-5-4-15",  "payout":1674.00, "total_pool":420567.00},
                {"bet_type":"Pick 6",        "combination":"8-10-2-5-4-15","payout":69.50,   "total_pool":0.00},
                {"bet_type":"Pick 6",        "combination":"8-10-2-5-4-15","payout":4218.00, "total_pool":127766.00},
                {"bet_type":"Daily Double",  "combination":"4-15",         "payout":12.20,   "total_pool":167441.00},
            ],

            "race_times": {
                "fraction_1": 23.9,
                "fraction_2": 47.08,
                "fraction_3": 72.66,
                "final_time": 99.09,   # 1:39.09
            },
        }
    ]
}

db = Database()
n  = db.insert_day(race_day)
print(f"Inserted {n} race(s)")
db.export_parquet()
print("Parquet updated")
print("\nDB summary:")
for k, v in db.summary().items():
    print(f"  {k:<18} {v:>6,} rows")