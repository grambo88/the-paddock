"""
insert_2025_08_20_race9.py
==========================
Manually inserts Saratoga Race 9 from 2025-08-20 directly into the DB.
Run once:
  python insert_2025_08_20_race9.py
"""

import logging
from database import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

race_day = {
    "date": "2025-08-20",
    "races": [
        {
            "date":              "2025-08-20",
            "track":             "Saratoga",
            "race_num":          9,
            "post_time":         "5:44 PM",
            "distance_furlongs": 5.5,
            "surface":           "Turf",
            "surface_code":      1,
            "race_type":         "Maiden Claiming",
            "race_type_code":    1,
            "purse":             55000,
            "conditions":        "Fillies & Mares | 3 Year Olds And Up",
            "field_size":        14,
            "available_bets":    "Exacta ($1), Trifecta (.50), Super (.10)",

            "entries": [
                {"program_num":1,  "post_position":1,  "horse_name":"Aerialist",        "sire":"Daredevil",       "trainer":"Patrick L. Reynolds",    "jockey":"Flavien Prat",           "morning_line_odds":"15/1", "morning_line_decimal":15.0,  "hrn_speed_figure":79,  "scratched":False},
                {"program_num":2,  "post_position":2,  "horse_name":"Darty Time",        "sire":"Not This Time",   "trainer":"Rudy R. Rodriguez",      "jockey":"Jose Lezcano",           "morning_line_odds":"8/1",  "morning_line_decimal":8.0,   "hrn_speed_figure":98,  "scratched":False},
                {"program_num":3,  "post_position":3,  "horse_name":"Bolt House",        "sire":"Bolt d'Oro",      "trainer":"William I. Mott",        "jockey":"Junior Alvarado",        "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":89,  "scratched":False},
                {"program_num":4,  "post_position":4,  "horse_name":"Aunt Nona",         "sire":"Uncle Mo",        "trainer":"John C. Kimmel",         "jockey":"Kendrick Carmouche",     "morning_line_odds":"8/1",  "morning_line_decimal":8.0,   "hrn_speed_figure":68,  "scratched":False},
                {"program_num":5,  "post_position":5,  "horse_name":"Kate Barry",        "sire":"Constitution",    "trainer":"Bruce R. Brown",         "jockey":"Romero Ramsay Maragh",   "morning_line_odds":"12/1", "morning_line_decimal":12.0,  "hrn_speed_figure":76,  "scratched":False},
                {"program_num":6,  "post_position":6,  "horse_name":"Gratefully",        "sire":"Laoban",          "trainer":"Raymond Handal",         "jockey":"Dylan Davis",            "morning_line_odds":"12/1", "morning_line_decimal":12.0,  "hrn_speed_figure":105, "scratched":False},
                {"program_num":7,  "post_position":7,  "horse_name":"New Attitude (IRE)","sire":"New Bay (GB)",    "trainer":"Fernando Abreu",         "jockey":"Ricardo Santana, Jr.",   "morning_line_odds":"8/1",  "morning_line_decimal":8.0,   "hrn_speed_figure":71,  "scratched":False},
                {"program_num":8,  "post_position":8,  "horse_name":"Cara's Chianti",    "sire":"Vino Rosso",      "trainer":"Mitchell E. Friedman",   "jockey":"Lane J. Luzzi",          "morning_line_odds":"30/1", "morning_line_decimal":30.0,  "hrn_speed_figure":53,  "scratched":True},
                {"program_num":9,  "post_position":9,  "horse_name":"Sod Siren (FR)",    "sire":"Pinatubo (IRE)",  "trainer":"Saffie A. Joseph, Jr.",  "jockey":"Irad Ortiz, Jr.",        "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":115, "scratched":False},
                {"program_num":10, "post_position":10, "horse_name":"Veola",             "sire":"Vekoma",          "trainer":"Keri Brion",             "jockey":"Luis Saez",              "morning_line_odds":"7/2",  "morning_line_decimal":3.5,   "hrn_speed_figure":70,  "scratched":False},
                {"program_num":11, "post_position":11, "horse_name":"Victoriously",      "sire":"Win Win Win",     "trainer":"Adam Rice",              "jockey":"Jose L. Ortiz",          "morning_line_odds":"9/2",  "morning_line_decimal":4.5,   "hrn_speed_figure":102, "scratched":False},
                {"program_num":12, "post_position":12, "horse_name":"Grand Crossing",    "sire":"Street Boss",     "trainer":"Linda Rice",             "jockey":"Jose Lezcano",           "morning_line_odds":"5/1",  "morning_line_decimal":5.0,   "hrn_speed_figure":89,  "scratched":False},
                {"program_num":13, "post_position":13, "horse_name":"Miss Im Pulsive",   "sire":"Into Mischief",   "trainer":"Amelia J. Green",        "jockey":"Irad Ortiz, Jr.",        "morning_line_odds":"6/1",  "morning_line_decimal":6.0,   "hrn_speed_figure":90,  "scratched":True},
                {"program_num":14, "post_position":14, "horse_name":"Laughing Lady",     "sire":"Practical Joke",  "trainer":"Robert N. Falcone, Jr.", "jockey":"Ricardo Santana, Jr.",   "morning_line_odds":"3/1",  "morning_line_decimal":3.0,   "hrn_speed_figure":71,  "scratched":True},
            ],

            "results": [
                {"horse_name":"Gratefully",         "finish_position":1, "win_payout":15.60, "place_payout":5.10,  "show_payout":4.70,  "hrn_speed_figure_post":100},
                {"horse_name":"Bolt House",         "finish_position":2, "win_payout":None,  "place_payout":3.20,  "show_payout":2.70,  "hrn_speed_figure_post":87},
                {"horse_name":"Kate Barry",         "finish_position":3, "win_payout":None,  "place_payout":None,  "show_payout":8.20,  "hrn_speed_figure_post":70},
                {"horse_name":"New Attitude (IRE)", "finish_position":4, "win_payout":None,  "place_payout":None,  "show_payout":None,  "hrn_speed_figure_post":61},
            ],

            "exotic_payouts": [
                {"bet_type":"Daily Double",      "combination":"1-6",       "payout":43.00,    "total_pool":143662.00},
                {"bet_type":"Exacta",            "combination":"6-3",       "payout":60.00,    "total_pool":304557.00},
                {"bet_type":"Superfecta",        "combination":"6-3-5-7",   "payout":3489.00,  "total_pool":77660.00},
                {"bet_type":"Trifecta",          "combination":"6-3-5",     "payout":479.00,   "total_pool":136107.00},
                {"bet_type":"Pick 3",            "combination":"2-1-6",     "payout":139.00,   "total_pool":197245.00},
                {"bet_type":"Consolation Pick 3","combination":"2-4-6",     "payout":74.00,    "total_pool":0.00},
                {"bet_type":"Pick 4",            "combination":"1-2-1-6",   "payout":1275.00,  "total_pool":209544.00},
                {"bet_type":"Pick 5",            "combination":"3-1-2-1-6", "payout":3852.00,  "total_pool":348096.00},
                {"bet_type":"Pick 6",            "combination":"4-3-1-2-1-6","payout":79.50,   "total_pool":0.00},
                {"bet_type":"Pick 6",            "combination":"4-3-1-2-1-6","payout":7009.00, "total_pool":104979.00},
            ],

            "race_times": {
                "fraction_1": 21.82,
                "fraction_2": 45.16,
                "fraction_3": 57.47,
                "final_time": 63.98,   # 1:03.98
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