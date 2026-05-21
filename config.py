from loguru import logger

PARK_FACTORS = {
    "COL": 1.18, "BOS": 1.09, "CIN": 1.07, "TEX": 1.06, "BAL": 1.05,
    "PHI": 1.04, "MIL": 1.03, "CHC": 1.03, "NYY": 1.02, "ATL": 1.01,
    "TOR": 1.01, "DET": 1.00, "MIN": 1.00, "HOU": 0.99, "WSH": 0.99,
    "STL": 0.99, "CLE": 0.98, "LAA": 0.98, "TB":  0.98, "ARI": 0.97,
    "NYM": 0.97, "MIA": 0.97, "PIT": 0.97, "KC":  0.96, "CWS": 0.96,
    "SEA": 0.96, "OAK": 0.95, "SD":  0.94, "LAD": 0.94, "SF":  0.92,
}

STADIUM_COORDS = {
    "COL": (39.7559, -104.9942), "BOS": (42.3467, -71.0972),
    "CIN": (39.0979, -84.5082),  "TEX": (32.7512, -97.0832),
    "BAL": (39.2838, -76.6217),  "PHI": (39.9061, -75.1665),
    "MIL": (43.0280, -87.9712),  "CHC": (41.9484, -87.6553),
    "NYY": (40.8296, -73.9262),  "ATL": (33.8908, -84.4678),
    "TOR": (43.6414, -79.3894),  "DET": (42.3390, -83.0485),
    "MIN": (44.9817, -93.2776),  "HOU": (29.7573, -95.3555),
    "WSH": (38.8730, -77.0074),  "STL": (38.6226, -90.1928),
    "CLE": (41.4962, -81.6852),  "LAA": (33.8003, -117.8827),
    "TB":  (27.7682, -82.6534),  "ARI": (33.4455, -112.0667),
    "NYM": (40.7571, -73.8458),  "MIA": (25.7781, -80.2197),
    "PIT": (40.4469, -80.0058),  "KC":  (39.0517, -94.4803),
    "CWS": (41.8299, -87.6338),  "SEA": (47.5914, -122.3325),
    "OAK": (37.7516, -122.2005), "SD":  (32.7076, -117.1570),
    "LAD": (34.0739, -118.2400), "SF":  (37.7786, -122.3893),
}

# Degrees (from North) representing the direction toward the outfield at each park.
# Wind blowing FROM the opposite direction (±60°) is "blowing out."
STADIUM_OUT_DIRECTION = {
    "COL": 25,   # Coors — outfield roughly NNE
    "BOS": 90,   # Fenway — outfield toward E (Green Monster in LF means RF out)
    "CIN": 0,    # GABP — outfield N
    "TEX": 0,    # Globe Life — outfield N
    "BAL": 350,  # Camden — outfield NNW
    "PHI": 10,   # Citizens Bank — outfield N
    "MIL": 340,  # American Family — outfield NNW
    "CHC": 45,   # Wrigley — outfield NE (famous wind out to LF/CF from SW)
    "NYY": 0,    # Yankee Stadium — outfield N
    "ATL": 20,   # Truist — outfield NNE
    "TOR": 0,    # Rogers Centre — dome, wind neutral
    "DET": 350,  # Comerica — outfield NNW
    "MIN": 0,    # Target Field — outfield N
    "HOU": 10,   # Minute Maid — dome/retractable, wind neutral
    "WSH": 340,  # Nationals Park — outfield NNW
    "STL": 350,  # Busch — outfield NNW
    "CLE": 15,   # Progressive — outfield NNE
    "LAA": 350,  # Angel Stadium — outfield NNW
    "TB":  0,    # Tropicana — dome, wind neutral
    "ARI": 0,    # Chase Field — dome/retractable, wind neutral
    "NYM": 350,  # Citi Field — outfield NNW
    "MIA": 0,    # Loan Depot — retractable, wind neutral
    "PIT": 10,   # PNC Park — outfield N
    "KC":  340,  # Kauffman — outfield NNW
    "CWS": 0,    # Guaranteed Rate — outfield N
    "SEA": 15,   # T-Mobile — outfield NNE
    "OAK": 340,  # Oakland Coliseum — outfield NNW
    "SD":  350,  # Petco Park — outfield NNW
    "LAD": 330,  # Dodger Stadium — outfield NNW
    "SF":  315,  # Oracle Park — outfield NW (McCovey Cove)
}

TEAM_ID_MAP = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}

TEAM_ABBR_TO_ID = {v: k for k, v in TEAM_ID_MAP.items()}

PITCH_TYPES = {
    "FF": "4-Seam Fastball", "SI": "Sinker",    "FC": "Cutter",
    "SL": "Slider",          "ST": "Sweeper",   "CU": "Curveball",
    "KC": "Knuckle Curve",   "CH": "Changeup",  "FS": "Splitter",
    "SV": "Slurve",          "CS": "Slow Curve",
}

FEATURE_WEIGHTS = {
    "batter_h_pct_14day":          0.14,
    "batter_h_pct_7day":           0.08,
    "batter_h_pct_season":         0.07,
    "batter_xba_season":           0.06,
    "babip_regression_delta":      0.06,
    "career_h_ab_vs_pitcher":      0.08,
    "arsenal_weighted_ba":         0.10,
    "arsenal_weighted_whiff":      0.04,
    "pitcher_h9_season":           0.05,
    "pitcher_xfip":                0.05,
    "pitcher_babip_against":       0.03,
    "pitcher_last3_hits_avg":      0.04,
    "pitcher_fatigue_score":       0.03,
    "times_through_order":         0.04,
    "bullpen_h9":                  0.03,
    "park_factor":                 0.03,
    "game_total":                  0.04,
    "weather_score":               0.03,
    "handedness_split":            0.03,
    "lineup_position_pa_weight":   0.03,
    "umpire_zone_score":           0.02,
    "exit_velocity_14day":         0.02,
}

MIN_SAMPLES = {
    "career_vs_pitcher_ab":     15,
    "batter_vs_pitch_type_ab":  25,
    "pitcher_starts_for_form":   3,
}

LEAGUE_AVG = {
    "ba":           0.248,
    "xba":          0.250,
    "babip":        0.295,
    "h9":           8.7,
    "whiff_pct":    0.245,
    "hard_hit_pct": 0.365,
}

# Spec weights sum to 1.10 — normalize to 1.0 while preserving relative importance
_raw_sum = sum(FEATURE_WEIGHTS.values())
FEATURE_WEIGHTS = {k: round(v / _raw_sum, 6) for k, v in FEATURE_WEIGHTS.items()}
_weight_sum = round(sum(FEATURE_WEIGHTS.values()), 6)
assert abs(_weight_sum - 1.0) < 1e-4, f"FEATURE_WEIGHTS sum to {_weight_sum}, must be 1.0"

if __name__ == "__main__":
    logger.info(f"Park factors loaded: {len(PARK_FACTORS)} teams")
    logger.info(f"Stadium coords loaded: {len(STADIUM_COORDS)} stadiums")
    logger.info(f"Feature weights sum: {_weight_sum:.4f} ✓")
    logger.info(f"Team ID map: {len(TEAM_ID_MAP)} teams")
