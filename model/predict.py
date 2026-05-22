import math
import pickle
from pathlib import Path

from loguru import logger

from config import FEATURE_WEIGHTS, LEAGUE_AVG

_ARTIFACTS = Path(__file__).parent / "artifacts"
_MODEL_PATH = _ARTIFACTS / "hit_model.pkl"
_SCALER_PATH = _ARTIFACTS / "scaler.pkl"

_model = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is not None:
        return True
    if _MODEL_PATH.exists() and _SCALER_PATH.exists():
        try:
            with open(_MODEL_PATH, "rb") as f:
                _model = pickle.load(f)
            with open(_SCALER_PATH, "rb") as f:
                _scaler = pickle.load(f)
            logger.info("Loaded trained model from artifacts/")
            return True
        except Exception as e:
            logger.warning(f"Failed to load model artifacts: {e}")
    return False


# Normalization ranges for each feature (min, max) — used for sigmoid fallback
_NORM_RANGES = {
    "batter_h_pct_14day":          (0.10, 0.55),
    "batter_h_pct_7day":           (0.10, 0.60),
    "batter_h_pct_season":         (0.15, 0.40),
    "batter_xba_season":           (0.15, 0.38),
    "babip_regression_delta":      (-0.08, 0.08),
    "batter_k_rate":               (0.10, 0.38),  # low K rate = elite contact
    "career_h_ab_vs_pitcher":      (0.0, 0.50),
    "arsenal_weighted_ba":         (0.15, 0.40),
    "arsenal_weighted_whiff":      (0.10, 0.45),
    "pitcher_h9_season":           (5.0, 13.0),
    "pitcher_xfip":                (2.5, 6.0),
    "pitcher_k9":                  (5.0, 14.0),   # high K9 = tough pitcher
    "pitcher_babip_against":       (0.25, 0.36),
    "pitcher_last3_hits_avg":      (3.0, 12.0),
    "pitcher_fatigue_score":       (0.0, 1.0),
    "times_through_order":         (1.0, 1.12),
    "bullpen_h9":                  (6.0, 11.0),
    "park_factor":                 (0.92, 1.18),
    "game_total":                  (6.0, 12.0),
    "weather_score":               (-0.09, 0.10),
    "handedness_split":            (0.18, 0.32),
    "lineup_position_pa_weight":   (0.66, 1.0),
    "umpire_zone_score":           (-0.10, 0.10),
    "exit_velocity_14day":         (82.0, 96.0),
}

# Higher value = more hits (True) or fewer hits (True = inverted)
_INVERTED = {
    "arsenal_weighted_whiff",  # higher whiff = harder to hit
    "pitcher_h9_season",        # higher H/9 = actually MORE hits, but this is pitcher stat (bad for batter)
    # Note: pitcher_h9 high = more hits allowed by pitcher = GOOD for batter, so NOT inverted
    # But arsenal_weighted_whiff: higher whiff against batter = BAD for batter
}

# Features where higher value = fewer hits (must be inverted in scoring)
_FEWER_HITS_IS_HIGHER = {
    "arsenal_weighted_whiff",
    "pitcher_xfip",         # lower xFIP = better pitcher (fewer hits)
    "pitcher_babip_against", # higher = pitcher gives up more hits
    "pitcher_fatigue_score", # higher = worse pitcher (more hits expected) — NOT inverted
}

# Actually re-think: normalization should map each feature so that
# 0.0 = worst for batter, 1.0 = best for batter
_BATTER_FAVORABLE_HIGH = {
    "batter_h_pct_14day",
    "batter_h_pct_7day",
    "batter_h_pct_season",
    "batter_xba_season",
    "babip_regression_delta",   # positive delta = more hits coming
    "career_h_ab_vs_pitcher",
    "arsenal_weighted_ba",
    "pitcher_h9_season",        # higher H/9 allowed = more hits for batter
    "pitcher_babip_against",    # higher BABIP against = more hits
    "pitcher_last3_hits_avg",   # higher hits allowed recently = better for batter
    "pitcher_fatigue_score",    # higher fatigue = better for batter
    "times_through_order",      # higher TTO multiplier = better
    "bullpen_h9",               # higher bullpen H/9 = better for batter
    "park_factor",
    "game_total",
    "weather_score",            # higher weather score = better for batters
    "handedness_split",
    "lineup_position_pa_weight",
    "umpire_zone_score",        # positive = batter-friendly
    "exit_velocity_14day",
}
_BATTER_FAVORABLE_LOW = {
    "arsenal_weighted_whiff",   # lower whiff = batter makes more contact
    "pitcher_xfip",             # lower xFIP = better pitcher = fewer hits (bad for batter)
    "batter_k_rate",            # lower K rate = more contact = more hits
    "pitcher_k9",               # lower K9 = more hittable pitcher
}


def _normalize(feature: str, value: float) -> float:
    lo, hi = _NORM_RANGES.get(feature, (0.0, 1.0))
    if hi == lo:
        return 0.5
    norm = (value - lo) / (hi - lo)
    norm = max(0.0, min(1.0, norm))
    if feature in _BATTER_FAVORABLE_LOW:
        norm = 1.0 - norm
    return norm


def compute_hit_probability(features: dict) -> dict:
    if _load_model() and _model and _scaler:
        return _ml_predict(features)
    return _sigmoid_predict(features)


def _sigmoid_predict(features: dict) -> dict:
    weighted_sum = 0.0
    used_features = 0
    fallback_count = 0

    for feat, weight in FEATURE_WEIGHTS.items():
        val = features.get(feat)
        if val is None:
            val = _default_value(feat)
            fallback_count += 1
        try:
            norm = _normalize(feat, float(val))
            weighted_sum += weight * norm
            used_features += 1
        except (TypeError, ValueError):
            fallback_count += 1

    raw_score = weighted_sum
    prob = 1.0 / (1.0 + math.exp(-((raw_score - 0.5) * 6)))
    prob = max(0.40, min(0.92, prob))

    total_features = len(FEATURE_WEIGHTS)
    fallback_pct = fallback_count / total_features if total_features > 0 else 0
    confidence = "HIGH" if fallback_pct < 0.20 else ("MEDIUM" if fallback_pct < 0.40 else "LOW")

    return {
        "hit_probability": round(prob, 4),
        "confidence":      confidence,
        "fallback_pct":    round(fallback_pct, 2),
        "model":           "sigmoid",
    }


def _ml_predict(features: dict) -> dict:
    import numpy as np
    try:
        feat_names = list(FEATURE_WEIGHTS.keys())
        row = []
        fallback_count = 0
        for feat in feat_names:
            val = features.get(feat)
            if val is None:
                val = _default_value(feat)
                fallback_count += 1
            row.append(float(val) if val is not None else 0.0)

        X = np.array(row).reshape(1, -1)
        X_scaled = _scaler.transform(X)
        prob = float(_model.predict_proba(X_scaled)[0][1])
        prob = max(0.40, min(0.92, prob))

        fallback_pct = fallback_count / len(feat_names)
        confidence = "HIGH" if fallback_pct < 0.20 else ("MEDIUM" if fallback_pct < 0.40 else "LOW")

        return {
            "hit_probability": round(prob, 4),
            "confidence":      confidence,
            "fallback_pct":    round(fallback_pct, 2),
            "model":           "xgboost",
        }
    except Exception as e:
        logger.warning(f"ML predict failed, falling back to sigmoid: {e}")
        return _sigmoid_predict(features)


def _default_value(feature: str) -> float:
    defaults = {
        "batter_h_pct_14day":          LEAGUE_AVG["ba"],
        "batter_h_pct_7day":           LEAGUE_AVG["ba"],
        "batter_h_pct_season":         LEAGUE_AVG["ba"],
        "batter_xba_season":           LEAGUE_AVG["xba"],
        "babip_regression_delta":      0.0,
        "batter_k_rate":               LEAGUE_AVG["k_rate"],
        "career_h_ab_vs_pitcher":      LEAGUE_AVG["ba"],
        "arsenal_weighted_ba":         LEAGUE_AVG["ba"],
        "arsenal_weighted_whiff":      LEAGUE_AVG["whiff_pct"],
        "pitcher_h9_season":           LEAGUE_AVG["h9"],
        "pitcher_xfip":                4.0,
        "pitcher_k9":                  LEAGUE_AVG["k9"],
        "pitcher_babip_against":       LEAGUE_AVG["babip"],
        "pitcher_last3_hits_avg":      LEAGUE_AVG["h9"],
        "pitcher_fatigue_score":       0.0,
        "times_through_order":         1.0,
        "bullpen_h9":                  LEAGUE_AVG["h9"],
        "park_factor":                 1.0,
        "game_total":                  8.5,
        "weather_score":               0.0,
        "handedness_split":            LEAGUE_AVG["ba"],
        "lineup_position_pa_weight":   0.85,
        "umpire_zone_score":           0.0,
        "exit_velocity_14day":         88.5,
    }
    return defaults.get(feature, 0.0)


def _top_factors(features: dict, n: int = 3) -> list[str]:
    scores = {}
    for feat, weight in FEATURE_WEIGHTS.items():
        val = features.get(feat)
        if val is None:
            continue
        try:
            norm = _normalize(feat, float(val))
            deviation = abs(norm - 0.5)
            scores[feat] = weight * deviation
        except (TypeError, ValueError):
            pass
    return sorted(scores, key=scores.get, reverse=True)[:n]


def rank_predictions(predictions: list[dict]) -> list[dict]:
    for p in predictions:
        edge = p.get("edge")
        if edge is None:
            p["rating"] = "NO LINE"
            continue
        data_ok = p.get("confidence") != "LOW"
        fallback_pct = p.get("fallback_pct", 0)
        if fallback_pct > 0.40 or not data_ok:
            p["rating"] = "SKIP - LOW DATA"
        elif edge > 0.08:
            p["rating"] = "STRONG BET"
        elif edge > 0.05:
            p["rating"] = "VALUE BET"
        elif edge > 0.02:
            p["rating"] = "LEAN"
        else:
            p["rating"] = "PASS"

    return sorted(predictions, key=lambda x: x.get("edge") or -999, reverse=True)
