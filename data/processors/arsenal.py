import pandas as pd
from loguru import logger

from config import LEAGUE_AVG


def compute_arsenal_score(
    pitcher_arsenal: pd.DataFrame | None,
    batter_pitch_splits: dict | None,
    min_usage_threshold: float = 0.05,
) -> dict:
    default = {
        "arsenal_weighted_ba":    LEAGUE_AVG["ba"],
        "arsenal_weighted_whiff": LEAGUE_AVG["whiff_pct"],
        "pitch_count":            0,
        "breakdown":              [],
    }
    if pitcher_arsenal is None or pitcher_arsenal.empty:
        return default

    try:
        included = pitcher_arsenal[pitcher_arsenal["usage_pct"] >= min_usage_threshold].copy()
        if included.empty:
            return default

        total_usage = included["usage_pct"].sum()
        breakdown = []
        weighted_ba = 0.0
        weighted_whiff = 0.0

        for _, row in included.iterrows():
            pt = str(row["pitch_type"])
            usage = float(row["usage_pct"])
            weight = usage / total_usage

            pitcher_ba = row.get("ba_against")
            if isinstance(pitcher_ba, float) and not pd.isna(pitcher_ba):
                pitcher_ba = float(pitcher_ba)
            else:
                pitcher_ba = None

            whiff = row.get("whiff_pct")
            if isinstance(whiff, float) and not pd.isna(whiff):
                whiff = float(whiff)
            else:
                whiff = LEAGUE_AVG["whiff_pct"]

            # Fallback chain: batter actual → pitcher ba_against → LEAGUE_AVG
            source = "actual"
            batter_ba = None
            if batter_pitch_splits and pt in batter_pitch_splits:
                entry = batter_pitch_splits[pt]
                if entry is not None and entry.get("ba") is not None:
                    batter_ba = float(entry["ba"])

            if batter_ba is None:
                if pitcher_ba is not None:
                    batter_ba = pitcher_ba
                    source = "pitcher_fallback"
                else:
                    batter_ba = LEAGUE_AVG["ba"]
                    source = "league_avg"

            weighted_ba += weight * batter_ba
            weighted_whiff += weight * whiff
            breakdown.append({
                "pitch_type": pt,
                "pitch_name": str(row.get("pitch_name", pt)),
                "usage":      round(usage, 3),
                "batter_ba":  round(batter_ba, 3),
                "whiff_pct":  round(whiff, 3),
                "source":     source,
            })

        return {
            "arsenal_weighted_ba":    round(weighted_ba, 3),
            "arsenal_weighted_whiff": round(weighted_whiff, 3),
            "pitch_count":            len(breakdown),
            "breakdown":              breakdown,
        }
    except Exception as e:
        logger.warning(f"compute_arsenal_score failed: {e}")
        return default
