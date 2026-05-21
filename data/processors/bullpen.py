from loguru import logger

from config import LEAGUE_AVG


def compute_bullpen_features(bullpen_stats: dict | None) -> dict:
    default = {
        "bullpen_h9":           LEAGUE_AVG["h9"],
        "bullpen_whip":         1.30,
        "bullpen_era":          4.20,
        "bullpen_k9":           9.0,
        "bullpen_workload_3day": None,
    }
    if not bullpen_stats:
        return default
    try:
        return {
            "bullpen_h9":            bullpen_stats.get("bullpen_h9") or LEAGUE_AVG["h9"],
            "bullpen_whip":          bullpen_stats.get("bullpen_whip") or 1.30,
            "bullpen_era":           bullpen_stats.get("bullpen_era") or 4.20,
            "bullpen_k9":            bullpen_stats.get("bullpen_k9") or 9.0,
            "bullpen_workload_3day": bullpen_stats.get("recent_workload"),
        }
    except Exception as e:
        logger.warning(f"compute_bullpen_features failed: {e}")
        return default
