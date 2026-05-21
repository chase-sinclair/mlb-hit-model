from loguru import logger

from config import LEAGUE_AVG

_PA_BY_POSITION = {
    1: 4.7, 2: 4.5, 3: 4.3, 4: 4.1,
    5: 3.9, 6: 3.7, 7: 3.5, 8: 3.3, 9: 3.1,
}
_MAX_PA = 4.7


def compute_babip_regression_delta(actual_babip: float | None, xbabip: float | None) -> float:
    if actual_babip is None:
        return 0.0
    if xbabip is None:
        xbabip = LEAGUE_AVG["babip"]
    try:
        delta = actual_babip - xbabip
        # delta range ~[-0.08, +0.08] → adjustment [-0.04, +0.04]
        # Negative delta (unlucky) → positive adjustment
        adjustment = -delta / 0.08 * 0.04
        return round(max(-0.08, min(0.08, adjustment)), 5)
    except Exception as e:
        logger.warning(f"compute_babip_regression_delta failed: {e}")
        return 0.0


def compute_handedness_split_score(
    batter_bats: str | None,
    pitcher_throws: str | None,
    batter_splits: dict | None,
) -> float:
    fallback = LEAGUE_AVG["ba"]
    if not batter_splits:
        return fallback
    try:
        bats = (batter_bats or "R").upper()
        throws = (pitcher_throws or "R").upper()

        if bats == "S":
            if throws == "R":
                return batter_splits.get("vs_lhp_avg") or fallback
            else:
                return batter_splits.get("vs_rhp_avg") or fallback

        if bats == throws:
            return batter_splits.get("vs_rhp_avg" if throws == "R" else "vs_lhp_avg") or (fallback * 0.95)
        else:
            return batter_splits.get("vs_lhp_avg" if throws == "L" else "vs_rhp_avg") or (fallback * 1.05)
    except Exception as e:
        logger.warning(f"compute_handedness_split_score failed: {e}")
        return fallback


def compute_lineup_pa_weight(lineup_position: int) -> float:
    try:
        pa = _PA_BY_POSITION.get(lineup_position, 3.1)
        return round(pa / _MAX_PA, 4)
    except Exception as e:
        logger.warning(f"compute_lineup_pa_weight failed: {e}")
        return 0.66
