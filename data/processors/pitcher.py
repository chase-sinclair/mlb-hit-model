from loguru import logger


def compute_pitcher_fatigue_score(game_log: list[dict], velo_delta: float | None = None) -> float:
    if not game_log:
        return 0.0
    try:
        last = game_log[0]
        score = 0.0
        days_rest = last.get("days_rest")
        if days_rest is not None and days_rest <= 4:
            score += 0.25

        pitches = last.get("pitches_thrown", 0) or 0
        if pitches > 100:
            score += 0.20
        if pitches > 110:
            score += 0.10

        recent = game_log[:3]
        ips = [g.get("ip") for g in recent if g.get("ip") is not None]
        if ips and (sum(ips) / len(ips)) < 5.0:
            score += 0.15

        if velo_delta is not None and velo_delta < -1.5:
            score += 0.20

        return min(score, 1.0)
    except Exception as e:
        logger.warning(f"compute_pitcher_fatigue_score failed: {e}")
        return 0.0


def compute_times_through_order_weight(lineup_position: int, pitcher_avg_ip: float | None) -> float:
    if pitcher_avg_ip is None:
        pitcher_avg_ip = 5.5  # league average
    try:
        batters_per_inning = 3.3
        total_batters = pitcher_avg_ip * batters_per_inning
        tto = min(total_batters / max(lineup_position, 1), 3.0)

        # Interpolate multiplier between TTO tiers
        if tto <= 1.0:
            return 1.00
        elif tto <= 2.0:
            frac = tto - 1.0
            return round(1.00 + frac * 0.06, 4)
        else:
            frac = tto - 2.0
            return round(1.06 + frac * 0.06, 4)
    except Exception as e:
        logger.warning(f"compute_times_through_order_weight failed: {e}")
        return 1.0


def compute_pitcher_luck_adjustment(xfip: float | None, era: float | None) -> float:
    if xfip is None or era is None:
        return 0.0
    try:
        delta = xfip - era
        delta = max(-1.5, min(1.5, delta))
        return round(delta / 1.5 * 0.05, 5)
    except Exception as e:
        logger.warning(f"compute_pitcher_luck_adjustment failed: {e}")
        return 0.0
