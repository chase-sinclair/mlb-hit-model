from loguru import logger


def compute_game_context_features(
    game_total: float | None,
    park_factor: float,
    weather: dict | None,
    umpire: dict | None,
    is_home: bool,
    is_day_game: bool,
) -> dict:
    try:
        if game_total is None:
            game_total = 8.5

        normalized = (game_total - 8.5) / 2.0
        normalized = max(-1.0, min(1.0, normalized))
        game_total_score = round(normalized * 0.04, 5)

        park_score = round((park_factor - 1.0) * 0.3, 5)

        w_score = 0.0
        if weather:
            w_score = float(weather.get("weather_score", 0.0) or 0.0)

        u_score = 0.0
        if umpire:
            u_score = float(umpire.get("zone_score", 0.0) or 0.0)

        return {
            "game_total_score":  game_total_score,
            "park_factor_score": park_score,
            "weather_score":     round(w_score, 5),
            "umpire_zone_score": round(u_score, 5),
            "home_field_bonus":  0.01 if is_home else 0.0,
            "day_game_penalty":  -0.01 if is_day_game else 0.0,
        }
    except Exception as e:
        logger.warning(f"compute_game_context_features failed: {e}")
        return {
            "game_total_score":  0.0,
            "park_factor_score": 0.0,
            "weather_score":     0.0,
            "umpire_zone_score": 0.0,
            "home_field_bonus":  0.0,
            "day_game_penalty":  0.0,
        }
