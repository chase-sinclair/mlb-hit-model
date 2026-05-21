import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

WEATHER_BASE = "https://api.open-meteo.com/v1/forecast"
_CACHE_BASE = Path(__file__).parent.parent / "cache"


def _cache_path(key: str, date_str: str) -> Path:
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return d / f"weather_{h}.json"


def get_game_weather(lat: float, lon: float, game_time_utc: str, date_str: str = None) -> dict:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"{lat}_{lon}_{game_time_utc}"
    cp = _cache_path(cache_key, date_str)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass

    try:
        time.sleep(0.3)
        r = requests.get(
            WEATHER_BASE,
            params={
                "latitude":  lat,
                "longitude": lon,
                "hourly":    "temperature_2m,windspeed_10m,winddirection_10m,relativehumidity_2m,precipitation_probability",
                "timezone":  "auto",
                "forecast_days": 2,
            },
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"Weather API failed for ({lat},{lon}): {e}")
        return _neutral_weather()

    try:
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return _neutral_weather()

        # Match game hour
        game_hour = _parse_game_hour(game_time_utc)
        game_date = game_time_utc[:10] if game_time_utc else date_str

        best_idx = None
        for i, t in enumerate(times):
            if t.startswith(game_date):
                h = int(t[11:13]) if len(t) >= 13 else 0
                if best_idx is None or abs(h - game_hour) < abs(int(times[best_idx][11:13]) - game_hour):
                    best_idx = i

        if best_idx is None:
            best_idx = 0

        def _val(key, idx):
            vals = hourly.get(key, [])
            return float(vals[idx]) if idx < len(vals) and vals[idx] is not None else None

        temp_c = _val("temperature_2m", best_idx)
        wind_kmh = _val("windspeed_10m", best_idx)
        wind_dir = _val("winddirection_10m", best_idx)
        humidity = _val("relativehumidity_2m", best_idx)
        precip_prob = _val("precipitation_probability", best_idx)

        temp_f = round(temp_c * 9 / 5 + 32, 1) if temp_c is not None else None
        wind_mph = round(wind_kmh * 0.621371, 1) if wind_kmh is not None else None
        precip_frac = precip_prob / 100 if precip_prob is not None else 0.0

        result = {
            "temp_f":         temp_f,
            "wind_speed_mph": wind_mph,
            "wind_direction": int(wind_dir) if wind_dir is not None else None,
            "humidity_pct":   humidity,
            "precip_prob":    round(precip_frac, 2),
            "weather_score":  0.0,
        }
        result["weather_score"] = _compute_weather_score(
            lat, lon, wind_mph, wind_dir, temp_f, humidity
        )
        cp.write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception as e:
        logger.warning(f"Weather parsing failed for ({lat},{lon}): {e}")
        return _neutral_weather()


def _parse_game_hour(game_time_utc: str) -> int:
    try:
        dt = datetime.fromisoformat(game_time_utc.replace("Z", "+00:00"))
        return dt.hour
    except Exception:
        return 19  # default 7 PM UTC


def _compute_weather_score(
    lat: float, lon: float,
    wind_mph: float | None,
    wind_dir: float | None,
    temp_f: float | None,
    humidity: float | None,
) -> float:
    score = 0.0
    if wind_mph and wind_dir is not None:
        out_dir = _get_stadium_out_direction(lat, lon)
        if out_dir is not None:
            blowing_out = _is_blowing_out(wind_dir, out_dir, tolerance=60)
            if blowing_out and wind_mph >= 10:
                score += 0.06
            if blowing_out and wind_mph >= 15:
                score += 0.04
    if temp_f is not None:
        if temp_f < 45:
            score -= 0.05
        if temp_f < 35:
            score -= 0.03
        if temp_f > 80:
            score += 0.01
    if humidity is not None and humidity > 70:
        score -= 0.01
    return round(score, 4)


def _is_blowing_out(wind_from_dir: float, out_dir: float, tolerance: int = 60) -> bool:
    # Wind FROM direction means ball travels IN that direction.
    # Wind blows OUT if wind_from_dir ≈ out_dir (wind pushes from home plate toward outfield).
    diff = abs((wind_from_dir - out_dir + 180) % 360 - 180)
    return diff <= tolerance


def _get_stadium_out_direction(lat: float, lon: float) -> int | None:
    try:
        from config import STADIUM_COORDS, STADIUM_OUT_DIRECTION
        best_team = None
        best_dist = float("inf")
        for team, (slat, slon) in STADIUM_COORDS.items():
            dist = (lat - slat) ** 2 + (lon - slon) ** 2
            if dist < best_dist:
                best_dist = dist
                best_team = team
        return STADIUM_OUT_DIRECTION.get(best_team)
    except Exception:
        return None


def _neutral_weather() -> dict:
    return {
        "temp_f":         72.0,
        "wind_speed_mph": 5.0,
        "wind_direction": 180,
        "humidity_pct":   50.0,
        "precip_prob":    0.0,
        "weather_score":  0.0,
    }


if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Testing weather_api.py — Yankee Stadium")
    # Tonight's game time at Yankee Stadium, approximate 7 PM ET = 23:00 UTC
    game_utc = f"{today}T23:05:00Z"
    weather = get_game_weather(40.8296, -73.9262, game_utc, date_str=today)
    logger.info(f"Weather result: {weather}")
