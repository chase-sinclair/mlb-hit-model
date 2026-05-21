import json
import time
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger

ODDS_BASE = "https://api.the-odds-api.com/v4"
_CACHE_BASE = Path(__file__).parent.parent / "cache"

_TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",         "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",      "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",     "Detroit Tigers": "DET",
    "Houston Astros": "HOU",       "Kansas City Royals": "KC",
    "Los Angeles Angels": "LAA",   "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",        "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",      "New York Mets": "NYM",
    "New York Yankees": "NYY",     "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI","Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SD",      "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA",     "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TB",        "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",    "Washington Nationals": "WSH",
    "Athletics": "OAK",
}


def _cache_path(key: str, date_str: str) -> Path:
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d / f"odds_{key}.json"


def _team_abbr(name: str) -> str:
    return _TEAM_NAME_MAP.get(name, name[:3].upper())


def get_game_totals(api_key: str, date_str: str = None) -> dict:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cp = _cache_path("totals", date_str)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not api_key:
        logger.warning("No ODDS_API_KEY set — skipping game totals")
        return {}
    try:
        time.sleep(0.5)
        r = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/odds/",
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "totals",
                "dateFormat":  "iso",
                "oddsFormat":  "american",
            },
            timeout=15,
        )
        if r.status_code == 401:
            logger.warning("Odds API: invalid key")
            return {}
        if r.status_code == 422:
            logger.warning("Odds API quota likely exhausted (422)")
            return {}
        r.raise_for_status()
        games = r.json()
        result = {}
        for game in games:
            home = _team_abbr(game.get("home_team", ""))
            away = _team_abbr(game.get("away_team", ""))
            best_total = None
            best_book = None
            for bookie in game.get("bookmakers", []):
                for market in bookie.get("markets", []):
                    if market.get("key") != "totals":
                        continue
                    for outcome in market.get("outcomes", []):
                        if outcome.get("name") == "Over":
                            pt = outcome.get("point")
                            if best_total is None:
                                best_total = pt
                                best_book = bookie.get("key")
            if best_total:
                over_odds = under_odds = None
                for bookie in game.get("bookmakers", []):
                    if bookie.get("key") != best_book:
                        continue
                    for market in bookie.get("markets", []):
                        if market.get("key") != "totals":
                            continue
                        for outcome in market.get("outcomes", []):
                            if outcome.get("name") == "Over":
                                over_odds = outcome.get("price")
                            elif outcome.get("name") == "Under":
                                under_odds = outcome.get("price")
                result[(home, away)] = {
                    "total":      best_total,
                    "over_odds":  over_odds,
                    "under_odds": under_odds,
                    "book":       best_book,
                }
        cp.write_text(json.dumps({str(k): v for k, v in result.items()}), encoding="utf-8")
        logger.info(f"Odds API: fetched totals for {len(result)} games")
        return result
    except Exception as e:
        logger.warning(f"Odds API game totals failed: {e}")
        return {}


def get_hit_prop_lines(api_key: str, date_str: str = None) -> dict:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cp = _cache_path("hit_props", date_str)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    if not api_key:
        logger.warning("No ODDS_API_KEY set — skipping hit prop lines")
        return {}
    try:
        time.sleep(0.5)
        r = requests.get(
            f"{ODDS_BASE}/sports/baseball_mlb/odds/",
            params={
                "apiKey":      api_key,
                "regions":     "us",
                "markets":     "batter_hits",
                "oddsFormat":  "american",
            },
            timeout=15,
        )
        if r.status_code in (401, 422, 404):
            logger.warning(f"Odds API hit props: HTTP {r.status_code}")
            return {}
        r.raise_for_status()
        games = r.json()
        result = {}
        for game in games:
            for bookie in game.get("bookmakers", []):
                for market in bookie.get("markets", []):
                    if market.get("key") != "batter_hits":
                        continue
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("description") or outcome.get("name", "")
                        price = outcome.get("price")
                        side = outcome.get("name", "")  # "Over" or "Under"
                        if name not in result:
                            result[name] = {"yes_odds": None, "no_odds": None, "implied_prob": None, "book": bookie.get("key")}
                        if side == "Over" and price:
                            result[name]["yes_odds"] = price
                            result[name]["implied_prob"] = round(_american_to_prob(price), 3)
                        elif side == "Under" and price:
                            result[name]["no_odds"] = price
        cp.write_text(json.dumps(result), encoding="utf-8")
        logger.info(f"Odds API: fetched hit props for {len(result)} players")
        return result
    except Exception as e:
        logger.warning(f"Odds API hit props failed: {e}")
        return {}


def _american_to_prob(odds: int) -> float:
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / ".env")
    api_key = os.getenv("ODDS_API_KEY", "")
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Testing odds_api.py")
    totals = get_game_totals(api_key, date_str=today)
    logger.info(f"Totals returned: {len(totals)} games")
    for k, v in list(totals.items())[:3]:
        logger.info(f"  {k}: {v}")
    props = get_hit_prop_lines(api_key, date_str=today)
    logger.info(f"Hit props returned: {len(props)} players")
