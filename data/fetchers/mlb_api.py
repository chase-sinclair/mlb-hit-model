import hashlib
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests
from loguru import logger

BASE = "https://statsapi.mlb.com/api/v1"
_DELAY = 0.5
_CACHE_BASE = Path(__file__).parent.parent / "cache"


def _cache_path(url: str, date_str: str) -> Path:
    key = hashlib.md5(url.encode()).hexdigest()
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _get(url: str, params: dict = None, date_str: str = None, skip_cache: bool = False) -> dict | list | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    full_url = url if not params else url + "?" + "&".join(f"{k}={v}" for k, v in params.items())
    cp = _cache_path(full_url, date_str)
    if not skip_cache and cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        time.sleep(_DELAY)
        r = requests.get(url, params=params, timeout=15)
        if r.status_code == 404:
            logger.warning(f"404 from {full_url}")
            return None
        r.raise_for_status()
        data = r.json()
        cp.write_text(json.dumps(data), encoding="utf-8")
        return data
    except Exception as e:
        logger.warning(f"MLB API request failed for {full_url}: {e}")
        return None


def get_today_schedule(date_str: str) -> list[dict]:
    data = _get(
        f"{BASE}/schedule",
        {"sportId": "1", "date": date_str, "hydrate": "probablePitcher,lineups,team"},
        date_str=date_str,
    )
    if not data:
        return []
    games = []
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            home = g.get("teams", {}).get("home", {})
            away = g.get("teams", {}).get("away", {})

            def _pitcher(side):
                pp = side.get("probablePitcher")
                if not pp:
                    return None
                return {
                    "id": pp.get("id"),
                    "name": pp.get("fullName"),
                    "throws": pp.get("pitchHand", {}).get("code"),
                }

            def _lineup(side):
                players = side.get("battingOrder", [])
                if not players:
                    return []
                return [{"player_id": p.get("id"), "name": p.get("fullName")} for p in players]

            home_team = home.get("team", {})
            away_team = away.get("team", {})

            from config import TEAM_ID_MAP
            home_abbr = TEAM_ID_MAP.get(home_team.get("id"), home_team.get("abbreviation", "UNK"))
            away_abbr = TEAM_ID_MAP.get(away_team.get("id"), away_team.get("abbreviation", "UNK"))

            game_time_raw = g.get("gameDate", "")
            try:
                dt = datetime.fromisoformat(game_time_raw.replace("Z", "+00:00"))
                et_offset = timedelta(hours=-4)
                dt_et = dt + et_offset
                hour = dt_et.hour % 12 or 12
                ampm = "PM" if dt_et.hour >= 12 else "AM"
                game_time_str = f"{hour}:{dt_et.minute:02d} {ampm} ET"
            except Exception:
                game_time_str = game_time_raw

            games.append({
                "game_id": g.get("gamePk"),
                "game_time": game_time_str,
                "game_time_utc": game_time_raw,
                "home_team": home_abbr,
                "away_team": away_abbr,
                "home_team_id": home_team.get("id"),
                "away_team_id": away_team.get("id"),
                "venue": g.get("venue", {}).get("name"),
                "home_probable_pitcher": _pitcher(home),
                "away_probable_pitcher": _pitcher(away),
                "home_lineup": _lineup(home),
                "away_lineup": _lineup(away),
                "status": g.get("status", {}).get("abstractGameState"),
            })
    return games


def get_lineup(game_id: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    # Always skip cache — lineups post throughout the morning and cache would return stale empty data
    data = _get(f"{BASE}/game/{game_id}/boxscore", date_str=date_str, skip_cache=True)
    if not data:
        return None
    teams = data.get("teams", {})
    result = {}
    for side in ("home", "away"):
        t = teams.get(side, {})
        order = t.get("battingOrder", [])
        batters = t.get("players", {})
        if not order:
            return None
        lineup = []
        for pos, pid in enumerate(order, 1):
            key = f"ID{pid}"
            p = batters.get(key, {})
            info = p.get("person", {})
            pos_info = p.get("position", {})
            batting_hand = p.get("batter", {})
            lineup.append({
                "batting_order": pos,
                "player_id": pid,
                "name": info.get("fullName"),
                "position": pos_info.get("abbreviation"),
                "bats": p.get("person", {}).get("batSide", {}).get("code"),
            })
        result[side] = lineup
    return result if result.get("home") and result.get("away") else None


def get_batter_season_stats(player_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "season", "season": season, "group": "hitting"},
        date_str=date_str,
    )
    if not data:
        return None
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        return {
            "avg": _f(s.get("avg")),
            "obp": _f(s.get("obp")),
            "slg": _f(s.get("slg")),
            "hits": s.get("hits", 0),
            "at_bats": s.get("atBats", 0),
            "games": s.get("gamesPlayed", 0),
            "babip": _f(s.get("babip")),
        }
    return None


def get_batter_last_n_games(player_id: int, n: int = 14, season: int = None, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if season is None:
        season = int(date_str[:4])
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "gameLog", "season": season, "group": "hitting"},
        date_str=date_str,
    )
    if not data:
        return None
    splits = data.get("stats", [{}])[0].get("splits", [])
    if not splits:
        return None

    cutoff_7 = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=7)
    cutoff_14 = datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=14)

    def _agg(games):
        h = sum(g.get("stat", {}).get("hits", 0) for g in games)
        ab = sum(g.get("stat", {}).get("atBats", 0) for g in games)
        return h, ab, len(games)

    recent_7 = []
    recent_14 = []
    for split in splits:
        gd = split.get("date", "")
        try:
            gdt = datetime.strptime(gd, "%Y-%m-%d")
        except Exception:
            continue
        if gdt >= cutoff_14:
            recent_14.append(split)
        if gdt >= cutoff_7:
            recent_7.append(split)

    h7, ab7, g7 = _agg(recent_7)
    h14, ab14, g14 = _agg(recent_14)

    return {
        "h_pct_7day":  round(h7 / ab7, 3) if ab7 > 0 else None,
        "h_pct_14day": round(h14 / ab14, 3) if ab14 > 0 else None,
        "hits_14day":  h14,
        "ab_14day":    ab14,
        "games_14day": g14,
    }


def get_batter_vs_pitcher(batter_id: int, pitcher_id: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/people/{batter_id}/stats",
        {"stats": "vsPlayer", "opposingPlayerId": pitcher_id, "group": "hitting"},
        date_str=date_str,
    )
    if not data:
        return {"ab": 0, "hits": 0, "avg": None, "hr": 0, "bb": 0}
    from config import MIN_SAMPLES
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        ab = s.get("atBats", 0)
        hits = s.get("hits", 0)
        return {
            "ab":   ab,
            "hits": hits,
            "avg":  round(hits / ab, 3) if ab >= MIN_SAMPLES["career_vs_pitcher_ab"] else None,
            "hr":   s.get("homeRuns", 0),
            "bb":   s.get("baseOnBalls", 0),
        }
    return {"ab": 0, "hits": 0, "avg": None, "hr": 0, "bb": 0}


def get_batter_splits(player_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "statSplits", "season": season, "group": "hitting"},
        date_str=date_str,
    )
    if not data:
        return None
    result = {
        "vs_rhp_avg": None, "vs_lhp_avg": None,
        "home_avg": None,   "away_avg": None,
        "day_avg": None,    "night_avg": None,
    }
    split_map = {
        "vs. RHP": "vs_rhp_avg",
        "vs. LHP": "vs_lhp_avg",
        "Home":    "home_avg",
        "Away":    "away_avg",
        "Day":     "day_avg",
        "Night":   "night_avg",
    }
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            desc = split.get("split", {}).get("description", "")
            key = split_map.get(desc)
            if key:
                result[key] = _f(split.get("stat", {}).get("avg"))
    return result


def get_pitcher_season_stats(player_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "season", "season": season, "group": "pitching"},
        date_str=date_str,
    )
    if not data:
        return None
    for split in data.get("stats", [{}])[0].get("splits", []):
        s = split.get("stat", {})
        ip = _f(s.get("inningsPitched"))
        gs = s.get("gamesStarted", 0)
        return {
            "era":   _f(s.get("era")),
            "whip":  _f(s.get("whip")),
            "h9":    _f(s.get("hitsPer9Inn")),
            "k9":    _f(s.get("strikeoutsPer9Inn")),
            "bb9":   _f(s.get("walksPer9Inn")),
            "avg_ip_per_start": round(ip / gs, 2) if ip and gs > 0 else None,
            "games_started": gs,
            "babip": _f(s.get("babip")),
        }
    return None


def get_pitcher_game_log(player_id: int, season: int, last_n: int = 5, date_str: str = None) -> list[dict]:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/people/{player_id}/stats",
        {"stats": "gameLog", "season": season, "group": "pitching"},
        date_str=date_str,
    )
    if not data:
        return []
    splits = data.get("stats", [{}])[0].get("splits", [])
    starts = [s for s in splits if s.get("stat", {}).get("gamesStarted", 0) > 0]
    starts_sorted = sorted(starts, key=lambda x: x.get("date", ""), reverse=True)[:last_n]

    result = []
    for i, start in enumerate(starts_sorted):
        s = start.get("stat", {})
        gd = start.get("date", "")
        days_rest = None
        if i < len(starts_sorted) - 1:
            prev_date = starts_sorted[i + 1].get("date", "")
            try:
                d1 = datetime.strptime(gd, "%Y-%m-%d")
                d2 = datetime.strptime(prev_date, "%Y-%m-%d")
                days_rest = (d1 - d2).days
            except Exception:
                pass
        result.append({
            "game_date":      gd,
            "opponent":       start.get("opponent", {}).get("name", ""),
            "ip":             _f(s.get("inningsPitched")),
            "hits_allowed":   s.get("hits", 0),
            "er":             s.get("earnedRuns", 0),
            "pitches_thrown": s.get("numberOfPitches", 0),
            "days_rest":      days_rest,
        })
    return result


def get_player_info(player_id: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(f"{BASE}/people/{player_id}", date_str=date_str)
    if not data:
        return None
    people = data.get("people", [])
    if not people:
        return None
    p = people[0]
    from config import TEAM_ID_MAP
    team_id = p.get("currentTeam", {}).get("id")
    return {
        "id":       p.get("id"),
        "name":     p.get("fullName"),
        "team":     TEAM_ID_MAP.get(team_id, "UNK"),
        "position": p.get("primaryPosition", {}).get("abbreviation"),
        "bats":     p.get("batSide", {}).get("code"),
        "throws":   p.get("pitchHand", {}).get("code"),
    }


def get_bullpen_stats(team_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(
        f"{BASE}/teams/{team_id}/stats",
        {"stats": "season", "season": season, "group": "pitching", "sportId": "1"},
        date_str=date_str,
    )
    if not data:
        return None
    relief_stats = []
    for stat_group in data.get("stats", []):
        for split in stat_group.get("splits", []):
            pos = split.get("player", {}).get("primaryPosition", {}).get("abbreviation", "")
            if pos in ("RP", "P"):
                gs = split.get("stat", {}).get("gamesStarted", 1)
                if gs == 0:
                    relief_stats.append(split.get("stat", {}))

    if not relief_stats:
        # Fall back to team aggregate
        for stat_group in data.get("stats", []):
            for split in stat_group.get("splits", []):
                s = split.get("stat", {})
                if s:
                    return {
                        "bullpen_h9":       _f(s.get("hitsPer9Inn")),
                        "bullpen_whip":     _f(s.get("whip")),
                        "bullpen_era":      _f(s.get("era")),
                        "bullpen_k9":       _f(s.get("strikeoutsPer9Inn")),
                        "recent_workload":  None,
                    }
        return None

    def _avg(key):
        vals = [_f(s.get(key)) for s in relief_stats if _f(s.get(key)) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None

    return {
        "bullpen_h9":      _avg("hitsPer9Inn"),
        "bullpen_whip":    _avg("whip"),
        "bullpen_era":     _avg("era"),
        "bullpen_k9":      _avg("strikeoutsPer9Inn"),
        "recent_workload": None,
    }


def get_umpire_for_game(game_id: int, date_str: str = None) -> str | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    data = _get(f"{BASE}/game/{game_id}/boxscore", date_str=date_str)
    if not data:
        return None
    for official in data.get("officials", []):
        if official.get("officialType") == "Home Plate":
            return official.get("official", {}).get("fullName")
    return None


def _f(val) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"Testing mlb_api.py for {today}")
    games = get_today_schedule(today)
    logger.info(f"Games today: {len(games)}")
    if games:
        g = games[0]
        logger.info(f"First game: {g['away_team']} @ {g['home_team']} — {g['game_time']}")
        logger.info(f"  Home pitcher: {g['home_probable_pitcher']}")
        logger.info(f"  Away pitcher: {g['away_probable_pitcher']}")
        logger.info(f"  Home lineup posted: {bool(g['home_lineup'])}")
