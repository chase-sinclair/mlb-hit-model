import hashlib
import io
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from loguru import logger

SAVANT_CSV = "https://baseballsavant.mlb.com/statcast_search/csv"
SAVANT_PITCHING = "https://baseballsavant.mlb.com/player-services/statcast-pitching"
_DELAY = 1.0
_CACHE_BASE = Path(__file__).parent.parent / "cache"

_HIT_EVENTS = {"single", "double", "triple", "home_run"}
_AB_EVENTS = {
    "single", "double", "triple", "home_run",
    "strikeout", "strikeout_double_play", "field_out",
    "grounded_into_double_play", "force_out", "double_play",
    "fielders_choice", "fielders_choice_out",
}


def _cache_path(key: str, date_str: str) -> Path:
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return d / f"{h}.json"


def _cache_df_path(key: str, date_str: str) -> Path:
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return d / f"{h}.csv"


def _get_savant_csv(params: dict, date_str: str) -> pd.DataFrame | None:
    cache_key = "savant_csv_" + str(sorted(params.items()))
    cp = _cache_df_path(cache_key, date_str)
    if cp.exists():
        try:
            df = pd.read_csv(cp, low_memory=False)
            return df if not df.empty else None
        except Exception:
            pass
    try:
        time.sleep(_DELAY)
        r = requests.get(SAVANT_CSV, params=params, timeout=30)
        r.raise_for_status()
        content = r.content
        # Savant occasionally returns an HTML error page instead of CSV
        if b"<!DOCTYPE" in content[:200] or b"<html" in content[:200]:
            logger.warning(f"Savant returned HTML instead of CSV for params {params}")
            return None
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
        if df.empty:
            return None
        df.to_csv(cp, index=False)
        return df
    except Exception as e:
        logger.warning(f"Savant CSV fetch failed for {params}: {e}")
        return None


def get_batter_statcast_season(player_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    params = {
        "player_type": "batter",
        "player_id": player_id,
        "season": season,
        "min_pitches": 1,
        "type": "details",
    }
    df = _get_savant_csv(params, date_str)
    if df is None or df.empty:
        return None
    return _aggregate_batter_statcast(df)


def get_batter_statcast_rolling(player_id: int, days: int = 14, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=days)).strftime("%Y-%m-%d")
    params = {
        "player_type": "batter",
        "player_id": player_id,
        "game_date_gt": start_date,
        "game_date_lt": date_str,
        "min_pitches": 1,
        "type": "details",
    }
    df = _get_savant_csv(params, date_str)
    if df is None or df.empty:
        return None
    base = _aggregate_batter_statcast(df)
    if base is None:
        return None
    ab_col = df["events"].isin(_AB_EVENTS) if "events" in df.columns else pd.Series([False] * len(df))
    return {
        "exit_velo_14day": base.get("exit_velo_avg"),
        "hard_hit_14day":  base.get("hard_hit_pct"),
        "xba_14day":       base.get("xba"),
        "ab_14day":        int(ab_col.sum()),
    }


def _aggregate_batter_statcast(df: pd.DataFrame) -> dict | None:
    try:
        result = {}
        if "estimated_ba_using_speedangle" in df.columns:
            xba_vals = pd.to_numeric(df["estimated_ba_using_speedangle"], errors="coerce").dropna()
            result["xba"] = round(float(xba_vals.mean()), 3) if not xba_vals.empty else None
        else:
            result["xba"] = None

        if "babip_value" in df.columns:
            babip_vals = pd.to_numeric(df["babip_value"], errors="coerce").dropna()
            result["babip"] = round(float(babip_vals.mean()), 3) if not babip_vals.empty else None
        else:
            # Compute BABIP from events
            result["babip"] = _compute_babip(df)

        result["xbabip"] = None  # not reliably available in raw CSV

        if "launch_speed" in df.columns:
            ev = pd.to_numeric(df["launch_speed"], errors="coerce").dropna()
            contact = ev[ev > 0]
            result["exit_velo_avg"] = round(float(contact.mean()), 1) if not contact.empty else None
            result["hard_hit_pct"] = round(float((contact >= 95).sum() / len(contact)), 3) if not contact.empty else None
        else:
            result["exit_velo_avg"] = None
            result["hard_hit_pct"] = None

        if "launch_speed_angle" in df.columns:
            barrel = pd.to_numeric(df["launch_speed_angle"], errors="coerce")
            result["barrel_pct"] = round(float((barrel == 6).sum() / max(len(df), 1)), 3)
        else:
            result["barrel_pct"] = None

        if "sprint_speed" in df.columns:
            ss = pd.to_numeric(df["sprint_speed"], errors="coerce").dropna()
            result["sprint_speed"] = round(float(ss.mean()), 1) if not ss.empty else None
        else:
            result["sprint_speed"] = None

        if "hit_location" in df.columns and "stand" in df.columns:
            batted = df[df["launch_speed"].notna()] if "launch_speed" in df.columns else df
            if len(batted) > 0:
                # pull = hit to same side as batter's box (simplified: location 1-3 for RHB)
                result["pull_pct"] = None
                result["oppo_pct"] = None
            else:
                result["pull_pct"] = None
                result["oppo_pct"] = None
        else:
            result["pull_pct"] = None
            result["oppo_pct"] = None

        return result
    except Exception as e:
        logger.warning(f"Error aggregating batter Statcast: {e}")
        return None


def _compute_babip(df: pd.DataFrame) -> float | None:
    if "events" not in df.columns:
        return None
    events = df["events"].fillna("")
    hits = events.isin({"single", "double", "triple"}).sum()
    hr = events.isin({"home_run"}).sum()
    ab = events.isin(_AB_EVENTS).sum()
    k = events.isin({"strikeout", "strikeout_double_play"}).sum()
    sf = events.isin({"sac_fly"}).sum() if "sac_fly" in events.values else 0
    denom = ab - k - hr + sf
    return round(float((hits) / denom), 3) if denom > 0 else None


def get_pitcher_arsenal(pitcher_id: int, season: int, date_str: str = None) -> pd.DataFrame | None:
    """Build pitcher arsenal from Statcast CSV — pitch-level data aggregated by pitch_type."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"arsenal_csv_{pitcher_id}_{season}"
    cp = _cache_df_path(cache_key, date_str)
    if cp.exists():
        try:
            df = pd.read_csv(cp)
            return df if not df.empty else None
        except Exception:
            pass

    params = {
        "player_type": "pitcher",
        "player_id": pitcher_id,
        "season": season,
        "type": "details",
        "min_pitches": 1,
    }
    df = _get_savant_csv(params, date_str)
    if df is None or df.empty:
        return None

    try:
        from config import PITCH_TYPES
        if "pitch_type" not in df.columns:
            return None

        total_pitches = len(df)
        rows = []
        for pt, group in df.groupby("pitch_type"):
            if not pt or pd.isna(pt):
                continue
            usage = len(group) / total_pitches
            if usage < 0.03:
                continue

            # Whiff rate: swinging strikes / pitches thrown
            if "description" in group.columns:
                swings = group["description"].isin(
                    {"swinging_strike", "swinging_strike_blocked", "foul_tip"}
                ).sum()
                whiff_pct = round(float(swings / len(group)), 3) if len(group) > 0 else None
            else:
                whiff_pct = None

            # BA against: hits / at-bat events on balls in play
            ba_against = None
            if "events" in group.columns:
                ab = group["events"].isin(_AB_EVENTS).sum()
                hits = group["events"].isin(_HIT_EVENTS).sum()
                ba_against = round(float(hits / ab), 3) if ab > 0 else None

            avg_velocity = None
            if "release_speed" in group.columns:
                v = pd.to_numeric(group["release_speed"], errors="coerce").dropna()
                avg_velocity = round(float(v.mean()), 1) if not v.empty else None

            spin_rate = None
            if "release_spin_rate" in group.columns:
                sp = pd.to_numeric(group["release_spin_rate"], errors="coerce").dropna()
                spin_rate = round(float(sp.mean()), 0) if not sp.empty else None

            rows.append({
                "pitch_type":   str(pt),
                "pitch_name":   PITCH_TYPES.get(str(pt), str(pt)),
                "usage_pct":    round(usage, 3),
                "ba_against":   ba_against,
                "whiff_pct":    whiff_pct,
                "avg_velocity": avg_velocity,
                "spin_rate":    spin_rate,
            })

        if not rows:
            return None
        result = pd.DataFrame(rows).sort_values("usage_pct", ascending=False).reset_index(drop=True)
        result.to_csv(cp, index=False)
        return result
    except Exception as e:
        logger.warning(f"Arsenal aggregation failed for {pitcher_id}/{season}: {e}")
        return None


def get_batter_vs_pitch_types(player_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    params = {
        "player_type": "batter",
        "player_id": player_id,
        "season": season,
        "type": "details",
    }
    df = _get_savant_csv(params, date_str)
    if df is None or df.empty:
        return None
    if "pitch_type" not in df.columns or "events" not in df.columns:
        return None

    from config import MIN_SAMPLES
    result = {}
    for pt, group in df.groupby("pitch_type"):
        if not pt or pd.isna(pt):
            continue
        ab = int(group["events"].isin(_AB_EVENTS).sum())
        hits = int(group["events"].isin(_HIT_EVENTS).sum())
        if ab < MIN_SAMPLES["batter_vs_pitch_type_ab"]:
            continue
        result[str(pt)] = {
            "ba":   round(hits / ab, 3) if ab > 0 else 0.0,
            "ab":   ab,
            "hits": hits,
        }
    return result if result else None


def get_pitcher_xfip(pitcher_id: int, season: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    cache_key = f"xfip_{pitcher_id}_{season}"
    cp = _cache_path(cache_key, date_str)
    if cp.exists():
        try:
            return json.loads(cp.read_text(encoding="utf-8"))
        except Exception:
            pass
    try:
        from pybaseball import pitching_stats
        time.sleep(_DELAY)
        df = pitching_stats(season, qual=0)
        if df is None or df.empty:
            return None

        # Try to match by MLB ID via playerid_lookup or IDfangraphs column
        row = None
        if "IDfangraphs" in df.columns:
            pass  # Fangraphs ID != MLB ID; need to use name matching fallback

        # Use playerid_lookup if available to map MLB ID to Fangraphs ID
        try:
            from pybaseball import playerid_reverse_lookup
            id_map = playerid_reverse_lookup([pitcher_id], key_type="mlbam")
            if not id_map.empty:
                fg_id = id_map.iloc[0].get("key_fangraphs")
                if fg_id and "IDfangraphs" in df.columns:
                    rows = df[df["IDfangraphs"] == fg_id]
                    if not rows.empty:
                        row = rows.iloc[0]
        except Exception:
            pass

        if row is None:
            return None

        fip = _safe_float(row.get("FIP"))
        xfip = _safe_float(row.get("xFIP"))
        era = _safe_float(row.get("ERA"))
        result = {
            "fip":            fip,
            "xfip":           xfip,
            "era":            era,
            "era_minus_xfip": round(era - xfip, 3) if era and xfip else None,
        }
        cp.write_text(json.dumps(result), encoding="utf-8")
        return result
    except Exception as e:
        logger.warning(f"pybaseball xFIP failed for {pitcher_id}/{season}: {e}")
        return None


def get_pitcher_velocity_trend(pitcher_id: int, date_str: str = None) -> dict | None:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    season = int(date_str[:4])
    recent_start = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=21)).strftime("%Y-%m-%d")

    season_params = {
        "player_type": "pitcher",
        "player_id": pitcher_id,
        "season": season,
        "type": "details",
        "pitch_type": "FF,SI",
    }
    recent_params = {
        "player_type": "pitcher",
        "player_id": pitcher_id,
        "game_date_gt": recent_start,
        "game_date_lt": date_str,
        "type": "details",
        "pitch_type": "FF,SI",
    }

    season_df = _get_savant_csv(season_params, date_str)
    recent_df = _get_savant_csv(recent_params, date_str)

    def _avg_velo(df):
        if df is None or df.empty or "release_speed" not in df.columns:
            return None
        v = pd.to_numeric(df["release_speed"], errors="coerce").dropna()
        return round(float(v.mean()), 1) if not v.empty else None

    season_avg = _avg_velo(season_df)
    recent_avg = _avg_velo(recent_df)
    delta = round(recent_avg - season_avg, 2) if season_avg and recent_avg else None

    return {
        "season_avg_velo": season_avg,
        "recent_avg_velo": recent_avg,
        "velo_delta":      delta,
        "fatigue_flag":    delta is not None and delta < -1.5,
    }


def _safe_float(val) -> float | None:
    if val is None:
        return None
    try:
        f = float(val)
        return None if pd.isna(f) else f
    except (ValueError, TypeError):
        return None


if __name__ == "__main__":
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    logger.info(f"Testing savant.py for {today}")

    # Test with Aaron Nola (pitcher ID 605400) from today's schedule
    pitcher_id = 605400
    season = int(today[:4])
    logger.info(f"Fetching arsenal for pitcher {pitcher_id}...")
    arsenal = get_pitcher_arsenal(pitcher_id, season, date_str=today)
    if arsenal is not None:
        logger.info(f"Arsenal for pitcher {pitcher_id}:\n{arsenal.to_string()}")
    else:
        logger.warning("Arsenal returned None")

    logger.info("Fetching velocity trend...")
    vt = get_pitcher_velocity_trend(pitcher_id, date_str=today)
    logger.info(f"Velocity trend: {vt}")
