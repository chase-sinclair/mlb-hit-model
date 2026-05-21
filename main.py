import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv(Path(__file__).parent / ".env")

# Configure loguru
_today = datetime.now().strftime("%Y-%m-%d")
_log_dir = Path(__file__).parent / "outputs" / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}")
logger.add(_log_dir / f"{_today}.log", level="DEBUG", rotation="1 day")

from config import PARK_FACTORS, STADIUM_COORDS, TEAM_ID_MAP, LEAGUE_AVG
from data.fetchers.mlb_api import (
    get_today_schedule, get_lineup, get_batter_season_stats,
    get_batter_last_n_games, get_batter_vs_pitcher, get_batter_splits,
    get_pitcher_season_stats, get_pitcher_game_log, get_player_info,
    get_bullpen_stats, get_umpire_for_game,
)
from data.fetchers.savant import (
    get_batter_statcast_season, get_batter_statcast_rolling,
    get_pitcher_arsenal, get_batter_vs_pitch_types, get_pitcher_xfip,
    get_pitcher_velocity_trend,
)
from data.fetchers.odds_api import get_game_totals, get_hit_prop_lines
from data.fetchers.weather_api import get_game_weather
from data.fetchers.umpire import get_umpire_tendencies
from data.processors.arsenal import compute_arsenal_score
from data.processors.pitcher import (
    compute_pitcher_fatigue_score, compute_times_through_order_weight,
    compute_pitcher_luck_adjustment,
)
from data.processors.batter import (
    compute_babip_regression_delta, compute_handedness_split_score,
    compute_lineup_pa_weight,
)
from data.processors.game_context import compute_game_context_features
from data.processors.bullpen import compute_bullpen_features
from model.predict import compute_hit_probability, rank_predictions, _top_factors


def _get_pitcher_data(pitcher_info: dict | None, season: int, date_str: str) -> dict:
    if not pitcher_info or not pitcher_info.get("id"):
        return {}
    pid = pitcher_info["id"]
    season_stats = get_pitcher_season_stats(pid, season, date_str=date_str) or {}
    game_log = get_pitcher_game_log(pid, season, last_n=5, date_str=date_str) or []
    xfip_data = get_pitcher_xfip(pid, season, date_str=date_str) or {}
    velo_trend = get_pitcher_velocity_trend(pid, date_str=date_str) or {}
    arsenal = get_pitcher_arsenal(pid, season, date_str=date_str)
    player = get_player_info(pid, date_str=date_str) or {}

    last3_hits = None
    if game_log:
        hits = [g.get("hits_allowed", 0) for g in game_log[:3] if g.get("hits_allowed") is not None]
        last3_hits = round(sum(hits) / len(hits), 2) if hits else None

    return {
        "id":           pid,
        "name":         pitcher_info.get("name") or player.get("name"),
        "throws":       pitcher_info.get("throws") or player.get("throws"),
        "season_stats": season_stats,
        "game_log":     game_log,
        "xfip_data":    xfip_data,
        "velo_trend":   velo_trend,
        "arsenal":      arsenal,
        "last3_hits":   last3_hits,
        "avg_ip":       season_stats.get("avg_ip_per_start"),
    }


def _get_batter_data(player_id: int, season: int, date_str: str) -> dict:
    season_stats = get_batter_season_stats(player_id, season, date_str=date_str) or {}
    rolling = get_batter_last_n_games(player_id, n=14, season=season, date_str=date_str) or {}
    statcast = get_batter_statcast_season(player_id, season, date_str=date_str) or {}
    statcast_rolling = get_batter_statcast_rolling(player_id, days=14, date_str=date_str) or {}
    splits = get_batter_splits(player_id, season, date_str=date_str) or {}
    pitch_splits = get_batter_vs_pitch_types(player_id, season, date_str=date_str) or {}
    player_info = get_player_info(player_id, date_str=date_str) or {}

    return {
        "season_stats":     season_stats,
        "rolling":          rolling,
        "statcast":         statcast,
        "statcast_rolling": statcast_rolling,
        "splits":           splits,
        "pitch_splits":     pitch_splits,
        "player_info":      player_info,
    }


def _build_features(
    batter_data: dict,
    pitcher_data: dict,
    vs_pitcher: dict,
    game_context: dict,
    bullpen_feats: dict,
    lineup_pos: int,
) -> dict:
    rolling = batter_data.get("rolling", {})
    season = batter_data.get("season_stats", {})
    statcast = batter_data.get("statcast", {})
    sc_rolling = batter_data.get("statcast_rolling", {})
    splits = batter_data.get("splits", {})
    pitch_splits = batter_data.get("pitch_splits", {})
    pitcher_season = pitcher_data.get("season_stats", {})
    xfip_d = pitcher_data.get("xfip_data", {})
    velo_d = pitcher_data.get("velo_trend", {})
    pi = batter_data.get("player_info", {})

    arsenal_result = compute_arsenal_score(pitcher_data.get("arsenal"), pitch_splits)
    fatigue = compute_pitcher_fatigue_score(
        pitcher_data.get("game_log", []),
        velo_delta=velo_d.get("velo_delta")
    )
    tto_weight = compute_times_through_order_weight(lineup_pos, pitcher_data.get("avg_ip"))
    luck_adj = compute_pitcher_luck_adjustment(
        xfip_d.get("xfip"), pitcher_season.get("era") or xfip_d.get("era")
    )
    babip_delta = compute_babip_regression_delta(
        season.get("babip"), statcast.get("xbabip")
    )
    handedness_ba = compute_handedness_split_score(
        pi.get("bats"), pitcher_data.get("throws"), splits
    )
    pa_weight = compute_lineup_pa_weight(lineup_pos)

    h_pct_season = None
    if season.get("at_bats") and season.get("at_bats", 0) > 0:
        h_pct_season = round(season.get("hits", 0) / season["at_bats"], 3)

    career_ba_vs = vs_pitcher.get("avg") if vs_pitcher else None

    return {
        "batter_h_pct_14day":        rolling.get("h_pct_14day"),
        "batter_h_pct_7day":         rolling.get("h_pct_7day"),
        "batter_h_pct_season":       h_pct_season,
        "batter_xba_season":         statcast.get("xba"),
        "babip_regression_delta":    babip_delta,
        "career_h_ab_vs_pitcher":    career_ba_vs,
        "arsenal_weighted_ba":       arsenal_result.get("arsenal_weighted_ba"),
        "arsenal_weighted_whiff":    arsenal_result.get("arsenal_weighted_whiff"),
        "pitcher_h9_season":         pitcher_season.get("h9"),
        "pitcher_xfip":              xfip_d.get("xfip"),
        "pitcher_babip_against":     pitcher_season.get("babip"),
        "pitcher_last3_hits_avg":    pitcher_data.get("last3_hits"),
        "pitcher_fatigue_score":     fatigue,
        "times_through_order":       tto_weight,
        "bullpen_h9":                bullpen_feats.get("bullpen_h9"),
        "park_factor":               game_context.get("park_factor"),
        "game_total":                game_context.get("game_total"),
        "weather_score":             game_context.get("weather_score_raw"),
        "handedness_split":          handedness_ba,
        "lineup_position_pa_weight": pa_weight,
        "umpire_zone_score":         game_context.get("umpire_zone_score"),
        "exit_velocity_14day":       sc_rolling.get("exit_velo_14day"),
        "_arsenal_breakdown":        arsenal_result.get("breakdown", []),
        "_luck_adj":                 luck_adj,
    }


def _process_game(game: dict, season: int, date_str: str, odds_totals: dict, hit_props: dict) -> list[dict]:
    game_id = game["game_id"]
    home_team = game["home_team"]
    away_team = game["away_team"]

    logger.info(f"Processing {away_team} @ {home_team} (game {game_id})")

    # Get lineup
    lineup = get_lineup(game_id, date_str=date_str)
    if not lineup:
        logger.warning(f"  No lineup for game {game_id} — skipping")
        return []

    # Game-level context
    park_factor = PARK_FACTORS.get(home_team, 1.0)
    coords = STADIUM_COORDS.get(home_team)
    weather = {}
    if coords:
        weather = get_game_weather(coords[0], coords[1], game.get("game_time_utc", ""), date_str=date_str)

    umpire_name = get_umpire_for_game(game_id, date_str=date_str)
    umpire = get_umpire_tendencies(umpire_name, date_str=date_str) if umpire_name else {}

    total_key = (home_team, away_team)
    total_info = odds_totals.get(total_key) or odds_totals.get(str(total_key)) or {}
    game_total = total_info.get("total")

    is_day_game = False
    try:
        hour = int(game.get("game_time_utc", "T19")[11:13])
        is_day_game = hour < 18  # before 6 PM UTC (2 PM ET)
    except Exception:
        pass

    predictions = []

    for side in ("home", "away"):
        side_lineup = lineup.get(side, [])
        pitcher_info = game.get(f"{side}_probable_pitcher") if side == "home" else game.get(f"{'away' if side == 'home' else 'home'}_probable_pitcher")

        # Opponent pitcher is who the batters face
        opp_side = "away" if side == "home" else "home"
        opp_pitcher_info = game.get(f"{opp_side}_probable_pitcher")
        team = home_team if side == "home" else away_team
        opp_team = away_team if side == "home" else home_team
        is_home = side == "home"

        if not opp_pitcher_info:
            opp_pitcher_info = {"id": None, "name": "TBD", "throws": "R"}

        pitcher_data = _get_pitcher_data(opp_pitcher_info, season, date_str)
        team_id_map_rev = {v: k for k, v in TEAM_ID_MAP.items()}
        opp_team_id = team_id_map_rev.get(opp_team)
        bullpen_raw = get_bullpen_stats(opp_team_id, season, date_str=date_str) if opp_team_id else None
        bullpen_feats = compute_bullpen_features(bullpen_raw)

        context_features = {
            "park_factor": park_factor,
            "game_total":  game_total,
            "weather_score_raw": weather.get("weather_score", 0.0),
            "umpire_zone_score": umpire.get("zone_score", 0.0) if umpire else 0.0,
        }
        game_ctx = compute_game_context_features(
            game_total, park_factor, weather, umpire, is_home, is_day_game
        )

        for batter_entry in side_lineup:
            player_id = batter_entry.get("player_id")
            if not player_id:
                continue
            lineup_pos = batter_entry.get("batting_order", 5)
            player_name = batter_entry.get("name", f"Player {player_id}")

            batter_data = _get_batter_data(player_id, season, date_str)
            pi = batter_data.get("player_info", {})

            # Check minimum PA (50 PA this season)
            season_ab = batter_data.get("season_stats", {}).get("at_bats", 0) or 0
            if season_ab < 50:
                logger.debug(f"  {player_name}: insufficient sample ({season_ab} AB) — flagging")

            vs_pitcher = get_batter_vs_pitcher(
                player_id, pitcher_data.get("id") or 0, date_str=date_str
            ) if pitcher_data.get("id") else {}

            features = _build_features(
                batter_data, pitcher_data, vs_pitcher,
                context_features, bullpen_feats, lineup_pos
            )

            prediction = compute_hit_probability(features)

            # Book implied probability
            book_entry = hit_props.get(player_name) or hit_props.get(pi.get("name", ""))
            implied_prob = None
            book_odds = None
            edge = None
            if book_entry:
                implied_prob = book_entry.get("implied_prob")
                book_odds = book_entry.get("yes_odds")
                if implied_prob:
                    edge = round(prediction["hit_probability"] - implied_prob, 4)

            top_3 = _top_factors(features)

            predictions.append({
                "date":            date_str,
                "player_name":     player_name,
                "player_id":       player_id,
                "team":            team,
                "opponent":        opp_team,
                "game_time":       game.get("game_time", ""),
                "is_home":         is_home,
                "pitcher_name":    pitcher_data.get("name", "TBD"),
                "pitcher_throws":  pitcher_data.get("throws", "R"),
                "bats":            pi.get("bats"),
                "lineup_position": lineup_pos,

                # Core prediction
                "hit_probability":    prediction["hit_probability"],
                "implied_book_prob":  implied_prob,
                "edge":               edge,
                "book_odds":          book_odds,
                "rating":             None,
                "data_confidence":    prediction["confidence"],
                "model":              prediction["model"],

                # Batter form
                "h_pct_7day":    features.get("batter_h_pct_7day"),
                "h_pct_14day":   features.get("batter_h_pct_14day"),
                "h_pct_season":  features.get("batter_h_pct_season"),
                "xba_season":    features.get("batter_xba_season"),
                "babip_actual":  batter_data.get("season_stats", {}).get("babip"),
                "xbabip":        batter_data.get("statcast", {}).get("xbabip"),
                "babip_delta":   features.get("babip_regression_delta"),
                "exit_velo_14day": features.get("exit_velocity_14day"),
                "hard_hit_14day": batter_data.get("statcast_rolling", {}).get("hard_hit_14day"),

                # Matchup
                "career_ab_vs_pitcher":  vs_pitcher.get("ab"),
                "career_h_vs_pitcher":   vs_pitcher.get("hits"),
                "career_avg_vs_pitcher": vs_pitcher.get("avg"),
                "arsenal_weighted_ba":   features.get("arsenal_weighted_ba"),
                "arsenal_weighted_whiff": features.get("arsenal_weighted_whiff"),
                "handedness_split_ba":   features.get("handedness_split"),
                "times_through_order_est": features.get("times_through_order"),

                # Pitcher
                "pitcher_h9":           features.get("pitcher_h9_season"),
                "pitcher_whip":         pitcher_data.get("season_stats", {}).get("whip"),
                "pitcher_xfip":         features.get("pitcher_xfip"),
                "pitcher_era":          pitcher_data.get("season_stats", {}).get("era"),
                "pitcher_days_rest":    pitcher_data.get("game_log", [{}])[0].get("days_rest") if pitcher_data.get("game_log") else None,
                "pitcher_last_pitch_count": pitcher_data.get("game_log", [{}])[0].get("pitches_thrown") if pitcher_data.get("game_log") else None,
                "pitcher_avg_ip":       pitcher_data.get("avg_ip"),
                "pitcher_fatigue_score": features.get("pitcher_fatigue_score"),
                "pitcher_velo_delta":   pitcher_data.get("velo_trend", {}).get("velo_delta"),

                # Context
                "park_factor":       park_factor,
                "game_total":        game_total,
                "weather_score":     weather.get("weather_score"),
                "wind_speed":        weather.get("wind_speed_mph"),
                "wind_direction":    weather.get("wind_direction"),
                "temp_f":            weather.get("temp_f"),
                "umpire_name":       umpire_name,
                "umpire_zone_score": umpire.get("zone_score", 0.0) if umpire else 0.0,

                # Bullpen
                "bullpen_h9":             bullpen_feats.get("bullpen_h9"),
                "bullpen_workload_3day":   bullpen_feats.get("bullpen_workload_3day"),

                # Top factors
                "top_factor_1": top_3[0] if len(top_3) > 0 else None,
                "top_factor_2": top_3[1] if len(top_3) > 1 else None,
                "top_factor_3": top_3[2] if len(top_3) > 2 else None,
            })

    return predictions


def _print_terminal_summary(predictions: list[dict], today: str, run_ts: str):
    width = 69
    print("\n" + "=" * width)
    print(f"MLB HIT PROPS -- {today}    Model run: {run_ts}")
    print("=" * width)

    strong = [p for p in predictions if p.get("rating") == "STRONG BET"]
    value = [p for p in predictions if p.get("rating") == "VALUE BET"]

    def _fmt_row(p):
        name   = (p["player_name"] or "")[:18].ljust(18)
        teams  = f"{p['team'][:3]} vs {p['opponent'][:3]}".ljust(10)
        pitcher = f"vs {(p['pitcher_name'] or 'TBD')[:12]}".ljust(16)
        prob   = f"{p['hit_probability']*100:.1f}%".rjust(6)
        odds   = str(p["book_odds"] or "N/A").rjust(5)
        edge   = f"+{p['edge']*100:.1f}%" if p.get("edge") and p["edge"] > 0 else (f"{p['edge']*100:.1f}%" if p.get("edge") else "  N/A")
        star   = " ***" if p.get("rating") == "STRONG BET" else ""
        return f"{name} {teams} {pitcher} {prob} {odds} {edge.rjust(7)}{star}"

    if strong:
        print(f"\nSTRONG BETS (edge > 8%)")
        print("-" * width)
        for p in strong:
            print(_fmt_row(p))

    if value:
        print(f"\nVALUE BETS (edge 5-8%)")
        print("-" * width)
        for p in value:
            print(_fmt_row(p))

    has_lines = any(p.get("edge") is not None for p in predictions)
    sort_label = "edge" if has_lines else "hit probability"
    print(f"\nALL PLAYS (top 15 by {sort_label})")
    print("-" * width)
    header = f"{'Player':<18} {'Teams':<10} {'Pitcher':<16} {'Prob':>6} {'Odds':>5} {'Edge':>7}"
    print(header)
    print("-" * width)
    if has_lines:
        top15 = [p for p in predictions if p.get("edge") is not None][:15]
    else:
        top15 = sorted(predictions, key=lambda x: x.get("hit_probability") or 0, reverse=True)[:15]
    for p in top15:
        print(_fmt_row(p))

    no_line = [p for p in predictions if p.get("edge") is None]
    if no_line:
        print(f"\n  + {len(no_line)} players with no book line available")
    print("=" * width + "\n")


def run_daily_pipeline(date_str: str = None, max_lineup_polls: int = 18):
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    season = int(date_str[:4])
    run_ts = datetime.now().strftime("%H:%M:%S")

    logger.info(f"=== MLB Hit Model — {date_str} ===")

    # Schedule
    games = get_today_schedule(date_str)
    logger.info(f"Games today: {len(games)}")
    if not games:
        logger.info("No games today — exiting")
        return

    # Lineup poll (up to 90 min, 5 min intervals)
    games_with_lineups = []
    games_no_lineup = []
    for g in games:
        if g.get("home_lineup") or g.get("away_lineup"):
            games_with_lineups.append(g)
        else:
            games_no_lineup.append(g)

    if games_no_lineup:
        logger.info(f"{len(games_no_lineup)} games without lineups — polling...")
        max_polls = max_lineup_polls  # default 18 × 5 min = 90 min
        for poll in range(max_polls):
            still_missing = []
            for g in games_no_lineup:
                lineup = get_lineup(g["game_id"], date_str=date_str)
                if lineup:
                    g["home_lineup"] = [b for b in lineup.get("home", [])]
                    g["away_lineup"] = [b for b in lineup.get("away", [])]
                    games_with_lineups.append(g)
                    logger.info(f"  Got lineup for {g['away_team']} @ {g['home_team']}")
                else:
                    still_missing.append(g)
            games_no_lineup = still_missing
            if not games_no_lineup:
                break
            if poll < max_polls - 1:
                logger.info(f"  {len(games_no_lineup)} still missing — waiting 5 min...")
                time.sleep(300)

    if games_no_lineup:
        logger.warning(f"{len(games_no_lineup)} games still lack lineups after timeout — skipping")

    if not games_with_lineups:
        logger.warning("No lineups available — exiting")
        return

    # Bulk data pulls
    api_key = os.getenv("ODDS_API_KEY", "")
    logger.info("Fetching game totals and hit prop lines...")
    odds_totals = get_game_totals(api_key, date_str=date_str)
    hit_props = get_hit_prop_lines(api_key, date_str=date_str)
    logger.info(f"  Totals: {len(odds_totals)}, Props: {len(hit_props)}")

    # Process each game
    all_predictions = []
    for game in games_with_lineups:
        try:
            preds = _process_game(game, season, date_str, odds_totals, hit_props)
            all_predictions.extend(preds)
        except Exception as e:
            logger.error(f"Game {game.get('game_id')} failed: {e}")

    logger.info(f"Generated {len(all_predictions)} batter predictions")

    if not all_predictions:
        logger.warning("No predictions generated")
        return

    # Rank
    all_predictions = rank_predictions(all_predictions)

    # Save CSV
    import pandas as pd
    out_dir = Path(__file__).parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"predictions_{date_str}.csv"
    df = pd.DataFrame(all_predictions)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved predictions to {out_path}")

    # Terminal summary
    _print_terminal_summary(all_predictions, date_str, run_ts)
    logger.info(f"Pipeline complete — {len(all_predictions)} predictions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-polls", type=int, default=18)
    parser.add_argument("--date", type=str, default=None)
    args = parser.parse_args()
    run_daily_pipeline(date_str=args.date, max_lineup_polls=args.max_polls)
