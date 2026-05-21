"""
model/train.py -- Phase 4: Build training dataset + train XGBoost hit model.

Usage:
  python model/train.py                          # pull data + train
  python model/train.py --skip-build             # train from existing CSVs
  python model/train.py --skip-train             # pull data only
  python model/train.py --start-season 2023      # fewer seasons (faster)
"""
import sys
import time
import pickle
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import FEATURE_WEIGHTS, LEAGUE_AVG, PARK_FACTORS

TRAINING_DIR = Path(__file__).parent.parent / "data" / "training"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"

HIT_EVENTS = frozenset({"single", "double", "triple", "home_run"})
AB_EVENTS = frozenset({
    "single", "double", "triple", "home_run", "strikeout",
    "strikeout_double_play", "field_out", "grounded_into_double_play",
    "force_out", "double_play", "fielders_choice", "fielders_choice_out",
})


# ── Data download ─────────────────────────────────────────────────────────────

def _pull_statcast_season(season: int) -> pd.DataFrame | None:
    """Download full-season Statcast data and cache as parquet."""
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    cache = TRAINING_DIR / f"statcast_raw_{season}.parquet"

    if cache.exists():
        logger.info(f"  Loading cached Statcast {season}")
        return pd.read_parquet(cache)

    try:
        from pybaseball import statcast
    except ImportError:
        logger.error("pybaseball not installed — run: pip install pybaseball")
        return None

    months = [
        (f"{season}-04-01", f"{season}-04-30"),
        (f"{season}-05-01", f"{season}-05-31"),
        (f"{season}-06-01", f"{season}-06-30"),
        (f"{season}-07-01", f"{season}-07-31"),
        (f"{season}-08-01", f"{season}-08-31"),
        (f"{season}-09-01", f"{season}-09-30"),
        (f"{season}-10-01", f"{season}-10-05"),
    ]
    chunks = []
    for start, end in months:
        try:
            logger.info(f"  Pulling {start} to {end}...")
            chunk = statcast(start, end)
            if chunk is not None and not chunk.empty:
                chunks.append(chunk)
                logger.info(f"    {len(chunk):,} rows")
            time.sleep(3.0)
        except Exception as e:
            logger.warning(f"  Pull {start}-{end} failed: {e}")
            time.sleep(5.0)

    if not chunks:
        return None

    df = pd.concat(chunks, ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"])
    try:
        df.to_parquet(cache)
        logger.info(f"  Cached {len(df):,} rows → {cache.name}")
    except Exception as e:
        logger.warning(f"  Parquet cache failed: {e}")
    return df


# ── Pitcher feature precompute ────────────────────────────────────────────────

def _pitcher_snapshot(r_hits, r_outs, r_bip, pitch_stats, game_log) -> dict:
    """Build pitcher feature dict from cumulative running state."""
    total_ip = r_outs / 3.0
    h9 = (r_hits / total_ip * 9.0) if total_ip >= 9.0 else LEAGUE_AVG["h9"]
    babip = r_hits / r_bip if r_bip >= 30 else LEAGUE_AVG["babip"]

    recent = game_log[-3:]
    last3_hits_avg = float(np.mean([g["hits"] for g in recent])) if recent else LEAGUE_AVG["h9"]

    fatigue = 0.0
    if game_log:
        last = game_log[-1]
        if last["pitches"] > 100:
            fatigue += 0.20
        if last["pitches"] > 110:
            fatigue += 0.10
        if len(game_log) >= 3:
            avg_ip_3 = np.mean([g["outs"] / 3.0 for g in game_log[-3:]])
            if avg_ip_3 < 5.0:
                fatigue += 0.15
    fatigue = min(1.0, fatigue)

    # Arsenal-weighted BA and whiff
    total_cnt = sum(v["cnt"] for v in pitch_stats.values())
    arsenal_ba = LEAGUE_AVG["ba"]
    arsenal_whiff = LEAGUE_AVG["whiff_pct"]
    if total_cnt >= 50:
        rows = []
        for s in pitch_stats.values():
            usage = s["cnt"] / total_cnt
            if usage < 0.05:
                continue
            ba = s["hits"] / s["ab"] if s["ab"] >= 10 else LEAGUE_AVG["ba"]
            whiff = s["whiffs"] / s["cnt"] if s["cnt"] >= 10 else LEAGUE_AVG["whiff_pct"]
            rows.append((usage, ba, whiff))
        if rows:
            tot_u = sum(r[0] for r in rows)
            arsenal_ba = sum(r[0] * r[1] for r in rows) / tot_u
            arsenal_whiff = sum(r[0] * r[2] for r in rows) / tot_u

    # Times-through-order weight (position 5 = middle of lineup default)
    starts = [g for g in game_log if g["outs"] >= 9]
    avg_ip = float(np.mean([g["outs"] / 3.0 for g in starts[-5:]])) if starts else 6.0
    tto_raw = min(avg_ip * 3.3 / 5.0, 3.0)
    tto_weight = 1.0 + max(0.0, tto_raw - 1.0) * 0.06

    return {
        "arsenal_weighted_ba":     round(float(arsenal_ba), 4),
        "arsenal_weighted_whiff":  round(float(arsenal_whiff), 4),
        "pitcher_h9_season":       round(float(h9), 3),
        "pitcher_xfip":            4.0,
        "pitcher_babip_against":   round(float(babip), 4),
        "pitcher_last3_hits_avg":  round(float(last3_hits_avg), 3),
        "pitcher_fatigue_score":   round(fatigue, 3),
        "times_through_order":     round(tto_weight, 4),
    }


def _precompute_pitcher_features(df: pd.DataFrame) -> dict:
    """
    Returns {(pitcher_id, game_date): feature_dict} using only data
    from games BEFORE each game_date (no lookahead).
    """
    cache: dict = {}

    df = df.copy()
    df["is_hit"] = df["events"].isin(HIT_EVENTS).fillna(False).astype(int)
    df["is_ab"] = df["events"].isin(AB_EVENTS).fillna(False).astype(int)
    df["is_bip"] = (df["type"] == "X").astype(int)
    df["is_whiff"] = (
        (df["description"] == "swinging_strike").astype(int)
        if "description" in df.columns else 0
    )

    for pitcher_id, pdata in tqdm(df.groupby("pitcher"), desc="  Pitcher precompute", leave=False):
        pdata = pdata.sort_values("game_date")

        r_hits, r_outs, r_bip = 0, 0, 0
        pitch_stats: dict = defaultdict(lambda: {"cnt": 0, "hits": 0, "ab": 0, "whiffs": 0})
        game_log: list = []

        for game_date, gdata in pdata.groupby("game_date"):
            cache[(int(pitcher_id), game_date)] = _pitcher_snapshot(
                r_hits, r_outs, r_bip, pitch_stats, game_log
            )

            gpa = gdata[gdata["events"].notna()]
            g_hits = int(gpa["is_hit"].sum())
            g_outs = int(gpa["is_ab"].sum()) - g_hits
            r_hits += g_hits
            r_outs += max(0, g_outs)
            r_bip += int(gdata["is_bip"].sum())

            for pt, ptdata in gdata.groupby("pitch_type"):
                if not pt or str(pt) in ("nan", ""):
                    continue
                ptpa = ptdata[ptdata["events"].notna()]
                pitch_stats[pt]["cnt"] += len(ptdata)
                pitch_stats[pt]["hits"] += int(ptpa["is_hit"].sum())
                pitch_stats[pt]["ab"] += int(ptpa["is_ab"].sum())
                pitch_stats[pt]["whiffs"] += int(ptdata["is_whiff"].sum())

            game_log.append({
                "date":   game_date,
                "hits":   g_hits,
                "outs":   max(0, g_outs),
                "pitches": len(gdata),
            })

    return cache


# ── Feature extraction ────────────────────────────────────────────────────────

_PA_WEIGHTS = {1: 4.7, 2: 4.5, 3: 4.3, 4: 4.1, 5: 3.9,
               6: 3.7, 7: 3.5, 8: 3.3, 9: 3.1}


def _extract_features(df: pd.DataFrame, season: int) -> list[dict]:
    """One training row per (batter, game), features computed from prior data only."""
    pa_df = df[df["events"].notna()].copy()
    pa_df["is_hit"] = pa_df["events"].isin(HIT_EVENTS).astype(int)
    pa_df["is_ab"] = pa_df["events"].isin(AB_EVENTS).astype(int)
    pa_df["is_bip"] = (pa_df["type"] == "X").astype(int)
    pa_df["ev"] = pd.to_numeric(pa_df["launch_speed"] if "launch_speed" in pa_df.columns else np.nan, errors="coerce")
    pa_df["xba"] = pd.to_numeric(
        pa_df["estimated_ba_using_speedangle"] if "estimated_ba_using_speedangle" in pa_df.columns else np.nan,
        errors="coerce"
    )

    logger.info("  Precomputing pitcher features...")
    pitcher_cache = _precompute_pitcher_features(df)
    logger.info(f"  Pitcher cache: {len(pitcher_cache):,} entries")

    rows: list[dict] = []

    for batter_id, bdata in tqdm(pa_df.groupby("batter"), desc=f"  {season} batters"):
        bdata = bdata.sort_values("game_date").reset_index(drop=True)

        # Primary pitcher per game (most PAs)
        opp_pitcher = (
            bdata.groupby("game_pk")["pitcher"]
            .agg(lambda x: x.value_counts().index[0])
            .rename("opp_pitcher")
        )

        # Game-level aggregates
        game_agg = bdata.groupby(["game_pk", "game_date"]).agg(
            hits=("is_hit", "sum"),
            ab=("is_ab", "sum"),
            bip=("is_bip", "sum"),
            ev_avg=("ev", "mean"),
            xba_avg=("xba", "mean"),
        ).reset_index().join(opp_pitcher, on="game_pk")

        meta = (
            bdata.groupby("game_pk")[["home_team", "away_team", "inning_topbot", "stand"]]
            .first()
        )
        game_agg = game_agg.join(meta, on="game_pk")
        # pitcher throws: look up from bdata per game
        p_throws_map = bdata.groupby("game_pk")["p_throws"].first() if "p_throws" in bdata.columns else {}
        game_agg["p_throws"] = game_agg["game_pk"].map(p_throws_map).fillna("R")

        game_agg = game_agg.sort_values("game_date").reset_index(drop=True)
        game_agg["did_get_hit"] = (game_agg["hits"] > 0).astype(int)

        if len(game_agg) < 5:
            continue

        # Running season stats (updated AFTER each row is appended)
        s_hits = s_ab = s_bip = s_bip_hits = s_xba_n = 0
        s_xba_sum = 0.0
        game_hist: list = []  # dicts

        for _, game in game_agg.iterrows():
            gdate = pd.Timestamp(game["game_date"])

            # Season stats before this game
            s_h_pct  = s_hits / s_ab if s_ab >= 10 else LEAGUE_AVG["ba"]
            s_xba    = s_xba_sum / s_xba_n if s_xba_n >= 10 else LEAGUE_AVG["xba"]
            s_babip  = s_bip_hits / s_bip if s_bip >= 20 else LEAGUE_AVG["babip"]

            # Rolling windows
            cut14 = gdate - timedelta(days=14)
            cut7  = gdate - timedelta(days=7)
            h14 = [g for g in game_hist if g["date"] >= cut14]
            h7  = [g for g in game_hist if g["date"] >= cut7]

            h14_h, h14_ab = sum(g["hits"] for g in h14), sum(g["ab"] for g in h14)
            h7_h,  h7_ab  = sum(g["hits"] for g in h7),  sum(g["ab"] for g in h7)
            h_pct_14 = h14_h / h14_ab if h14_ab >= 5 else s_h_pct
            h_pct_7  = h7_h  / h7_ab  if h7_ab  >= 3 else h_pct_14

            ev_vals = [g["ev"] for g in h14 if g["ev"] is not None and not np.isnan(g["ev"])]
            ev_14 = float(np.mean(ev_vals)) if ev_vals else 88.5

            bip14 = sum(g["bip"] for g in h14)
            bip14_h = sum(g["bip_hit"] for g in h14)
            babip14 = bip14_h / bip14 if bip14 >= 5 else s_babip
            babip_delta = float(np.clip(babip14 - s_xba, -0.08, 0.08))

            # Pitcher features
            opp_pid = game.get("opp_pitcher")
            try:
                p_key = (int(opp_pid), gdate)
            except (TypeError, ValueError):
                p_key = None
            pf = pitcher_cache.get(p_key, {}) if p_key else {}

            # Career H/AB vs pitcher (same-season prior data)
            try:
                pid_int = int(opp_pid)
                vs = bdata[(bdata["pitcher"] == pid_int) & (bdata["game_date"] < gdate)]
                career_avg = vs["is_hit"].sum() / vs["is_ab"].sum() if vs["is_ab"].sum() >= 15 else LEAGUE_AVG["ba"]
            except (TypeError, ValueError):
                career_avg = LEAGUE_AVG["ba"]

            # Handedness
            stand    = str(game.get("stand", "R"))
            p_throws = str(game.get("p_throws", "R"))
            if stand == "S":
                handedness_split = LEAGUE_AVG["ba"] * (1.05 if p_throws == "R" else 1.03)
            elif stand != p_throws:
                handedness_split = LEAGUE_AVG["ba"] * 1.05
            else:
                handedness_split = LEAGUE_AVG["ba"] * 0.95

            # Park factor
            park_factor = PARK_FACTORS.get(str(game.get("home_team", "")), 1.0)

            # Lineup position weight from at_bat_number
            gm_pas = bdata[bdata["game_pk"] == game["game_pk"]]
            if "at_bat_number" in gm_pas.columns and not gm_pas.empty:
                abn = gm_pas["at_bat_number"].median()
                lpos = max(1, min(9, int(abn % 9) or 9))
            else:
                lpos = 5
            lineup_weight = _PA_WEIGHTS.get(lpos, 3.9) / 4.7

            rows.append({
                "did_get_hit":               int(game["did_get_hit"]),
                "batter_id":                 int(batter_id),
                "game_pk":                   int(game["game_pk"]),
                "game_date":                 str(gdate.date()),
                "season":                    season,
                "batter_h_pct_14day":        round(h_pct_14, 4),
                "batter_h_pct_7day":         round(h_pct_7, 4),
                "batter_h_pct_season":       round(s_h_pct, 4),
                "batter_xba_season":         round(s_xba, 4),
                "babip_regression_delta":    round(babip_delta, 4),
                "career_h_ab_vs_pitcher":    round(float(career_avg), 4),
                "arsenal_weighted_ba":       pf.get("arsenal_weighted_ba",    LEAGUE_AVG["ba"]),
                "arsenal_weighted_whiff":    pf.get("arsenal_weighted_whiff", LEAGUE_AVG["whiff_pct"]),
                "pitcher_h9_season":         pf.get("pitcher_h9_season",      LEAGUE_AVG["h9"]),
                "pitcher_xfip":              pf.get("pitcher_xfip",           4.0),
                "pitcher_babip_against":     pf.get("pitcher_babip_against",  LEAGUE_AVG["babip"]),
                "pitcher_last3_hits_avg":    pf.get("pitcher_last3_hits_avg", LEAGUE_AVG["h9"]),
                "pitcher_fatigue_score":     pf.get("pitcher_fatigue_score",  0.0),
                "times_through_order":       pf.get("times_through_order",    1.05),
                "bullpen_h9":                LEAGUE_AVG["h9"],
                "park_factor":               park_factor,
                "game_total":                8.5,
                "weather_score":             0.0,
                "handedness_split":          round(handedness_split, 4),
                "lineup_position_pa_weight": round(lineup_weight, 4),
                "umpire_zone_score":         0.0,
                "exit_velocity_14day":       round(ev_14, 2),
            })

            # Update history AFTER recording features
            xba_val = float(game["xba_avg"]) if pd.notna(game.get("xba_avg")) else None
            ev_val  = float(game["ev_avg"])  if pd.notna(game.get("ev_avg"))  else None
            g_bip = int(game["bip"])
            g_bip_hit = 1 if int(game["hits"]) > 0 and g_bip > 0 else 0

            game_hist.append({
                "date": gdate, "hits": int(game["hits"]), "ab": int(game["ab"]),
                "ev": ev_val, "bip": g_bip, "bip_hit": g_bip_hit,
            })
            s_hits += int(game["hits"])
            s_ab   += int(game["ab"])
            s_bip  += g_bip
            s_bip_hits += g_bip_hit
            if xba_val is not None:
                s_xba_sum += xba_val
                s_xba_n += 1

    return rows


# ── Public API ────────────────────────────────────────────────────────────────

def build_training_dataset(start_season: int = 2022, end_season: int = 2024) -> pd.DataFrame | None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    all_dfs: list[pd.DataFrame] = []

    for season in range(start_season, end_season + 1):
        csv_path = TRAINING_DIR / f"training_data_{season}.csv"
        if csv_path.exists():
            logger.info(f"Season {season} cached at {csv_path.name}")
            all_dfs.append(pd.read_csv(csv_path))
            continue

        raw = _pull_statcast_season(season)
        if raw is None or raw.empty:
            logger.warning(f"No data for {season}, skipping")
            continue

        logger.info(f"Extracting features for {season} ({len(raw):,} pitch rows)...")
        rows = _extract_features(raw, season)
        if not rows:
            logger.warning(f"No rows extracted for {season}")
            continue

        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved {len(df):,} rows → {csv_path.name}")
        all_dfs.append(df)

    if not all_dfs:
        logger.error("No training data available.")
        return None

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined):,} rows, hit rate: {combined['did_get_hit'].mean():.3f}")
    return combined


def train_model(training_data_path: str | None = None) -> None:
    from sklearn.preprocessing import StandardScaler
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
    from xgboost import XGBClassifier

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    if training_data_path:
        df = pd.read_csv(training_data_path)
    else:
        csvs = sorted(TRAINING_DIR.glob("training_data_*.csv"))
        if not csvs:
            logger.error("No training CSVs found — run build first.")
            return
        df = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    feat_cols = list(FEATURE_WEIGHTS.keys())
    for col in feat_cols:
        if col not in df.columns:
            logger.warning(f"Missing feature column: {col} — filling with 0")
            df[col] = 0.0

    X = df[feat_cols].copy()
    y = df["did_get_hit"]

    for col in feat_cols:
        X[col] = X[col].fillna(X[col].median())

    # Temporal split: train on earlier seasons, test on 2024
    if "season" in df.columns and df["season"].max() >= 2024:
        train_mask = df["season"] < 2024
    else:
        n = len(df)
        split = int(n * 0.8)
        train_mask = pd.Series([True] * split + [False] * (n - split), index=df.index)

    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]
    logger.info(f"Train: {len(X_train):,} | Test: {len(X_test):,} | Hit rate: {y_train.mean():.3f}")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    logger.info("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )
    xgb.fit(X_tr, y_train)

    importances = sorted(zip(feat_cols, xgb.feature_importances_), key=lambda x: x[1], reverse=True)
    logger.info("Feature importances (top 8):")
    for feat, imp in importances[:8]:
        logger.info(f"  {feat:<35} {imp:.4f}")

    logger.info("Calibrating (isotonic, cv=5)...")
    model = CalibratedClassifierCV(xgb, cv=5, method="isotonic")
    model.fit(X_tr, y_train)

    y_pred = model.predict_proba(X_te)[:, 1]
    logger.info(f"AUC-ROC:        {roc_auc_score(y_test, y_pred):.4f}")
    logger.info(f"Brier Score:    {brier_score_loss(y_test, y_pred):.4f}")
    logger.info(f"Baseline Brier: {brier_score_loss(y_test, [y_train.mean()] * len(y_test)):.4f}")
    logger.info("\n" + classification_report(y_test, (y_pred >= 0.5).astype(int)))

    model_path  = ARTIFACTS_DIR / "hit_model.pkl"
    scaler_path = ARTIFACTS_DIR / "scaler.pkl"
    with open(model_path,  "wb") as f: pickle.dump(model,  f)
    with open(scaler_path, "wb") as f: pickle.dump(scaler, f)
    logger.info(f"Saved: {model_path}")
    logger.info(f"Saved: {scaler_path}")


if __name__ == "__main__":
    import argparse
    from pathlib import Path as _P
    _P("outputs/logs").mkdir(parents=True, exist_ok=True)
    logger.add("outputs/logs/train_{time}.log", level="INFO")

    ap = argparse.ArgumentParser(description="MLB hit model training pipeline")
    ap.add_argument("--start-season", type=int, default=2022)
    ap.add_argument("--end-season",   type=int, default=2024)
    ap.add_argument("--skip-build",  action="store_true", help="Skip data pull, use existing CSVs")
    ap.add_argument("--skip-train",  action="store_true", help="Pull data only, skip model training")
    args = ap.parse_args()

    if not args.skip_build:
        build_training_dataset(args.start_season, args.end_season)

    if not args.skip_train:
        train_model()
