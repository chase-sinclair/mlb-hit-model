# MLB Hit Probability Model

## Project Goal

Automated MLB player prop model predicting probability of a batter getting 1+ hit in a game.
Runs daily at 11:30 AM ET, pulls free data sources, outputs ranked +EV bets to CSV and terminal.

---

## Project Structure

```
mlb_hit_model/
├── main.py                     # daily pipeline orchestrator
├── scheduler.py                # runs main.py at 11:30 AM ET (not yet built)
├── config.py                   # park factors, stadium coords, feature weights, constants
├── data/
│   ├── fetchers/
│   │   ├── mlb_api.py          # MLB Stats API — schedule, lineups, player/pitcher stats
│   │   ├── savant.py           # Baseball Savant Statcast — xBA, EV, arsenal, velocity trend
│   │   ├── odds_api.py         # The Odds API — game totals, hit prop lines
│   │   ├── weather_api.py      # Open-Meteo — hourly forecast per stadium
│   │   └── umpire.py           # UmpScorecards scraper (currently returns neutral/zeros — site 404s)
│   ├── processors/
│   │   ├── arsenal.py          # weighted BA/whiff across pitcher's pitch mix
│   │   ├── pitcher.py          # fatigue score, TTO multiplier, luck adjustment
│   │   ├── batter.py           # BABIP regression delta, handedness split, PA weight
│   │   ├── game_context.py     # park, weather, total, umpire → feature dict
│   │   └── bullpen.py          # bullpen H/9 and workload features
│   └── cache/                  # daily JSON/CSV cache keyed by date + URL hash
├── model/
│   ├── predict.py              # sigmoid scorer (V1) + XGBoost loader (after training)
│   ├── train.py                # historical data build + XGBoost training ✅
│   ├── evaluate.py             # backtest against actual outcomes (not yet built)
│   └── artifacts/              # hit_model.pkl + scaler.pkl (populated after training)
├── outputs/
│   ├── predictions_YYYY-MM-DD.csv
│   └── logs/
└── requirements.txt
```

---

## Tech Stack

Python 3.13 (originally spec'd 3.11 — works fine on 3.13 with relaxed version pins)

```
requests==2.31.0       pandas>=2.2.0          numpy>=2.0.0
pybaseball==2.2.5      scikit-learn>=1.4.0    xgboost>=2.0.3
schedule==1.2.1        tqdm==4.66.0            loguru==0.7.2
python-dotenv==1.0.0   beautifulsoup4==4.12.3  lxml>=5.3.0
```

---

## Environment Variables

`.env` in project root (never commit):
```
ODDS_API_KEY=your_free_key_from_the-odds-api.com
```

Sign up at the-odds-api.com — free tier, 500 requests/month, no credit card.
All other sources (MLB API, Savant, Open-Meteo) require no key.

---

## Running the Pipeline

```bash
# Activate venv (Windows — required before any python/pip command)
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& c:\Users\chase\Documents\mlb_hit_model\.venv\Scripts\Activate.ps1

# Daily run (waits up to 90 min for lineups)
python main.py

# Dry run / fast test (2 lineup polls then proceeds)
python main.py --max-polls 2

# Specific date
python main.py --date 2026-05-20
```

---

## Key Constraints

- **Zero paid APIs.** MLB Stats, Savant, Open-Meteo, Odds API free tier only.
- **No lookahead bias in training.** Features for game on date D must only use data before D.
- **Lineup polling.** Polls boxscore endpoint every 5 min. `get_lineup` always skips cache (lineups change throughout the morning — caching stale empty responses was a bug, fixed).
- **Odds API quota.** Max 2 calls/day (totals + hit props). Cache immediately after fetch.
- **Minimum sample.** Players with <50 PA flagged; >40% fallback features → `data_confidence=LOW`.
- **Probability clamp.** Output range [0.40, 0.92] — no MLB batter should be outside this.

---

## Error Handling

Every fetcher catches all exceptions → logs warning → returns None. Pipeline never crashes on a single bad data point. Cache lives at `data/cache/{YYYY-MM-DD}/{md5(url)}.json`.

Rate limits: 0.5s between MLB API calls, 1.0s between Savant calls.

---

## Build Log

### Phase 1 — Data Fetchers ✅ COMPLETE

**config.py**
- All constants implemented: PARK_FACTORS, STADIUM_COORDS, TEAM_ID_MAP, PITCH_TYPES, FEATURE_WEIGHTS, MIN_SAMPLES, LEAGUE_AVG
- Added STADIUM_OUT_DIRECTION (per-park outfield bearing for wind scoring) — not in original spec
- Added TEAM_ABBR_TO_ID reverse map
- **Adaptation:** FEATURE_WEIGHTS in spec summed to 1.10, not 1.00. Normalized at load time by dividing each weight by raw sum. Relative importance preserved.

**data/fetchers/mlb_api.py**
- All 10 functions implemented + `get_umpire_for_game()` helper
- File cache with md5-keyed JSON files, 0.5s delay, graceful 404 handling
- `get_lineup()` uses `skip_cache=True` — boxscore must be re-fetched on every poll or stale empty lineups poison the retry loop
- Game time formatted manually (Windows doesn't support `%-I` strftime format; uses `%#I` equivalent via manual f-string)
- Pitcher `throws` not included in schedule hydration — fetched separately via `get_player_info()`
- **Live test:** 15 games found on 2026-05-20, schedule and lineup polling both working

**data/fetchers/savant.py**
- `get_pitcher_arsenal()`: spec called `player-services/statcast-pitching` endpoint — returns 404 for current season. **Adapted:** build arsenal from main Statcast CSV endpoint, group by pitch_type and aggregate usage/BA/whiff/velocity/spin.
- `get_pitcher_xfip()`: uses pybaseball `pitching_stats()` + `playerid_reverse_lookup()` to map MLB ID → Fangraphs ID
- `get_pitcher_velocity_trend()`: working — e.g., Aaron Nola 86.1 mph season avg, 85.8 recent, -0.3 delta
- Savant occasionally returns HTML error pages instead of CSV — detected and handled
- **Live test:** Aaron Nola arsenal (KC 31%, FF 28%, SI 18%, CH 13%, FC 10%) with correct BA/whiff values

**data/fetchers/odds_api.py**
- Both endpoints implemented with graceful empty-dict fallback when no key set
- **Live test:** Graceful fallback confirmed with empty ODDS_API_KEY

**data/fetchers/weather_api.py**
- Open-Meteo hourly forecast, matched to game hour UTC
- `weather_score` computation: wind-out bonus, cold penalty, humidity penalty
- Wind "blowing out" determined by comparing wind direction vs `STADIUM_OUT_DIRECTION` (±60° tolerance)
- **Live test:** Yankee Stadium — 69.8°F, 7.6 mph NW wind, -0.01 score

**data/fetchers/umpire.py**
- BeautifulSoup scraper implemented for umpscorecards.com
- **Adaptation:** umpscorecards.com returns 404 on all requests. Falls back to neutral zeros as spec requires. Pipeline continues unaffected.

---

### Phase 2 — Processors ✅ COMPLETE

All 5 processors built per spec formulas:
- `arsenal.py`: fallback chain batter-actual → pitcher-ba-against → LEAGUE_AVG
- `pitcher.py`: fatigue score (rest/pitches/IP/velo), TTO interpolation, luck adjustment
- `batter.py`: BABIP regression delta, handedness split selection, PA weight by lineup spot
- `game_context.py`: 6-feature context dict (total, park, weather, umpire, home, day)
- `bullpen.py`: wraps bullpen API data into standardized feature dict

---

### Phase 3 — Prediction + Pipeline ✅ COMPLETE (dry run confirmed)

**model/predict.py**
- V1: weighted normalized feature sum → sigmoid → clamp [0.40, 0.92]
- Loads XGBoost model from `model/artifacts/` if present; falls back to sigmoid
- `_top_factors()` identifies top 3 high-deviation features per prediction
- `rank_predictions()`: sorts by edge; tiers: STRONG BET (>8%), VALUE BET (>5%), LEAN (>2%), PASS, NO LINE, SKIP-LOW DATA

**main.py**
- Full orchestrator: schedule fetch → lineup poll loop → bulk data pulls → feature computation → prediction → CSV + terminal output
- `--max-polls N` arg to control lineup wait (default 18 = 90 min; use `--max-polls 2` for fast runs)
- `--date YYYY-MM-DD` arg for historical/test runs
- Terminal table falls back to sorting by hit probability when no book lines are available (Odds API key not set)
- **Unicode fix:** replaced `═` / `─` box-drawing chars with ASCII `=` / `-` (Windows cp1252 terminal can't encode them)

**Dry run result (2026-05-21, 2 games):**
- 36 predictions generated, all HIGH confidence
- Top plays: Riley Greene 55.9% (45% 14-day H%), Alec Burleson 52.5%, Spencer Horwitz 52.1% (vs Dustin May, 9.99 H/9)
- CSV saved to `outputs/predictions_2026-05-21.csv`
- ~30 sec re-run time from cache; ~5 min/game on cold run

---

### Phase 4 — Training ✅ COMPLETE

**model/train.py**
- `build_training_dataset(2022, 2024)`: pulls Statcast pitch-by-pitch data monthly via pybaseball, caches each season as parquet (`data/training/statcast_raw_{season}.parquet`), extracts one feature row per batter-game with strict no-lookahead (all rolling stats computed from games strictly before the target date)
- `_precompute_pitcher_features()`: processes each pitcher's game log chronologically, building a `{(pitcher_id, game_date): features}` lookup dict — avoids O(n²) re-filtering the full dataset
- `train_model()`: XGBClassifier (n_estimators=300, max_depth=4, lr=0.05) + CalibratedClassifierCV isotonic cv=5, 2022–2023 train / 2024 holdout test, saves `model/artifacts/hit_model.pkl` + `scaler.pkl`
- Runtime: ~50 min total (download + feature extraction + training); re-runs skip to feature extraction via parquet cache

**Training results (2024 holdout):**
- Dataset: 146,029 rows, 57.6% hit rate
- AUC-ROC: 0.6036 | Brier: 0.2340 vs baseline 0.2446
- `main.py` now auto-loads XGBoost probabilities instead of sigmoid fallback

**Feature importances (XGBoost):**
| Feature | Importance | Note |
|---|---|---|
| pitcher_last3_hits_avg | 0.281 | Dominant signal |
| times_through_order | 0.088 | |
| handedness_split | 0.081 | |
| pitcher_fatigue_score | 0.074 | |
| batter_h_pct_season | 0.060 | |
| babip_regression_delta | 0.050 | |
| arsenal_weighted_whiff | 0.045 | |
| batter_xba_season | 0.042 | |
| pitcher_h9_season | 0.039 | |
| batter_h_pct_14day | 0.038 | |
| exit_velocity_14day | 0.038 | |
| park_factor | 0.036 | |
| arsenal_weighted_ba | 0.034 | |
| pitcher_babip_against | 0.034 | |
| batter_h_pct_7day | 0.032 | |
| lineup_position_pa_weight | 0.028 | |
| career_h_ab_vs_pitcher | 0.000 | Constant in training (insufficient matchup history) |
| pitcher_xfip | 0.000 | Constant in training (always 4.0) |
| bullpen_h9 | 0.000 | Constant in training (always league avg) |
| game_total | 0.000 | Constant in training (always 8.5) |
| weather_score | 0.000 | Constant in training (always 0.0) |
| umpire_zone_score | 0.000 | Constant in training (always 0.0) |

**Adaptation:** Six features were constants in training data (no historical odds, weather, umpire, or bullpen data available). XGBoost learned nothing from them — they contribute only via the sigmoid fallback path. Live pipeline feeds real values for all six. Future retraining with richer data would unlock these signals.

---

### Phase 5 — Evaluation + Scheduler ⏳ NOT YET BUILT

- `model/evaluate.py`: backtest predictions vs actual box scores, ROI simulation at edge > 0.05
- `scheduler.py`: `schedule.every().day.at("11:30").do(job)` — already spec'd, straightforward

---

## Feature Weights (22 features, normalized to sum 1.0)

| Feature | Weight | Notes |
|---|---|---|
| batter_h_pct_14day | 0.127 | Strongest signal |
| career_h_ab_vs_pitcher | 0.073 | Requires 15+ AB |
| arsenal_weighted_ba | 0.091 | Pitch-mix weighted BA |
| batter_h_pct_7day | 0.073 | |
| batter_xba_season | 0.055 | Statcast xBA |
| babip_regression_delta | 0.055 | Luck adjustment |
| batter_h_pct_season | 0.064 | |
| arsenal_weighted_whiff | 0.036 | |
| pitcher_h9_season | 0.045 | |
| pitcher_xfip | 0.045 | pybaseball |
| game_total | 0.036 | Defaults to 8.5 if no line |
| times_through_order | 0.036 | TTO multiplier |
| pitcher_last3_hits_avg | 0.036 | Recent form |
| pitcher_fatigue_score | 0.027 | |
| bullpen_h9 | 0.027 | |
| park_factor | 0.027 | |
| weather_score | 0.027 | |
| handedness_split | 0.027 | |
| lineup_position_pa_weight | 0.027 | |
| pitcher_babip_against | 0.027 | |
| umpire_zone_score | 0.018 | Currently always 0 |
| exit_velocity_14day | 0.018 | |
