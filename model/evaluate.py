from pathlib import Path
from datetime import date, timedelta
import sys
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from data.fetchers.mlb_api import get_today_schedule, _get


def fetch_actuals(date_str: str) -> dict:
    """Return {player_id: did_get_hit} for all batters on date_str."""
    games = get_today_schedule(date_str)
    if not games:
        return {}

    actuals = {}
    for game in games:
        gid = game['game_id']
        data = _get(f'https://statsapi.mlb.com/api/v1/game/{gid}/boxscore')
        if not data:
            continue
        for side in ('home', 'away'):
            players = data.get('teams', {}).get(side, {}).get('players', {})
            for _, pdata in players.items():
                player_id = pdata.get('person', {}).get('id')
                if not player_id:
                    continue
                hits = pdata.get('stats', {}).get('batting', {}).get('hits')
                if hits is None:
                    continue
                actuals[player_id] = int(hits >= 1)

    return actuals


def _roi_simulation(df: pd.DataFrame, edge_threshold: float = 0.05, bet_size: float = 100.0) -> dict:
    """Flat $100 bet on every play with edge > threshold, assuming -110 vig."""
    bets = df[(df['edge'].notna()) & (df['edge'] > edge_threshold)].copy()
    if bets.empty:
        return {'n_bets': 0, 'wins': 0, 'losses': 0, 'profit': 0.0, 'roi_pct': None}

    wins = bets[bets['actual_hit'] == 1]
    losses = bets[bets['actual_hit'] == 0]
    profit = len(wins) * bet_size * (100 / 110) - len(losses) * bet_size
    roi_pct = profit / (len(bets) * bet_size) * 100

    return {
        'n_bets': len(bets),
        'wins': len(wins),
        'losses': len(losses),
        'profit': round(profit, 2),
        'roi_pct': round(roi_pct, 2),
    }


def evaluate_day(date_str: str, predictions_dir: str = 'outputs') -> dict | None:
    """Evaluate one day's predictions against actual box score outcomes."""
    pred_path = Path(predictions_dir) / f'predictions_{date_str}.csv'
    if not pred_path.exists():
        logger.warning(f'No predictions file for {date_str}')
        return None

    preds = pd.read_csv(pred_path)
    actuals = fetch_actuals(date_str)

    if not actuals:
        logger.warning(f'No actuals found for {date_str} — game may not be complete yet')
        return None

    preds['player_id'] = preds['player_id'].astype(int)
    preds['actual_hit'] = preds['player_id'].map(actuals)
    matched = preds.dropna(subset=['actual_hit']).copy()
    matched['actual_hit'] = matched['actual_hit'].astype(int)

    if matched.empty:
        logger.warning(f'No player IDs matched actuals for {date_str}')
        return None

    tier_stats = {}
    for tier in ['STRONG BET', 'VALUE BET', 'LEAN', 'PASS', 'NO LINE']:
        grp = matched[matched['rating'] == tier]
        if len(grp):
            tier_stats[tier] = {
                'n': len(grp),
                'hit_rate': round(grp['actual_hit'].mean(), 4),
                'avg_prob': round(grp['hit_probability'].mean(), 4),
            }

    actual_rate = matched['actual_hit'].mean()
    pred_rate = matched['hit_probability'].mean()

    return {
        'date': date_str,
        'n_predictions': len(matched),
        'actual_hit_rate': round(actual_rate, 4),
        'predicted_hit_rate': round(pred_rate, 4),
        'calibration_error': round(abs(actual_rate - pred_rate), 4),
        'tier_stats': tier_stats,
        'roi': _roi_simulation(matched),
    }


def backtest(start_date: str, end_date: str, predictions_dir: str = 'outputs') -> pd.DataFrame:
    """Backtest all prediction CSVs between start_date and end_date inclusive."""
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    rows = []
    d = start
    while d <= end:
        ds = d.isoformat()
        result = evaluate_day(ds, predictions_dir)
        if result:
            rows.append(result)
            logger.info(
                f'{ds}: n={result["n_predictions"]}  '
                f'actual={result["actual_hit_rate"]:.3f}  '
                f'pred={result["predicted_hit_rate"]:.3f}  '
                f'err={result["calibration_error"]:.4f}'
            )
        d += timedelta(days=1)

    if not rows:
        logger.warning('No results found in date range')
        return pd.DataFrame()

    summary = pd.DataFrame([{
        'date': r['date'],
        'n': r['n_predictions'],
        'actual_hit_rate': r['actual_hit_rate'],
        'predicted_hit_rate': r['predicted_hit_rate'],
        'calibration_error': r['calibration_error'],
        'roi_n_bets': r['roi']['n_bets'],
        'roi_wins': r['roi']['wins'],
        'roi_losses': r['roi']['losses'],
        'roi_profit': r['roi']['profit'],
        'roi_pct': r['roi'].get('roi_pct'),
    } for r in rows])

    total_bets = int(summary['roi_n_bets'].sum())
    total_profit = summary['roi_profit'].sum()

    logger.info(f'\nBacktest summary ({len(rows)} days):')
    logger.info(f'  Avg actual hit rate:  {summary["actual_hit_rate"].mean():.3f}')
    logger.info(f'  Avg predicted:        {summary["predicted_hit_rate"].mean():.3f}')
    logger.info(f'  Avg calibration err:  {summary["calibration_error"].mean():.4f}')
    if total_bets > 0:
        logger.info(
            f'  ROI bets: {total_bets}  '
            f'profit: ${total_profit:.2f}  '
            f'ROI: {total_profit / (total_bets * 100) * 100:.1f}%'
        )
    else:
        logger.info('  No ROI bets (no book lines available in this date range)')

    return summary


if __name__ == '__main__':
    if len(sys.argv) == 2:
        result = evaluate_day(sys.argv[1])
        if result:
            print(f'\nDate: {result["date"]}')
            print(f'Predictions matched: {result["n_predictions"]}')
            print(f'Actual hit rate:     {result["actual_hit_rate"]:.3f}')
            print(f'Predicted hit rate:  {result["predicted_hit_rate"]:.3f}')
            print(f'Calibration error:   {result["calibration_error"]:.4f}')
            print('\nTier breakdown:')
            for tier, stats in result['tier_stats'].items():
                print(f'  {tier:<12}  n={stats["n"]:>3}  actual={stats["hit_rate"]:.3f}  pred={stats["avg_prob"]:.3f}')
            roi = result['roi']
            print('\nROI simulation (edge > 5%, -110 vig):')
            if roi['n_bets']:
                print(f'  {roi["n_bets"]} bets  {roi["wins"]}W / {roi["losses"]}L  profit=${roi["profit"]:.2f}  ROI={roi["roi_pct"]:.1f}%')
            else:
                print('  No bets qualified (no book lines or edge <= 5%)')
    elif len(sys.argv) == 3:
        df = backtest(sys.argv[1], sys.argv[2])
        if not df.empty:
            print(df.to_string(index=False))
    else:
        print('Usage:')
        print('  python evaluate.py YYYY-MM-DD                   # single day')
        print('  python evaluate.py YYYY-MM-DD YYYY-MM-DD        # date range backtest')
