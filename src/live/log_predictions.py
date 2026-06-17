"""
Generate and log player goal-scoring predictions for upcoming fixtures.

Reads the upcoming World Cup fixtures (un-scored rows in the martj42 results
cache), predicts 1+/2+/3+ probabilities for every modelled player whose team is
in our universe, and writes a timestamped CSV to predictions/.

The commit timestamp of that file is the integrity proof for out-of-sample
validation: predictions must be committed (and pushed) BEFORE kickoff. We
therefore also record, in the file, the data cutoff (`as_of`) and the date of
the last scored match the model could have seen, so the information set is
auditable after the fact.

Usage:
    python -m src.live.log_predictions [--as-of YYYY-MM-DD] [--days N]
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.live.predict_match import (
    build_player_features,
    load_config,
    load_team_strength,
    predict_match,
)

logger = logging.getLogger(__name__)


def upcoming_fixtures(cache_path: Path, as_of: pd.Timestamp,
                      days: int) -> pd.DataFrame:
    """Un-scored fixtures in [as_of, as_of + days]."""
    df = pd.read_parquet(cache_path)
    df["date"] = pd.to_datetime(df["date"])
    horizon = as_of + pd.Timedelta(days=days)
    mask = (df["home_score"].isna() & (df["date"] >= as_of)
            & (df["date"] <= horizon))
    return df[mask].sort_values("date").reset_index(drop=True)


def run(as_of: pd.Timestamp, days: int, cfg: dict) -> Path:
    feats = build_player_features(cfg)
    universe_teams = set(feats["team"].unique())
    strength = load_team_strength(cfg, as_of=as_of)

    cache = Path(cfg["data"]["raw_dir"]) / "international_results.parquet"
    fixtures = upcoming_fixtures(cache, as_of, days)

    results = pd.read_parquet(cache)
    last_scored = pd.to_datetime(
        results.dropna(subset=["home_score"])["date"]).max()

    rows = []
    skipped = []
    for _, fx in fixtures.iterrows():
        home, away = fx["home_team"], fx["away_team"]
        if home not in universe_teams and away not in universe_teams:
            skipped.append(f"{home} v {away}")
            continue
        venue_home = "neutral" if bool(fx["neutral"]) else "home"
        pred = predict_match(home, away, feats=feats, strength=strength,
                             venue_a=venue_home)
        pred.insert(0, "match_date", fx["date"].date())
        pred.insert(1, "home_team", home)
        pred.insert(2, "away_team", away)
        rows.append(pred)

    if not rows:
        raise SystemExit("No fixtures with universe teams in the window.")

    out = pd.concat(rows, ignore_index=True)
    # Provenance columns (same for every row in this run).
    gen_ts = datetime.now(timezone.utc)
    out["model"] = "baseline_poisson_v1"
    out["as_of"] = as_of.date()
    out["last_scored_match"] = last_scored.date()
    out["generated_at_utc"] = gen_ts.isoformat()

    out_dir = Path(cfg["predictions"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = out_dir / f"predictions_{gen_ts.strftime('%Y%m%dT%H%M%SZ')}_baseline.csv"
    out.to_csv(fname, index=False)
    logger.info("Wrote %d player predictions across %d fixtures to %s",
                len(out), len(rows), fname)
    if skipped:
        logger.info("Skipped (no universe team): %s", ", ".join(skipped))
    return fname


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    help="Data cutoff / 'now' date (YYYY-MM-DD).")
    ap.add_argument("--days", type=int, default=1,
                    help="Predict fixtures from as-of through as-of + N days.")
    args = ap.parse_args()
    cfg = load_config()
    fname = run(pd.Timestamp(args.as_of), args.days, cfg)

    # Brief human-readable peek at the most confident scorers.
    df = pd.read_csv(fname)
    top = df.sort_values("p1plus", ascending=False).head(12)
    print("\nTop 1+ goal probabilities in this run:")
    print(top[["match_date", "team", "player_name", "opponent",
               "lam", "p1plus", "p2plus", "p3plus"]].to_string(
        index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
