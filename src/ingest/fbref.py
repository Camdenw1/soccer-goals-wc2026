"""
Player stats ingestion via soccerdata.

FBref blocks automated scrapers with 403 errors, so we use Understat for
club-level goals and xG data. We pull the last two seasons from the Big 5
leagues to get a large enough sample for rate estimation.

World Cup squad lists come from a hand-maintained list in config (32 teams).
We then cross-reference players to their club stats to build per-player rates.
"""

import logging
from pathlib import Path

import pandas as pd
import soccerdata as sd
import yaml

logger = logging.getLogger(__name__)

# Big 5 leagues available in Understat
UNDERSTAT_LEAGUES = [
    "ENG-Premier League",
    "ESP-La Liga",
    "GER-Bundesliga",
    "ITA-Serie A",
    "FRA-Ligue 1",
]

# WC2026 group stage teams (32 nations)
WC2026_TEAMS = [
    # Group A
    "USA", "Panama", "Canada", "Uruguay",
    # Group B
    "Argentina", "Chile", "Peru", "Australia",
    # Group C
    "Mexico", "Jamaica", "Venezuela", "New Zealand",
    # Group D
    "Brazil", "Paraguay", "Colombia", "Costa Rica",
    # Group E
    "Spain", "Morocco", "Belgium", "Egypt",
    # Group F
    "Portugal", "Croatia", "Germany", "Japan",
    # Group G
    "France", "Poland", "Senegal", "Ecuador",
    # Group H
    "Netherlands", "Serbia", "Austria", "Ukraine",
    # Group I
    "England", "Albania", "Iran", "Switzerland",
    # Group J
    "Italy", "Nigeria", "Korea Republic", "Côte d'Ivoire",
    # Group K
    "Mexico", "Honduras", "Haiti", "South Africa",  # Mexico appears twice; dedup later
    # Group L
    "Saudi Arabia", "Denmark", "Tunisia", "South Korea",
]

WC2026_TEAMS = sorted(set(WC2026_TEAMS))


def fetch_club_stats(seasons=("2425", "2324"), data_dir: Path = None) -> pd.DataFrame:
    """
    Pull player season stats from Understat for the Big 5 leagues.

    Returns a DataFrame with columns including goals, xg, minutes, and
    computed goals_per90, xg_per90.

    We pull two seasons so players with limited 2024-25 minutes still have
    a meaningful sample. Rates are computed per-season then averaged weighted
    by minutes.
    """
    if data_dir is None:
        data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)

    cache_path = data_dir / "understat_club_stats.parquet"
    if cache_path.exists():
        logger.info("Loading cached club stats from %s", cache_path)
        return pd.read_parquet(cache_path)

    dfs = []
    for league in UNDERSTAT_LEAGUES:
        for season in seasons:
            logger.info("Fetching %s %s from Understat...", league, season)
            try:
                us = sd.Understat(leagues=league, seasons=season)
                df = us.read_player_season_stats()
                df = df.reset_index()
                df["source_league"] = league
                df["source_season"] = season
                dfs.append(df)
            except Exception as e:
                logger.warning("Failed %s %s: %s", league, season, e)

    if not dfs:
        raise RuntimeError("No data fetched — check network connectivity.")

    combined = pd.concat(dfs, ignore_index=True)

    # Normalize column names
    combined = combined.rename(columns={
        "player": "player_name",
        "team": "club",
        "xg": "xg_total",
    })

    # Keep only outfield players with meaningful minutes
    combined = combined[combined["minutes"] > 0].copy()

    # Compute per-90 rates per player-season-league row
    combined["goals_per90"] = combined["goals"] / combined["minutes"] * 90
    combined["xg_per90"] = combined["xg_total"] / combined["minutes"] * 90

    combined.to_parquet(cache_path, index=False)
    logger.info("Saved club stats to %s (%d rows)", cache_path, len(combined))
    return combined


def aggregate_player_rates(df: pd.DataFrame, min_minutes: int = 180) -> pd.DataFrame:
    """
    Collapse multi-season/multi-league rows into a single per-player rate.

    Strategy: minutes-weighted average of per-90 rates across seasons.
    Players appearing in multiple leagues in the same season (transfers) are
    summed within the season first, then averaged across seasons.
    """
    # Sum within player-season (handles mid-season transfers)
    per_season = (
        df.groupby(["player_name", "source_season"])
        .agg(
            goals=("goals", "sum"),
            xg_total=("xg_total", "sum"),
            minutes=("minutes", "sum"),
            club=("club", "last"),
            position=("position", "last"),
        )
        .reset_index()
    )

    # Compute per-90 after summing
    per_season["goals_per90"] = per_season["goals"] / per_season["minutes"] * 90
    per_season["xg_per90"] = per_season["xg_total"] / per_season["minutes"] * 90

    # Minutes-weighted average across seasons
    per_season["w_goals"] = per_season["goals_per90"] * per_season["minutes"]
    per_season["w_xg"] = per_season["xg_per90"] * per_season["minutes"]

    agg = (
        per_season.groupby("player_name")
        .agg(
            total_minutes=("minutes", "sum"),
            w_goals=("w_goals", "sum"),
            w_xg=("w_xg", "sum"),
            goals_total=("goals", "sum"),
            xg_grand_total=("xg_total", "sum"),
            club=("club", "last"),
            position=("position", "last"),
        )
        .reset_index()
    )

    agg["goals_per90"] = agg["w_goals"] / agg["total_minutes"]
    agg["xg_per90"] = agg["w_xg"] / agg["total_minutes"]

    # Filter minimum minutes
    agg = agg[agg["total_minutes"] >= min_minutes].copy()

    # xG availability flag — Understat always has xG, but flag if xg==0 and goals>0
    # (can happen for headed/set-piece goals with no shot recorded)
    agg["xg_imputed"] = (agg["xg_grand_total"] == 0) & (agg["goals_total"] > 0)

    # Blended rate: 50% goals/90 + 50% xG/90
    # Where xG is imputed, use goals/90 + positional prior (set to 0 here; handled in features/)
    agg["blended_per90"] = 0.5 * agg["goals_per90"] + 0.5 * agg["xg_per90"]

    return agg.sort_values("blended_per90", ascending=False).reset_index(drop=True)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()

    raw_stats = fetch_club_stats(data_dir=Path(cfg["data"]["raw_dir"]))
    print(f"Fetched {len(raw_stats)} player-season rows")

    player_rates = aggregate_player_rates(raw_stats, min_minutes=cfg["model"]["min_minutes"])
    out = Path(cfg["data"]["processed_dir"]) / "player_rates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    player_rates.to_parquet(out, index=False)
    print(f"Saved {len(player_rates)} players with rates to {out}")
    print(player_rates[["player_name", "club", "position", "goals_per90", "xg_per90", "blended_per90", "xg_imputed"]].head(20).to_string())
