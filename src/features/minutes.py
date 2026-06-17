"""
Expected-minutes feature.

The third factor in lambda is expected_minutes / 90: a player who is rotated or
plays as a substitute should have a lower per-match goal expectation than a
nailed-on starter, even at the same per-90 rate.

We estimate expected minutes from each player's recent club appearance record
in the Understat data: average minutes per appearance, recency-weighted toward
the latest season, capped at 90. This is a proxy for national-team minutes
(we don't have per-match NT lineups in the baseline), but club availability and
fitness track national-team starts closely for these top players.

Manual-supplement players (non-Big-5, no Understat record) are first-choice
internationals by construction, so they get `default_expected_minutes`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def estimate_expected_minutes(
    raw_understat: pd.DataFrame,
    season_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Return per-player expected minutes from club appearance data.

    expected_minutes = weighted mean of (minutes / matches) across seasons,
    weighting recent seasons more, capped at 90.
    """
    df = raw_understat[raw_understat["matches"] > 0].copy()
    df["min_per_match"] = (df["minutes"] / df["matches"]).clip(upper=90)

    # Recency weight: latest season counts double by default.
    if season_weights is None:
        seasons = sorted(df["season"].unique())
        season_weights = {s: (2.0 if s == seasons[-1] else 1.0) for s in seasons}
    df["w"] = df["season"].map(season_weights).fillna(1.0) * df["matches"]

    grp = df.groupby("player_name")
    exp = grp.apply(
        lambda g: np.average(g["min_per_match"], weights=g["w"]),
        include_groups=False,
    ).rename("expected_minutes")
    apps = grp["matches"].sum().rename("total_appearances")
    out = pd.concat([exp, apps], axis=1).reset_index()
    return out


def add_expected_minutes(
    universe: pd.DataFrame,
    raw_understat: pd.DataFrame,
    default_minutes: float = 85.0,
    starter_threshold: float = 60.0,
) -> pd.DataFrame:
    """
    Attach `expected_minutes` to the universe, falling back to default_minutes
    for manual-supplement / unmatched players. Flags rotation risk and the
    imputed fallback.
    """
    est = estimate_expected_minutes(raw_understat)
    df = universe.merge(est, on="player_name", how="left")

    df["minutes_imputed"] = df["expected_minutes"].isna()
    df["expected_minutes"] = df["expected_minutes"].fillna(default_minutes)
    df["rotation_risk"] = df["expected_minutes"] < starter_threshold

    n_imp = int(df["minutes_imputed"].sum())
    if n_imp:
        logger.info("Imputed default minutes for %d players (no club record)", n_imp)
    return df


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    universe = pd.read_csv(cfg["data"]["player_universe"])
    raw = pd.read_parquet(Path(cfg["data"]["raw_dir"]) / "understat_club_stats.parquet")
    out = add_expected_minutes(
        universe,
        raw,
        default_minutes=cfg["model"]["default_expected_minutes"],
        starter_threshold=cfg["features"]["starter_minutes_threshold"],
    )
    print(out[["team", "player_name", "expected_minutes", "rotation_risk", "minutes_imputed"]].head(20).to_string(index=False))
