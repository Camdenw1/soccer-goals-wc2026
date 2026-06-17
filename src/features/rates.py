"""
Player scoring-rate features for the goal model.

The match expected-goals decomposition we use is:

    lambda_player = E[team goals vs opponent]            (opponent_strength.py)
                    * player's share of team goals       (this module)
                    * expected_minutes / 90              (minutes.py)

This module turns the player universe (which carries each player's blended
goals+xG per-90 rate) into a *share of team goals*.

We don't observe national-team goal distributions per player directly in the
baseline, so we apportion using club scoring ability, which is exactly what the
blended rate measures:

    within_top3_share_i = blended_per90_i / sum_j(blended_per90_j)   (j in team top-3)
    goal_share_i        = TOP_GROUP_SHARE * within_top3_share_i

TOP_GROUP_SHARE (config: model.top_group_goal_share, default 0.65) is the
fraction of a nation's goals scored by its three modelled players; the
remaining ~0.35 comes from the rest of the XI, substitutes, and own goals.
Holding that constant keeps the three players' shares summing to 0.65 and
prevents us from implicitly assuming a team scores *only* through its top 3.

Empirical per-player national-team goal shares (from the martj42 goalscorers
dataset) are a planned refinement for later models; the baseline keeps the
apportioning transparent and dependency-light.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def add_goal_shares(
    universe: pd.DataFrame,
    top_group_share: float = 0.65,
) -> pd.DataFrame:
    """
    Add `within_top3_share` and `goal_share` columns to the player universe.

    `goal_share` is the player's estimated share of ALL their national team's
    goals; the three modelled players per team sum to `top_group_share`.
    """
    df = universe.copy()
    team_rate_sum = df.groupby("team")["blended_per90"].transform("sum")
    df["within_top3_share"] = df["blended_per90"] / team_rate_sum
    df["goal_share"] = top_group_share * df["within_top3_share"]
    return df


def build_rate_features(
    universe_path: str,
    output_path: str,
    top_group_share: float = 0.65,
) -> pd.DataFrame:
    universe = pd.read_csv(universe_path)
    feats = add_goal_shares(universe, top_group_share=top_group_share)

    # Sanity: each team's modelled shares should sum to top_group_share.
    sums = feats.groupby("team")["goal_share"].sum()
    bad = sums[(sums - top_group_share).abs() > 1e-6]
    if len(bad):
        logger.warning("Goal-share sums off target for: %s", bad.to_dict())

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(output_path, index=False)
    logger.info("Saved rate features (%d players) to %s", len(feats), output_path)
    return feats


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    feats = build_rate_features(
        universe_path=cfg["data"]["player_universe"],
        output_path=str(Path(cfg["data"]["processed_dir"]) / "rate_features.csv"),
        top_group_share=cfg["model"]["top_group_goal_share"],
    )
    print(feats[["team", "player_name", "blended_per90", "within_top3_share", "goal_share"]].head(15).to_string(index=False))
