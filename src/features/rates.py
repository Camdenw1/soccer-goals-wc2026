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
    top_n: int = 3,
    missing_slot_rate: float = 0.25,
) -> pd.DataFrame:
    """
    Add `within_top3_share` and `goal_share` columns to the player universe.

    `goal_share` is the player's estimated share of ALL their national team's
    goals. For a team with all `top_n` players present, the modelled players'
    shares sum to `top_group_share`.

    When fewer than `top_n` players are available (teams with thin Big-5 /
    supplement coverage), we DON'T hand the whole top-group share to the one or
    two we have — that would wildly overstate a lone striker. Instead we pad the
    share denominator with the missing slots at a typical 3rd-choice scorer rate
    (`missing_slot_rate`), so the present players keep a realistic share and the
    unmodelled remainder is simply left unassigned:

        denom_team = sum(observed rates) + (top_n - n_observed) * missing_slot_rate
        within_share_i = rate_i / denom_team
        goal_share_i   = top_group_share * within_share_i
    """
    df = universe.copy()
    n_obs = df.groupby("team")["blended_per90"].transform("size")
    rate_sum = df.groupby("team")["blended_per90"].transform("sum")
    pad = (top_n - n_obs).clip(lower=0) * missing_slot_rate
    denom = rate_sum + pad
    df["within_top3_share"] = df["blended_per90"] / denom
    df["goal_share"] = top_group_share * df["within_top3_share"]
    return df


def build_rate_features(
    universe_path: str,
    output_path: str,
    top_group_share: float = 0.65,
    top_n: int = 3,
    missing_slot_rate: float = 0.25,
) -> pd.DataFrame:
    universe = pd.read_csv(universe_path)
    feats = add_goal_shares(
        universe, top_group_share=top_group_share,
        top_n=top_n, missing_slot_rate=missing_slot_rate)

    # Sanity: teams with a full top-n should sum to top_group_share; thinner
    # teams sum to less (the unmodelled slots are intentionally unassigned).
    n_obs = feats.groupby("team")["player_name"].transform("size")
    full = feats[n_obs == top_n]
    sums = full.groupby("team")["goal_share"].sum()
    bad = sums[(sums - top_group_share).abs() > 1e-6]
    if len(bad):
        logger.warning("Goal-share sums off target for full teams: %s", bad.to_dict())

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
        top_n=cfg["model"]["top_n_players_per_team"],
        missing_slot_rate=cfg["model"]["missing_slot_rate"],
    )
    print(feats[["team", "player_name", "blended_per90", "within_top3_share", "goal_share"]].head(15).to_string(index=False))
