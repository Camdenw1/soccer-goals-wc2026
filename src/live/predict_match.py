"""
Predict 1+/2+/3+ scoring probabilities for the modelled players in one match.

Wires the feature modules and the baseline Poisson model together:
  - opponent_strength.TeamStrength  -> E[team goals vs opponent]
  - rates.add_goal_shares           -> player share of team goals
  - minutes.add_expected_minutes    -> expected minutes / 90
  - baseline_poisson                -> Poisson tail probabilities
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

from src.features import minutes as minutes_mod
from src.features import opponent_strength as opp_mod
from src.features import rates as rates_mod
from src.models.baseline_poisson import PlayerPrediction

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_player_features(cfg: dict) -> pd.DataFrame:
    """Assemble per-player goal_share + expected_minutes for every team."""
    universe = pd.read_csv(cfg["data"]["player_universe"])
    feats = rates_mod.add_goal_shares(
        universe, top_group_share=cfg["model"]["top_group_goal_share"],
        top_n=cfg["model"]["top_n_players_per_team"],
        missing_slot_rate=cfg["model"]["missing_slot_rate"])
    raw = pd.read_parquet(
        Path(cfg["data"]["raw_dir"]) / "understat_club_stats.parquet")
    feats = minutes_mod.add_expected_minutes(
        feats, raw,
        default_minutes=cfg["model"]["default_expected_minutes"],
        starter_threshold=cfg["features"]["starter_minutes_threshold"],
    )
    return feats


def load_team_strength(cfg: dict, as_of: str | pd.Timestamp | None = None):
    oc = cfg["opponent"]
    results = opp_mod.fetch_results(
        oc["results_url"],
        Path(cfg["data"]["raw_dir"]) / "international_results.parquet",
    )
    return opp_mod.TeamStrength.fit(
        results,
        history_start=oc["history_start"],
        halflife_days=oc["recent_halflife_days"],
        home_advantage_goals=oc["home_advantage_goals"],
        as_of=pd.Timestamp(as_of) if as_of is not None else None,
    )


def predict_team(team: str, opponent: str, team_goals: float,
                 feats: pd.DataFrame) -> list[PlayerPrediction]:
    rows = feats[feats["team"] == team]
    preds = []
    for _, r in rows.iterrows():
        preds.append(PlayerPrediction.build(
            team=team, player_name=r["player_name"], opponent=opponent,
            team_goals=team_goals, goal_share=r["goal_share"],
            expected_minutes=r["expected_minutes"],
            xg_imputed=bool(r.get("xg_imputed", False)),
            manual_source=bool(r.get("manual_source", False)),
        ))
    return preds


def predict_match(team_a: str, team_b: str, *, feats: pd.DataFrame,
                  strength: opp_mod.TeamStrength,
                  venue_a: str = "neutral") -> pd.DataFrame:
    """Return a tidy DataFrame of player predictions for both sides."""
    lam_a, lam_b = strength.expected_goals_both(team_a, team_b, venue_a)
    preds = (predict_team(team_a, team_b, lam_a, feats)
             + predict_team(team_b, team_a, lam_b, feats))
    df = pd.DataFrame([p.__dict__ for p in preds])
    return df.sort_values(["team", "p1plus"], ascending=[True, False])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    feats = build_player_features(cfg)
    strength = load_team_strength(cfg)
    out = predict_match("Argentina", "Saudi Arabia", feats=feats, strength=strength)
    cols = ["team", "player_name", "opponent", "team_goals", "goal_share",
            "expected_minutes", "lam", "p1plus", "p2plus", "p3plus"]
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
