"""
Baseline model (ladder step 1): empirical-rate Poisson.

For a player with expected goals `lam` in a match, we model goals as
Poisson(lam) and read off the tail probabilities:

    P(scores >= k) = 1 - P(X <= k-1),   X ~ Poisson(lam)

`lam` is composed from the three feature factors:

    lam = E[team goals vs opponent]      (opponent_strength)
          * player's share of team goals (rates)
          * expected_minutes / 90        (minutes)

This is deliberately the simplest sensible model: independent Poisson, no
opponent-specific defensive interaction beyond team strength, no correlation
between team goals and which player scores. It is the benchmark every later
model (Dixon-Coles, hierarchical, boosted) must beat on calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import poisson


def player_lambda(team_goals: float, goal_share: float,
                  expected_minutes: float) -> float:
    """Compose a player's expected goals for the match."""
    return team_goals * goal_share * (expected_minutes / 90.0)


def tail_probs(lam: float, thresholds=(1, 2, 3)) -> dict:
    """P(player scores >= k) for each k in thresholds."""
    return {f"p{k}plus": float(poisson.sf(k - 1, lam)) for k in thresholds}


@dataclass
class PlayerPrediction:
    team: str
    player_name: str
    opponent: str
    team_goals: float
    goal_share: float
    expected_minutes: float
    lam: float
    p1plus: float
    p2plus: float
    p3plus: float
    xg_imputed: bool = False
    manual_source: bool = False

    @classmethod
    def build(cls, *, team, player_name, opponent, team_goals, goal_share,
              expected_minutes, xg_imputed=False, manual_source=False):
        lam = player_lambda(team_goals, goal_share, expected_minutes)
        tp = tail_probs(lam)
        return cls(
            team=team, player_name=player_name, opponent=opponent,
            team_goals=team_goals, goal_share=goal_share,
            expected_minutes=expected_minutes, lam=lam,
            p1plus=tp["p1plus"], p2plus=tp["p2plus"], p3plus=tp["p3plus"],
            xg_imputed=bool(xg_imputed), manual_source=bool(manual_source),
        )
