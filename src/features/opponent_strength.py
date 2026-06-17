"""
Opponent-adjusted expected team goals.

The first factor in lambda is E[team goals vs the specific opponent]. We
estimate it from international match results (martj42 public dataset; chosen
over the originally-planned football-data.co.uk because the latter has NO
international fixtures and cannot rate national teams).

Baseline method (a recency-weighted ratio / "Maher"-style attack-defence
model, the standard precursor to the Dixon-Coles model that comes next in the
ladder):

    mu          = weighted mean goals scored by one team in one match
    attack_i    = (weighted mean goals i scores)   / mu
    defence_i   = (weighted mean goals i concedes)  / mu
    E[goals i vs j, neutral] = mu * attack_i * defence_j

Matches are weighted by exponential time decay (recent form counts more) and we
restrict to a recent history window. Home advantage is added for the true host
side only; World Cup group games on neutral-ish ground get no bump.

Dixon-Coles (Model 2) will replace the ratio estimates with a joint MLE fit,
add the low-score dependence correction, and a fitted home term.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Map our universe team names -> names used in the martj42 results dataset.
TEAM_ALIASES = {
    "Korea Republic": "South Korea",
    "USA": "United States",
    "Côte d'Ivoire": "Ivory Coast",
}


def fetch_results(url: str, cache_path: Path) -> pd.DataFrame:
    """Download (and cache) the international results CSV."""
    if cache_path.exists():
        logger.info("Loading cached international results from %s", cache_path)
        return pd.read_parquet(cache_path)
    logger.info("Downloading international results from %s", url)
    df = pd.read_csv(url, parse_dates=["date"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


class TeamStrength:
    """Recency-weighted attack/defence ratings for international teams."""

    def __init__(self, mu: float, attack: dict, defence: dict,
                 home_adv: float = 0.35):
        self.mu = mu
        self.attack = attack
        self.defence = defence
        self.home_adv = home_adv
        # League-average fallback for teams with no rated history.
        self._default = 1.0

    @classmethod
    def fit(cls, results: pd.DataFrame, history_start: str,
            halflife_days: float, home_advantage_goals: float,
            as_of: pd.Timestamp | None = None) -> "TeamStrength":
        df = results[results["date"] >= pd.Timestamp(history_start)].copy()
        as_of = as_of or df["date"].max()
        # Only train on matches strictly before as_of (avoids leakage in backtests).
        df = df[df["date"] <= as_of]

        age_days = (as_of - df["date"]).dt.days.clip(lower=0)
        df["w"] = 0.5 ** (age_days / halflife_days)

        # Long form: one row per (team, goals_for, goals_against).
        home = df.rename(columns={
            "home_team": "team", "away_team": "opp",
            "home_score": "gf", "away_score": "ga"})[["team", "opp", "gf", "ga", "w"]]
        away = df.rename(columns={
            "away_team": "team", "home_team": "opp",
            "away_score": "gf", "home_score": "ga"})[["team", "opp", "gf", "ga", "w"]]
        long = pd.concat([home, away], ignore_index=True).dropna(subset=["gf", "ga"])

        mu = np.average(long["gf"], weights=long["w"])

        def wmean(g, col):
            return np.average(g[col], weights=g["w"])

        grp = long.groupby("team")
        attack = (grp.apply(lambda g: wmean(g, "gf"), include_groups=False) / mu).to_dict()
        defence = (grp.apply(lambda g: wmean(g, "ga"), include_groups=False) / mu).to_dict()
        logger.info("Fitted strengths for %d teams (mu=%.3f, as_of=%s)",
                    len(attack), mu, as_of.date())
        return cls(mu, attack, defence, home_adv=home_advantage_goals)

    def _resolve(self, name: str) -> str:
        return TEAM_ALIASES.get(name, name)

    def expected_goals(self, team: str, opponent: str,
                       venue: str = "neutral") -> float:
        """
        Expected goals for `team` against `opponent`.
        venue: 'home' | 'away' | 'neutral' (from team's perspective).
        """
        t, o = self._resolve(team), self._resolve(opponent)
        a = self.attack.get(t, self._default)
        d = self.defence.get(o, self._default)
        lam = self.mu * a * d
        if venue == "home":
            lam += self.home_adv
        elif venue == "away":
            lam = max(lam - self.home_adv, 0.05)
        return float(lam)

    def expected_goals_both(self, team_a: str, team_b: str,
                            venue_a: str = "neutral") -> tuple[float, float]:
        venue_b = {"home": "away", "away": "home", "neutral": "neutral"}[venue_a]
        return (self.expected_goals(team_a, team_b, venue_a),
                self.expected_goals(team_b, team_a, venue_b))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    oc = cfg["opponent"]
    results = fetch_results(
        oc["results_url"],
        Path(cfg["data"]["raw_dir"]) / "international_results.parquet",
    )
    ts = TeamStrength.fit(
        results,
        history_start=oc["history_start"],
        halflife_days=oc["recent_halflife_days"],
        home_advantage_goals=oc["home_advantage_goals"],
    )
    # Demo: a few WC2026 matchups.
    for a, b in [("Argentina", "Saudi Arabia"), ("Brazil", "Serbia"),
                 ("USA", "Panama"), ("France", "Australia")]:
        la, lb = ts.expected_goals_both(a, b)
        print(f"{a} {la:.2f} - {lb:.2f} {b}")
