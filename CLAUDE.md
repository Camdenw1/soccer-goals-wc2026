# CLAUDE.md — Soccer Goals (World Cup 2026 player goal model)

Context for future Claude Code sessions. Read this first.

## Goal
A Poisson model predicting, for each WC2026 team's top ~3 scorers, their
probability of scoring **1+, 2+, 3+ goals in a given match**.

Core method — for a player with expected goals `lambda` in a match,
goals ~ Poisson(lambda), so `P(>=k) = poisson.sf(k-1, lambda)`. We compose:

    lambda = E[team goals vs the specific opponent]   # src/features/opponent_strength.py
             * player's share of team goals           # src/features/rates.py
             * expected_minutes / 90                   # src/features/minutes.py

## Working agreements (from the project owner, Camden)
- **Commits authored solely by Camden Weber <camdenweber18@gmail.com>.** Never
  add Co-Authored-By: Claude or "Generated with Claude Code". Global setting
  `attribution.commit/pr` is "" in ~/.claude/settings.json.
- Commit incrementally with clear messages; **commit AND push predictions
  before kickoff** — the push timestamp is the out-of-sample integrity proof.
- Explain every modeling choice (Camden defends this in interviews — no black
  boxes). Pause at the end of each phase item to check in.
- Data sources must be **free, no account**. If anything needs a paid plan or
  signup, STOP and ask.
- GitHub: repo `Camdenw1/soccer-goals-wc2026` (public), authed via `gh`.

## Design decisions (made — follow, don't relitigate)
- Player rates: blend goals/90 and xG/90 50/50. No xG -> goals/90 + flag
  `xg_imputed`.
- Player universe: top 3 per team by blended rate. Built in
  `src/ingest/player_universe.py`.
- Validation: train through 2022-11-01, backtest Qatar 2022 (Brier, log-loss,
  reliability curves for 1+/2+/3+). Then retrain through today for WC2026.
- Model ladder (each benchmarks the next): (1) baseline empirical-rate Poisson
  [DONE], (2) Dixon-Coles team model + player allocation, (3) Bayesian
  hierarchical (PyMC), (4) gradient boosting (XGBoost/LightGBM),
  (5) ensemble + calibration comparison table.
- Market benchmark: Polymarket public read API for closing-line value — later.

## Data sources (all free, no account)
- **Player club goals/xG**: Understat via `soccerdata` — BIG 5 LEAGUES ONLY.
  Cached at `data/raw/understat_club_stats.parquet`.
  - FBref is 403-blocked by scrapers; Understat/Sofascore are Big-5 only;
    FotMob has internationals but NO player stats. So player rates can only come
    from the Big 5.
- **International team results**: martj42 GitHub dataset (results.csv,
  goalscorers.csv). **Deviation from the original plan**: football-data.co.uk
  was specified but has NO international matches, so it cannot rate national
  teams. Cached at `data/raw/international_results.parquet`. This file also
  contains UPCOMING WC fixtures as un-scored (NaN) rows — that's our fixture
  list for predictions, and `dropna` keeps them out of ratings.

## Repo layout
    src/ingest/        fbref.py (Understat rates), player_universe.py
    src/features/      rates.py, minutes.py, opponent_strength.py
    src/models/        baseline_poisson.py  (+ future: dixon_coles, hierarchical, ...)
    src/live/          predict_match.py, log_predictions.py
    src/eval/          (backtest harness — TODO)
    data/raw/          gitignored caches (regenerate)
    data/processed/    player_rates.parquet, rate_features.csv (tracked)
    data/player_universe.csv, data/manual_player_rates.csv (tracked)
    predictions/       timestamped prediction CSVs (the integrity artifacts)
    config.yaml        all tunable params

## Key implementation notes / gotchas
- **Python is 3.9.6** (system), NOT the spec's 3.11+ — the venv was scaffolded
  on system Python. Code uses `from __future__ import annotations` so modern
  type hints work. **Rebuild the venv on 3.11 before the PyMC phase.**
- **Name matching** (`player_universe.normalize_name` / `_name_keys`): Understat
  is inconsistent with accents/case and uses Western order for Korean names.
  We strip diacritics + try token rotations (surname-position swaps). Three
  international-results aliases live in `opponent_strength.TEAM_ALIASES`
  (Korea Republic->South Korea, USA->United States, Cote d'Ivoire->Ivory Coast).
- **Manual supplement** (`data/manual_player_rates.csv`): hand-sourced rates for
  7 non-Big-5 stars (Messi, Ronaldo, Enner Valencia, Ugalde, Ismael Diaz,
  Al-Buraikan, Duke). Every row flagged `manual_source` + `xg_imputed` with a
  `source_note`. Covers all 45 WC nations present in our universe.
- **Goal share with <3 players**: lone/2-player teams would otherwise hand the
  whole `top_group_goal_share` (0.65) to one player. `rates.add_goal_shares`
  pads the denominator with missing slots at `missing_slot_rate` (0.25).

## Known limitations (the ladder is designed to fix these)
- **Schedule-strength confound**: the baseline opponent model uses a naive
  recency-weighted ratio method that understates strong teams with hard
  schedules (e.g. France ~1.2 vs Australia, too low). **Dixon-Coles (Model 2)
  fixes this** via joint MLE + low-score correction + fitted home term.
- **Club-minutes proxy**: `expected_minutes` comes from club appearances, so
  young squad players who are bench at their club but start internationally are
  understated (e.g. Endrick ~15 min). Consider NT-lineup data or a floor.
- **24 teams have <3 modelled players** (non-Big-5 squad members unsourced);
  each has at least its top scorer. Ecuador's #2 (Caicedo) is a DM with ~0 rate.
- `top_group_goal_share=0.65` is a documented assumption; the goalscorers
  dataset could ground it empirically in a later model.

## How to run
    python -m src.ingest.fbref                 # refresh Understat rates
    python -m src.ingest.player_universe       # rebuild universe (+ supplement)
    python -m src.live.log_predictions --as-of YYYY-MM-DD --days N
        # writes predictions/predictions_<UTC>_baseline.csv  (then commit+push)

## Status
Phase 0 complete: scaffold, ingestion, universe, features, baseline model, first
logged predictions (8 fixtures 2026-06-17/18), this file.
Next: Phase 1 — backtest harness (`src/eval/`) on Qatar 2022, then Dixon-Coles.
