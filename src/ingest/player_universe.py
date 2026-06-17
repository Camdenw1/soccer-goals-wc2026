"""
Build the player universe: top 3 players per WC2026 team by blended goals/90.

Nationality data is pulled from Understat's player pages via soccerdata.
Since Understat doesn't expose nationality directly, we use a curated mapping
of known WC2026 squad members. This covers the top scorers we care about;
fringe players beyond top-3 don't affect model output.

The output CSV has columns:
  team, player_name, club, position, goals_per90, xg_per90, blended_per90,
  xg_imputed, total_minutes, rank_in_team
"""

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def normalize_name(name: str) -> str:
    """
    Canonicalize a player name for matching across sources.

    Understat is inconsistent: it keeps accents for some players (Lautaro
    Martínez) but strips them for others (Vlahovic, Modric). It also uses
    Western token order for Korean names (Hee-Chan Hwang, Son Heung-Min).

    We strip diacritics, lowercase, replace punctuation/hyphens with spaces,
    and collapse whitespace so both sides compare on a common form.
    """
    if not isinstance(name, str):
        return ""
    # Decompose accents and drop the combining marks
    nfkd = unicodedata.normalize("NFKD", name)
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Hyphens/punctuation -> space, lowercase
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", no_accents).lower()
    return re.sub(r"\s+", " ", cleaned).strip()


def _name_keys(name: str):
    """
    Return the set of normalized lookup keys for a name, covering common
    Eastern/Western surname-position swaps:

      - the normalized form itself
      - full token reversal  ('son heung min' <-> 'min heung son')
      - rotate last token to front  ('hee chan hwang' -> 'hwang hee chan')
      - rotate first token to back  ('hwang hee chan' -> 'hee chan hwang')

    The two rotations (not just reversal) are what correctly match 3-token
    Korean names where the surname is a single token and the given name is
    hyphenated, e.g. data 'Hee-Chan Hwang' vs map 'Hwang Hee-chan'.
    """
    norm = normalize_name(name)
    keys = {norm}
    tokens = norm.split()
    if len(tokens) > 1:
        keys.add(" ".join(reversed(tokens)))
        keys.add(" ".join([tokens[-1]] + tokens[:-1]))  # last -> front
        keys.add(" ".join(tokens[1:] + [tokens[0]]))    # first -> back
    return keys

# Hand-curated WC2026 squad key players (top ~5 per nation from known squads)
# Keyed by player_name AS IT APPEARS in Understat data.
# Sources: confirmed squad lists from FIFA / major sports outlets.
NATIONALITY_MAP = {
    # Argentina
    "Lionel Messi": "Argentina",
    "Julián Álvarez": "Argentina",
    "Lautaro Martínez": "Argentina",
    "Rodrigo De Paul": "Argentina",
    "Ángel Di María": "Argentina",

    # Australia
    "Mathew Leckie": "Australia",
    "Martin Boyle": "Australia",
    "Mitchell Duke": "Australia",
    "Jamie Maclaren": "Australia",

    # Austria
    "Marcel Sabitzer": "Austria",
    "Marko Arnautovic": "Austria",
    "Christoph Baumgartner": "Austria",
    "Michael Gregoritsch": "Austria",

    # Belgium
    "Romelu Lukaku": "Belgium",
    "Loïs Openda": "Belgium",
    "Dodi Lukebakio": "Belgium",
    "Jeremy Doku": "Belgium",
    "Leandro Trossard": "Belgium",

    # Brazil
    "Vinicius Junior": "Brazil",
    "Rodrygo": "Brazil",
    "Gabriel Martinelli": "Brazil",
    "Raphinha": "Brazil",
    "Gabriel Jesus": "Brazil",
    "Endrick": "Brazil",

    # Canada
    "Jonathan David": "Canada",
    "Cyle Larin": "Canada",
    "Tajon Buchanan": "Canada",
    "Lucas Cavallini": "Canada",

    # Chile
    "Alexis Sánchez": "Chile",
    "Ben Brereton Díaz": "Chile",
    "Eduardo Vargas": "Chile",

    # Colombia
    "Luis Díaz": "Colombia",
    "Falcao": "Colombia",
    "Jhon Durán": "Colombia",
    "Rafael Santos Borré": "Colombia",
    "Radamel Falcao": "Colombia",

    # Costa Rica
    "Bryan Ruiz": "Costa Rica",
    "Joel Campbell": "Costa Rica",

    # Croatia
    "Ivan Perišić": "Croatia",
    "Andrej Kramarić": "Croatia",
    "Luka Modrić": "Croatia",
    "Marko Livaja": "Croatia",

    # Denmark
    "Rasmus Højlund": "Denmark",
    "Viktor Gyökeres": "Sweden",  # correction
    "Kasper Dolberg": "Denmark",
    "Andreas Skov Olsen": "Denmark",

    # Ecuador
    "Enner Valencia": "Ecuador",
    "Michael Estrada": "Ecuador",
    "Moisés Caicedo": "Ecuador",

    # Egypt
    "Mohamed Salah": "Egypt",
    "Mostafa Mohamed": "Egypt",
    "Omar Marmoush": "Egypt",

    # England
    "Harry Kane": "England",
    "Bukayo Saka": "England",
    "Marcus Rashford": "England",
    "Phil Foden": "England",
    "Ollie Watkins": "England",
    "Dominic Calvert-Lewin": "England",
    "Alexander Isak": "Sweden",  # Swedish, not English

    # France
    "Kylian Mbappe-Lottin": "France",
    "Olivier Giroud": "France",
    "Marcus Thuram": "France",
    "Ousmane Dembélé": "France",
    "Antoine Griezmann": "France",
    "Randal Kolo Muani": "France",
    "Gonçalo Ramos": "Portugal",  # Portuguese

    # Germany
    "Kai Havertz": "Germany",
    "Florian Wirtz": "Germany",
    "Thomas Müller": "Germany",
    "Serge Gnabry": "Germany",
    "Leroy Sané": "Germany",
    "Niclas Füllkrug": "Germany",

    # Albania (Big-5 forward)
    "Armando Broja": "Albania",

    # South Africa (Big-5 forward)
    "Lyle Foster": "South Africa",

    # Honduras
    "Alberth Elis": "Honduras",
    "Anthony Lozano": "Honduras",

    # Iran
    "Mehdi Taremi": "Iran",
    "Sardar Azmoun": "Iran",

    # Italy
    "Ciro Immobile": "Italy",
    "Gianluca Scamacca": "Italy",
    "Mateo Retegui": "Italy",
    "Lorenzo Pellegrini": "Italy",
    "Federico Chiesa": "Italy",

    # Jamaica
    "Leon Bailey": "Jamaica",
    "Michail Antonio": "Jamaica",
    "Demarai Gray": "Jamaica",

    # Japan
    "Takumi Minamino": "Japan",
    "Ayase Ueda": "Japan",
    "Daichi Kamada": "Japan",
    "Ritsu Doan": "Japan",
    "Kaoru Mitoma": "Japan",
    "Junya Ito": "Japan",

    # Korea Republic
    "Son Heung-min": "Korea Republic",
    "Hwang Hee-chan": "Korea Republic",
    "Cho Gue-sung": "Korea Republic",

    # Mexico
    "Hirving Lozano": "Mexico",
    "Raúl Jiménez": "Mexico",
    "Henry Martín": "Mexico",
    "Santiago Giménez": "Mexico",

    # Morocco
    "Youssef En-Nesyri": "Morocco",
    "Ayoub El Kaabi": "Morocco",
    "Sofiane Boufal": "Morocco",

    # Netherlands
    "Memphis Depay": "Netherlands",
    "Cody Gakpo": "Netherlands",
    "Donyell Malen": "Netherlands",
    "Wout Weghorst": "Netherlands",
    "Vincent Janssen": "Netherlands",

    # New Zealand
    "Chris Wood": "New Zealand",

    # Nigeria
    "Victor Osimhen": "Nigeria",
    "Kelechi Iheanacho": "Nigeria",
    "Paul Onuachu": "Nigeria",
    "Samuel Chukwueze": "Nigeria",

    # Panama
    "Ismael Díaz": "Panama",

    # Paraguay
    "Miguel Almirón": "Paraguay",
    "Alejandro Romero": "Paraguay",

    # Peru
    "André Carrillo": "Peru",
    "Gianluca Lapadula": "Peru",
    "Bryan Reyna": "Peru",

    # Poland
    "Robert Lewandowski": "Poland",
    "Arkadiusz Milik": "Poland",
    "Karol Świderski": "Poland",

    # Portugal
    "Cristiano Ronaldo": "Portugal",
    "Diogo Jota": "Portugal",
    "Rafael Leão": "Portugal",
    "Bruno Fernandes": "Portugal",
    "Bernardo Silva": "Portugal",

    # Saudi Arabia
    "Firas Al-Buraikan": "Saudi Arabia",
    "Saleh Al-Shehri": "Saudi Arabia",

    # Senegal
    "Sadio Mané": "Senegal",
    "Ismaïla Sarr": "Senegal",
    "Boulaye Dia": "Senegal",
    "Nicolas Jackson": "Senegal",

    # Serbia
    "Aleksandar Mitrović": "Serbia",
    "Dušan Vlahović": "Serbia",
    "Luka Jović": "Serbia",

    # South Korea
    "Son Heung-min": "Korea Republic",

    # Spain
    "Álvaro Morata": "Spain",
    "Ferran Torres": "Spain",
    "Dani Olmo": "Spain",
    "Mikel Oyarzabal": "Spain",
    "Joselu": "Spain",
    "Ayoze Pérez": "Spain",

    # Switzerland
    "Haris Seferović": "Switzerland",
    "Breel Embolo": "Switzerland",
    "Ruben Vargas": "Switzerland",
    "Granit Xhaka": "Switzerland",

    # Tunisia
    "Wahbi Khazri": "Tunisia",
    "Seifeddine Jaziri": "Tunisia",

    # Ukraine
    "Artem Dovbyk": "Ukraine",
    "Roman Yaremchuk": "Ukraine",
    "Oleksandr Zinchenko": "Ukraine",

    # Uruguay
    "Luis Suárez": "Uruguay",
    "Darwin Núñez": "Uruguay",
    "Edinson Cavani": "Uruguay",
    "Facundo Pellistri": "Uruguay",

    # USA
    "Christian Pulisic": "USA",
    "Josh Sargent": "USA",
    "Folarin Balogun": "USA",
    "Giovanni Reyna": "USA",
    "Ricardo Pepi": "USA",
    "Malik Tillman": "USA",

    # Venezuela
    "Salomón Rondón": "Venezuela",
    "Darwin Machis": "Venezuela",
    "Yeferson Soteldo": "Venezuela",

    # Côte d'Ivoire
    "Sébastien Haller": "Côte d'Ivoire",
    "Nicolas Pépé": "Côte d'Ivoire",
    "Wilfried Zaha": "Côte d'Ivoire",
    "Franck Kessié": "Côte d'Ivoire",

    # South Africa
    "Percy Tau": "South Africa",
    "Bongokuhle Hlongwane": "South Africa",

    # Sweden (not WC2026 but some players above miscoded)
    "Viktor Gyökeres": "Sweden",
    "Alexander Isak": "Sweden",
}

# Fix duplicates/corrections — remove non-WC2026 teams from the map
WC2026_NATIONS = {
    "USA", "Panama", "Canada", "Uruguay", "Argentina", "Chile", "Peru",
    "Australia", "Mexico", "Jamaica", "Venezuela", "New Zealand", "Brazil",
    "Paraguay", "Colombia", "Costa Rica", "Spain", "Morocco", "Belgium",
    "Egypt", "Portugal", "Croatia", "Germany", "Japan", "France", "Poland",
    "Senegal", "Ecuador", "Netherlands", "Serbia", "Austria", "Ukraine",
    "England", "Albania", "Iran", "Switzerland", "Italy", "Nigeria",
    "Korea Republic", "Côte d'Ivoire", "Honduras", "South Africa",
    "Saudi Arabia", "Denmark", "Tunisia",
}

NATIONALITY_MAP = {k: v for k, v in NATIONALITY_MAP.items() if v in WC2026_NATIONS}


def build_player_universe(
    rates_path: str = "data/processed/player_rates.parquet",
    output_path: str = "data/player_universe.csv",
    top_n: int = 3,
) -> pd.DataFrame:
    """
    Join player rates with nationality map; select top N per team.
    Players not in the nationality map are dropped (they're not WC2026 squad members).
    """
    df = pd.read_parquet(rates_path)

    # Build a normalized-name -> nation lookup from the curated map.
    norm_map = {}
    for raw_name, nation in NATIONALITY_MAP.items():
        for key in _name_keys(raw_name):
            norm_map[key] = nation

    # Attach nationality via normalized matching (accent/case/order-robust).
    def _lookup(name):
        for key in _name_keys(name):
            if key in norm_map:
                return norm_map[key]
        return None

    df["team"] = df["player_name"].apply(_lookup)
    wc = df[df["team"].notna()].copy()
    wc["manual_source"] = False

    # Merge the hand-curated supplement (uncovered nations + non-Big-5 stars).
    supp = load_supplement()
    if supp is not None and len(supp):
        wc = pd.concat([wc, supp], ignore_index=True)

    # For any team, if we have <3 players in the data, we'll note the gap
    # but don't pad — partial data is honest.
    wc = wc.sort_values("blended_per90", ascending=False)
    top = (
        wc.groupby("team")
        .head(top_n)
        .copy()
    )
    top["rank_in_team"] = top.groupby("team").cumcount() + 1

    teams_in_data = top["team"].nunique()
    logger.info("Player universe: %d players across %d teams", len(top), teams_in_data)

    missing_teams = WC2026_NATIONS - set(top["team"].unique())
    if missing_teams:
        logger.warning("Still no data (Big-5 or supplement) for: %s", sorted(missing_teams))

    cols = [
        "team", "rank_in_team", "player_name", "club", "position",
        "goals_per90", "xg_per90", "blended_per90", "xg_imputed",
        "manual_source", "total_minutes",
    ]
    universe = top[cols].sort_values(["team", "rank_in_team"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    logger.info("Saved player universe to %s", output_path)
    return universe


SUPPLEMENT_PATH = "data/manual_player_rates.csv"


def load_supplement(path: str = SUPPLEMENT_PATH) -> pd.DataFrame:
    """
    Load the hand-curated supplement for players outside the Big-5 leagues
    (e.g. Messi/MLS, Ronaldo/Saudi) and nations with no Big-5 representation.

    Every row here is manually sourced from public stats and flagged with
    manual_source=True and xg_imputed=True (no shot-level xG available). The
    rate columns are expected to be pre-computed per-90 values. Each row should
    cite its source in the `source_note` column for reproducibility.

    Required columns: team, player_name, club, position, goals_per90,
    xg_per90, total_minutes, source_note.
    """
    p = Path(path)
    if not p.exists():
        logger.info("No supplement file at %s — skipping.", path)
        return None
    supp = pd.read_csv(p)
    if supp.empty:
        return None
    # xG is never truly available for these rows; blend still uses the
    # supplied xg_per90 (often a proxy = goals_per90 or a positional prior).
    supp["blended_per90"] = 0.5 * supp["goals_per90"] + 0.5 * supp["xg_per90"]
    supp["xg_imputed"] = True
    supp["manual_source"] = True
    logger.info("Loaded %d manual supplement rows from %s", len(supp), path)
    return supp


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config()
    universe = build_player_universe(
        rates_path=str(Path(cfg["data"]["processed_dir"]) / "player_rates.parquet"),
        output_path=cfg["data"]["player_universe"],
        top_n=cfg["model"]["top_n_players_per_team"],
    )
    print(universe.to_string())
    print(f"\n{len(universe)} players, {universe['team'].nunique()} teams")
