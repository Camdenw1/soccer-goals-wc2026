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
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

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

    # Haiti": no top big-5 scorers

    # Honduras
    "Alberth Elis": "Honduras",

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

    # Attach nationality
    df["team"] = df["player_name"].map(NATIONALITY_MAP)
    wc = df[df["team"].notna()].copy()

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
        logger.warning("No Understat data found for: %s", sorted(missing_teams))

    cols = [
        "team", "rank_in_team", "player_name", "club", "position",
        "goals_per90", "xg_per90", "blended_per90", "xg_imputed", "total_minutes",
    ]
    universe = top[cols].sort_values(["team", "rank_in_team"]).reset_index(drop=True)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(output_path, index=False)
    logger.info("Saved player universe to %s", output_path)
    return universe


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
