
import io
import math
import random
import re
from pathlib import Path

from datetime import datetime

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Launch Monitor Analyzer", layout="wide")
alt.data_transformers.disable_max_rows()

DIRECTIONAL_COLS = {
    "Offline",
    "Curve",
    "Launch Direction",
    "Side Spin",
    "Spin Axis Tilt",
}

NUMERIC_COLS = {
    "Carry",
    "Total",
    "Peak Height",
    "Descent Angle",
    "Hang Time",
    "Ball Speed",
    "Launch Angle",
    "Back Spin",
    "Total Spin",
    "Club Speed",
    "Smash Factor",
}

METRIC_PRIORITY = [
    "Carry",
    "Total",
    "Ball Speed",
    "Club Speed",
    "Smash Factor",
    "Launch Angle",
    "Back Spin",
    "Total Spin",
    "Peak Height",
    "Descent Angle",
    "Hang Time",
    "Offline Num",
    "Curve Num",
    "Launch Direction Num",
    "Side Spin Num",
    "Spin Axis Tilt Num",
]

AUTO_FILE_CANDIDATES = [
    "Launch.csv",
    "Launch .csv",
    "launch.csv",
    "launch_data.csv",
]
MASTER_OUTPUT_FILE = "Launch.csv"

TRIVIA_FACTS = [
    ("Baseball", "Cy Young won 511 games, a record that is almost certainly untouchable in the modern era."),
    ("Baseball", "Walter Johnson threw 110 career shutouts, still one of the most staggering pitching totals ever."),
    ("Baseball", "Babe Ruth once hit 60 home runs in 1927, more than every team in the American League except one."),
    ("Baseball", "Lou Gehrig played 2,130 straight games before his streak ended in 1939."),
    ("Baseball", "Joe DiMaggio's 56-game hitting streak in 1941 still stands as the major league record."),
    ("Baseball", "Ted Williams finished his career with a .344 batting average and reached base in over 48% of his plate appearances."),
    ("Baseball", "Stan Musial had exactly 1,815 hits at home and 1,815 hits on the road."),
    ("Baseball", "Willie Mays combined elite power, speed, and defense so completely that many historians call him the greatest all-around player ever."),
    ("Baseball", "Hank Aaron hit 755 home runs and also collected more than 3,700 hits."),
    ("Baseball", "Bob Gibson posted a 1.12 ERA in 1968, one of the greatest pitching seasons in MLB history."),
    ("Baseball", "Sandy Koufax threw four no-hitters in four seasons and won three Cy Young Awards in a row."),
    ("Baseball", "Roberto Clemente recorded exactly 3,000 hits, with his final hit becoming a perfect round-number milestone."),
    ("Baseball", "Mickey Mantle hit 18 World Series home runs, still the record."),
    ("Baseball", "Ty Cobb finished with a .366 career batting average, the highest in modern major league history."),
    ("Baseball", "Honus Wagner's tobacco card became one of the most famous and valuable sports cards in the world."),
    ("Baseball", "Christy Mathewson won 373 games and threw three shutouts in the 1905 World Series."),
    ("Baseball", "Lefty Grove led the American League in ERA nine times."),
    ("Baseball", "Yogi Berra played on 10 World Series champions with the Yankees."),
    ("Baseball", "Jackie Robinson won Rookie of the Year, an MVP, and changed the sport forever after breaking MLB's color barrier."),
    ("Baseball", "Brooks Robinson won 16 straight Gold Gloves at third base."),
    ("Baseball", "Tom Seaver struck out 10 straight Padres in 1970, tying a major league record."),
    ("Baseball", "Nolan Ryan recorded 5,714 strikeouts, the highest total in MLB history."),
    ("Baseball", "Reggie Jackson hit three home runs on three consecutive swings in Game 6 of the 1977 World Series."),
    ("Notre Dame", "Knute Rockne coached Notre Dame to 105 wins and helped popularize the forward pass."),
    ("Notre Dame", "The Four Horsemen backfield of 1924 became one of the most famous nicknames in sports history."),
    ("Notre Dame", "George Gipp inspired the immortal phrase 'Win one for the Gipper.'"),
    ("Notre Dame", "Ara Parseghian led Notre Dame to national titles in 1966 and 1973."),
    ("Notre Dame", "Dan Devine coached Notre Dame to the 1977 national championship."),
    ("Notre Dame", "Lou Holtz guided Notre Dame to the 1988 national championship and a 12-0 season."),
    ("Notre Dame", "The Fighting Irish have produced seven Heisman Trophy winners, one of the highest totals in college football."),
    ("Notre Dame", "The 1973 Sugar Bowl win over Alabama remains one of Notre Dame's most celebrated title-clinching performances."),
    ("Notre Dame", "Tim Brown won the 1987 Heisman Trophy primarily as a wide receiver and return specialist, a rare combination."),
    ("Notre Dame", "The phrase 'Play Like a Champion Today' became an iconic Notre Dame tunnel tradition."),
    ("Wrestling", "Dan Gable finished his college career at Iowa State with a 117-1 record."),
    ("Wrestling", "Dan Gable then won Olympic gold in 1972 without surrendering a single point."),
    ("Wrestling", "Cael Sanderson went 159-0 in college, the only four-time undefeated NCAA Division I champion."),
    ("Wrestling", "Iowa under Dan Gable won 15 NCAA team titles in a 21-year span."),
    ("Wrestling", "John Smith won two Olympic gold medals and four World titles in freestyle wrestling."),
    ("Wrestling", "Bruce Baumgartner became one of America's greatest heavyweights with multiple Olympic medals and world titles."),
    ("Wrestling", "Kurt Angle won Olympic gold in 1996 despite competing with a severely injured neck."),
    ("Wrestling", "Aleksandr Karelin went nearly 13 years without losing an international match."),
    ("Wrestling", "Jordan Burroughs won an Olympic gold medal and multiple world titles with one of the sport's most explosive double legs."),
    ("Wrestling", "Kyle Snyder became one of the youngest American Olympic wrestling champions ever when he won gold in 2016."),
    ("Olympics", "Jesse Owens won four gold medals at the 1936 Berlin Olympics."),
    ("Olympics", "Nadia Comaneci scored the first perfect 10 in Olympic gymnastics in 1976."),
    ("Olympics", "Michael Phelps won 23 Olympic gold medals, the most by any athlete."),
    ("Olympics", "Usain Bolt swept the 100m and 200m at three straight Olympics from 2008 through 2016."),
    ("Olympics", "Eric Heiden won five gold medals in five speed skating events at the 1980 Winter Olympics."),
    ("Golf", "Jack Nicklaus won 18 professional major championships, still the men's record."),
    ("Golf", "Tiger Woods won the 2000 U.S. Open by 15 shots, the largest margin in major championship history."),
    ("Golf", "Ben Hogan returned from a near-fatal car crash and still won six majors afterward."),
    ("Golf", "Arnold Palmer's charge at the 1960 U.S. Open helped define modern televised golf drama."),
    ("Golf", "Gary Player became one of the first global golf superstars and completed the career Grand Slam."),
    ("Golf", "Bobby Jones won the Grand Slam in 1930, before the modern professional major structure existed."),
    ("Golf", "Tom Watson won five Open Championships and nearly won another at age 59 in 2009."),
    ("Golf", "Phil Mickelson became the oldest major champion when he won the 2021 PGA Championship at age 50."),
    ("Golf", "Nick Faldo won six majors and was known for major-week precision and discipline."),
    ("Golf", "Padraig Harrington won back-to-back Open Championships in 2007 and 2008."),
    ("Golf", "Seve Ballesteros brought imagination and recovery artistry to major championship golf like few players ever have."),
    ("Golf", "Jordan Spieth nearly completed the Grand Slam by age 24 and won the 2015 Masters by tying the scoring record."),
    ("Golf", "Brooks Koepka won four majors in just eight starts across the U.S. Open and PGA Championship from 2017 to 2019."),
    ("Golf", "Annika Sorenstam won 10 majors and once shot 59 in competition."),
    ("Golf", "Mickey Wright's swing was praised by Ben Hogan as one of the best he had ever seen."),
    ("Golf", "Patty Berg won 15 majors, still the all-time LPGA record."),
    ("Movies", "Katharine Hepburn won four Academy Awards for acting, more than any other performer."),
    ("Movies", "Meryl Streep has received more Oscar acting nominations than any other performer."),
    ("Movies", "The Godfather, Casablanca, and Lawrence of Arabia are frequent fixtures on greatest-film lists."),
    ("Movies", "Ben-Hur, Titanic, and The Lord of the Rings: The Return of the King each won 11 Oscars."),
    ("Movies", "Citizen Kane is often cited as one of the most influential films ever made."),
    ("Movies", "2001: A Space Odyssey reshaped science-fiction filmmaking with its scale, visuals, and ambition."),
    ("Movies", "Blade Runner became more influential over time and helped define the look of cinematic cyberpunk."),
    ("Movies", "The Empire Strikes Back was initially just a sequel, but it grew into one of the most acclaimed blockbusters ever."),
    ("Movies", "Alien blended science fiction and horror so effectively that it launched one of film's great franchises."),
    ("Movies", "Jaws is often credited with helping invent the modern summer blockbuster."),
    ("Movies", "Schindler's List won Best Picture and remains one of Steven Spielberg's most acclaimed films."),
    ("Movies", "No Country for Old Men won Best Picture and is celebrated for its tension, restraint, and unforgettable villain."),
    ("Movies", "Everything Everywhere All at Once swept the 2023 Oscars with seven wins including Best Picture."),
    ("Movies", "The Silence of the Lambs won the 'Big Five' Oscars: Picture, Director, Actor, Actress, and Screenplay."),
    ("Movies", "Mad Max: Fury Road won six Oscars and became one of the most acclaimed action films of the 21st century."),
    ("Movies", "Parasite became the first non-English-language film to win Best Picture at the Oscars."),
]

# Illustrative benchmark baselines for comparison mode.
# These are meant as useful reference targets, not absolute truths.
BENCHMARKS = {
    "Tour / Pro Reference": {
        "DR": {"Carry": 275, "Ball Speed": 167, "Club Speed": 113, "Smash Factor": 1.48},
        "3W": {"Carry": 242, "Ball Speed": 156, "Club Speed": 107, "Smash Factor": 1.46},
        "5W": {"Carry": 230, "Ball Speed": 151, "Club Speed": 103, "Smash Factor": 1.46},
        "4I": {"Carry": 215, "Ball Speed": 141, "Club Speed": 98, "Smash Factor": 1.44},
        "5I": {"Carry": 205, "Ball Speed": 136, "Club Speed": 95, "Smash Factor": 1.43},
        "6I": {"Carry": 193, "Ball Speed": 131, "Club Speed": 92, "Smash Factor": 1.42},
        "7I": {"Carry": 181, "Ball Speed": 126, "Club Speed": 89, "Smash Factor": 1.41},
        "8I": {"Carry": 169, "Ball Speed": 120, "Club Speed": 85, "Smash Factor": 1.40},
        "9I": {"Carry": 157, "Ball Speed": 114, "Club Speed": 80, "Smash Factor": 1.39},
        "PW": {"Carry": 145, "Ball Speed": 108, "Club Speed": 76, "Smash Factor": 1.38},
        "GW": {"Carry": 130, "Ball Speed": 100, "Club Speed": 72, "Smash Factor": 1.37},
        "SW": {"Carry": 115, "Ball Speed": 92, "Club Speed": 68, "Smash Factor": 1.35},
        "LW": {"Carry": 100, "Ball Speed": 86, "Club Speed": 64, "Smash Factor": 1.34},
    },
    "Scratch Reference": {
        "DR": {"Carry": 255, "Ball Speed": 154, "Club Speed": 105, "Smash Factor": 1.47},
        "3W": {"Carry": 228, "Ball Speed": 145, "Club Speed": 100, "Smash Factor": 1.45},
        "5W": {"Carry": 216, "Ball Speed": 139, "Club Speed": 96, "Smash Factor": 1.45},
        "4I": {"Carry": 198, "Ball Speed": 129, "Club Speed": 91, "Smash Factor": 1.42},
        "5I": {"Carry": 188, "Ball Speed": 124, "Club Speed": 88, "Smash Factor": 1.41},
        "6I": {"Carry": 177, "Ball Speed": 119, "Club Speed": 85, "Smash Factor": 1.40},
        "7I": {"Carry": 166, "Ball Speed": 114, "Club Speed": 82, "Smash Factor": 1.39},
        "8I": {"Carry": 155, "Ball Speed": 108, "Club Speed": 78, "Smash Factor": 1.38},
        "9I": {"Carry": 144, "Ball Speed": 102, "Club Speed": 74, "Smash Factor": 1.38},
        "PW": {"Carry": 132, "Ball Speed": 96, "Club Speed": 70, "Smash Factor": 1.37},
        "GW": {"Carry": 118, "Ball Speed": 90, "Club Speed": 66, "Smash Factor": 1.36},
        "SW": {"Carry": 103, "Ball Speed": 84, "Club Speed": 62, "Smash Factor": 1.35},
        "LW": {"Carry": 90, "Ball Speed": 79, "Club Speed": 59, "Smash Factor": 1.34},
    },
    "10 Handicap Reference": {
        "DR": {"Carry": 225, "Ball Speed": 140, "Club Speed": 96, "Smash Factor": 1.46},
        "3W": {"Carry": 205, "Ball Speed": 132, "Club Speed": 92, "Smash Factor": 1.43},
        "5W": {"Carry": 194, "Ball Speed": 126, "Club Speed": 88, "Smash Factor": 1.43},
        "4I": {"Carry": 180, "Ball Speed": 118, "Club Speed": 84, "Smash Factor": 1.40},
        "5I": {"Carry": 171, "Ball Speed": 113, "Club Speed": 81, "Smash Factor": 1.39},
        "6I": {"Carry": 162, "Ball Speed": 108, "Club Speed": 78, "Smash Factor": 1.38},
        "7I": {"Carry": 153, "Ball Speed": 103, "Club Speed": 75, "Smash Factor": 1.37},
        "8I": {"Carry": 143, "Ball Speed": 98, "Club Speed": 72, "Smash Factor": 1.36},
        "9I": {"Carry": 133, "Ball Speed": 92, "Club Speed": 68, "Smash Factor": 1.35},
        "PW": {"Carry": 122, "Ball Speed": 87, "Club Speed": 64, "Smash Factor": 1.35},
        "GW": {"Carry": 108, "Ball Speed": 81, "Club Speed": 60, "Smash Factor": 1.34},
        "SW": {"Carry": 95, "Ball Speed": 76, "Club Speed": 57, "Smash Factor": 1.33},
        "LW": {"Carry": 82, "Ball Speed": 71, "Club Speed": 54, "Smash Factor": 1.31},
    },
}


def _clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_directional(value):
    s = _clean_text(value)
    if s == "":
        return np.nan
    if s in {"0", "0.0"}:
        return 0.0

    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z\-]+)?\s*$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return np.nan

    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()

    negative_tags = {"L", "DN", "O-I", "CLOSED", "IN"}
    positive_tags = {"R", "UP", "I-O", "OPEN", "OUT"}

    if suffix in negative_tags:
        return -abs(num)
    if suffix in positive_tags:
        return abs(num)
    return num


def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def detect_table_type(df):
    cols = [str(c).strip() for c in df.columns]
    normalized_markers = {"Player", "Club", "Carry"}
    if normalized_markers.issubset(set(cols)):
        return "normalized"

    rows = df.fillna("")
    for _, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        if "Date" in vals and "Carry" in vals and "Offline" in vals:
            return "monitor_export"
    return "unknown"


def normalize_shot_table(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "Player" not in out.columns:
        out["Player"] = "Unknown Player"
    if "Club" not in out.columns:
        out["Club"] = "Unknown Club"
    if "Source File" not in out.columns:
        out["Source File"] = "uploaded file"

    for col in DIRECTIONAL_COLS:
        if col in out.columns:
            out[col + " Num"] = out[col].apply(parse_directional)

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    if "Date" in out.columns:
        out["Date Parsed"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        out["Date Parsed"] = pd.NaT

    out["Session Label"] = np.where(
        out["Date Parsed"].notna(),
        out["Date Parsed"].dt.strftime("%Y-%m-%d"),
        out["Source File"].astype(str),
    )

    club_order = {
        "DR": 1, "DRIVER": 1, "3W": 2, "5W": 3, "7W": 4,
        "3H": 5, "4H": 6, "5H": 7,
        "3I": 8, "4I": 9, "5I": 10, "6I": 11, "7I": 12,
        "8I": 13, "9I": 14, "PW": 15, "GW": 16, "AW": 17,
        "SW": 18, "LW": 19,
    }

    def club_key(x):
        s = str(x).strip().upper().replace(" ", "")
        return club_order.get(s, 999)

    out["Club Clean"] = out["Club"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    out["Club Sort"] = out["Club"].map(club_key)
    return out


def parse_monitor_export(df, source_name="Launch.csv"):
    rows = df.fillna("")
    header_idxs = []
    for i, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        if "Date" in vals and "Carry" in vals and "Offline" in vals:
            header_idxs.append(i)

    player_name = str(df.columns[0]).strip()
    if player_name.startswith("Unnamed"):
        player_name = ""

    records = []
    current_club = None
    current_header = None

    for idx, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        nonempty = [v for v in vals if v != ""]
        if not nonempty:
            continue

        if idx in header_idxs:
            current_header = vals
            continue

        if player_name and vals[0] == player_name and len(nonempty) <= 2:
            continue

        if vals[0] and len(nonempty) == 1 and vals[0] not in {"Average", "Std Dev", "Longest", "Shortest"}:
            current_club = vals[0]
            continue

        if vals[0] in {"Average", "Std Dev", "Longest", "Shortest"}:
            continue

        if re.fullmatch(r"\d+", vals[0] or "") and current_header is not None:
            rec = {}
            for c, v in zip(current_header, vals):
                if c:
                    rec[c] = v
            rec["Club"] = current_club
            rec["Player"] = player_name or "Unknown Player"
            rec["Source File"] = source_name
            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return normalize_shot_table(out)


def load_single_csv(file_like, source_name="Launch.csv"):
    raw = pd.read_csv(file_like)
    t = detect_table_type(raw)
    if t == "normalized":
        return normalize_shot_table(raw)
    if t == "monitor_export":
        return parse_monitor_export(raw, source_name=source_name)
    return normalize_shot_table(raw)




def ensure_shot_schema(df):
    out = df.copy()
    if "Player" not in out.columns:
        out["Player"] = "Unknown Player"
    if "Club" not in out.columns:
        out["Club"] = "Unknown Club"

    club_order = {
        "DR": 1, "DRIVER": 1, "3W": 2, "5W": 3, "7W": 4,
        "3H": 5, "4H": 6, "5H": 7,
        "3I": 8, "4I": 9, "5I": 10, "6I": 11, "7I": 12,
        "8I": 13, "9I": 14, "PW": 15, "GW": 16, "AW": 17,
        "SW": 18, "LW": 19,
    }

    def club_key(x):
        s = str(x).strip().upper().replace(" ", "")
        return club_order.get(s, 999)

    if "Club Clean" not in out.columns:
        out["Club Clean"] = out["Club"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    if "Club Sort" not in out.columns:
        out["Club Sort"] = out["Club"].map(club_key)
    else:
        out["Club Sort"] = pd.to_numeric(out["Club Sort"], errors="coerce")
        missing_sort = out["Club Sort"].isna()
        if missing_sort.any():
            out.loc[missing_sort, "Club Sort"] = out.loc[missing_sort, "Club"].map(club_key)

    if "Session Label" not in out.columns:
        if "Date" in out.columns:
            dt = pd.to_datetime(out["Date"], errors="coerce")
            out["Session Label"] = np.where(dt.notna(), dt.dt.strftime("%Y-%m-%d"), "Session")
        else:
            out["Session Label"] = "Session"

    if "Is Clean Shot" not in out.columns:
        out["Is Clean Shot"] = True
    if "Exclusion Reason" not in out.columns:
        out["Exclusion Reason"] = ""

    return out

def add_quality_flags(df):
    out = df.copy()
    needed = {"Carry", "Ball Speed", "Smash Factor"}
    if not needed.issubset(out.columns):
        out["Is Clean Shot"] = True
        out["Exclusion Reason"] = ""
        return out

    out["Exclusion Reason"] = ""
    missing_core = out["Carry"].isna() | out["Ball Speed"].isna() | out["Smash Factor"].isna()
    out.loc[missing_core, "Exclusion Reason"] = "Missing core data"

    hard_floor = (
        (out["Carry"] <= 0)
        | (out["Ball Speed"] < 40)
        | (out["Smash Factor"] < 0.80)
    )
    out.loc[hard_floor & (out["Exclusion Reason"] == ""), "Exclusion Reason"] = "Invalid / obvious bad read"

    for _, idx in out.groupby("Club").groups.items():
        g = out.loc[idx]
        if len(g) < 6:
            continue
        carry_cut = g["Carry"].quantile(0.10)
        speed_cut = g["Ball Speed"].quantile(0.10)
        smash_cut = g["Smash Factor"].quantile(0.10)
        low_tail = (
            out.index.isin(idx)
            & (out["Exclusion Reason"] == "")
            & (
                (out["Carry"] < carry_cut)
                | (out["Ball Speed"] < speed_cut)
                | (out["Smash Factor"] < smash_cut)
            )
        )
        out.loc[low_tail, "Exclusion Reason"] = "Low-end outlier / likely mishit"

    out["Is Clean Shot"] = out["Exclusion Reason"].eq("")
    return out


def pct(n, d):
    return 0 if d == 0 else 100.0 * n / d


def consistency_score(group):
    if len(group) < 3 or group["Carry"].dropna().empty:
        return np.nan
    med = group["Carry"].median()
    sd = group["Carry"].std(ddof=0)
    if pd.isna(sd) or med <= 0:
        return np.nan
    score = 10 - (sd / med) * 35
    return max(1, min(10, score))


def pattern_label(group):
    if group.empty or "Offline Num" not in group.columns or group["Offline Num"].dropna().empty:
        return "Need more direction data"

    off_mean = group["Offline Num"].mean()
    off_sd = group["Offline Num"].std(ddof=0)
    carry_sd = group["Carry"].std(ddof=0) if "Carry" in group.columns else np.nan

    labels = []
    if abs(off_mean) <= 3:
        labels.append("🎯 Neutral")
    elif off_mean > 3:
        labels.append("➡️ Right bias")
    else:
        labels.append("⬅️ Left bias")

    if pd.notna(off_sd):
        if off_sd <= 6:
            labels.append("🧵 Tight")
        elif off_sd >= 12:
            labels.append("🌪 Wide")

    if pd.notna(carry_sd):
        if carry_sd <= 4:
            labels.append("📏 Stable distance")
        elif carry_sd >= 9:
            labels.append("📉 Loose distance")

    return " • ".join(labels) if labels else "Pattern still developing"


@st.cache_data(show_spinner=False)
def prepare_data_from_bytes(file_bytes, file_name):
    buffer = io.BytesIO(file_bytes)
    return ensure_shot_schema(add_quality_flags(load_single_csv(buffer, source_name=file_name)))


@st.cache_data(show_spinner=False)
def prepare_data_from_path(path_str):
    path = Path(path_str)
    with open(path, "rb") as f:
        file_bytes = f.read()
    return prepare_data_from_bytes(file_bytes, path.name)


def make_download(df):
    return df.to_csv(index=False).encode("utf-8")


def find_auto_file():
    search_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path("/mnt/data")]
    for directory in search_dirs:
        for name in AUTO_FILE_CANDIDATES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def available_metrics(df):
    metrics = []
    for metric in METRIC_PRIORITY:
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any():
            metrics.append(metric)
    return metrics


def metric_axis_label(metric):
    mapping = {
        "Offline Num": "Offline (Left negative / Right positive)",
        "Curve Num": "Curve",
        "Launch Direction Num": "Launch Direction",
        "Side Spin Num": "Side Spin",
        "Spin Axis Tilt Num": "Spin Axis Tilt",
    }
    return mapping.get(metric, metric)


def nice_format(metric):
    if metric in {"Smash Factor"}:
        return ".2f"
    return ".1f"


def delta_text(player_val, baseline_val, unit=""):
    if pd.isna(player_val) or pd.isna(baseline_val):
        return "—"
    diff = player_val - baseline_val
    sign = "+" if diff > 0 else ""
    suffix = f" {unit}" if unit else ""
    return f"{sign}{diff:.1f}{suffix}"


def baseline_table_for_set(name):
    rows = []
    for club, metrics in BENCHMARKS[name].items():
        row = {"Club Clean": club}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_unit(metric):
    if metric in {"Carry", "Total", "Peak Height", "Offline Num", "Curve Num", "Launch Direction Num"}:
        return "yds"
    if metric in {"Ball Speed", "Club Speed"}:
        return "mph"
    if metric in {"Launch Angle", "Descent Angle", "Spin Axis Tilt Num"}:
        return "°"
    if metric in {"Back Spin", "Total Spin", "Side Spin Num"}:
        return "rpm"
    return ""


def player_baseline_summary(filtered, benchmark_name):
    club_player = (
        filtered.groupby(["Club", "Club Clean", "Club Sort"], dropna=False)
        .agg(
            Shots=("Club", "size"),
            Carry=("Carry", "median"),
            BallSpeed=("Ball Speed", "median"),
            ClubSpeed=("Club Speed", "median"),
            Smash=("Smash Factor", "median"),
        )
        .reset_index()
    )
    club_player = club_player.rename(columns={"BallSpeed": "Ball Speed", "ClubSpeed": "Club Speed", "Smash": "Smash Factor"})

    base = baseline_table_for_set(benchmark_name)
    merged = club_player.merge(base, on="Club Clean", how="left", suffixes=(" Player", " Baseline"))
    rename_map = {
        "Carry Player": "Carry Player", "Carry Baseline": "Carry Baseline",
        "Ball Speed Player": "Ball Speed Player", "Ball Speed Baseline": "Ball Speed Baseline",
        "Club Speed Player": "Club Speed Player", "Club Speed Baseline": "Club Speed Baseline",
        "Smash Factor Player": "Smash Factor Player", "Smash Factor Baseline": "Smash Factor Baseline",
    }
    # normalize columns after merge
    for raw in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        if raw in merged.columns and f"{raw} Baseline" not in merged.columns:
            merged[f"{raw} Baseline"] = np.nan
        if raw in merged.columns and f"{raw} Player" not in merged.columns:
            merged[f"{raw} Player"] = merged[raw]
    for raw in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        if raw in merged.columns:
            merged.drop(columns=[raw], inplace=True)
    merged = merged.sort_values(["Club Sort", "Club"]).copy()
    return merged


def metric_eval_label(delta, metric):
    if pd.isna(delta):
        return "No baseline"
    abs_delta = abs(delta)
    if metric == "Smash Factor":
        if abs_delta <= 0.02:
            return "Right there"
        if delta > 0:
            return "Above baseline"
        if delta >= -0.05:
            return "Close"
        return "Needs strike quality"
    if abs_delta <= 3:
        return "Right there"
    if delta > 0:
        return "Above baseline"
    if delta >= -8:
        return "Close"
    return "Gap to close"


def metric_card_html(club, stock, p25, p75, shots, consistency, pattern, baseline=None):
    baseline_row = ""
    if baseline is not None and not pd.isna(baseline):
        diff = stock - baseline
        baseline_row = f"<div class='club-sub'>vs baseline: <b>{diff:+.0f}</b></div>"
    consistency_txt = "—" if pd.isna(consistency) else f"{consistency:.1f}/10"
    width = "—" if pd.isna(p25) or pd.isna(p75) else f"{p25:.0f}–{p75:.0f}"
    stock_txt = "—" if pd.isna(stock) else f"{stock:.0f}"
    return f"""
    <div class='club-card'>
        <div class='club-top'>
            <div class='club-name'>{club}</div>
            <div class='club-shots'>{int(shots)} shots</div>
        </div>
        <div class='club-stock'>{stock_txt}<span class='club-unit'> yds</span></div>
        <div class='club-sub'>stock window: <b>{width}</b></div>
        {baseline_row}
        <div class='club-sub'>consistency: <b>{consistency_txt}</b></div>
        <div class='club-pattern'>{pattern}</div>
    </div>
    """



def _club_strength_reason(row):
    reasons = []
    cons = row.get("Consistency", np.nan)
    if pd.notna(cons):
        if cons >= 8:
            reasons.append("tight carry window")
        elif cons <= 5.5:
            reasons.append("loose carry spread")
    avg_off = row.get("AvgOffline", np.nan)
    if pd.notna(avg_off):
        if abs(avg_off) <= 3:
            reasons.append("starts near center")
        elif abs(avg_off) >= 8:
            reasons.append("direction bias shows up")
    smash = row.get("Smash", np.nan)
    baseline_smash = row.get("Baseline Smash", np.nan)
    if pd.notna(smash) and pd.notna(baseline_smash):
        diff = smash - baseline_smash
        if diff >= 0.01:
            reasons.append("solid strike efficiency")
        elif diff <= -0.03:
            reasons.append("strike efficiency trails benchmark")
    elif pd.notna(smash):
        if smash >= 1.35:
            reasons.append("solid strike efficiency")
        elif smash <= 1.25:
            reasons.append("strike efficiency is inconsistent")
    carry_delta = row.get("Carry Delta vs Benchmark", np.nan)
    if pd.notna(carry_delta):
        if carry_delta >= 3:
            reasons.append("carry is above baseline")
        elif carry_delta <= -8:
            reasons.append("carry trails baseline")
    if not reasons:
        reasons.append("small sample / still forming")
    return ", ".join(reasons[:3])


def build_best_worst_lists(club_summary):
    if club_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = club_summary.copy()
    work["CarryStdFill"] = work["CarryStd"].fillna(work["CarryStd"].median())
    work["ConsistencyFill"] = work["Consistency"].fillna(work["Consistency"].median())
    work["AbsOffline"] = work["AvgOffline"].abs().fillna(work["AvgOffline"].abs().median())
    work["SmashDelta"] = (work["Smash"] - work["Baseline Smash"]).fillna(0)
    # Higher is better
    work["Club Score"] = (
        work["ConsistencyFill"] * 1.7
        - work["CarryStdFill"] * 0.22
        - work["AbsOffline"] * 0.16
        + work["SmashDelta"] * 40
        + work["Carry Delta vs Benchmark"].fillna(0) * 0.04
    )
    work["Why"] = work.apply(_club_strength_reason, axis=1)
    best = work.sort_values(["Club Score", "Shots"], ascending=[False, False]).head(3).copy()
    worst = work.sort_values(["Club Score", "Shots"], ascending=[True, False]).head(3).copy()
    return best, worst


def benchmark_reason(row, benchmark_name):
    bits = []
    for metric in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        pcol = f"{metric} Player"
        bcol = f"{metric} Baseline"
        if pcol not in row or bcol not in row:
            continue
        pv, bv = row[pcol], row[bcol]
        if pd.isna(pv) or pd.isna(bv):
            continue
        delta = pv - bv
        if metric == "Smash Factor":
            if delta >= 0.01:
                bits.append("strike efficiency is at or above the benchmark")
            elif delta >= -0.02:
                bits.append("strike efficiency is basically even")
            else:
                bits.append("strike efficiency is behind the benchmark")
        elif metric == "Carry":
            if abs(delta) <= 3:
                bits.append("carry is basically even")
            elif delta > 0:
                bits.append("carry is beating the benchmark")
            else:
                bits.append("carry is trailing the benchmark")
        elif metric == "Ball Speed":
            if abs(delta) <= 2:
                bits.append("ball speed is right there")
            elif delta > 0:
                bits.append("ball speed is strong")
            else:
                bits.append("ball speed is the main gap")
        elif metric == "Club Speed":
            if abs(delta) <= 2:
                bits.append("speed is in range")
            elif delta > 0:
                bits.append("speed is plenty")
            else:
                bits.append("speed is a little light")
    if not bits:
        return f"Not enough data to compare to {benchmark_name.lower()}."
    return "; ".join(bits[:3]).capitalize() + "."


def overall_session_score(club_summary):
    if club_summary.empty:
        return np.nan
    vals = (
        club_summary["Consistency"].fillna(club_summary["Consistency"].median()) * 0.55
        + np.clip(10 - club_summary["AvgOffline"].abs().fillna(0) * 0.35, 1, 10) * 0.20
        + np.clip(10 - club_summary["CarryStd"].fillna(club_summary["CarryStd"].median()) * 0.28, 1, 10) * 0.25
    )
    return float(np.clip(vals.mean(), 1, 10))


def gap_table(club_summary):
    g = club_summary.sort_values(["Club Sort", "Club"]).copy()
    g["Next Club"] = g["Club"].shift(-1)
    g["Gap To Next"] = g["MedianCarry"] - g["MedianCarry"].shift(-1)
    return g


def benchmark_bucket(delta, metric):
    if pd.isna(delta):
        return "No baseline"
    tol = 0.02 if metric == "Smash Factor" else 3
    close = 0.05 if metric == "Smash Factor" else 8
    if abs(delta) <= tol:
        return "About even"
    if delta > 0:
        return "Better"
    if delta >= -close:
        return "Close"
    return "Worse"



def _safe_num(s):
    return pd.to_numeric(s, errors="coerce")


def append_to_master_csv(new_df, master_path):
    """Append normalized shots into the local master CSV, with light de-duping."""
    if new_df is None or new_df.empty:
        return 0, 0
    master_path = Path(master_path)
    incoming = ensure_shot_schema(new_df.copy())
    if master_path.exists():
        try:
            existing = prepare_data_from_path(str(master_path))
            existing = ensure_shot_schema(existing)
        except Exception:
            existing = pd.DataFrame()
    else:
        existing = pd.DataFrame()

    key_cols = [c for c in ["Player", "Session Label", "Club", "Carry", "Ball Speed", "Club Speed", "Smash Factor", "Offline Num"] if c in incoming.columns]
    if key_cols:
        for c in key_cols:
            if c in incoming.columns:
                if pd.api.types.is_numeric_dtype(incoming[c]) or c.endswith("Num") or c in {"Carry", "Ball Speed", "Club Speed", "Smash Factor"}:
                    incoming[c] = _safe_num(incoming[c]).round(3)
                else:
                    incoming[c] = incoming[c].fillna("").astype(str).str.strip()
            if c in existing.columns:
                if pd.api.types.is_numeric_dtype(existing[c]) or c.endswith("Num") or c in {"Carry", "Ball Speed", "Club Speed", "Smash Factor"}:
                    existing[c] = _safe_num(existing[c]).round(3)
                else:
                    existing[c] = existing[c].fillna("").astype(str).str.strip()
        existing_keys = set(pd.util.hash_pandas_object(existing[key_cols], index=False).astype(str)) if not existing.empty else set()
        incoming_hash = pd.util.hash_pandas_object(incoming[key_cols], index=False).astype(str)
        keep_mask = ~incoming_hash.isin(existing_keys)
        to_add = incoming.loc[keep_mask].copy()
    else:
        to_add = incoming.copy()

    if existing.empty:
        combined = incoming.copy()
    else:
        aligned_cols = list(dict.fromkeys(list(existing.columns) + list(incoming.columns)))
        combined = pd.concat([existing.reindex(columns=aligned_cols), to_add.reindex(columns=aligned_cols)], ignore_index=True)

    combined.to_csv(master_path, index=False)
    return int(len(to_add)), int(len(incoming))


def build_time_key(df):
    d = df.copy()
    if "Session Label" not in d.columns:
        d["Session Label"] = "Session"
    label = d["Session Label"].fillna("").astype(str)
    dt = pd.to_datetime(label, errors="coerce")
    if dt.notna().any():
        d["_TimeKey"] = dt
        d["_TimeLabel"] = dt.dt.strftime("%Y-%m-%d")
    else:
        cats = pd.Categorical(label)
        d["_TimeKey"] = pd.Series(cats.codes, index=d.index)
        d["_TimeLabel"] = label.replace("", "Session")
    return d


def build_gap_optimizer(gtab):
    out = []
    if gtab is None or gtab.empty:
        return pd.DataFrame(columns=["Club", "Next Club", "Gap To Next", "Read", "Why"])
    for _, r in gtab.dropna(subset=["Gap To Next"]).iterrows():
        gap = r["Gap To Next"]
        if pd.isna(gap):
            continue
        if gap < 6:
            read = "Overlap"
            why = "These two clubs are living too close together. Either the loft gap is small, strike pattern overlaps, or one club is not producing enough separation."
        elif gap <= 15:
            read = "Healthy"
            why = "That is a pretty usable distance gap. It should create cleaner on-course decisions without forcing you to manufacture yardage."
        else:
            read = "Big jump"
            why = "This gap is wider than ideal. A loft tweak, different model, or a dedicated in-between club could eventually tighten the ladder."
        out.append({"Club": r["Club"], "Next Club": r["Next Club"], "Gap To Next": gap, "Read": read, "Why": why})
    return pd.DataFrame(out)


def best_strike_of_session(df):
    if df is None or df.empty:
        return None
    work = df.copy()
    for c in ["Ball Speed", "Smash Factor", "Carry", "Offline Num"]:
        if c not in work.columns:
            work[c] = np.nan
    work["bs_z"] = (_safe_num(work["Ball Speed"]) - _safe_num(work["Ball Speed"]).mean()) / (_safe_num(work["Ball Speed"]).std(ddof=0) or 1)
    work["sm_z"] = (_safe_num(work["Smash Factor"]) - _safe_num(work["Smash Factor"]).mean()) / (_safe_num(work["Smash Factor"]).std(ddof=0) or 1)
    carry = _safe_num(work["Carry"])
    work["carry_center"] = 1 - ((carry - carry.median()).abs() / (carry.std(ddof=0) or 1)).clip(lower=0)
    work["offline_center"] = 1 - (_safe_num(work["Offline Num"]).abs() / (_safe_num(work["Offline Num"]).std(ddof=0) or 1)).clip(lower=0)
    work["Strike Score"] = work[["bs_z", "sm_z", "carry_center", "offline_center"]].fillna(0).sum(axis=1)
    idx = work["Strike Score"].idxmax()
    if pd.isna(idx):
        return None
    return work.loc[idx].to_dict()


def ellipse_points(df, xcol="Offline Num", ycol="Carry", n=120):
    d = df[[xcol, ycol]].copy()
    d[xcol] = _safe_num(d[xcol])
    d[ycol] = _safe_num(d[ycol])
    d = d.dropna()
    if len(d) < 3:
        return pd.DataFrame(columns=["x", "y"])
    cov = np.cov(d[xcol], d[ycol])
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    theta = np.linspace(0, 2*np.pi, n)
    circle = np.array([np.cos(theta), np.sin(theta)])
    scale = 1.8  # about 1.8 std gives a nice readable shot cloud envelope
    ell = vecs @ np.diag(np.sqrt(np.maximum(vals, 1e-9)) * scale) @ circle
    center = np.array([d[xcol].mean(), d[ycol].mean()]).reshape(2, 1)
    pts = ell + center
    return pd.DataFrame({"x": pts[0], "y": pts[1]})


def confidence_circle_rings(df, offline_col="Offline Num", carry_col="Carry", target_offline=0.0, target_carry=None):
    cols = [offline_col, carry_col]
    d = df[cols].copy()
    d[offline_col] = _safe_num(d[offline_col])
    d[carry_col] = _safe_num(d[carry_col])
    d = d.dropna()
    if d.empty:
        return pd.DataFrame(columns=["Ring", "Radius", "Pct"]), pd.DataFrame(columns=["x", "y", "Ring"]), pd.DataFrame(columns=["x", "y"])
    if target_carry is None or pd.isna(target_carry):
        target_carry = d[carry_col].median()
    d["dx"] = d[offline_col] - float(target_offline)
    d["dy"] = d[carry_col] - float(target_carry)
    d["r"] = np.sqrt(d["dx"]**2 + d["dy"]**2)
    ring_specs = [("50%", 0.50), ("80%", 0.80), ("95%", 0.95)]
    ring_rows = []
    circle_rows = []
    theta = np.linspace(0, 2 * np.pi, 181)
    for label, pct in ring_specs:
        radius = float(np.nanpercentile(d["r"], pct * 100)) if len(d) else np.nan
        ring_rows.append({"Ring": label, "Radius": radius, "Pct": pct})
        for t in theta:
            circle_rows.append({"x": np.cos(t) * radius, "y": np.sin(t) * radius, "Ring": label})
    pts = d[["dx", "dy"]].rename(columns={"dx": "x", "dy": "y"}).copy()
    return pd.DataFrame(ring_rows), pd.DataFrame(circle_rows), pts


def confidence_circle_insights(rings_df):
    if rings_df is None or rings_df.empty:
        return []
    r50 = float(rings_df.loc[rings_df["Ring"] == "50%", "Radius"].iloc[0]) if (rings_df["Ring"] == "50%").any() else np.nan
    r80 = float(rings_df.loc[rings_df["Ring"] == "80%", "Radius"].iloc[0]) if (rings_df["Ring"] == "80%").any() else np.nan
    r95 = float(rings_df.loc[rings_df["Ring"] == "95%", "Radius"].iloc[0]) if (rings_df["Ring"] == "95%").any() else np.nan
    out = []
    if not pd.isna(r50):
        out.append(("50% circle", f"Half of your shots finish within {r50:.1f} yards of your stock target."))
    if not pd.isna(r80):
        out.append(("80% circle", f"About four out of five shots finish within {r80:.1f} yards. That is a strong on-course expectation band."))
    if not pd.isna(r95):
        out.append(("95% circle", f"Almost every shot sits within {r95:.1f} yards, so that is your real outer cone for strategy."))
    if not pd.isna(r80):
        for diam in [20, 25, 30]:
            prob = (r80 <= diam / 2)
            label = "Likely playable" if prob else "Needs caution"
            out.append((f"{diam}-yd green", f"{label}: an {diam}-yard green has a radius of {diam/2:.1f} yards versus your 80% circle of {r80:.1f}."))
    return out


def modeled_birdie_putt_make_pct(feet):
    if pd.isna(feet):
        return np.nan
    feet = float(feet)
    if feet <= 3:
        return 0.96
    if feet <= 6:
        return 0.72
    if feet <= 10:
        return 0.42
    if feet <= 15:
        return 0.24
    if feet <= 20:
        return 0.16
    if feet <= 30:
        return 0.09
    if feet <= 40:
        return 0.06
    return 0.03


def modeled_bogey_risk_pct(feet):
    if pd.isna(feet):
        return np.nan
    feet = float(feet)
    if feet <= 6:
        return 0.01
    if feet <= 10:
        return 0.02
    if feet <= 15:
        return 0.04
    if feet <= 25:
        return 0.07
    if feet <= 35:
        return 0.10
    if feet <= 45:
        return 0.13
    if feet <= 60:
        return 0.16
    return 0.20


def build_approach_proximity_model(df, low_carry, high_carry, target_carry=None):
    needed = {"Carry", "Offline Num", "Club"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return {}, pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d["Carry"] = _safe_num(d["Carry"])
    d["Offline Num"] = _safe_num(d["Offline Num"])
    d = d.dropna(subset=["Carry", "Offline Num"])
    d = d[(d["Carry"] >= float(low_carry)) & (d["Carry"] <= float(high_carry))].copy()
    if d.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    if target_carry is None or pd.isna(target_carry):
        target_carry = (float(low_carry) + float(high_carry)) / 2.0
    d["Target Carry"] = float(target_carry)
    d["Carry Delta"] = d["Carry"] - float(target_carry)
    d["Proximity Yds"] = np.sqrt(d["Offline Num"]**2 + d["Carry Delta"]**2)
    d["Proximity Ft"] = d["Proximity Yds"] * 3.0
    d["Birdie Model %"] = d["Proximity Ft"].apply(lambda x: modeled_birdie_putt_make_pct(x) * 100 if pd.notna(x) else np.nan)
    d["Bogey Model %"] = d["Proximity Ft"].apply(lambda x: modeled_bogey_risk_pct(x) * 100 if pd.notna(x) else np.nan)
    summary = {
        "Shots": int(len(d)),
        "Target Carry": float(target_carry),
        "Avg Proximity Ft": float(d["Proximity Ft"].mean()),
        "Median Proximity Ft": float(d["Proximity Ft"].median()),
        "Birdie Chance %": float(d["Birdie Model %"].mean()),
        "Bogey Risk %": float(d["Bogey Model %"].mean()),
        "Best Leave Ft": float(d["Proximity Ft"].min()),
    }
    club_table = (
        d.groupby("Club", dropna=False)
        .agg(
            Shots=("Club", "size"),
            AvgProximityFt=("Proximity Ft", "mean"),
            MedianProximityFt=("Proximity Ft", "median"),
            BirdieChancePct=("Birdie Model %", "mean"),
            BogeyRiskPct=("Bogey Model %", "mean"),
        )
        .reset_index()
        .sort_values(["MedianProximityFt", "AvgProximityFt", "Shots"], ascending=[True, True, False])
    )
    return summary, d, club_table


def build_wedge_distance_control_model(df, low_carry, high_carry, target_carry=None):
    needed = {"Carry", "Club"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return {}, pd.DataFrame(), pd.DataFrame()
    d = df.copy()
    d["Carry"] = _safe_num(d["Carry"])
    d = d.dropna(subset=["Carry"])
    d = d[(d["Carry"] >= float(low_carry)) & (d["Carry"] <= float(high_carry))].copy()
    if d.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    if target_carry is None or pd.isna(target_carry):
        target_carry = (float(low_carry) + float(high_carry)) / 2.0
    d["Target Carry"] = float(target_carry)
    d["Carry Delta"] = d["Carry"] - float(target_carry)
    d["Abs Carry Delta"] = d["Carry Delta"].abs()
    avg_carry = float(d["Carry"].mean())
    std_carry = float(d["Carry"].std(ddof=0)) if len(d) > 1 else 0.0
    mean_abs_delta = float(d["Abs Carry Delta"].mean())
    p80 = float(d["Abs Carry Delta"].quantile(0.8)) if len(d) > 1 else mean_abs_delta
    score = 10 - (std_carry * 0.45 + mean_abs_delta * 0.20)
    score = max(1.0, min(10.0, score))
    summary = {
        "Shots": int(len(d)),
        "Target Carry": float(target_carry),
        "Carry Avg": avg_carry,
        "Carry Median": float(d["Carry"].median()),
        "Carry Std": std_carry,
        "Mean Abs Delta": mean_abs_delta,
        "80% Miss Window": p80,
        "Distance Control Score": float(score),
        "Best Shot Delta": float(d["Abs Carry Delta"].min()),
    }
    club_table = (
        d.groupby("Club", dropna=False)
        .agg(
            Shots=("Club", "size"),
            AvgCarry=("Carry", "mean"),
            MedianCarry=("Carry", "median"),
            CarryStd=("Carry", lambda s: float(_safe_num(s).std(ddof=0)) if len(_safe_num(s).dropna()) > 1 else 0.0),
            MeanAbsDelta=("Abs Carry Delta", "mean"),
        )
        .reset_index()
        .sort_values(["MeanAbsDelta", "CarryStd", "Shots"], ascending=[True, True, False])
    )
    return summary, d, club_table


def build_approach_scoring_expectation(df, low_carry, high_carry, target_carry=None, hole_par=4):
    summary, d, club_table = build_approach_proximity_model(df, low_carry, high_carry, target_carry)
    if d.empty:
        return {}, pd.DataFrame(), pd.DataFrame()
    d = d.copy()
    d["Birdie P"] = _safe_num(d["Birdie Model %"]) / 100.0
    d["Bogey P"] = _safe_num(d["Bogey Model %"]) / 100.0
    d["Par P"] = (1.0 - d["Birdie P"] - d["Bogey P"]).clip(lower=0.0, upper=1.0)
    denom = d[["Birdie P", "Par P", "Bogey P"]].sum(axis=1).replace(0, np.nan)
    d["Birdie P"] = d["Birdie P"] / denom
    d["Par P"] = d["Par P"] / denom
    d["Bogey P"] = d["Bogey P"] / denom
    d["Expected Score"] = (hole_par - 1) * d["Birdie P"] + hole_par * d["Par P"] + (hole_par + 1) * d["Bogey P"]
    summary = {
        "Shots": int(len(d)),
        "Target Carry": float(summary.get("Target Carry", target_carry if target_carry is not None else (float(low_carry)+float(high_carry))/2.0)),
        "Avg Proximity Ft": float(d["Proximity Ft"].mean()),
        "Birdie %": float(d["Birdie P"].mean() * 100.0),
        "Par %": float(d["Par P"].mean() * 100.0),
        "Bogey+ %": float(d["Bogey P"].mean() * 100.0),
        "Expected Score": float(d["Expected Score"].mean()),
        "Hole Par": int(hole_par),
    }
    club_table = (
        d.groupby("Club", dropna=False)
        .agg(
            Shots=("Club", "size"),
            AvgProximityFt=("Proximity Ft", "mean"),
            BirdiePct=("Birdie P", lambda s: float(_safe_num(s).mean() * 100.0)),
            ParPct=("Par P", lambda s: float(_safe_num(s).mean() * 100.0)),
            BogeyPct=("Bogey P", lambda s: float(_safe_num(s).mean() * 100.0)),
            ExpectedScore=("Expected Score", "mean"),
        )
        .reset_index()
        .sort_values(["ExpectedScore", "AvgProximityFt", "Shots"], ascending=[True, True, False])
    )
    return summary, d, club_table


def trend_summary(df):
    if df is None or df.empty:
        return pd.DataFrame()
    d = build_time_key(df)
    group_cols = ["_TimeKey", "_TimeLabel"]
    if "Player" in d.columns:
        group_cols.append("Player")
    out = (
        d.groupby(group_cols, dropna=False)
        .agg(
            Shots=("Club", "size"),
            Carry=("Carry", "median"),
            BallSpeed=("Ball Speed", "median"),
            Smash=("Smash Factor", "median"),
            Offline=("Offline Num", lambda s: _safe_num(s).abs().median()),
        )
        .reset_index()
        .sort_values("_TimeKey")
    )
    return out


def trend_delta_text(series):
    s = _safe_num(series).dropna()
    if len(s) < 2:
        return "Need at least two sessions."
    delta = s.iloc[-1] - s.iloc[0]
    sign = "+" if delta > 0 else ""
    return f"{sign}{delta:.1f}"



def render_trivia_block(page_key, heading="Fun trivia corner"):
    state_key = f"trivia_idx_{page_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = random.randrange(len(TRIVIA_FACTS))
    c1, c2 = st.columns([6,1])
    with c1:
        st.markdown(f"#### {heading}")
    with c2:
        if st.button("New fact", key=f"btn_{page_key}"):
            st.session_state[state_key] = random.randrange(len(TRIVIA_FACTS))
    cat, fact = TRIVIA_FACTS[st.session_state[state_key]]
    st.markdown(f"<div class='trivia-card'><div class='trivia-cat'>{cat}</div><div class='trivia-text'>{fact}</div></div>", unsafe_allow_html=True)


st.markdown(
    """
    <style>
    .club-card {
        background: linear-gradient(180deg, #10131a 0%, #171b23 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 14px 16px;
        min-height: 158px;
        box-shadow: 0 8px 22px rgba(0,0,0,.16);
        margin-bottom: 10px;
    }
    .club-top {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom: 10px;
    }
    .club-name {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .club-shots {
        font-size: 0.82rem;
        opacity: 0.75;
    }
    .club-stock {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 8px;
    }
    .club-unit {
        font-size: .95rem;
        font-weight: 600;
        opacity: .8;
    }
    .club-sub {
        font-size: .9rem;
        margin-bottom: 4px;
        opacity: .92;
    }
    .club-pattern {
        margin-top: 10px;
        font-size: .88rem;
        opacity: .92;
    }
    .note-box {
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        margin-bottom: 10px;
    }
    .analysis-stat-card {
        background: linear-gradient(180deg, #10131a 0%, #171b23 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        padding: 12px 14px;
        min-height: 104px;
        box-shadow: 0 8px 22px rgba(0,0,0,.14);
        margin-bottom: 8px;
    }
    .analysis-stat-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: rgba(255,255,255,.68);
        margin-bottom: 8px;
        font-weight: 700;
    }
    .analysis-stat-value {
        font-size: 1.55rem;
        line-height: 1.12;
        font-weight: 800;
        margin-bottom: 6px;
        word-break: break-word;
    }
    .analysis-stat-value.pattern {
        font-size: 1.05rem;
        font-weight: 700;
    }
    .analysis-stat-note {
        font-size: 0.84rem;
        line-height: 1.28;
        color: rgba(255,255,255,.72);
    }
    .hero-card {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 18px;
        padding: 14px 16px;
        min-height: 146px;
        box-shadow: 0 10px 24px rgba(0,0,0,.14);
        margin-bottom: 12px;
    }
    .hero-title {font-size: .85rem; opacity: .72; margin-bottom: 8px;}
    .hero-value {font-size: 1.9rem; font-weight: 800; line-height:1.05; margin-bottom:6px;}
    .hero-sub {font-size: .92rem; opacity:.88;}
    .rank-card {
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .rank-label {font-size: .8rem; opacity: .72; text-transform: uppercase; letter-spacing: .04em;}
    .rank-club {font-size: 1.12rem; font-weight: 700; margin: 4px 0 6px 0;}
    .rank-why {font-size: .9rem; opacity: .92; line-height: 1.35;}
    .bench-chip {
        display:inline-block; padding: 4px 8px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.05);
        margin-right: 6px; margin-bottom: 6px; font-size: .82rem;
    }
    .trivia-card {
        background: linear-gradient(180deg, rgba(49,46,129,.28) 0%, rgba(17,24,39,.78) 100%);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 16px;
        padding: 14px 16px;
        margin-top: 8px;
        margin-bottom: 10px;
    }
    .trivia-cat {
        font-size: .78rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        opacity: .72;
        margin-bottom: 8px;
    }
    .trivia-text {
        font-size: 1rem;
        line-height: 1.45;
        font-weight: 500;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⛳ Launch Monitor Analyzer")
st.caption("Auto-loads Launch.csv, keeps the dispersion views, upgrades the carry section, and adds a benchmark comparison tab.")

with st.sidebar:
    st.header("Data Source")
    auto_file = find_auto_file()
    source_mode = st.radio(
        "How should the app load data?",
        ["Auto-load Launch.csv", "Upload and analyze only", "Upload and append into Launch.csv"],
        index=0,
    )

    shots = None
    source_label = None

    if source_mode == "Auto-load Launch.csv":
        if auto_file is None:
            st.warning("No Launch.csv file was found next to the app or in the working folder.")
        else:
            shots = prepare_data_from_path(str(auto_file))
            source_label = str(auto_file)
            st.success(f"Loaded: {Path(auto_file).name}")
    elif source_mode == "Upload and analyze only":
        uploaded = st.file_uploader("Upload one master CSV file", type=["csv"])
        if uploaded is not None:
            shots = prepare_data_from_bytes(uploaded.getvalue(), uploaded.name)
            source_label = uploaded.name
    else:
        uploaded = st.file_uploader("Upload session CSV to append into Launch.csv", type=["csv"])
        target_path = Path.cwd() / MASTER_OUTPUT_FILE
        st.caption(f"New sessions will be appended into: {target_path}")
        if uploaded is not None:
            incoming = prepare_data_from_bytes(uploaded.getvalue(), uploaded.name)
            added, incoming_total = append_to_master_csv(incoming, target_path)
            st.success(f"Added {added} new shots out of {incoming_total} parsed rows into {target_path.name}.")
            shots = prepare_data_from_path(str(target_path))
            source_label = str(target_path)

if shots is None:
    st.info("Put Launch.csv in the same folder as this app, or choose a file from the sidebar.")
    st.stop()

shots = ensure_shot_schema(shots)

if shots.empty:
    st.error("Could not parse any shot rows from that file.")
    st.stop()

st.caption(f"Current source: {source_label}")

metrics = available_metrics(shots)
if not metrics:
    st.error("No usable numeric metrics were found in this file.")
    st.stop()

st.subheader("Filters")
f1, f2, f3, f4, f5, f6 = st.columns(6)
players = ["All Players"] + sorted(shots["Player"].dropna().astype(str).unique().tolist())
sessions = ["All Sessions"] + sorted(shots["Session Label"].dropna().astype(str).unique().tolist())
clubs = ["All Clubs"] + sorted(
    shots["Club"].dropna().astype(str).unique().tolist(),
    key=lambda x: (shots.loc[shots["Club"] == x, "Club Sort"].min(), str(x)),
)

with f1:
    player_sel = st.selectbox("Player", players)
with f2:
    session_sel = st.selectbox("Session / Date", sessions)
with f3:
    club_sel = st.selectbox("Club", clubs)
with f4:
    quality_sel = st.selectbox("Shot Set", ["Clean Shots Only", "All Shots"])
with f5:
    metric_sel = st.selectbox("Analysis Metric", metrics, index=0)
with f6:
    benchmark_sel = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=1)

filtered = shots.copy()
if player_sel != "All Players":
    filtered = filtered[filtered["Player"] == player_sel]
if session_sel != "All Sessions":
    filtered = filtered[filtered["Session Label"] == session_sel]
if club_sel != "All Clubs":
    filtered = filtered[filtered["Club"] == club_sel]
if quality_sel == "Clean Shots Only":
    filtered = filtered[filtered["Is Clean Shot"]]

if filtered.empty:
    st.warning("No shots match the current filters.")
    st.stop()

all_count = len(shots)
clean_count = int(shots["Is Clean Shot"].sum())
filtered_count = len(filtered)
metric_series = pd.to_numeric(filtered[metric_sel], errors="coerce") if metric_sel in filtered.columns else pd.Series(dtype=float)
metric_fmt = nice_format(metric_sel)
metric_median = metric_series.median() if not metric_series.dropna().empty else np.nan

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Shots in file", f"{all_count}")
m2.metric("Clean shots", f"{clean_count}")
m3.metric("Shots in current view", f"{filtered_count}")
m4.metric("Exclusion rate", f"{pct(all_count-clean_count, all_count):.1f}%")
m5.metric(f"Median {metric_sel}", "—" if pd.isna(metric_median) else format(metric_median, metric_fmt))

club_summary = (
    filtered.groupby(["Club", "Club Clean", "Club Sort"], dropna=False)
    .agg(
        Shots=("Club", "size"),
        CarryP10=("Carry", lambda s: s.quantile(0.10)),
        CarryP25=("Carry", lambda s: s.quantile(0.25)),
        MedianCarry=("Carry", "median"),
        CarryP75=("Carry", lambda s: s.quantile(0.75)),
        CarryP90=("Carry", lambda s: s.quantile(0.90)),
        AvgCarry=("Carry", "mean"),
        CarryStd=("Carry", "std"),
        AvgOffline=("Offline Num", "mean"),
        BallSpeed=("Ball Speed", "mean"),
        Smash=("Smash Factor", "mean"),
        MetricMean=(metric_sel, "mean"),
        MetricMedian=(metric_sel, "median"),
        MetricStd=(metric_sel, "std"),
        MetricP25=(metric_sel, lambda s: s.quantile(0.25)),
        MetricP75=(metric_sel, lambda s: s.quantile(0.75)),
    )
    .reset_index()
    .sort_values(["Club Sort", "Club"])
)

if club_summary.empty:
    st.warning("No summary rows available for the current filters.")
    st.stop()

cons_map = filtered.groupby("Club").apply(consistency_score).to_dict()
pat_map = filtered.groupby("Club").apply(pattern_label).to_dict()
club_summary["Consistency"] = club_summary["Club"].map(cons_map)
club_summary["Pattern"] = club_summary["Club"].map(pat_map)

base_df = baseline_table_for_set(benchmark_sel)
base_carry_map = base_df.set_index("Club Clean")["Carry"].to_dict()
club_summary["Baseline Carry"] = club_summary["Club Clean"].map(base_carry_map)
base_smash_map = base_df.set_index("Club Clean")["Smash Factor"].to_dict() if "Smash Factor" in base_df.columns else {}
club_summary["Baseline Smash"] = club_summary["Club Clean"].map(base_smash_map)
club_summary["Carry Delta vs Benchmark"] = club_summary["MedianCarry"] - club_summary["Baseline Carry"]

summary_show = club_summary.copy()
for c in [
    "CarryP10", "CarryP25", "MedianCarry", "CarryP75", "CarryP90", "AvgCarry", "CarryStd",
    "AvgOffline", "BallSpeed", "Smash", "Consistency", "MetricMean", "MetricMedian", "MetricStd",
    "MetricP25", "MetricP75", "Baseline Carry", "Baseline Smash", "Carry Delta vs Benchmark",
]:
    if c in summary_show.columns:
        summary_show[c] = summary_show[c].round(2 if c in {"Smash", "Baseline Smash"} else 1)

best_three, worst_three = build_best_worst_lists(club_summary)

overview_tab, analysis_tab, progress_tab, benchmark_tab, data_tab = st.tabs(["Overview", "Analysis Lab", "Progress Over Time", "Benchmark Analysis", "Shot Tables"])

with overview_tab:
    st.subheader("At-a-Glance Read")
    h1, h2, h3 = st.columns(3)
    best_name = best_three.iloc[0]["Club"] if not best_three.empty else "—"
    worst_name = worst_three.iloc[0]["Club"] if not worst_three.empty else "—"
    stock_gap = club_summary["MedianCarry"].max() - club_summary["MedianCarry"].min() if len(club_summary) > 1 else np.nan
    with h1:
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Best current gamer</div><div class='hero-value'>{best_name}</div><div class='hero-sub'>{best_three.iloc[0]['Why'] if not best_three.empty else 'Need more data.'}</div></div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Needs the most love</div><div class='hero-value'>{worst_name}</div><div class='hero-sub'>{worst_three.iloc[0]['Why'] if not worst_three.empty else 'Need more data.'}</div></div>",
            unsafe_allow_html=True,
        )
    with h3:
        sub = "Need at least two clubs." if pd.isna(stock_gap) else f"{stock_gap:.0f} yds from shortest median to longest median in this filtered view."
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Current gapping window</div><div class='hero-value'>{'—' if pd.isna(stock_gap) else f'{stock_gap:.0f} yds'}</div><div class='hero-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Stock Carry Cards")
    st.markdown(
        "<div class='note-box'>This version makes stock carry the star and pushes the range into a cleaner horizontal view. Median carry is the main number. The range bar is your middle 50% window. Average carry is there for context only.</div>",
        unsafe_allow_html=True,
    )

    rows = [club_summary.iloc[i:i+4] for i in range(0, len(club_summary), 4)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, r) in zip(cols, chunk.iterrows()):
            with col:
                st.markdown(
                    metric_card_html(
                        r["Club"],
                        r["MedianCarry"],
                        r["CarryP25"],
                        r["CarryP75"],
                        r["Shots"],
                        r["Consistency"],
                        r["Pattern"],
                        r.get("Baseline Carry", np.nan),
                    ),
                    unsafe_allow_html=True,
                )

    st.subheader("Carry Window by Club")
    carry_plot_df = club_summary.sort_values(["Club Sort", "Club"], ascending=[False, True]).copy()
    carry_base = alt.Chart(carry_plot_df)
    range_bar = carry_base.mark_bar(size=16, cornerRadius=8, opacity=0.42).encode(
        y=alt.Y("Club:N", sort=None, title="Club"),
        x=alt.X("CarryP25:Q", title="Carry Distance (yds)"),
        x2="CarryP75:Q",
        tooltip=[
            "Club",
            alt.Tooltip("MedianCarry:Q", title="Stock Carry", format=".0f"),
            alt.Tooltip("CarryP25:Q", title="25th %", format=".0f"),
            alt.Tooltip("CarryP75:Q", title="75th %", format=".0f"),
            alt.Tooltip("AvgCarry:Q", title="Average", format=".1f"),
            alt.Tooltip("Carry Delta vs Benchmark:Q", title="vs Benchmark", format="+.0f"),
        ],
    )
    median_dot = carry_base.mark_circle(size=220, filled=True).encode(
        y=alt.Y("Club:N", sort=None),
        x=alt.X("MedianCarry:Q"),
        color=alt.Color("Club:N", legend=None),
        tooltip=["Club", alt.Tooltip("MedianCarry:Q", title="Stock Carry", format=".0f")],
    )
    avg_tick = carry_base.mark_tick(size=24, thickness=2, color="white").encode(
        y=alt.Y("Club:N", sort=None),
        x="AvgCarry:Q",
        tooltip=["Club", alt.Tooltip("AvgCarry:Q", title="Average Carry", format=".1f")],
    )
    carry_label = carry_base.mark_text(dx=22, fontSize=12, align="left", fontWeight="bold").encode(
        y=alt.Y("Club:N", sort=None),
        x="MedianCarry:Q",
        text=alt.Text("MedianCarry:Q", format=".0f"),
    )
    st.altair_chart((range_bar + median_dot + avg_tick + carry_label).properties(height=max(260, 42 * len(carry_plot_df))), use_container_width=True)
    st.caption("Dot = stock carry. Thick bar = middle 50% carry window. White tick = average carry. This should read closer to a real club-gapping board.")

    st.subheader("What you hit best right now")
    if best_three.empty:
        st.info("Need more club data to rank your best clubs.")
    else:
        cols = st.columns(len(best_three))
        medals = ["🥇", "🥈", "🥉"]
        for col, (_, r), medal in zip(cols, best_three.iterrows(), medals):
            with col:
                st.markdown(
                    f"<div class='rank-card'><div class='rank-label'>{medal} Best club</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>Stock carry: {r['MedianCarry']:.0f} yds</div><div class='bench-chip'>Consistency: {r['Consistency']:.1f}/10</div><div class='bench-chip'>Smash: {r['Smash']:.2f}</div></div>",
                    unsafe_allow_html=True,
                )

    st.subheader("What needs work")
    if worst_three.empty:
        st.info("Need more club data to rank the weaker clubs.")
    else:
        cols = st.columns(len(worst_three))
        badges = ["🛠️ 1", "🛠️ 2", "🛠️ 3"]
        for col, (_, r), badge in zip(cols, worst_three.iterrows(), badges):
            with col:
                st.markdown(
                    f"<div class='rank-card'><div class='rank-label'>{badge} Focus club</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>Carry SD: {0 if pd.isna(r['CarryStd']) else r['CarryStd']:.1f}</div><div class='bench-chip'>Offline avg: {0 if pd.isna(r['AvgOffline']) else r['AvgOffline']:+.1f}</div><div class='bench-chip'>vs benchmark: {0 if pd.isna(r['Carry Delta vs Benchmark']) else r['Carry Delta vs Benchmark']:+.0f}</div></div>",
                    unsafe_allow_html=True,
                )


    st.subheader("Best strike of the session")
    strike = best_strike_of_session(filtered)
    if not strike:
        st.info("Need a few more usable shots before I can confidently flag the best strike.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Club", strike.get("Club", "—"))
        c2.metric("Carry", "—" if pd.isna(strike.get("Carry", np.nan)) else f"{float(strike.get('Carry')):.1f} yds")
        c3.metric("Ball Speed", "—" if pd.isna(strike.get("Ball Speed", np.nan)) else f"{float(strike.get('Ball Speed')):.1f} mph")
        c4.metric("Smash", "—" if pd.isna(strike.get("Smash Factor", np.nan)) else f"{float(strike.get('Smash Factor')):.2f}")
        st.markdown(
            "<div class='note-box'>Best strike blends strong ball speed, efficient smash, carry near your playable stock window, and a start line that stayed near center. It is not just the longest shot — it is the best all-around gamer swing in this filtered set.</div>",
            unsafe_allow_html=True,
        )

    st.subheader(f"{metric_sel} Analysis")
    metric_df = filtered.dropna(subset=[metric_sel]).copy() if metric_sel in filtered.columns else pd.DataFrame()
    if metric_df.empty:
        st.info(f"No usable {metric_sel} values are available for the current selection.")
    else:
        left_metric, right_metric = st.columns(2)

        metric_by_club = alt.Chart(club_summary).mark_bar(size=24, cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.65).encode(
            x=alt.X("Club:N", sort=None, title="Club"),
            y=alt.Y("MetricMedian:Q", title=metric_axis_label(metric_sel)),
            color=alt.Color("Club:N", legend=None),
            tooltip=[
                "Club",
                alt.Tooltip("MetricMedian:Q", title="Median", format=metric_fmt),
                alt.Tooltip("MetricMean:Q", title="Mean", format=metric_fmt),
                alt.Tooltip("MetricP25:Q", title="25th %", format=metric_fmt),
                alt.Tooltip("MetricP75:Q", title="75th %", format=metric_fmt),
            ],
        )
        metric_text = alt.Chart(club_summary).mark_text(dy=-10, fontSize=12, fontWeight="bold").encode(
            x=alt.X("Club:N", sort=None),
            y=alt.Y("MetricMedian:Q"),
            text=alt.Text("MetricMedian:Q", format=metric_fmt),
        )

        with left_metric:
            st.markdown("**Metric by Club**")
            st.altair_chart((metric_by_club + metric_text).properties(height=340), use_container_width=True)

        compare_field = "Session Label" if session_sel == "All Sessions" else "Club"
        with right_metric:
            st.markdown(f"**{metric_sel} Distribution by {compare_field}**")
            metric_box = alt.Chart(metric_df).mark_boxplot(size=40).encode(
                x=alt.X(f"{compare_field}:N", title=compare_field, sort=None),
                y=alt.Y(f"{metric_sel}:Q", title=metric_axis_label(metric_sel)),
                color=alt.Color(f"{compare_field}:N", legend=None),
                tooltip=[compare_field],
            ).properties(height=340)
            st.altair_chart(metric_box, use_container_width=True)

    st.subheader("Dispersion")
    left_col, right_col = st.columns(2)

    has_offline = "Offline Num" in filtered.columns and filtered["Offline Num"].notna().any()
    has_carry = "Carry" in filtered.columns and filtered["Carry"].notna().any()

    with left_col:
        st.markdown("**Left / Right Pattern**")
        if has_offline:
            offline_df = filtered.dropna(subset=["Offline Num"]).copy()
            scatter_lr = alt.Chart(offline_df).mark_circle(size=82, opacity=0.68).encode(
                x=alt.X("Offline Num:Q", title="Offline (Left negative / Right positive)"),
                y=alt.Y("jitter:Q", axis=None),
                color=alt.Color("Club:N", legend=None),
                tooltip=[
                    "Player", "Club", "Session Label",
                    alt.Tooltip("Offline Num:Q", format=".1f"),
                    alt.Tooltip("Carry:Q", format=".1f"),
                    alt.Tooltip(metric_sel + ":Q", title=metric_sel, format=metric_fmt),
                ],
            ).transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
            zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], size=2).encode(x="x:Q")
            st.altair_chart((scatter_lr + zero_rule).properties(height=320), use_container_width=True)
        else:
            st.info("No usable offline column was found for this selection.")

    with right_col:
        st.markdown("**Carry + Direction Map**")
        if has_offline and has_carry:
            disp_df = filtered.dropna(subset=["Offline Num", "Carry"]).copy()
            color_field = "Club" if club_sel == "All Clubs" else "Player"
            scatter = alt.Chart(disp_df).mark_circle(size=95, opacity=0.72).encode(
                x=alt.X("Offline Num:Q", title="Offline"),
                y=alt.Y("Carry:Q", title="Carry"),
                color=alt.Color(f"{color_field}:N", legend=alt.Legend(title=color_field)),
                tooltip=[
                    "Player", "Club", "Session Label",
                    alt.Tooltip("Carry:Q", format=".1f"),
                    alt.Tooltip("Offline Num:Q", format=".1f"),
                    alt.Tooltip(metric_sel + ":Q", title=metric_sel, format=metric_fmt),
                ],
            )
            ellipse_df = ellipse_points(disp_df, "Offline Num", "Carry")
            layers = [scatter]
            if not ellipse_df.empty:
                ellipse = alt.Chart(ellipse_df).mark_line(strokeWidth=3, opacity=0.85).encode(
                    x=alt.X("x:Q", title="Offline"),
                    y=alt.Y("y:Q", title="Carry"),
                )
                layers.append(ellipse)
            vline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], size=2).encode(x="x:Q")
            layers.append(vline)
            chart = layers[0]
            for lyr in layers[1:]:
                chart = chart + lyr
            st.altair_chart(chart.properties(height=320), use_container_width=True)
            st.caption("The outline is a shot-shape ellipse. It gives you the quick feel for how wide and how deep the pattern really is, without staring at every dot.")
        else:
            st.info("Need both carry and offline data for the combo dispersion view.")

    st.subheader("Distance Consistency")
    box_df = filtered.dropna(subset=["Carry"]).copy()
    if not box_df.empty:
        box = alt.Chart(box_df).mark_boxplot(size=38).encode(
            x=alt.X("Club:N", sort=None, title="Club"),
            y=alt.Y("Carry:Q", title="Carry Distance"),
            color=alt.Color("Club:N", legend=None),
            tooltip=["Club"],
        ).properties(height=340)
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("No carry data available for the current selection.")

    st.subheader("Club Summary Table")
    st.dataframe(
        summary_show[[
            "Club", "Shots", "CarryP25", "MedianCarry", "CarryP75", "Baseline Carry",
            "Carry Delta vs Benchmark", "MetricMedian", "MetricMean", "MetricStd",
            "BallSpeed", "Smash", "Consistency", "Pattern"
        ]],
        use_container_width=True,
        hide_index=True,
    )

    render_trivia_block("overview", "Random sports & film trivia")

with analysis_tab:
    st.subheader("Analysis Lab")
    s1, s2, s3 = st.columns(3)
    session_score = overall_session_score(club_summary)
    top_bias = "No offline data"
    pattern_note = "No usable offline column in this filtered slice."
    if "Offline Num" in filtered.columns and filtered["Offline Num"].dropna().any():
        off_mean = filtered["Offline Num"].dropna().mean()
        off_std = filtered["Offline Num"].dropna().std()
        if abs(off_mean) <= 3:
            top_bias = "🎯 Neutral start line overall"
            pattern_note = "Average start line is sitting close to center."
        elif off_mean > 3:
            top_bias = "➡️ Overall right bias"
            pattern_note = "Average finish location is leaking right of target."
        else:
            top_bias = "⬅️ Overall left bias"
            pattern_note = "Average finish location is finishing left of target."
        if pd.notna(off_std):
            pattern_note += f" Typical spread: {off_std:.1f} yds."

    if len(club_summary) >= 2:
        gtab = gap_table(club_summary)
        valid_gaps = gtab["Gap To Next"].dropna()
        gap_value = "—" if valid_gaps.empty else f"{valid_gaps.mean():.1f} yds"
        gap_note = "Need at least two clubs with carry data." if valid_gaps.empty else f"Across {len(valid_gaps)} measured club gaps in this view."
    else:
        gap_value = "—"
        gap_note = "Need at least two clubs with carry data."

    score_value = "—" if pd.isna(session_score) else f"{session_score:.1f}/10"
    score_note = "Blend of carry consistency, strike quality, and direction control."

    with s1:
        st.markdown(f"""
        <div class='analysis-stat-card'>
            <div class='analysis-stat-title'>Overall ball-striking score</div>
            <div class='analysis-stat-value'>{score_value}</div>
            <div class='analysis-stat-note'>{score_note}</div>
        </div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown(f"""
        <div class='analysis-stat-card'>
            <div class='analysis-stat-title'>Primary pattern</div>
            <div class='analysis-stat-value pattern'>{top_bias}</div>
            <div class='analysis-stat-note'>{pattern_note}</div>
        </div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown(f"""
        <div class='analysis-stat-card'>
            <div class='analysis-stat-title'>Avg gap</div>
            <div class='analysis-stat-value'>{gap_value}</div>
            <div class='analysis-stat-note'>{gap_note}</div>
        </div>
        """, unsafe_allow_html=True)

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Distance ladder**")
        ladder_df = club_summary.sort_values(["Club Sort", "Club"], ascending=[False, True]).copy()
        ladder = alt.Chart(ladder_df).mark_bar(size=18, cornerRadius=6).encode(
            y=alt.Y("Club:N", sort=None, title="Club"),
            x=alt.X("MedianCarry:Q", title="Stock Carry (yds)"),
            color=alt.Color("MedianCarry:Q", legend=None),
            tooltip=["Club", alt.Tooltip("MedianCarry:Q", title="Stock Carry", format=".0f"), alt.Tooltip("CarryP25:Q", title="25th %", format=".0f"), alt.Tooltip("CarryP75:Q", title="75th %", format=".0f")],
        )
        labels = alt.Chart(ladder_df).mark_text(align="left", dx=8, fontWeight="bold").encode(
            y=alt.Y("Club:N", sort=None), x="MedianCarry:Q", text=alt.Text("MedianCarry:Q", format=".0f")
        )
        st.altair_chart((ladder + labels).properties(height=max(260, 36 * len(ladder_df))), use_container_width=True)

        gap_df = gap_table(club_summary)[["Club", "Next Club", "Gap To Next"]].dropna().copy()
        if not gap_df.empty:
            gap_df["Gap Label"] = gap_df.apply(lambda r: f"{r['Club']} → {r['Next Club']}: {r['Gap To Next']:.0f} yds", axis=1)
            st.markdown("**Gap reads**")
            for _, r in gap_df.iterrows():
                st.markdown(f"<div class='bench-chip'>{r['Gap Label']}</div>", unsafe_allow_html=True)
            gap_reads = build_gap_optimizer(gap_df)
            st.markdown("**Gap optimizer**")
            for _, r in gap_reads.iterrows():
                st.markdown(
                    f"<div class='rank-card'><div class='rank-label'>{r['Read']}</div><div class='rank-club'>{r['Club']} → {r['Next Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>Current gap: {r['Gap To Next']:.0f} yds</div></div>",
                    unsafe_allow_html=True,
                )

    with right_col:
        st.markdown("**Shot pattern heatmap**")
        if has_offline and has_carry:
            heat_df = filtered.dropna(subset=["Offline Num", "Carry"]).copy()
            heat = alt.Chart(heat_df).mark_rect().encode(
                x=alt.X("Offline Num:Q", bin=alt.Bin(maxbins=18), title="Offline"),
                y=alt.Y("Carry:Q", bin=alt.Bin(maxbins=18), title="Carry"),
                color=alt.Color("count():Q", title="Shots"),
                tooltip=[alt.Tooltip("count():Q", title="Shots")],
            )
            vline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4,4], color="white").encode(x="x:Q")
            st.altair_chart((heat + vline).properties(height=360), use_container_width=True)
        else:
            st.info("Need offline and carry data for the heatmap.")

    st.subheader("Shot dispersion confidence circle")
    conf_clubs = [c for c in club_summary["Club"].dropna().astype(str).tolist() if c != ""]
    if conf_clubs and has_offline and has_carry:
        default_conf_idx = 0
        if club_sel != "All Clubs" and club_sel in conf_clubs:
            default_conf_idx = conf_clubs.index(club_sel)
        conf_club = st.selectbox("Confidence circle club", conf_clubs, index=default_conf_idx, key="conf_circle_club")
        conf_df = filtered[filtered["Club"] == conf_club].dropna(subset=["Offline Num", "Carry"]).copy()
        if len(conf_df) >= 5:
            target_carry = float(conf_df["Carry"].median())
            rings_df, circles_df, conf_pts = confidence_circle_rings(conf_df, target_offline=0.0, target_carry=target_carry)
            cc1, cc2 = st.columns([1.35, 1])
            with cc1:
                base = alt.Chart(conf_pts).mark_circle(size=85, opacity=0.6).encode(
                    x=alt.X("x:Q", title="Offline vs target (yds)"),
                    y=alt.Y("y:Q", title="Carry delta vs stock (yds)"),
                    tooltip=[alt.Tooltip("x:Q", title="Offline", format="+.1f"), alt.Tooltip("y:Q", title="Carry Δ", format="+.1f")],
                )
                ring_lines = alt.Chart(circles_df).mark_line(strokeWidth=2.5).encode(
                    x="x:Q",
                    y="y:Q",
                    detail="Ring:N",
                    color=alt.Color("Ring:N", sort=["50%", "80%", "95%"], legend=alt.Legend(title="Confidence rings")),
                    tooltip=["Ring:N"],
                )
                v0 = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5,5], opacity=0.6).encode(x="x:Q")
                h0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[5,5], opacity=0.6).encode(y="y:Q")
                st.altair_chart((ring_lines + base + v0 + h0).properties(height=440), use_container_width=True)
                st.caption(f"Centered on {conf_club}'s stock target: 0 yards offline and {target_carry:.0f} yards carry. The circles show where roughly 50%, 80%, and 95% of your shots finish.")
            with cc2:
                st.markdown(f"**{conf_club} target-read**")
                ring_show = rings_df.copy()
                ring_show["Radius"] = ring_show["Radius"].map(lambda x: round(float(x), 1) if pd.notna(x) else x)
                st.dataframe(ring_show[["Ring", "Radius"]], use_container_width=True, hide_index=True)
                for title, msg in confidence_circle_insights(rings_df):
                    st.markdown(f"<div class='rank-card'><div class='rank-label'>{title}</div><div class='rank-why'>{msg}</div></div>", unsafe_allow_html=True)
        else:
            st.info("Need at least 5 clean shots with carry and offline data for the confidence circle.")
    else:
        st.info("Need at least one club plus carry and offline data to build the confidence circle.")

    st.subheader("Metric explorer")
    exp_left, exp_right = st.columns(2)
    with exp_left:
        metric_df = filtered.dropna(subset=[metric_sel]).copy()
        if not metric_df.empty:
            metric_chart = alt.Chart(metric_df).mark_boxplot(size=34).encode(
                x=alt.X("Club:N", sort=None, title="Club"),
                y=alt.Y(f"{metric_sel}:Q", title=metric_axis_label(metric_sel)),
                color=alt.Color("Club:N", legend=None),
                tooltip=["Club"],
            ).properties(height=340)
            st.altair_chart(metric_chart, use_container_width=True)
        else:
            st.info("No values available for the selected metric.")
    with exp_right:
        metric_summary = club_summary[["Club", "Shots", "MetricMedian", "MetricMean", "MetricStd", "Pattern"]].copy()
        metric_summary = metric_summary.sort_values("MetricMedian", ascending=False)
        st.dataframe(metric_summary, use_container_width=True, hide_index=True)

    st.subheader("Wedge distance control")
    if has_carry:
        wedge_source = filtered.dropna(subset=["Carry"]).copy()
        if not wedge_source.empty:
            wedge_min_data = int(math.floor(float(wedge_source["Carry"].min()) / 5.0) * 5)
            wedge_max_data = int(math.ceil(float(wedge_source["Carry"].max()) / 5.0) * 5)
            wedge_max = min(140, wedge_max_data)
            wedge_min = min(wedge_min_data, wedge_max - 10)
            if wedge_max - wedge_min >= 10:
                default_wedge_low = max(wedge_min, min(40, wedge_max - 10))
                default_wedge_high = min(wedge_max, max(default_wedge_low + 10, min(100, wedge_max)))
                if default_wedge_high <= default_wedge_low:
                    default_wedge_high = min(wedge_max, default_wedge_low + 10)
                wa, wb = st.columns([1.2, 1])
                with wa:
                    wedge_bucket = st.slider(
                        "Wedge carry bucket (yds)",
                        min_value=int(wedge_min),
                        max_value=int(wedge_max),
                        value=(int(default_wedge_low), int(default_wedge_high)),
                        step=5,
                        key="wedge_control_bucket",
                    )
                with wb:
                    wedge_target = st.number_input(
                        "Wedge target carry",
                        min_value=float(wedge_min),
                        max_value=float(wedge_max),
                        value=float((default_wedge_low + default_wedge_high) / 2),
                        step=1.0,
                        key="wedge_control_target",
                    )
                wedge_summary, wedge_points, wedge_clubs = build_wedge_distance_control_model(
                    wedge_source, wedge_bucket[0], wedge_bucket[1], wedge_target
                )
                if wedge_points.empty:
                    st.info("No shots landed inside that wedge carry bucket.")
                else:
                    w1, w2, w3, w4 = st.columns(4)
                    w1.metric("Carry avg", f"{wedge_summary['Carry Avg']:.1f} yds")
                    w2.metric("Std dev", f"{wedge_summary['Carry Std']:.1f} yds")
                    w3.metric("Avg miss to target", f"{wedge_summary['Mean Abs Delta']:.1f} yds")
                    w4.metric("Distance control score", f"{wedge_summary['Distance Control Score']:.1f}/10")
                    st.caption("Modeled from carry only. Lower standard deviation and lower average miss to target produce a stronger wedge distance-control score.")
                    wx, wy = st.columns([1.2, 1])
                    with wx:
                        wedge_points = wedge_points.reset_index(drop=True).copy()
                        wedge_points["Shot #"] = np.arange(1, len(wedge_points) + 1)
                        wedge_chart = alt.Chart(wedge_points).mark_circle(size=90, opacity=0.72).encode(
                            x=alt.X("Shot #:Q", title="Shot sequence"),
                            y=alt.Y("Carry Delta:Q", title="Carry delta vs target (yds)"),
                            color=alt.Color("Abs Carry Delta:Q", title="Miss (yds)", scale=alt.Scale(scheme="teals")),
                            tooltip=[
                                "Club",
                                alt.Tooltip("Carry:Q", title="Carry", format=".1f"),
                                alt.Tooltip("Carry Delta:Q", title="Carry Δ", format="+.1f"),
                                alt.Tooltip("Abs Carry Delta:Q", title="Abs miss", format=".1f"),
                            ],
                        )
                        h0 = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[5, 5], opacity=0.6).encode(y="y:Q")
                        st.altair_chart((wedge_chart + h0).properties(height=360), use_container_width=True)
                    with wy:
                        st.markdown(f"**{int(wedge_bucket[0])}-{int(wedge_bucket[1])} yds wedge read**")
                        if wedge_summary['Distance Control Score'] >= 8:
                            wedge_read = "Elite distance control"
                            wedge_why = "The bucket is staying tight around the target, which is exactly what you want from scoring clubs."
                        elif wedge_summary['Distance Control Score'] >= 6:
                            wedge_read = "Playable scoring control"
                            wedge_why = "There is enough consistency here to attack flags when the lie is clean, but you still have some spread to tighten."
                        else:
                            wedge_read = "Needs tighter wedge windows"
                            wedge_why = "The average miss and overall spread are still wide enough that distance control is leaking strokes."
                        st.markdown(
                            f"<div class='rank-card'><div class='rank-label'>{wedge_read}</div><div class='rank-why'>{wedge_why}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='rank-card'><div class='rank-label'>Best shot to target</div><div class='rank-club'>{wedge_summary['Best Shot Delta']:.1f} yds off</div><div class='rank-why'>Closest carry to the selected target in this wedge bucket.</div></div>",
                            unsafe_allow_html=True,
                        )
                    if not wedge_clubs.empty:
                        wedge_show = wedge_clubs.rename(columns={
                            "AvgCarry": "Avg Carry",
                            "MedianCarry": "Median Carry",
                            "CarryStd": "Std Dev",
                            "MeanAbsDelta": "Avg Miss",
                        }).copy()
                        wedge_show["Distance Control Score"] = (10 - (wedge_show["Std Dev"] * 0.45 + wedge_show["Avg Miss"] * 0.20)).clip(lower=1, upper=10)
                        st.markdown("**By club inside this wedge bucket**")
                        st.dataframe(wedge_show.round(1), use_container_width=True, hide_index=True)
            else:
                st.info("Need more short-carry data to build the wedge control model.")
        else:
            st.info("Need clean carry data to build the wedge control model.")
    else:
        st.info("Need carry data to build the wedge control model.")

    st.subheader("Approach scoring expectation")
    if has_offline and has_carry:
        score_source = filtered.dropna(subset=["Carry", "Offline Num"]).copy()
        if not score_source.empty:
            sc_min = int(math.floor(float(score_source["Carry"].min()) / 5.0) * 5)
            sc_max = int(math.ceil(float(score_source["Carry"].max()) / 5.0) * 5)
            if sc_max - sc_min >= 10:
                default_sc_low = max(sc_min, min(150, sc_max - 10))
                default_sc_high = min(sc_max, max(default_sc_low + 10, 170))
                if default_sc_high <= default_sc_low:
                    default_sc_high = min(sc_max, default_sc_low + 10)
                sa, sb, sc = st.columns([1.15, 1, 0.7])
                with sa:
                    score_bucket = st.slider(
                        "Scoring model carry bucket (yds)",
                        min_value=sc_min,
                        max_value=sc_max,
                        value=(default_sc_low, default_sc_high),
                        step=5,
                        key="approach_scoring_bucket",
                    )
                with sb:
                    score_target = st.number_input(
                        "Scoring model target carry",
                        min_value=float(sc_min),
                        max_value=float(sc_max),
                        value=float((default_sc_low + default_sc_high) / 2),
                        step=1.0,
                        key="approach_scoring_target",
                    )
                with sc:
                    hole_par = st.selectbox("Hole par", [3, 4, 5], index=1, key="approach_scoring_par")
                score_summary, score_points, score_clubs = build_approach_scoring_expectation(
                    score_source, score_bucket[0], score_bucket[1], score_target, hole_par=hole_par
                )
                if score_points.empty:
                    st.info("No clean shots landed inside that scoring bucket.")
                else:
                    s1, s2, s3, s4 = st.columns(4)
                    s1.metric("Expected score", f"{score_summary['Expected Score']:.2f}")
                    s2.metric("Birdie", f"{score_summary['Birdie %']:.1f}%")
                    s3.metric("Par", f"{score_summary['Par %']:.1f}%")
                    s4.metric("Bogey+", f"{score_summary['Bogey+ %']:.1f}%")
                    st.caption(
                        f"Modeled finish for a par-{int(score_summary['Hole Par'])} hole from this approach window. These are leave-based estimates built from proximity and dispersion, not tracked on-course outcomes."
                    )
                    sx, sy = st.columns([1.2, 1])
                    with sx:
                        plot_df = pd.DataFrame({
                            "Result": ["Birdie", "Par", "Bogey+"],
                            "Probability": [score_summary["Birdie %"] / 100.0, score_summary["Par %"] / 100.0, score_summary["Bogey+ %"] / 100.0],
                        })
                        prob_chart = alt.Chart(plot_df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                            x=alt.X("Result:N", sort=["Birdie", "Par", "Bogey+"], title=None),
                            y=alt.Y("Probability:Q", axis=alt.Axis(format="%"), title="Modeled probability"),
                            color=alt.Color("Result:N", legend=None),
                            tooltip=[alt.Tooltip("Probability:Q", title="Probability", format=".1%")],
                        )
                        st.altair_chart(prob_chart.properties(height=330), use_container_width=True)
                    with sy:
                        diff = score_summary['Expected Score'] - score_summary['Hole Par']
                        if diff <= -0.15:
                            score_read = "Scoring pressure window"
                            score_why = "The modeled birdie rate is high enough that this approach bucket should create real red-number chances."
                        elif diff <= 0.15:
                            score_read = "Par-forward window"
                            score_why = "This bucket is projecting mostly par golf with some birdie upside when the leave stays tight."
                        else:
                            score_read = "Bogey pressure window"
                            score_why = "Dispersion and leave quality are pushing too many outcomes away from stress-free pars."
                        st.markdown(
                            f"<div class='rank-card'><div class='rank-label'>{score_read}</div><div class='rank-why'>{score_why}</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='rank-card'><div class='rank-label'>Target</div><div class='rank-club'>{score_summary['Target Carry']:.0f} yds</div><div class='rank-why'>Built from {score_summary['Shots']} clean shots in this scoring bucket.</div></div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div class='rank-card'><div class='rank-label'>Avg proximity</div><div class='rank-club'>{score_summary['Avg Proximity Ft']:.0f} ft</div><div class='rank-why'>Closer average leaves push birdie probability up and bogey risk down.</div></div>",
                            unsafe_allow_html=True,
                        )
                    if not score_clubs.empty:
                        score_show = score_clubs.rename(columns={
                            "AvgProximityFt": "Avg Proximity (ft)",
                            "BirdiePct": "Birdie %",
                            "ParPct": "Par %",
                            "BogeyPct": "Bogey+ %",
                            "ExpectedScore": "Expected Score",
                        }).copy()
                        st.markdown("**By club inside this scoring bucket**")
                        st.dataframe(score_show.round(2), use_container_width=True, hide_index=True)
            else:
                st.info("Need a wider carry range in the current filters to build the scoring model.")
        else:
            st.info("Need clean carry and offline data to build the scoring model.")
    else:
        st.info("Need carry and offline data to build the scoring model.")

    render_trivia_block("analysis", "Random sports & culture trivia")


with progress_tab:
    st.subheader("Progress Over Time")
    trend_df = trend_summary(shots if player_sel == "All Players" else shots[shots["Player"] == player_sel])
    if trend_df.empty or trend_df["_TimeLabel"].nunique() < 2:
        st.info("Add at least two dated sessions to Launch.csv and this tab will start tracking improvement over time.")
    else:
        p1, p2, p3, p4 = st.columns(4)
        carry_delta = trend_delta_text(trend_df["Carry"])
        bs_delta = trend_delta_text(trend_df["BallSpeed"])
        smash_delta = trend_delta_text(trend_df["Smash"])
        offline_delta = trend_delta_text(-trend_df["Offline"])  # improvement if lower abs offline
        p1.metric("Carry trend", carry_delta, help="Last session median carry minus first tracked session median carry.")
        p2.metric("Ball speed trend", bs_delta)
        p3.metric("Smash trend", smash_delta)
        p4.metric("Dispersion trend", offline_delta, help="Positive means your absolute offline miss is shrinking.")

        metric_choice = st.selectbox(
            "Trend metric",
            ["Carry", "BallSpeed", "Smash", "Offline"],
            format_func=lambda x: {"Carry":"Carry", "BallSpeed":"Ball Speed", "Smash":"Smash Factor", "Offline":"Absolute Offline"}.get(x, x),
            key="trend_metric_choice",
        )
        title_map = {"Carry":"Carry", "BallSpeed":"Ball Speed", "Smash":"Smash Factor", "Offline":"Absolute Offline"}
        base = alt.Chart(trend_df).encode(
            x=alt.X("_TimeLabel:N", title="Session"),
            y=alt.Y(f"{metric_choice}:Q", title=title_map.get(metric_choice, metric_choice)),
            tooltip=["_TimeLabel", "Player", "Shots", alt.Tooltip(f"{metric_choice}:Q", title=title_map.get(metric_choice, metric_choice), format=".1f")],
        )
        line = base.mark_line(point=True, strokeWidth=3).encode(color=alt.Color("Player:N", legend=None if player_sel != "All Players" else alt.Legend(title="Player")))
        st.altair_chart(line.properties(height=360), use_container_width=True)

        st.markdown("**What is moving?**")
        latest = trend_df.sort_values("_TimeKey").groupby("Player", dropna=False).tail(1).copy()
        earliest = trend_df.sort_values("_TimeKey").groupby("Player", dropna=False).head(1).copy()
        merged = latest.merge(earliest, on="Player", suffixes=(" Latest", " First"))
        for _, r in merged.iterrows():
            bits = []
            cdelta = r["Carry Latest"] - r["Carry First"]
            bdelta = r["BallSpeed Latest"] - r["BallSpeed First"]
            sdelta = r["Smash Latest"] - r["Smash First"]
            odelta = r["Offline First"] - r["Offline Latest"]
            if cdelta > 3:
                bits.append(f"carry is up {cdelta:.1f} yds")
            elif cdelta < -3:
                bits.append(f"carry is down {abs(cdelta):.1f} yds")
            if bdelta > 1.5:
                bits.append(f"ball speed is up {bdelta:.1f} mph")
            if sdelta > 0.02:
                bits.append(f"smash is trending cleaner by {sdelta:.2f}")
            if odelta > 1:
                bits.append(f"dispersion tightened by {odelta:.1f} yds")
            if not bits:
                bits.append("trend line is mostly stable so far")
            who = r["Player"] if pd.notna(r["Player"]) and str(r["Player"]).strip() else "Current set"
            st.markdown(f"<div class='rank-card'><div class='rank-club'>{who}</div><div class='rank-why'>{'; '.join(bits).capitalize()}.</div></div>", unsafe_allow_html=True)

        st.dataframe(
            trend_df.rename(columns={"_TimeLabel":"Session", "BallSpeed":"Ball Speed", "Smash":"Smash Factor", "Offline":"Abs Offline"})[
                ["Session", "Player", "Shots", "Carry", "Ball Speed", "Smash Factor", "Abs Offline"]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )
    render_trivia_block("progress", "Progress-page trivia")

with benchmark_tab:
    st.subheader("Benchmark View")
    st.markdown(
        f"<div class='note-box'>This tab compares your filtered shot set against the <b>{benchmark_sel}</b> reference profile. Treat these as directional baselines you can tweak over time, not commandments carved in stone.</div>",
        unsafe_allow_html=True,
    )

    bench_summary = player_baseline_summary(filtered, benchmark_sel)
    if bench_summary.empty:
        st.info("No benchmark comparison is available for the current filters.")
    else:
        focus_metric = st.selectbox(
            "Benchmark metric",
            ["Carry", "Ball Speed", "Club Speed", "Smash Factor"],
            index=0,
            key="benchmark_metric",
        )
        player_col = f"{focus_metric} Player"
        base_col = f"{focus_metric} Baseline"
        if player_col not in bench_summary.columns:
            bench_summary[player_col] = np.nan
        if base_col not in bench_summary.columns:
            bench_summary[base_col] = np.nan
        bench_summary["Delta"] = bench_summary[player_col] - bench_summary[base_col]
        bench_summary["Read"] = bench_summary["Delta"].apply(lambda x: metric_eval_label(x, focus_metric))
        bench_summary["Bucket"] = bench_summary["Delta"].apply(lambda x: benchmark_bucket(x, focus_metric))
        bench_summary["Why"] = bench_summary.apply(lambda r: benchmark_reason(r, benchmark_sel), axis=1)

        b1, b2, b3, b4 = st.columns(4)
        overall_player = bench_summary[player_col].mean()
        overall_base = bench_summary[base_col].mean()
        delta = overall_player - overall_base if pd.notna(overall_player) and pd.notna(overall_base) else np.nan
        b1.metric(f"Avg {focus_metric}", "—" if pd.isna(overall_player) else f"{overall_player:.1f}")
        b2.metric(f"{benchmark_sel}", "—" if pd.isna(overall_base) else f"{overall_base:.1f}")
        b3.metric("Delta", "—" if pd.isna(delta) else delta_text(overall_player, overall_base, metric_unit(focus_metric)))
        b4.metric("At / above benchmark", f"{int((bench_summary['Bucket'].isin(['About even','Better'])).sum())}")

        same_or_better = bench_summary[bench_summary["Bucket"].isin(["About even", "Better"])]
        needs_work = bench_summary[bench_summary["Bucket"].isin(["Close", "Worse"])]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Where you are even or better**")
            if same_or_better.empty:
                st.info("Nothing is clearly at or above the selected benchmark on this filter yet.")
            else:
                for _, r in same_or_better.head(5).iterrows():
                    st.markdown(
                        f"<div class='rank-card'><div class='rank-label'>{r['Read']}</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>{focus_metric}: {r[player_col]:.1f} vs {r[base_col]:.1f}</div><div class='bench-chip'>Delta: {r['Delta']:+.1f}</div></div>",
                        unsafe_allow_html=True,
                    )
        with c2:
            st.markdown("**Where the gap still lives**")
            if needs_work.empty:
                st.info("Nothing major to flag here with the current benchmark and filter.")
            else:
                for _, r in needs_work.head(5).iterrows():
                    st.markdown(
                        f"<div class='rank-card'><div class='rank-label'>{r['Read']}</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>{focus_metric}: {r[player_col]:.1f} vs {r[base_col]:.1f}</div><div class='bench-chip'>Delta: {r['Delta']:+.1f}</div></div>",
                        unsafe_allow_html=True,
                    )

        compare_plot_df = bench_summary.dropna(subset=[player_col, base_col]).copy()
        if not compare_plot_df.empty:
            melted = pd.concat([
                compare_plot_df[["Club", player_col]].rename(columns={player_col: "Value"}).assign(Group="You"),
                compare_plot_df[["Club", base_col]].rename(columns={base_col: "Value"}).assign(Group=benchmark_sel),
            ], ignore_index=True)

            rules = alt.Chart(compare_plot_df).mark_rule(strokeWidth=4, opacity=0.30).encode(
                x=alt.X("Club:N", sort=None, title="Club"),
                y=alt.Y(base_col + ":Q", title=f"{focus_metric} ({metric_unit(focus_metric)})" if metric_unit(focus_metric) else focus_metric),
                y2=player_col + ":Q",
                tooltip=[
                    "Club",
                    alt.Tooltip(player_col + ":Q", title="You", format=nice_format(focus_metric)),
                    alt.Tooltip(base_col + ":Q", title=benchmark_sel, format=nice_format(focus_metric)),
                ],
            )
            points = alt.Chart(melted).mark_circle(size=155, opacity=0.88).encode(
                x=alt.X("Club:N", sort=None),
                y=alt.Y("Value:Q"),
                color=alt.Color("Group:N", title="Series"),
                tooltip=["Club", "Group", alt.Tooltip("Value:Q", format=nice_format(focus_metric))],
            )
            labels = alt.Chart(compare_plot_df).mark_text(dy=-12, fontSize=11, fontWeight="bold").encode(
                x=alt.X("Club:N", sort=None),
                y=alt.Y(player_col + ":Q"),
                text=alt.Text("Delta:Q", format="+.1f"),
            )
            st.altair_chart((rules + points + labels).properties(height=390), use_container_width=True)
            st.caption("Connector line shows the gap between your median and the chosen baseline. Positive label = you are above it. Negative = the remaining gap.")

        show_cols = [
            "Club", "Shots", player_col, base_col, "Delta", "Read", "Why",
            "Carry Player", "Carry Baseline", "Ball Speed Player", "Ball Speed Baseline",
            "Club Speed Player", "Club Speed Baseline", "Smash Factor Player", "Smash Factor Baseline",
        ]
        show_cols = [c for c in show_cols if c in bench_summary.columns]
        bench_show = bench_summary.loc[:, show_cols].copy()
        bench_show = bench_show.loc[:, ~bench_show.columns.duplicated()].copy()
        for c in list(bench_show.columns):
            if c in {"Club", "Read", "Why"}:
                continue
            col_obj = bench_show[c]
            if isinstance(col_obj, pd.DataFrame):
                col_obj = col_obj.iloc[:, 0]
            bench_show[c] = pd.to_numeric(col_obj, errors="coerce")
            if pd.api.types.is_numeric_dtype(bench_show[c]):
                bench_show[c] = bench_show[c].round(2 if "Smash" in c else 1)
        st.dataframe(bench_show, use_container_width=True, hide_index=True)

    render_trivia_block("benchmark", "Benchmark break trivia")

with data_tab:
    st.subheader("Shot Tables")
    t1, t2 = st.tabs(["Clean / Current View", "Excluded Shots"])
    with t1:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Download current filtered shots",
            data=make_download(filtered),
            file_name="launch_filtered_shots.csv",
            mime="text/csv",
        )
    with t2:
        excluded = shots[~shots["Is Clean Shot"]].copy()
        st.dataframe(excluded, use_container_width=True, hide_index=True)
        st.download_button(
            "Download excluded shots",
            data=make_download(excluded),
            file_name="launch_excluded_shots.csv",
            mime="text/csv",
        )

    render_trivia_block("tables", "One more fun fact before you leave")
