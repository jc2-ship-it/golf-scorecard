
import pandas as pd
import streamlit as st
import datetime
import altair as alt
import numpy as np
import os, json, random
import re
from pathlib import Path


def _safe_int_scalar(val, default=0):
    x = pd.to_numeric(val, errors="coerce")
    return default if pd.isna(x) else int(x)

def _safe_float_scalar(val, default=0.0):
    x = pd.to_numeric(val, errors="coerce")
    return default if pd.isna(x) else float(x)


# =========================================================
# Page
# =========================================================
st.set_page_config(layout="wide", page_title="Golf Round Scorecard + Round Breakdown")

# =========================================================
# Config
# =========================================================
CSV_FILE = "Hole Data-Grid view (18).csv"

APPROACH_BUCKETS = [
    (0, 50, "0-50"),
    (51, 75, "51-75"),
    (76, 100, "76-100"),
    (101, 115, "101-115"),
    (116, 130, "116-130"),
    (131, 145, "131-145"),
    (146, 160, "146-160"),
    (161, 175, "161-175"),
    (176, 190, "176-190"),
    (191, 205, "191-205"),
    (206, 9999, "206+"),
]
APPROACH_BUCKET_ORDER = [x[2] for x in APPROACH_BUCKETS]

PUTT_BUCKETS = [
    (0, 3, "0-3"),
    (4, 6, "4-6"),
    (7, 10, "7-10"),
    (11, 15, "11-15"),
    (16, 20, "16-20"),
    (21, 30, "21-30"),
    (31, 9999, "31+"),
]
PUTT_BUCKET_ORDER = [x[2] for x in PUTT_BUCKETS]

SHORT_GAME_BUCKETS = [
    (0, 5, "0-5"),
    (6, 10, "6-10"),
    (11, 20, "11-20"),
    (21, 9999, "21+"),
]
SHORT_GAME_BUCKET_ORDER = [x[2] for x in SHORT_GAME_BUCKETS]

# =========================================================
# Helpers
# =========================================================
def _num(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def _int(series, default=0):
    return _num(series, default=default).astype(int)

def _safe_col(df, col, default=0):
    if col in df.columns:
        return df[col]
    return pd.Series([default] * len(df), index=df.index)

def _fmt_frac_pct(made, att):
    return f"{int(made)}/{int(att)} ({(100 * made / att):.1f}%)" if att else "0/0 (-)"

def _format_golf_dataframe(df):
    """Return a Styler with golf-stat columns formatted for clean Streamlit tables.
    Percent columns are already stored as 0-100 values, so we add the % sign without rescaling.
    """
    if df is None or df.empty:
        return df

    fmt = {}
    for col in df.columns:
        col_s = str(col)
        col_l = col_s.lower()

        if any(token in col_l for token in ["%", "pct", "delta"]):
            if "arrow" in col_l:
                continue
            if "delta" in col_l or col_s.startswith("Δ"):
                fmt[col] = lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}"
            else:
                fmt[col] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%"
        elif any(token in col_l for token in ["avg prox", "proximity", "avg start", "start ft", "avg leave", "ft"]):
            fmt[col] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f}"
        elif any(token in col_l for token in ["attempt", "made", "putts", "count", "holes", "inside"]):
            fmt[col] = lambda x: "—" if pd.isna(x) else f"{int(round(float(x))):,}"

    try:
        return df.style.format(fmt, na_rep="—")
    except Exception:
        return df


def _render_golf_dataframe(df, **kwargs):
    """Streamlit dataframe wrapper with consistent percent/decimal formatting."""
    st.dataframe(_format_golf_dataframe(df), **kwargs)

def _fmt_to_par(n: int) -> str:
    return "E" if n == 0 else f"{'+' if n > 0 else ''}{n}"

def _fmt_par_float(x: float) -> str:
    if pd.isna(x):
        return "—"
    if abs(x) < 1e-9:
        return "E"
    return f"{'+' if x > 0 else ''}{x:.1f}"

def get_emoji(pct):
    if pct >= 50:
        return "🔥"
    elif pct < 25:
        return "❄️"
    else:
        return ""

def load_fun_facts():
    json_path = Path(__file__).parent / "fun_facts.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [fact for facts in data.values() for fact in facts]
    return [
        "Golf balls were once made of wood.",
        "The term 'birdie' originated at Atlantic City Country Club in 1903."
    ]


def render_trivia_block(page_key, heading="Fun trivia corner"):
    facts = load_fun_facts()
    if not facts:
        return

    state_key = f"trivia_idx_{page_key}"
    if state_key not in st.session_state:
        st.session_state[state_key] = random.randrange(len(facts))

    c1, c2 = st.columns([6, 1])
    with c1:
        st.markdown(f"#### {heading}")
    with c2:
        if st.button("New fact", key=f"btn_{page_key}"):
            st.session_state[state_key] = random.randrange(len(facts))

    fact = facts[st.session_state[state_key]]
    st.markdown(
        f"""
        <div style="background: linear-gradient(180deg, rgba(49,46,129,.28) 0%, rgba(17,24,39,.78) 100%); border: 1px solid rgba(255,255,255,.10); border-radius: 16px; padding: 14px 16px; margin-top: 8px; margin-bottom: 10px;">
          <div style="font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; opacity: .72; margin-bottom: 8px;">Golf Trivia</div>
          <div style="font-size: 1rem; line-height: 1.45; font-weight: 500;">{fact}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def segment_total(vals, start, end):
    return sum(vals[start:end])

def icon_total(icon_row, start, end, symbol):
    return sum(symbol in str(cell) for cell in icon_row[start:end])

def insert_segment_sums(row, skip_total=False):
    out = segment_total(row, 0, 9)
    inn = segment_total(row, 9, 18)
    return row[:9] + [out] + row[9:18] + [inn] + ([""] if skip_total else [out + inn])

def insert_icon_sums(row, symbol):
    out = icon_total(row, 0, 9, symbol)
    inn = icon_total(row, 9, 18, symbol)
    return row[:9] + [out] + row[9:18] + [inn, out + inn]

def _made_total_pct_by_par(df_block, metric_col, par_value):
    if metric_col not in df_block or "Par" not in df_block:
        return 0, 0, 0.0
    block = df_block[df_block["Par"] == par_value]
    total = int(block.shape[0])
    made = int(pd.to_numeric(block[metric_col], errors="coerce").fillna(0).sum())
    pct = (made / total * 100.0) if total else 0.0
    return made, total, pct

def _bucket_value(val, buckets):
    try:
        v = float(val)
    except Exception:
        return None
    if pd.isna(v) or v < 0:
        return None
    for lo, hi, label in buckets:
        if lo <= v <= hi:
            return label
    return None

def _normalize_club(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""

    key = s.upper().replace("IRON", "I").replace(" ", "").replace("-", "")
    key = key.replace("DEG", "").replace("°", "")

    named = {
        "PWEDGE": "PW",
        "PW": "PW",
        "PITCHINGWEDGE": "PW",
        "AWEDGE": "AW",
        "AW": "AW",
        "APPROACHWEDGE": "AW",
        "GWEDGE": "GW",
        "GW": "GW",
        "GAPWEDGE": "GW",
        "SWEDGE": "SW",
        "SW": "SW",
        "SANDWEDGE": "SW",
        "LWEDGE": "LW",
        "LW": "LW",
        "LOBWEDGE": "LW",
        "DRIVER": "DRIVER",
        "DR": "DRIVER",
        "1W": "DRIVER",
        "3W": "3W",
        "4W": "4W",
        "5W": "5W",
        "7W": "7W",
        "2H": "2H",
        "3H": "3H",
        "4H": "4H",
        "5H": "5H",
        "6H": "6H",
        "7H": "7H",
        "HY": "HY",
        "HYBRID": "HY",
    }
    if key in named:
        return named[key]

    # Loft wedges
    if key.isdigit():
        loft = int(key)
        if loft >= 58:
            return "LW"
        if loft >= 54:
            return "SW"
        if loft >= 50:
            return "GW"
        if loft >= 45:
            return "PW"
        # Plain single-digit numbers are usually irons
        if 1 <= loft <= 9:
            return f"{loft}I"

    # Iron-style labels: 9I, 8, etc.
    m_iron = re.fullmatch(r"([1-9])I?", key)
    if m_iron:
        return f"{m_iron.group(1)}I"

    # Hybrid / wood labels
    m_hybrid = re.fullmatch(r"([2-7])H", key)
    if m_hybrid:
        return f"{m_hybrid.group(1)}H"
    m_wood = re.fullmatch(r"([2-9])W", key)
    if m_wood:
        return f"{m_wood.group(1)}W"

    return key

def _normalize_direction(val):
    if pd.isna(val):
        return ""
    s = str(val).strip()
    if not s:
        return ""
    key = s.lower()
    mapping = {
        "←": "Left",
        "→": "Right",
        "↑": "Long",
        "↓": "Short",
        "short left": "Short Left",
        "short-right": "Short Right",
        "short right": "Short Right",
        "long left": "Long Left",
        "long right": "Long Right",
        "left": "Left",
        "right": "Right",
        "short": "Short",
        "long": "Long",
        "pin high left": "Left",
        "pin high right": "Right",
    }
    return mapping.get(key, s)

def build_benchmark_df(full_df, round_df, mode, selected_round_ids=None):
    player = round_df["Player Name"].iloc[0] if "Player Name" in round_df and not round_df.empty else None
    course = round_df["Course Name"].iloc[0] if "Course Name" in round_df and not round_df.empty else None
    round_date = pd.to_datetime(round_df["Date Played"].iloc[0], errors="coerce") if ("Date Played" in round_df and not round_df.empty) else pd.NaT
    year = int(round_date.year) if pd.notna(round_date) else None
    month = int(round_date.month) if pd.notna(round_date) else None

    base = full_df.copy()

    # Custom selected-round baselines should respect the exact chosen rounds,
    # including rounds from other players.
    if mode == "Selected Rounds":
        if selected_round_ids and "Round Link" in base.columns:
            return base[base["Round Link"].isin(selected_round_ids)].copy()
        return base.iloc[0:0].copy()

    if player and "Player Name" in base:
        base = base[base["Player Name"] == player]
    if mode == "All Time":
        return base
    elif mode == "Same Year":
        return base[base["Year"] == year]
    elif mode == "Same Month":
        return base[(base["Year"] == year) & (base["Date Played"].dt.month == month)]
    elif mode == "Same Course":
        return base[base["Course Name"] == course]
    return base

def _compare_mode_options():
    return ["All Time", "Same Year", "Same Month", "Same Course", "Selected Rounds"]


def _compare_mode_display_label(mode):
    return "Selected Rounds Avg" if mode == "Selected Rounds" else mode

def build_compare_long(round_summary, bench_summary, key_col, round_label="Round", bench_label="Baseline"):
    r = round_summary.copy()
    b = bench_summary.copy()

    if r.empty and b.empty:
        return pd.DataFrame(columns=[key_col, "Series", "Pct", "Attempts", "Made", "Label"])

    r = r.loc[:, ~r.columns.duplicated()].copy()
    b = b.loc[:, ~b.columns.duplicated()].copy()

    required = [key_col, "Attempts", "Made", "Pct", "Label"]
    for col in required:
        if col not in r.columns:
            r[col] = pd.Series(dtype="object")
        if col not in b.columns:
            b[col] = pd.Series(dtype="object")

    r = r[required].copy().drop_duplicates(subset=[key_col], keep="first")
    b = b[required].copy().drop_duplicates(subset=[key_col], keep="first")

    merged = pd.merge(
        r,
        b,
        on=key_col,
        how="outer",
        suffixes=(f"_{round_label}", f"_{bench_label}")
    )

    rows = []
    for _, row in merged.iterrows():
        key = row[key_col]

        r_attempts = row.get(f"Attempts_{round_label}", 0)
        r_made = row.get(f"Made_{round_label}", 0)
        r_pct = row.get(f"Pct_{round_label}", 0)
        r_label = row.get(f"Label_{round_label}", "0/0 • 0.0%")

        b_attempts = row.get(f"Attempts_{bench_label}", 0)
        b_made = row.get(f"Made_{bench_label}", 0)
        b_pct = row.get(f"Pct_{bench_label}", 0)
        b_label = row.get(f"Label_{bench_label}", "0/0 • 0.0%")

        rows.append({
            key_col: key,
            "Series": round_label,
            "Attempts": 0 if pd.isna(r_attempts) else int(r_attempts),
            "Made": 0 if pd.isna(r_made) else int(r_made),
            "Pct": 0 if pd.isna(r_pct) else float(r_pct),
            "Label": r_label if pd.notna(r_label) else "0/0 • 0.0%"
        })
        rows.append({
            key_col: key,
            "Series": bench_label,
            "Attempts": 0 if pd.isna(b_attempts) else int(b_attempts),
            "Made": 0 if pd.isna(b_made) else int(b_made),
            "Pct": 0 if pd.isna(b_pct) else float(b_pct),
            "Label": b_label if pd.notna(b_label) else "0/0 • 0.0%"
        })

    return pd.DataFrame(rows)

# =========================================================
# Approach
# =========================================================
def prepare_approach_frame(frame):
    d = frame.copy()
    d["Approach Distance"] = _num(_safe_col(d, "Approach Shot Distance (how far you had to the hole)", 0))
    d["Approach Club"] = _safe_col(d, "Approach Shot Club Used", "").fillna("").apply(_normalize_club)
    d["Approach GIR Flag"] = _int(_safe_col(d, "Approach GIR Value", 0))
    d["Approach Miss Direction Clean"] = _safe_col(d, "Approach Shot Direction Miss", "").fillna("").apply(_normalize_direction)
    d["Approach Proximity"] = _num(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", 0))
    d["Approach Bucket"] = d["Approach Distance"].apply(lambda x: _bucket_value(x, APPROACH_BUCKETS))
    d = d[d["Approach Distance"] > 0].copy()
    return d

def summarize_approach_by_bucket(frame):
    d = prepare_approach_frame(frame)
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out = (
        d.dropna(subset=["Approach Bucket"])
         .groupby("Approach Bucket", as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean")
         )
         .rename(columns={"Approach Bucket": "Bucket"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)[["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"]]

def summarize_approach_by_club(frame, min_attempts=1):
    d = prepare_approach_frame(frame)
    d = d[d["Approach Club"] != ""].copy()
    if d.empty:
        return pd.DataFrame(columns=["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out = (
        d.groupby("Approach Club", as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean")
         )
         .rename(columns={"Approach Club": "Club"})
    )
    out = out[out["Attempts"] >= min_attempts].copy()
    out["Pct"] = (pd.to_numeric(out["Made"], errors="coerce").fillna(0) / pd.to_numeric(out["Attempts"], errors="coerce").replace(0, pd.NA) * 100).fillna(0).round(1)
    out["Label"] = (
        pd.to_numeric(out["Made"], errors="coerce").fillna(0).astype(int).astype(str)
        + "/"
        + pd.to_numeric(out["Attempts"], errors="coerce").fillna(0).astype(int).astype(str)
        + " • "
        + out["Pct"].map(lambda x: f"{float(x):.1f}%")
    )
    return out.sort_values(["Attempts", "Pct", "Club"], ascending=[False, False, True]).reset_index(drop=True)[["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"]]

def summarize_approach_miss_direction(frame):
    d = prepare_approach_frame(frame)
    d = d[d["Approach Miss Direction Clean"] != ""].copy()
    if d.empty:
        return pd.DataFrame(columns=["Direction", "Count", "Pct"])

    out = (
        d.groupby("Approach Miss Direction Clean", as_index=False)
         .agg(Count=("Approach Miss Direction Clean", "size"))
         .rename(columns={"Approach Miss Direction Clean": "Direction"})
    )
    total = out["Count"].sum()
    out["Pct"] = (out["Count"] / total * 100).round(1)
    return out.sort_values(["Count", "Direction"], ascending=[False, True]).reset_index(drop=True)

# =========================================================
# Putting
# =========================================================
def prepare_putting_frame(frame):
    d = frame.copy()

    # Match the working Putting_Stats validation logic:
    # - remove hole-outs entirely (Putts == 0)
    # - exclude blank first-putt proximity rows
    # - bucket by first-putt proximity
    # - "Made" for this view = 1-putt from that starting distance
    d["First Putt Distance"] = pd.to_numeric(
        _safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", pd.NA),
        errors="coerce"
    )
    d["Putt Made Feet"] = pd.to_numeric(
        _safe_col(d, "Feet of Putt Made (How far was the putt you made)", pd.NA),
        errors="coerce"
    )
    d["Putts Clean"] = _int(_safe_col(d, "Putts", 0))

    d = d[(d["Putts Clean"] > 0) & (d["First Putt Distance"].notna())].copy()

    d["Putt Bucket"] = d["First Putt Distance"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))
    d["Putt Attempt"] = d["Putt Bucket"].notna().astype(int)
    d["Putt Made Flag"] = (d["Putts Clean"] == 1).astype(int)

    d = d[d["Putt Attempt"] == 1].copy()
    return d

def summarize_putting_by_bucket(frame):
    d = prepare_putting_frame(frame)
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label"])

    out = (
        d.dropna(subset=["Putt Bucket"])
         .groupby("Putt Bucket", as_index=False)
         .agg(
             Attempts=("Putt Attempt", "sum"),
             Made=("Putt Made Flag", "sum")
         )
         .rename(columns={"Putt Bucket": "Bucket"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)[["Bucket", "Attempts", "Made", "Pct", "Label"]]


def build_putting_skill_fingerprint(frame):
    d = prepare_putting_frame(frame)
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "OnePutt", "TwoPutt", "ThreePuttPlus", "MakePct", "TwoPuttPct", "ThreePuttPct", "DisplayLabel"])

    out = (
        d.dropna(subset=["Putt Bucket"])
         .groupby("Putt Bucket", as_index=False)
         .agg(
             Attempts=("Putt Attempt", "sum"),
             OnePutt=("Putt Made Flag", "sum"),
             TwoPutt=("Putts Clean", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) == 2).sum())),
             ThreePuttPlus=("Putts Clean", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) >= 3).sum())),
             AvgStartFt=("First Putt Distance", "mean")
         )
         .rename(columns={"Putt Bucket": "Bucket"})
    )
    out["MakePct"] = (out["OnePutt"] / out["Attempts"] * 100).round(1)
    out["TwoPuttPct"] = (out["TwoPutt"] / out["Attempts"] * 100).round(1)
    out["ThreePuttPct"] = (out["ThreePuttPlus"] / out["Attempts"] * 100).round(1)
    out["DisplayLabel"] = out.apply(
        lambda r: f"{int(r['OnePutt'])}/{int(r['Attempts'])} • {r['MakePct']:.1f}% | 3P {r['ThreePuttPct']:.0f}%",
        axis=1
    )
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)


def build_putting_zone_stats(frame):
    d = prepare_putting_frame(frame)
    if d.empty:
        return {
            "inside6": {"attempts": 0, "made": 0, "pct": 0.0},
            "six10": {"attempts": 0, "made": 0, "pct": 0.0},
            "ten20": {"attempts": 0, "made": 0, "pct": 0.0},
            "lag20": {"attempts": 0, "made": 0, "pct": 0.0, "three_putts": 0, "three_putt_pct": 0.0, "two_putt_or_better_pct": 0.0, "avg_start_ft": 0.0},
        }

    fpd = pd.to_numeric(d["First Putt Distance"], errors="coerce").fillna(0)

    def _basic(mask):
        block = d[mask].copy()
        attempts = int(len(block))
        made = int(pd.to_numeric(block["Putt Made Flag"], errors="coerce").fillna(0).sum()) if attempts else 0
        pct = (made / attempts * 100.0) if attempts else 0.0
        return {"attempts": attempts, "made": made, "pct": pct}

    lag_mask = fpd >= 20
    lag_block = d[lag_mask].copy()
    lag_attempts = int(len(lag_block))
    lag_three_putts = int((pd.to_numeric(lag_block["Putts Clean"], errors="coerce").fillna(0) >= 3).sum()) if lag_attempts else 0
    lag_two_or_better = int((pd.to_numeric(lag_block["Putts Clean"], errors="coerce").fillna(0) <= 2).sum()) if lag_attempts else 0

    return {
        "inside6": _basic(fpd <= 6),
        "six10": _basic((fpd > 6) & (fpd <= 10)),
        "ten20": _basic((fpd > 10) & (fpd <= 20)),
        "lag20": {
            "attempts": lag_attempts,
            "made": int(pd.to_numeric(lag_block["Putt Made Flag"], errors="coerce").fillna(0).sum()) if lag_attempts else 0,
            "pct": (pd.to_numeric(lag_block["Putt Made Flag"], errors="coerce").fillna(0).sum() / lag_attempts * 100.0) if lag_attempts else 0.0,
            "three_putts": lag_three_putts,
            "three_putt_pct": (lag_three_putts / lag_attempts * 100.0) if lag_attempts else 0.0,
            "two_putt_or_better_pct": (lag_two_or_better / lag_attempts * 100.0) if lag_attempts else 0.0,
            "avg_start_ft": float(pd.to_numeric(lag_block["First Putt Distance"], errors="coerce").fillna(0).mean()) if lag_attempts else 0.0,
        },
    }


def _resolve_round_col(frame):
    for col in ["Round Link", "Round Label", "Round ID"]:
        if col in frame.columns:
            return col
    return None


def _round_index(frame):
    round_col = _resolve_round_col(frame)
    if round_col and round_col in frame.columns:
        vals = frame[round_col].dropna()
        if len(vals):
            return pd.Index(vals.unique())
    return pd.Index(["__single_round__"])


def _mean_per_round(frame, series_like):
    if frame is None or frame.empty:
        return 0.0
    round_ids = _round_index(frame)
    round_col = _resolve_round_col(frame)
    values = pd.to_numeric(series_like, errors="coerce").fillna(0)
    if round_col and round_col in frame.columns:
        tmp = frame[[round_col]].copy()
        tmp["_v"] = values.values
        grouped = tmp.groupby(round_col, dropna=True)["_v"].sum().reindex(round_ids, fill_value=0)
        return float(grouped.mean()) if len(grouped) else 0.0
    return float(values.sum())


def _mean_attempts_per_round(frame, mask):
    if frame is None or frame.empty:
        return 0.0
    round_ids = _round_index(frame)
    round_col = _resolve_round_col(frame)
    mask = pd.Series(mask, index=frame.index).fillna(False).astype(bool)
    if round_col and round_col in frame.columns:
        counts = frame.loc[mask].groupby(round_col, dropna=True).size().reindex(round_ids, fill_value=0)
        return float(counts.mean()) if len(counts) else 0.0
    return float(mask.sum())


def _hole_count(frame):
    if frame is None or frame.empty:
        return 0
    return int(len(frame))


def _per18(total, holes):
    holes = float(holes or 0)
    if holes <= 0:
        return 0.0
    return float(total) / holes * 18.0


def _standardized_score_per18(frame):
    """Projected 18-hole score on a standard par-72 mix.

    Uses: Par 3 avg * 4 + Par 4 avg * 10 + Par 5 avg * 4.
    If a sample is missing a par type, it fills that par type with the
    player's/course sample average score-to-par added to that par value.
    That keeps 9-hole rounds and par-3-heavy samples from skewing Avg Score / 18.
    """
    if frame is None or frame.empty:
        return 0.0

    d = frame.copy()
    score = pd.to_numeric(_safe_col(d, "Hole Score", pd.NA), errors="coerce")
    par = pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")
    valid = score.notna() & par.notna() & par.isin([3, 4, 5])
    if not valid.any():
        return 0.0

    score = score[valid]
    par = par[valid]
    avg_to_par = float((score - par).mean()) if len(score) else 0.0

    def _avg_for_par(par_value):
        vals = score[par == par_value]
        if len(vals):
            return float(vals.mean())
        return float(par_value) + avg_to_par

    return (4.0 * _avg_for_par(3)) + (10.0 * _avg_for_par(4)) + (4.0 * _avg_for_par(5))


def _standardized_to_par_per18(frame):
    return _standardized_score_per18(frame) - 72.0


def _baseline_label(mode, per18=False):
    mapping = {
        "All Time": "All-Time Avg",
        "Same Year": "Year Avg",
        "Same Month": "Month Avg",
        "Same Course": "Course Avg",
        "Selected Rounds": "Selected Rounds Avg",
    }
    base = mapping.get(mode, mode)
    return f"{base} (Per 18)" if per18 else base


def _delta_arrow(delta, higher_better=True, tol=0.05):
    try:
        delta = float(delta)
    except Exception:
        return "➡️"
    if abs(delta) <= tol:
        return "➡️"
    good = delta > 0 if higher_better else delta < 0
    return "🔼" if good else "🔽"


def build_putting_baseline_snapshot(frame):
    d = frame.copy()
    lag = build_putting_zone_stats(d)["lag20"]
    putts_clean = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0)
    three_putt_flag = (putts_clean >= 3).astype(int)
    one_putt_flag = (putts_clean == 1).astype(int)
    holes = _hole_count(d)

    first_putt = pd.to_numeric(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", pd.NA), errors="coerce")
    lag_mask = (putts_clean > 0) & (first_putt >= 20)

    return {
        "holes": holes,
        "total_putts_per18": _per18(putts_clean.sum(), holes),
        "one_putt_pct": (float(one_putt_flag.sum()) / holes * 100.0) if holes else 0.0,
        "three_putts_per18": _per18(three_putt_flag.sum(), holes),
        "three_putt_pct": (float(three_putt_flag.sum()) / holes * 100.0) if holes else 0.0,
        "lag_attempts_avg": _mean_attempts_per_round(d, lag_mask),
        "lag_avg_start_ft": float(lag["avg_start_ft"]),
        "lag_two_putt_or_better_pct": float(lag["two_putt_or_better_pct"]),
        "lag_three_putt_pct": float(lag["three_putt_pct"]),
    }


def summarize_three_putt_by_bucket(frame):
    d = prepare_putting_frame(frame)
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label"])

    out = (
        d.dropna(subset=["Putt Bucket"])
         .groupby("Putt Bucket", as_index=False)
         .agg(
             Attempts=("Putt Attempt", "sum"),
             Made=("Putts Clean", lambda s: int((pd.to_numeric(s, errors="coerce").fillna(0) >= 3).sum())),
         )
         .rename(columns={"Putt Bucket": "Bucket"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)[["Bucket", "Attempts", "Made", "Pct", "Label"]]



def render_putting_overlay_line_chart(primary_df, overlay_df, bucket_order, title, value_col="Pct", primary_name="You", overlay_name="Overlay"):
    if primary_df is None or primary_df.empty:
        st.info(f"No usable {title.lower()} data found.")
        return

    p = primary_df.copy()
    p = p[p["Bucket"].notna()].copy()
    p["Bucket"] = p["Bucket"].astype(str)
    p["Value"] = pd.to_numeric(p[value_col], errors="coerce").fillna(0.0)
    p["Series"] = primary_name

    frames = [p[["Bucket", "Value", "Series"]].copy()]
    if overlay_df is not None and not overlay_df.empty:
        o = overlay_df.copy()
        o = o[o["Bucket"].notna()].copy()
        o["Bucket"] = o["Bucket"].astype(str)
        o["Value"] = pd.to_numeric(o[value_col], errors="coerce").fillna(0.0)
        o["Series"] = overlay_name
        frames.append(o[["Bucket", "Value", "Series"]].copy())

    plot_df = pd.concat(frames, ignore_index=True)
    plot_df["Bucket"] = pd.Categorical(plot_df["Bucket"], categories=bucket_order, ordered=True)
    plot_df = plot_df.sort_values(["Bucket", "Series"]).copy()

    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Bucket:N", sort=bucket_order, title=None),
            y=alt.Y("Value:Q", title="%"),
            color=alt.Color("Series:N", title="Overlay"),
            tooltip=[
                alt.Tooltip("Bucket:N", title="Bucket"),
                alt.Tooltip("Series:N", title="Series"),
                alt.Tooltip("Value:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=320, title=title)
        .configure_view(strokeOpacity=0)
        .configure_axis(
            labelColor="white",
            titleColor="white",
            gridColor="rgba(255,255,255,0.10)",
            tickColor="rgba(255,255,255,0.20)",
            domainColor="rgba(255,255,255,0.20)",
        )
    )
    st.altair_chart(chart, use_container_width=True)

def render_putting_distance_line_chart(summary_df, bucket_order, title, value_col="Pct", made_col="Made", attempts_col="Attempts", value_label="%", good_metric=True):
    if summary_df is None or summary_df.empty:
        st.info(f"No usable {title.lower()} data found.")
        return

    plot_df = summary_df.copy()
    plot_df = plot_df[plot_df["Bucket"].notna()].copy()
    if plot_df.empty:
        st.info(f"No usable {title.lower()} data found.")
        return

    plot_df["Bucket"] = pd.Categorical(plot_df["Bucket"].astype(str), categories=bucket_order, ordered=True)
    plot_df = plot_df.sort_values("Bucket").copy()
    plot_df["Value"] = pd.to_numeric(plot_df[value_col], errors="coerce").fillna(0.0)
    plot_df["MadeN"] = pd.to_numeric(plot_df[made_col], errors="coerce").fillna(0).astype(int)
    plot_df["AttemptsN"] = pd.to_numeric(plot_df[attempts_col], errors="coerce").fillna(0).astype(int)
    plot_df["Label"] = plot_df.apply(
        lambda r: f"{int(r['MadeN'])}/{int(r['AttemptsN'])} {float(r['Value']):.1f}%" if int(r["AttemptsN"]) else "—",
        axis=1
    )

    bar_color = "#3B82F6" if good_metric else "#EF4444"
    line_color = "#22C55E" if good_metric else "#F97316"

    base = alt.Chart(plot_df).encode(
        x=alt.X("Bucket:N", sort=bucket_order, title=None),
        tooltip=[
            alt.Tooltip("Bucket:N", title="Bucket"),
            alt.Tooltip("Value:Q", title=value_label, format=".1f"),
            alt.Tooltip("MadeN:Q", title="Makes", format=",.0f"),
            alt.Tooltip("AttemptsN:Q", title="Attempts", format=",.0f"),
        ],
    )

    bars = base.mark_bar(
        cornerRadiusTopLeft=7,
        cornerRadiusTopRight=7,
        opacity=0.92,
    ).encode(
        y=alt.Y("AttemptsN:Q", title="Attempts"),
        color=alt.value(bar_color),
    )

    line = base.mark_line(strokeWidth=4, opacity=0.92).encode(
        y=alt.Y("Value:Q", title=value_label, axis=alt.Axis(orient="right")),
        color=alt.value(line_color),
    )

    points = base.mark_point(size=140, filled=True, opacity=0.97).encode(
        y=alt.Y("Value:Q", axis=alt.Axis(orient="right")),
        color=alt.value(line_color),
    )

    labels = base.mark_text(
        dx=10,
        dy=-8,
        fontSize=12,
        fontWeight="bold",
        color="white"
    ).encode(
        y=alt.Y("Value:Q", axis=alt.Axis(orient="right")),
        text="Label:N"
    )

    chart = alt.layer(bars, line, points, labels).resolve_scale(
        y="independent"
    ).properties(height=380, title=title).configure_view(
        strokeOpacity=0
    ).configure_axis(
        labelColor="white",
        titleColor="white",
        gridColor="rgba(255,255,255,0.10)",
        tickColor="rgba(255,255,255,0.20)",
        domainColor="rgba(255,255,255,0.20)",
    )

    st.altair_chart(chart, use_container_width=True)



def _putting_fingerprint_for_chart(fp_df, series_label, metric="make"):
    """
    Convert build_putting_skill_fingerprint() output into a chart-friendly long frame.
    metric='make' uses 1-putt percentage.
    metric='three' uses 3+ putt percentage.
    """
    if fp_df is None or fp_df.empty:
        return pd.DataFrame(columns=["Bucket", "Series", "Rate", "Attempts", "Count", "Label"])

    d = fp_df.copy()
    d = d[d["Bucket"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Series", "Rate", "Attempts", "Count", "Label"])

    d["Bucket"] = d["Bucket"].astype(str)
    d["Attempts"] = pd.to_numeric(d.get("Attempts", 0), errors="coerce").fillna(0).astype(int)

    if metric == "three":
        rate_col = "ThreePuttPct"
        count_col = "ThreePuttPlus"
    else:
        rate_col = "MakePct"
        count_col = "OnePutt"

    d["Rate"] = pd.to_numeric(d.get(rate_col, 0), errors="coerce").fillna(0.0)
    d["Count"] = pd.to_numeric(d.get(count_col, 0), errors="coerce").fillna(0).astype(int)
    d["Series"] = series_label
    d["Label"] = d.apply(
        lambda r: f"{int(r['Count'])}/{int(r['Attempts'])} • {float(r['Rate']):.1f}%" if int(r["Attempts"]) else "0/0 • 0.0%",
        axis=1
    )
    return d[["Bucket", "Series", "Rate", "Attempts", "Count", "Label"]]


def render_putting_distance_comparison_dashboard(
    round_fp,
    bench_fp,
    bucket_order,
    compare_label="Baseline",
    overlay_fps=None,
    show_round=True,
    show_baseline=True,
    title="Putting Distance Control — Player Comparison",
):
    """
    Putting distance view:
    - 1-putt % line by starting-distance bucket
    - 3+ putt % line by starting-distance bucket
    - optional selected-round line
    - optional selected baseline line
    - multiple player comparison overlays
    - compact table with makes/attempts and percentages
    """
    overlay_fps = overlay_fps or []

    r = round_fp.copy() if round_fp is not None else pd.DataFrame()
    b = bench_fp.copy() if bench_fp is not None else pd.DataFrame()

    st.markdown(f"#### {title}")
    st.caption("Lines show conversion/risk by first-putt distance. Use the controls above to hide the selected round and compare multiple players cleanly.")

    def _clean_fp(fp_df):
        if fp_df is None or fp_df.empty:
            return pd.DataFrame()
        d = fp_df.copy()
        if "Bucket" not in d.columns:
            return pd.DataFrame()
        return d[d["Bucket"].notna()].copy()

    r = _clean_fp(r)
    b = _clean_fp(b)

    series_defs = []
    if show_round and not r.empty:
        series_defs.append(("This Round", r))
    if show_baseline and not b.empty:
        series_defs.append((compare_label, b))

    for overlay_label, overlay_df in overlay_fps:
        o = _clean_fp(overlay_df)
        if overlay_label and not o.empty:
            series_defs.append((str(overlay_label), o))

    if not series_defs:
        st.info("No usable putting distance data found for the selected comparison.")
        return

    def _build_metric_long(metric):
        frames = []
        for label, fp in series_defs:
            frames.append(_putting_fingerprint_for_chart(fp, label, metric=metric))
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if out.empty:
            return out
        out = out[out["Bucket"].notna()].copy()
        out["Bucket"] = pd.Categorical(out["Bucket"].astype(str), categories=bucket_order, ordered=True)
        return out.sort_values(["Bucket", "Series"]).copy()

    def _render_line_chart(plot_df, metric_title, y_title, height=300):
        if plot_df is None or plot_df.empty:
            st.info(f"No usable {metric_title.lower()} data found.")
            return

        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Starting Distance"),
                y=alt.Y("Rate:Q", title=y_title, scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[
                    alt.Tooltip("Bucket:N", title="Distance"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Count:Q", title="Count", format=",.0f"),
                    alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
                    alt.Tooltip("Rate:Q", title=y_title, format=".1f"),
                    alt.Tooltip("Label:N", title="Summary"),
                ],
            )
            .properties(height=height, title=metric_title)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.20)",
                domainColor="rgba(255,255,255,0.20)",
            )
        )
        st.altair_chart(chart, use_container_width=True)

    make_df = _build_metric_long("make")
    three_df = _build_metric_long("three")

    _render_line_chart(make_df, "1-Putt Conversion by Starting Distance", "1-Putt %", height=320)
    _render_line_chart(three_df, "3+ Putt Risk by Starting Distance", "3+ Putt %", height=280)

    # ---------------------------
    # Compact comparison table
    # ---------------------------
    table = pd.DataFrame({"Bucket": bucket_order})
    for label, fp in series_defs:
        block = fp[["Bucket", "Attempts", "OnePutt", "MakePct", "ThreePuttPlus", "ThreePuttPct"]].copy()
        block["Bucket"] = block["Bucket"].astype(str)
        block = block.rename(columns={
            "Attempts": f"{label} Attempts",
            "OnePutt": f"{label} 1-Putts",
            "MakePct": f"{label} 1-Putt %",
            "ThreePuttPlus": f"{label} 3+ Putts",
            "ThreePuttPct": f"{label} 3+ Putt %",
        })
        table = table.merge(block, on="Bucket", how="left")

    table["Bucket"] = pd.Categorical(table["Bucket"].astype(str), categories=bucket_order, ordered=True)
    table = table.sort_values("Bucket").reset_index(drop=True)

    # Keep empty buckets visible for context, but don't clutter with NaN.
    for col in table.columns:
        if col != "Bucket":
            table[col] = pd.to_numeric(table[col], errors="coerce")

    _render_golf_dataframe(table, use_container_width=True, hide_index=True)


def summarize_approach_bucket_for_chart(frame):
    """
    Bucketed approach summary that works on either raw hole data or an already-prepared
    frame returned by apply_approach_filters().
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    if "Approach Bucket" in frame.columns and "Approach GIR Flag" in frame.columns:
        d = frame.copy()
    else:
        d = prepare_approach_frame(frame)

    if d.empty or "Approach Bucket" not in d.columns:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out = (
        d.dropna(subset=["Approach Bucket"])
         .groupby("Approach Bucket", as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean"),
         )
         .rename(columns={"Approach Bucket": "Bucket"})
    )
    if out.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out["Pct"] = (pd.to_numeric(out["Made"], errors="coerce").fillna(0) / pd.to_numeric(out["Attempts"], errors="coerce").replace(0, pd.NA) * 100).fillna(0).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {float(r['Pct']):.1f}% | Prox {float(r['AvgProx']):.1f} ft" if int(r["Attempts"]) else "0/0 • 0.0%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"].astype(str), categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)[["Bucket", "Attempts", "Made", "Pct", "Label", "AvgProx"]]


def _approach_summary_for_chart(summary_df, series_label, metric="gir"):
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=["Bucket", "Series", "Rate", "Attempts", "Count", "Label"])

    d = summary_df.copy()
    d = d[d["Bucket"].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Series", "Rate", "Attempts", "Count", "Label"])

    d["Bucket"] = d["Bucket"].astype(str)
    d["Attempts"] = pd.to_numeric(d.get("Attempts", 0), errors="coerce").fillna(0).astype(int)
    d["Made"] = pd.to_numeric(d.get("Made", 0), errors="coerce").fillna(0).astype(int)
    d["Pct"] = pd.to_numeric(d.get("Pct", 0), errors="coerce").fillna(0.0)
    d["AvgProx"] = pd.to_numeric(d.get("AvgProx", 0), errors="coerce").fillna(0.0)

    if metric == "prox":
        d["Rate"] = d["AvgProx"]
        d["Count"] = d["Attempts"]
        d["Label"] = d.apply(lambda r: f"{float(r['AvgProx']):.1f} ft • {int(r['Attempts'])} shots" if int(r["Attempts"]) else "—", axis=1)
    else:
        d["Rate"] = d["Pct"]
        d["Count"] = d["Made"]
        d["Label"] = d.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {float(r['Pct']):.1f}%" if int(r["Attempts"]) else "0/0 • 0.0%", axis=1)

    d["Series"] = series_label
    return d[["Bucket", "Series", "Rate", "Attempts", "Count", "Label"]]


def render_approach_distance_player_compare(
    round_summary,
    baseline_summary,
    bucket_order,
    compare_label="Baseline",
    overlay_summaries=None,
    show_round=True,
    show_baseline=True,
    title="Approach Distance Control — Player Comparison",
):
    """
    Analysis-tab player comparison:
    - GIR % by approach distance bucket
    - average proximity by approach distance bucket
    - optional selected-round line
    - optional selected baseline line
    - multiple player overlays
    """
    overlay_summaries = overlay_summaries or []

    series_defs = []
    if show_round and round_summary is not None and not round_summary.empty:
        series_defs.append(("This Round", round_summary.copy()))
    if show_baseline and baseline_summary is not None and not baseline_summary.empty:
        series_defs.append((compare_label, baseline_summary.copy()))
    for label, summary in overlay_summaries:
        if label and summary is not None and not summary.empty:
            series_defs.append((str(label), summary.copy()))

    st.markdown(f"#### {title}")
    st.caption("Lines show approach performance by distance bucket. Hide the selected round for a cleaner player-vs-player view.")

    if not series_defs:
        st.info("No usable approach distance data found for this comparison.")
        return

    def _build_long(metric):
        frames = [_approach_summary_for_chart(summary, label, metric=metric) for label, summary in series_defs]
        out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if out.empty:
            return out
        out = out[out["Bucket"].notna()].copy()
        out["Bucket"] = pd.Categorical(out["Bucket"].astype(str), categories=bucket_order, ordered=True)
        return out.sort_values(["Bucket", "Series"]).copy()

    def _render_line(plot_df, metric_title, y_title, height=300, pct_axis=True):
        if plot_df is None or plot_df.empty:
            st.info(f"No usable {metric_title.lower()} data found.")
            return
        y_encoding = alt.Y("Rate:Q", title=y_title, scale=alt.Scale(domain=[0, 100])) if pct_axis else alt.Y("Rate:Q", title=y_title)
        chart = (
            alt.Chart(plot_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("Bucket:N", sort=bucket_order, title="Approach Distance"),
                y=y_encoding,
                color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                tooltip=[
                    alt.Tooltip("Bucket:N", title="Distance"),
                    alt.Tooltip("Series:N", title="Series"),
                    alt.Tooltip("Count:Q", title="Made / Shots", format=",.0f"),
                    alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
                    alt.Tooltip("Rate:Q", title=y_title, format=".1f"),
                    alt.Tooltip("Label:N", title="Summary"),
                ],
            )
            .properties(height=height, title=metric_title)
            .configure_view(strokeOpacity=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.20)",
                domainColor="rgba(255,255,255,0.20)",
            )
        )
        st.altair_chart(chart, use_container_width=True)

    gir_long = _build_long("gir")
    prox_long = _build_long("prox")
    _render_line(gir_long, "Approach GIR % by Distance", "GIR %", height=320, pct_axis=True)
    _render_line(prox_long, "Average Proximity by Distance", "Avg Proximity (ft)", height=290, pct_axis=False)

    table = pd.DataFrame({"Bucket": bucket_order})
    for label, summary in series_defs:
        block = summary[["Bucket", "Attempts", "Made", "Pct", "AvgProx"]].copy()
        block["Bucket"] = block["Bucket"].astype(str)
        block = block.rename(columns={
            "Attempts": f"{label} Attempts",
            "Made": f"{label} GIR Made",
            "Pct": f"{label} GIR %",
            "AvgProx": f"{label} Avg Prox",
        })
        table = table.merge(block, on="Bucket", how="left")

    table["Bucket"] = pd.Categorical(table["Bucket"].astype(str), categories=bucket_order, ordered=True)
    table = table.sort_values("Bucket").reset_index(drop=True)
    for col in table.columns:
        if col != "Bucket":
            table[col] = pd.to_numeric(table[col], errors="coerce")
    st.dataframe(table, use_container_width=True, hide_index=True)



def summarize_approach_club_for_chart(frame, min_attempts=1):
    """
    Club approach summary for player-comparison charts.
    Works on raw hole data or already-prepared approach frames.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    if "Approach Club" in frame.columns and "Approach GIR Flag" in frame.columns:
        d = frame.copy()
    else:
        d = prepare_approach_frame(frame)

    if d.empty or "Approach Club" not in d.columns:
        return pd.DataFrame(columns=["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    d = d[d["Approach Club"].astype(str).str.strip() != ""].copy()
    if d.empty:
        return pd.DataFrame(columns=["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out = (
        d.groupby("Approach Club", as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean"),
         )
         .rename(columns={"Approach Club": "Club"})
    )
    out = out[pd.to_numeric(out["Attempts"], errors="coerce").fillna(0) >= int(min_attempts)].copy()
    if out.empty:
        return pd.DataFrame(columns=["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"])

    out["Pct"] = (
        pd.to_numeric(out["Made"], errors="coerce").fillna(0)
        / pd.to_numeric(out["Attempts"], errors="coerce").replace(0, pd.NA)
        * 100
    ).fillna(0).round(1)
    out["Label"] = out.apply(
        lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {float(r['Pct']):.1f}%",
        axis=1,
    )
    out["_sort"] = out["Club"].astype(str).apply(_club_sort_key) if "_club_sort_key" in globals() else out["Club"].astype(str)
    return out.sort_values(["_sort", "Club"]).drop(columns=["_sort"], errors="ignore").reset_index(drop=True)[["Club", "Attempts", "Made", "Pct", "Label", "AvgProx"]]


def _approach_category_summary_for_chart(summary_df, series_label, category_col):
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=[category_col, "Series", "GIRPct", "Attempts", "Made", "AvgProx", "Label"])

    d = summary_df.copy()
    if category_col not in d.columns:
        return pd.DataFrame(columns=[category_col, "Series", "GIRPct", "Attempts", "Made", "AvgProx", "Label"])

    d = d[d[category_col].notna()].copy()
    if d.empty:
        return pd.DataFrame(columns=[category_col, "Series", "GIRPct", "Attempts", "Made", "AvgProx", "Label"])

    d[category_col] = d[category_col].astype(str)
    d["Attempts"] = pd.to_numeric(d.get("Attempts", 0), errors="coerce").fillna(0).astype(int)
    d["Made"] = pd.to_numeric(d.get("Made", 0), errors="coerce").fillna(0).astype(int)
    d["GIRPct"] = pd.to_numeric(d.get("Pct", 0), errors="coerce").fillna(0.0)
    d["AvgProx"] = pd.to_numeric(d.get("AvgProx", 0), errors="coerce").fillna(0.0)
    d["Series"] = series_label
    d["Label"] = d.apply(
        lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {float(r['GIRPct']):.1f}%" if int(r["Attempts"]) else "0/0 • 0.0%",
        axis=1,
    )
    return d[[category_col, "Series", "GIRPct", "Attempts", "Made", "AvgProx", "Label"]]


def render_approach_gir_player_compare_bars(
    distance_series_defs,
    club_series_defs,
    bucket_order,
    min_attempts=3,
    max_clubs=14,
):
    """
    Player-comparison bar charts for:
    - GIR % by yardage bucket
    - GIR % by approach club

    This is intentionally separate from the round-review approach charts so it can be
    used as a clean player-vs-player comparison, like the putting section.
    """
    st.markdown("#### Player Compare — GIR % by Yardage and Club")
    st.caption("Grouped bars compare GIR % between selected players. Tooltips include makes, attempts, and average proximity so sample size is visible.")

    # ---------- Yardage range chart ----------
    dist_frames = []
    for label, summary in distance_series_defs:
        if label and summary is not None and not summary.empty:
            block = summary.copy()
            block = block[pd.to_numeric(block.get("Attempts", 0), errors="coerce").fillna(0) >= int(min_attempts)].copy()
            dist_frames.append(_approach_category_summary_for_chart(block, label, "Bucket"))

    dist_long = pd.concat(dist_frames, ignore_index=True) if dist_frames else pd.DataFrame()
    if not dist_long.empty:
        dist_long["Bucket"] = pd.Categorical(dist_long["Bucket"].astype(str), categories=bucket_order, ordered=True)
        dist_long = dist_long.sort_values(["Bucket", "Series"]).copy()
        dist_present = [b for b in bucket_order if b in dist_long["Bucket"].astype(str).unique().tolist()]

        dist_chart = (
            alt.Chart(dist_long)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Bucket:N", sort=dist_present, title="Yardage Range"),
                y=alt.Y("GIRPct:Q", title="GIR %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                xOffset=alt.XOffset("Series:N"),
                tooltip=[
                    alt.Tooltip("Bucket:N", title="Yardage"),
                    alt.Tooltip("Series:N", title="Player / Series"),
                    alt.Tooltip("Made:Q", title="GIR Made", format=",.0f"),
                    alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
                    alt.Tooltip("GIRPct:Q", title="GIR %", format=".1f"),
                    alt.Tooltip("AvgProx:Q", title="Avg Prox", format=".1f"),
                    alt.Tooltip("Label:N", title="Summary"),
                ],
            )
            .properties(height=360, title="GIR % by Yardage Range — Player Compare")
            .configure_view(strokeOpacity=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.20)",
                domainColor="rgba(255,255,255,0.20)",
            )
        )
        st.altair_chart(dist_chart, use_container_width=True)

        dist_table = dist_long.rename(columns={"Bucket": "Yardage", "GIRPct": "GIR %", "AvgProx": "Avg Prox"})[
            ["Yardage", "Series", "Attempts", "Made", "GIR %", "Avg Prox"]
        ]
        _render_golf_dataframe(dist_table, use_container_width=True, hide_index=True)

        dist_matrix = dist_long.copy()
        dist_matrix["Summary"] = dist_matrix.apply(
            lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} ({float(r['GIRPct']):.1f}%) | Prox {float(r['AvgProx']):.1f} ft",
            axis=1,
        )
        dist_pivot = (
            dist_matrix.pivot_table(index="Bucket", columns="Series", values="Summary", aggfunc="first")
            .reset_index()
            .rename(columns={"Bucket": "Yardage"})
        )
        dist_pivot["Yardage"] = pd.Categorical(dist_pivot["Yardage"].astype(str), categories=bucket_order, ordered=True)
        dist_pivot = dist_pivot.sort_values("Yardage").reset_index(drop=True)
        st.markdown("##### Yardage Summary Matrix")
        st.caption("Each cell shows GIR made/attempts, GIR %, and average proximity. Buckets stay detailed so the comparison does not get too general.")
        st.dataframe(dist_pivot, use_container_width=True, hide_index=True)
    else:
        st.info("No yardage buckets qualify for the player comparison with the current minimum attempts.")

    # ---------- Club chart ----------
    club_frames = []
    for label, summary in club_series_defs:
        if label and summary is not None and not summary.empty:
            block = summary.copy()
            block = block[pd.to_numeric(block.get("Attempts", 0), errors="coerce").fillna(0) >= int(min_attempts)].copy()
            club_frames.append(_approach_category_summary_for_chart(block, label, "Club"))

    club_long = pd.concat(club_frames, ignore_index=True) if club_frames else pd.DataFrame()
    if not club_long.empty:
        club_totals = (
            club_long.groupby("Club", as_index=False)["Attempts"]
            .sum()
            .sort_values(["Attempts", "Club"], ascending=[False, True])
        )
        club_order = club_totals["Club"].head(int(max_clubs)).tolist()
        club_long = club_long[club_long["Club"].isin(club_order)].copy()
        club_order = sorted(club_order, key=_club_sort_key) if "_club_sort_key" in globals() else club_order

        club_chart = (
            alt.Chart(club_long)
            .mark_bar(cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
            .encode(
                x=alt.X("Club:N", sort=club_order, title="Club"),
                y=alt.Y("GIRPct:Q", title="GIR %", scale=alt.Scale(domain=[0, 100])),
                color=alt.Color("Series:N", title=None, legend=alt.Legend(orient="top")),
                xOffset=alt.XOffset("Series:N"),
                tooltip=[
                    alt.Tooltip("Club:N", title="Club"),
                    alt.Tooltip("Series:N", title="Player / Series"),
                    alt.Tooltip("Made:Q", title="GIR Made", format=",.0f"),
                    alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
                    alt.Tooltip("GIRPct:Q", title="GIR %", format=".1f"),
                    alt.Tooltip("AvgProx:Q", title="Avg Prox", format=".1f"),
                    alt.Tooltip("Label:N", title="Summary"),
                ],
            )
            .properties(height=360, title="GIR % by Club — Player Compare")
            .configure_view(strokeOpacity=0)
            .configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.20)",
                domainColor="rgba(255,255,255,0.20)",
            )
        )
        st.altair_chart(club_chart, use_container_width=True)

        club_table = club_long.rename(columns={"GIRPct": "GIR %", "AvgProx": "Avg Prox"})[
            ["Club", "Series", "Attempts", "Made", "GIR %", "Avg Prox"]
        ]
        _render_golf_dataframe(club_table, use_container_width=True, hide_index=True)

        club_matrix = club_long.copy()
        club_matrix["Summary"] = club_matrix.apply(
            lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} ({float(r['GIRPct']):.1f}%) | Prox {float(r['AvgProx']):.1f} ft",
            axis=1,
        )
        club_pivot = (
            club_matrix.pivot_table(index="Club", columns="Series", values="Summary", aggfunc="first")
            .reset_index()
        )
        club_pivot["_sort"] = club_pivot["Club"].astype(str).apply(_club_sort_key) if "_club_sort_key" in globals() else range(len(club_pivot))
        club_pivot = club_pivot.sort_values(["_sort", "Club"]).drop(columns=["_sort"], errors="ignore").reset_index(drop=True)
        st.markdown("##### Club Summary Matrix")
        st.caption("Each cell shows GIR made/attempts, GIR %, and average proximity for that club.")
        st.dataframe(club_pivot, use_container_width=True, hide_index=True)
    else:
        st.info("No clubs qualify for the player comparison with the current minimum attempts.")

def build_overview_snapshot(frame):
    d = frame.copy()
    holes = int(len(d))
    score = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
    par = pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)
    putts = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0)
    gir = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0)

    fw_block = d[pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0).isin([4, 5])].copy()
    fw = pd.to_numeric(_safe_col(fw_block, "Fairway", 0), errors="coerce").fillna(0) if not fw_block.empty else pd.Series(dtype=float)

    scramble_made = pd.to_numeric(_safe_col(d, "Scramble", 0), errors="coerce").fillna(0).sum()
    scramble_ops = pd.to_numeric(_safe_col(d, "Scramble Opportunity", 0), errors="coerce").fillna(0).sum()
    updown_made = (((gir == 0) & (putts == 1)).sum())

    prox_series = pd.to_numeric(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", pd.NA), errors="coerce")
    valid_prox = prox_series.notna() & (prox_series > 0)
    avg_prox = prox_series[valid_prox].mean()
    avg_prox_gir = prox_series[valid_prox & (gir == 1)].mean()
    avg_prox_no_gir = prox_series[valid_prox & (gir == 0)].mean()

    lost_balls = (
        pd.to_numeric(_safe_col(d, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(_safe_col(d, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0).sum()
    )
    arnies = pd.to_numeric(_safe_col(d, "Arnie", 0), errors="coerce").fillna(0).sum()
    seves = pd.to_numeric(_safe_col(d, "Seve", 0), errors="coerce").fillna(0).sum()

    total_score = float(score.sum())

    weighted_score_per18 = _standardized_score_per18(d)

    return {
        "holes": holes,
        "total_score": total_score,
        "score_per18": weighted_score_per18,
        "putts_per18": _per18(float(putts.sum()), holes),
        "gir_pct": (float(gir.sum()) / holes * 100.0) if holes else 0.0,
        "fw_pct": (float(fw.sum()) / len(fw) * 100.0) if len(fw) else 0.0,
        "scramble_pct": (float(scramble_made) / float(scramble_ops) * 100.0) if scramble_ops else 0.0,
        "updown_pct": (float(updown_made) / float(scramble_ops) * 100.0) if scramble_ops else 0.0,
        "arnies_per18": _per18(float(arnies), holes),
        "seves_per18": _per18(float(seves), holes),
        "avg_prox": float(avg_prox) if pd.notna(avg_prox) else 0.0,
        "avg_prox_gir": float(avg_prox_gir) if pd.notna(avg_prox_gir) else 0.0,
        "avg_prox_no_gir": float(avg_prox_no_gir) if pd.notna(avg_prox_no_gir) else 0.0,
        "lost_balls_per18": _per18(float(lost_balls), holes),
    }

    score = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0).sum()
    putts = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0).sum()
    gir = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0)
    par = pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")
    fw = pd.to_numeric(_safe_col(d, "Fairway", 0), errors="coerce").fillna(0)
    fw_block = d[par.isin([4, 5])] if "Par" in d.columns else d.iloc[0:0]
    fw_att = int(len(fw_block))
    fw_made = pd.to_numeric(_safe_col(fw_block, "Fairway", 0), errors="coerce").fillna(0).sum() if fw_att else 0

    scramble_made = pd.to_numeric(_safe_col(d, "Scramble", 0), errors="coerce").fillna(0).sum()
    scramble_att = pd.to_numeric(_safe_col(d, "Scramble Opportunity", 0), errors="coerce").fillna(0).sum()
    up_made = (((pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0) == 0) & (pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0) == 1)).sum())
    up_att = scramble_att
    prox = pd.to_numeric(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", pd.NA), errors="coerce")
    prox_avg = float(prox.dropna().mean()) if prox.notna().any() else 0.0
    lost_balls = pd.to_numeric(_safe_col(d, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0).sum() + pd.to_numeric(_safe_col(d, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0).sum()

    return {
        "holes": holes,
        "score_per18": float(score) / holes * 18.0,
        "putts_per18": float(putts) / holes * 18.0,
        "gir_pct": float(gir.sum()) / holes * 100.0 if holes else 0.0,
        "fw_pct": float(fw_made) / fw_att * 100.0 if fw_att else 0.0,
        "scramble_pct": float(scramble_made) / scramble_att * 100.0 if scramble_att else 0.0,
        "updown_pct": float(up_made) / up_att * 100.0 if up_att else 0.0,
        "avg_prox": prox_avg,
        "lost_balls_per18": float(lost_balls) / holes * 18.0,
    }


def build_putting_fingerprint_insights(frame):
    fp = build_putting_skill_fingerprint(frame)
    if fp.empty:
        return []

    qualified = fp[pd.to_numeric(fp["Attempts"], errors="coerce").fillna(0) >= 2].copy()
    if qualified.empty:
        qualified = fp.copy()

    insights = []
    if not qualified.empty:
        strong = qualified.sort_values(["MakePct", "Attempts"], ascending=[False, False]).iloc[0]
        weak = qualified.sort_values(["MakePct", "Attempts"], ascending=[True, False]).iloc[0]
        worst_3p = qualified.sort_values(["ThreePuttPct", "Attempts"], ascending=[False, False]).iloc[0]

        insights.append(f"🔥 Strongest putting range: {strong['Bucket']} ({strong['OnePutt']}/{strong['Attempts']}, {strong['MakePct']:.1f}%)")
        insights.append(f"❄️ Weakest conversion range: {weak['Bucket']} ({weak['OnePutt']}/{weak['Attempts']}, {weak['MakePct']:.1f}%)")
        if float(worst_3p.get("ThreePuttPct", 0)) > 0:
            insights.append(f"⚠️ Highest 3-putt pressure zone: {worst_3p['Bucket']} ({worst_3p['ThreePuttPlus']}/{worst_3p['Attempts']}, {worst_3p['ThreePuttPct']:.1f}%)")
    return insights[:3]


def build_short_game_proximity_stats(frame):
    d = prepare_short_game_frame(frame)
    if d.empty:
        return {
            "attempts": 0,
            "avg_leave_ft": 0.0,
            "inside3": 0,
            "inside3_pct": 0.0,
            "inside6": 0,
            "inside6_pct": 0.0,
            "inside10": 0,
            "inside10_pct": 0.0,
            "convert_inside10": 0,
            "convert_inside10_pct": 0.0,
        }

    prox = pd.to_numeric(d["SG Proximity"], errors="coerce").fillna(0)
    putts = pd.to_numeric(d["SG Putts"], errors="coerce").fillna(0)
    attempts = int(len(d))
    inside3 = int((prox <= 3).sum())
    inside6 = int((prox <= 6).sum())
    inside10 = int((prox <= 10).sum())
    convert_inside10 = int(((prox <= 10) & (putts == 1)).sum())

    return {
        "attempts": attempts,
        "avg_leave_ft": float(prox.mean()) if attempts else 0.0,
        "inside3": inside3,
        "inside3_pct": (inside3 / attempts * 100.0) if attempts else 0.0,
        "inside6": inside6,
        "inside6_pct": (inside6 / attempts * 100.0) if attempts else 0.0,
        "inside10": inside10,
        "inside10_pct": (inside10 / attempts * 100.0) if attempts else 0.0,
        "convert_inside10": convert_inside10,
        "convert_inside10_pct": (convert_inside10 / inside10 * 100.0) if inside10 else 0.0,
    }


def build_bogey_avoidance_stats(frame):
    d = frame.copy()
    d["GIR Clean"] = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0).astype(int)
    if "Score to Par" in d.columns:
        d["ScoreToParN"] = pd.to_numeric(_safe_col(d, "Score to Par", pd.NA), errors="coerce")
    else:
        d["ScoreToParN"] = pd.to_numeric(_safe_col(d, "Hole Score", pd.NA), errors="coerce") - pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")

    miss_gir = d[d["GIR Clean"] == 0].copy()
    attempts = int(len(miss_gir))
    bogey_or_better = int((pd.to_numeric(miss_gir["ScoreToParN"], errors="coerce").fillna(99) <= 1).sum()) if attempts else 0
    par_or_better = int((pd.to_numeric(miss_gir["ScoreToParN"], errors="coerce").fillna(99) <= 0).sum()) if attempts else 0
    bogey_or_worse = int((pd.to_numeric(miss_gir["ScoreToParN"], errors="coerce").fillna(-99) >= 1).sum()) if attempts else 0

    return {
        "attempts": attempts,
        "bogey_or_better": bogey_or_better,
        "bogey_or_better_pct": (bogey_or_better / attempts * 100.0) if attempts else 0.0,
        "par_or_better": par_or_better,
        "par_or_better_pct": (par_or_better / attempts * 100.0) if attempts else 0.0,
        "bogey_or_worse": bogey_or_worse,
        "bogey_or_worse_pct": (bogey_or_worse / attempts * 100.0) if attempts else 0.0,
    }


def build_short_game_hole_out_stats(frame):
    d = frame.copy()
    rel = _safe_col(d, "Approach Shot Direction Relative to Hole", "").fillna("").astype(str).str.strip().str.lower()
    gir = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0).astype(int)
    chip_opps = pd.to_numeric(_safe_col(d, "Chip Opportunity", 0), errors="coerce").fillna(0)

    # Short-game hole outs only: source-of-truth text is "Hole Out",
    # restricted to missed-GIR / chip-opportunity style holes.
    hole_out_mask = (rel == "hole out") & ((gir == 0) | (chip_opps > 0))
    count = int(hole_out_mask.sum())
    return {"count": count}


def build_short_game_insights(frame):
    prox = build_short_game_proximity_stats(frame)
    bogey = build_bogey_avoidance_stats(frame)
    extra = build_short_game_extra_stats(frame)
    hole_outs = build_short_game_hole_out_stats(frame)

    insights = []
    if prox["attempts"]:
        insights.append(f"🎯 Avg short-game leave: {prox['avg_leave_ft']:.1f} ft")
        insights.append(f"⛳ Inside 6 ft: {prox['inside6']}/{prox['attempts']} ({prox['inside6_pct']:.1f}%)")
        if prox["inside10"]:
            insights.append(f"🪄 Converted from inside 10 ft: {prox['convert_inside10']}/{prox['inside10']} ({prox['convert_inside10_pct']:.1f}%)")
    if bogey["attempts"]:
        insights.append(f"🛡️ Bogey avoidance after missed GIR: {bogey['par_or_better']}/{bogey['attempts']} ({bogey['par_or_better_pct']:.1f}%)")
    if hole_outs["count"]:
        insights.append(f"🕳️ Hole outs: {hole_outs['count']}")
    elif extra["opportunities"]:
        insights.append(f"⚠️ Holes needing 2+ chips: {extra['holes_2plus']}/{extra['opportunities']}")
    return insights[:5]

# =========================================================
# Short game
# =========================================================
def prepare_short_game_frame(frame):
    d = frame.copy()
    d["SG Proximity"] = _num(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", 0))
    d["SG GIR"] = _int(_safe_col(d, "GIR", 0))
    d["SG Putts"] = _int(_safe_col(d, "Putts", 0))
    d["SG Bucket"] = d["SG Proximity"].apply(lambda x: _bucket_value(x, SHORT_GAME_BUCKETS))
    d["SG Attempt"] = ((d["SG GIR"] == 0) & (d["SG Proximity"] > 0)).astype(int)
    d["SG OnePutt"] = ((d["SG GIR"] == 0) & (d["SG Proximity"] > 0) & (d["SG Putts"] == 1)).astype(int)
    d = d[d["SG Attempt"] == 1].copy()
    return d

def summarize_short_game_by_bucket(frame):
    d = prepare_short_game_frame(frame)
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label"])

    out = (
        d.dropna(subset=["SG Bucket"])
         .groupby("SG Bucket", as_index=False)
         .agg(
             Attempts=("SG Attempt", "sum"),
             Made=("SG OnePutt", "sum")
         )
         .rename(columns={"SG Bucket": "Bucket"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=SHORT_GAME_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)[["Bucket", "Attempts", "Made", "Pct", "Label"]]

# =========================================================
# Shared rendering helpers
# =========================================================
def render_bucket_compare_tab(round_summary, bench_summary, key_col, key_order, compare_mode, title, x_title, table_round_prefix="Round", table_bench_prefix=None):
    if table_bench_prefix is None:
        table_bench_prefix = compare_mode

    long_df = build_compare_long(round_summary, bench_summary, key_col, round_label="Round", bench_label=compare_mode)
    if long_df.empty:
        st.info(f"No usable {title.lower()} data found for this round / comparison group.")
        return

    long_df[key_col] = pd.Categorical(long_df[key_col], categories=key_order, ordered=True)
    long_df = long_df.sort_values([key_col, "Series"]).copy()
    long_df["DisplayLabel"] = long_df.apply(
        lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%",
        axis=1
    )

    max_pct = float(long_df["Pct"].max()) if "Pct" in long_df and not long_df.empty else 100.0
    label_pad = 22.0
    x_max = max(100.0, max_pct + label_pad)

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{key_col}:N", sort=key_order, title=title),
            x=alt.X("Pct:Q", title=x_title, scale=alt.Scale(domain=[0, x_max])),
            color=alt.Color("Series:N", title="Series"),
            xOffset="Series:N",
            tooltip=[
                alt.Tooltip(f"{key_col}:N"),
                alt.Tooltip("Series:N"),
                alt.Tooltip("Made:Q"),
                alt.Tooltip("Attempts:Q"),
                alt.Tooltip("Pct:Q", format=".1f"),
                alt.Tooltip("Label:N", title="Summary"),
            ],
        )
        .properties(height=max(320, len(key_order) * 28))
    )

    label_chart = (
        alt.Chart(long_df)
        .mark_text(align="left", dx=6, fontWeight="bold", clip=False)
        .encode(
            y=alt.Y(f"{key_col}:N", sort=key_order),
            x=alt.X("Pct:Q", scale=alt.Scale(domain=[0, x_max])),
            text="DisplayLabel:N",
        )
    )

    st.altair_chart((chart + label_chart).configure_view(clip=False), use_container_width=True)

    table = pd.merge(
        round_summary.rename(columns={
            "Attempts": f"{table_round_prefix} Attempts",
            "Made": f"{table_round_prefix} Made",
            "Pct": f"{table_round_prefix} %",
        }),
        bench_summary.rename(columns={
            "Attempts": f"{table_bench_prefix} Attempts",
            "Made": f"{table_bench_prefix} Made",
            "Pct": f"{table_bench_prefix} %",
        }),
        on=key_col,
        how="outer",
    )
    st.dataframe(table.sort_values(key_col), use_container_width=True, hide_index=True)

def render_debug_section(title, debug_df):
    with st.expander(title, expanded=False):
        if debug_df.empty:
            st.info("No rows to validate.")
        else:
            st.dataframe(debug_df, use_container_width=True, hide_index=True)


def render_paired_compare_bars(compare_df, key_col, key_order, compare_mode, title, x_title="GIR %"):
    """
    Reliable visible labels:
    - one row for Round
    - one row for selected baseline
    - explicit value shown at right, e.g. 3/5 60%
    - round label gets green/red tint vs baseline
    """
    if compare_df.empty:
        st.info(f"No usable {title.lower()} data found for this round / comparison group.")
        return

    plot_df = compare_df.copy()
    plot_df[key_col] = plot_df[key_col].astype(str)
    plot_df["DisplayLabel"] = plot_df.apply(
        lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%",
        axis=1
    )

    present = set(plot_df[key_col].tolist())
    ordered_categories = [str(k) for k in key_order if str(k) in present]

    if not ordered_categories:
        st.info(f"No usable {title.lower()} data found for this round / comparison group.")
        return

    html = """
    <style>
      .pair-wrap {background:#222; padding:10px 12px; border-radius:12px; margin-bottom:10px;}
      .pair-cat {margin:6px 0 9px 0; padding-bottom:7px; border-bottom:1px solid rgba(255,255,255,.08);}
      .pair-cat:last-child {border-bottom:none; margin-bottom:0; padding-bottom:0;}
      .pair-cat-title {font-weight:700; margin-bottom:4px; color:#fff; font-size:12px; line-height:1.15;}
      .pair-row {
        display:grid;
        grid-template-columns: 82px minmax(120px, 1fr) 116px 50px;
        gap:8px;
        align-items:center;
        margin:2px 0;
      }
      .pair-series {font-size:11px; color:#ddd; font-weight:700;}
      .pair-bar-bg {
        width:100%;
        background:#3a3a3a;
        border-radius:999px;
        height:14px;
        overflow:hidden;
        position:relative;
      }
      .pair-bar-fill-round {
        height:14px;
        border-radius:999px;
        background:#4f8cff;
      }
      .pair-bar-fill-base {
        height:14px;
        border-radius:999px;
        background:#8a8a8a;
      }
      .pair-value {
        font-size:11px;
        color:#fff;
        font-variant-numeric: tabular-nums;
        text-align:left;
        white-space:nowrap;
        font-weight:700;
      }
      .pair-delta {
        font-size:10px;
        font-weight:700;
        text-align:right;
        white-space:nowrap;
      }
      .pair-good {color:#64dfb5;}
      .pair-bad {color:#ee6c4d;}
      .pair-neutral {color:#aaa;}
    </style>
    <div class="pair-wrap">
    """

    for cat in ordered_categories:
        html += f'<div class="pair-cat"><div class="pair-cat-title">{cat}</div>'
        cat_df = plot_df[plot_df[key_col] == cat].copy()

        round_row = cat_df[cat_df["Series"] == "Round"]
        base_row = cat_df[cat_df["Series"] == compare_mode]

        round_pct = float(round_row["Pct"].iloc[0]) if not round_row.empty else 0.0
        base_pct = float(base_row["Pct"].iloc[0]) if not base_row.empty else 0.0
        delta = round_pct - base_pct

        for series in ["Round", compare_mode]:
            row = cat_df[cat_df["Series"] == series]
            if row.empty:
                pct = 0.0
                label = "0/0 0%"
            else:
                pct = float(row["Pct"].iloc[0])
                label = str(row["DisplayLabel"].iloc[0])

            fill_class = "pair-bar-fill-round" if series == "Round" else "pair-bar-fill-base"

            if series == "Round":
                if delta > 0.05:
                    delta_cls = "pair-good"
                    delta_txt = f"+{delta:.0f}%"
                elif delta < -0.05:
                    delta_cls = "pair-bad"
                    delta_txt = f"{delta:.0f}%"
                else:
                    delta_cls = "pair-neutral"
                    delta_txt = "0%"
            else:
                delta_cls = "pair-neutral"
                delta_txt = ""

            html += f"""
            <div class="pair-row">
              <div class="pair-series">{series}</div>
              <div class="pair-bar-bg">
                <div class="{fill_class}" style="width:{max(0, min(100, pct))}%;"></div>
              </div>
              <div class="pair-value">{label}</div>
              <div class="pair-delta {delta_cls}">{delta_txt}</div>
            </div>
            """

        html += "</div>"

    html += "</div>"

    import streamlit.components.v1 as components
    height_px = max(180, 38 + len(ordered_categories) * 58)
    components.html(html, height=height_px, scrolling=False)


def render_paired_compare_counts(round_df, bench_df, key_col, key_order, compare_mode, title, value_label="Count"):
    """
    Paired HTML bars for count-based comparisons like miss direction.
    Shows Round and baseline on separate rows with visible labels.
    """
    if round_df.empty and bench_df.empty:
        st.info(f"No usable {title.lower()} data found for this round / comparison group.")
        return

    r = round_df.copy()
    b = bench_df.copy()

    if key_col not in r.columns:
        r[key_col] = pd.Series(dtype="object")
    if key_col not in b.columns:
        b[key_col] = pd.Series(dtype="object")
    if "Count" not in r.columns:
        r["Count"] = 0
    if "Count" not in b.columns:
        b["Count"] = 0
    if "Pct" not in r.columns:
        r["Pct"] = 0.0
    if "Pct" not in b.columns:
        b["Pct"] = 0.0

    r[key_col] = r[key_col].astype(str)
    b[key_col] = b[key_col].astype(str)

    present = set(r[key_col].tolist()) | set(b[key_col].tolist())
    ordered_categories = [str(k) for k in key_order if str(k) in present]
    if not ordered_categories:
        ordered_categories = sorted(present)

    max_count = max(
        float(r["Count"].max()) if not r.empty else 0.0,
        float(b["Count"].max()) if not b.empty else 0.0,
        1.0
    )

    html = """
    <style>
      .pair-wrap {background:#222; padding:10px 12px; border-radius:12px; margin-bottom:10px;}
      .pair-cat {margin:6px 0 9px 0; padding-bottom:7px; border-bottom:1px solid rgba(255,255,255,.08);}
      .pair-cat:last-child {border-bottom:none; margin-bottom:0; padding-bottom:0;}
      .pair-cat-title {font-weight:700; margin-bottom:4px; color:#fff; font-size:12px; line-height:1.15;}
      .pair-row {
        display:grid;
        grid-template-columns: 82px minmax(120px, 1fr) 116px 50px;
        gap:8px;
        align-items:center;
        margin:2px 0;
      }
      .pair-series {font-size:11px; color:#ddd; font-weight:700;}
      .pair-bar-bg {
        width:100%;
        background:#3a3a3a;
        border-radius:999px;
        height:14px;
        overflow:hidden;
        position:relative;
      }
      .pair-bar-fill-round {
        height:14px;
        border-radius:999px;
        background:#4f8cff;
      }
      .pair-bar-fill-base {
        height:14px;
        border-radius:999px;
        background:#8a8a8a;
      }
      .pair-value {
        font-size:11px;
        color:#fff;
        font-variant-numeric: tabular-nums;
        text-align:left;
        white-space:nowrap;
        font-weight:700;
      }
      .pair-delta {
        font-size:10px;
        font-weight:700;
        text-align:right;
        white-space:nowrap;
      }
      .pair-good {color:#64dfb5;}
      .pair-bad {color:#ee6c4d;}
      .pair-neutral {color:#aaa;}
    </style>
    <div class="pair-wrap">
    """

    for cat in ordered_categories:
        html += f'<div class="pair-cat"><div class="pair-cat-title">{cat}</div>'

        r_row = r[r[key_col] == cat]
        b_row = b[b[key_col] == cat]

        round_count = float(r_row["Count"].iloc[0]) if not r_row.empty else 0.0
        round_pct = float(r_row["Pct"].iloc[0]) if not r_row.empty else 0.0
        base_count = float(b_row["Count"].iloc[0]) if not b_row.empty else 0.0
        base_pct = float(b_row["Pct"].iloc[0]) if not b_row.empty else 0.0
        delta = round_count - base_count

        for series, count, pct in [
            ("Round", round_count, round_pct),
            (compare_mode, base_count, base_pct),
        ]:
            width = 100.0 * count / max_count if max_count else 0.0
            label = f"{int(count)} ({pct:.0f}%)"
            fill_class = "pair-bar-fill-round" if series == "Round" else "pair-bar-fill-base"

            if series == "Round":
                if delta > 0.05:
                    delta_cls = "pair-good"
                    delta_txt = f"+{int(round(delta))}"
                elif delta < -0.05:
                    delta_cls = "pair-bad"
                    delta_txt = f"{int(round(delta))}"
                else:
                    delta_cls = "pair-neutral"
                    delta_txt = "0"
            else:
                delta_cls = "pair-neutral"
                delta_txt = ""

            html += f"""
            <div class="pair-row">
              <div class="pair-series">{series}</div>
              <div class="pair-bar-bg">
                <div class="{fill_class}" style="width:{max(0, min(100, width))}%;"></div>
              </div>
              <div class="pair-value">{label}</div>
              <div class="pair-delta {delta_cls}">{delta_txt}</div>
            </div>
            """

        html += "</div>"

    html += "</div>"

    import streamlit.components.v1 as components
    height_px = max(180, 38 + len(ordered_categories) * 58)
    components.html(html, height=height_px, scrolling=False)



def render_single_series_bars(summary_df, key_col, key_order, title, bar_label="GIR %"):
    """
    Clean single-series visible bar layout for dashboards like Shot Pattern.
    Always shows the explicit label at right (e.g. 3/5 60%).
    """
    if summary_df.empty:
        st.info(f"No usable {title.lower()} data found.")
        return

    df_plot = summary_df.copy()
    df_plot[key_col] = df_plot[key_col].astype(str)
    present = set(df_plot[key_col].tolist())
    ordered_categories = [str(k) for k in key_order if str(k) in present]
    if not ordered_categories:
        ordered_categories = sorted(present)

    html = """
    <style>
      .single-wrap {background:#222; padding:12px; border-radius:12px; margin-bottom:10px;}
      .single-cat {margin:8px 0 12px 0; padding-bottom:8px; border-bottom:1px solid rgba(255,255,255,.08);}
      .single-cat:last-child {border-bottom:none; margin-bottom:0; padding-bottom:0;}
      .single-row {
        display:grid;
        grid-template-columns: 110px minmax(140px, 1fr) 130px;
        gap:10px;
        align-items:center;
      }
      .single-name {font-size:12px; color:#fff; font-weight:700; white-space:nowrap;}
      .single-bar-bg {
        width:100%;
        background:#3a3a3a;
        border-radius:999px;
        height:18px;
        overflow:hidden;
        position:relative;
      }
      .single-bar-fill {
        height:18px;
        border-radius:999px;
        background:#4f8cff;
      }
      .single-value {
        font-size:12px;
        color:#fff;
        font-variant-numeric: tabular-nums;
        text-align:left;
        white-space:nowrap;
        font-weight:700;
      }
    </style>
    <div class="single-wrap">
    """

    for cat in ordered_categories:
        row = df_plot[df_plot[key_col] == cat]
        if row.empty:
            pct = 0.0
            label = "0/0 0%"
        else:
            pct = float(row["Pct"].iloc[0]) if "Pct" in row.columns else 0.0
            if "DisplayLabel" in row.columns:
                label = str(row["DisplayLabel"].iloc[0])
            elif all(c in row.columns for c in ["Made", "Attempts", "Pct"]):
                label = f"{int(row['Made'].iloc[0])}/{int(row['Attempts'].iloc[0])} {float(row['Pct'].iloc[0]):.0f}%"
            else:
                label = f"{pct:.0f}%"

        html += f"""
        <div class="single-cat">
          <div class="single-row">
            <div class="single-name">{cat}</div>
            <div class="single-bar-bg">
              <div class="single-bar-fill" style="width:{max(0, min(100, pct))}%;"></div>
            </div>
            <div class="single-value">{label}</div>
          </div>
        </div>
        """

    html += "</div>"
    import streamlit.components.v1 as components
    height_px = max(180, 26 + len(ordered_categories) * 48)
    components.html(html, height=height_px, scrolling=False)


def build_baseline_summary(frame, compare_mode="Baseline"):
    d = frame.copy()
    if d is None or d.empty:
        return {
            "rounds": 0,
            "players": 0,
            "courses": 0,
            "avg_score": None,
            "avg_to_par": None,
            "date_start": None,
            "date_end": None,
            "player_list": [],
            "course_list": [],
            "label": _compare_mode_display_label(compare_mode),
        }

    d["Date Played"] = pd.to_datetime(_safe_col(d, "Date Played", pd.NaT), errors="coerce")
    d["Hole Score"] = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
    d["Par"] = pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)

    round_col = _resolve_round_col(d)
    rounds = int(d[round_col].dropna().nunique()) if round_col and round_col in d.columns else 1
    players = sorted([str(x) for x in _safe_col(d, "Player Name", "").dropna().astype(str).unique().tolist() if str(x).strip()])
    courses = sorted([str(x) for x in _safe_col(d, "Course Name", "").dropna().astype(str).unique().tolist() if str(x).strip()])

    avg_score = None
    avg_to_par = None
    if round_col and round_col in d.columns:
        per_round_scores = []
        for _, block in d.groupby(round_col, dropna=True):
            per_round_scores.append(_standardized_score_per18(block))
        if per_round_scores:
            avg_score = float(pd.Series(per_round_scores).mean())
            avg_to_par = avg_score - 72.0
    else:
        avg_score = _standardized_score_per18(d)
        avg_to_par = avg_score - 72.0

    return {
        "rounds": rounds,
        "players": len(players),
        "courses": len(courses),
        "avg_score": avg_score,
        "avg_to_par": avg_to_par,
        "date_start": d["Date Played"].min() if "Date Played" in d.columns else None,
        "date_end": d["Date Played"].max() if "Date Played" in d.columns else None,
        "player_list": players,
        "course_list": courses,
        "label": _compare_mode_display_label(compare_mode),
    }


def render_baseline_summary_box(summary):
    label = summary.get("label", "Baseline")
    rounds = int(summary.get("rounds", 0) or 0)
    players = int(summary.get("players", 0) or 0)
    courses = int(summary.get("courses", 0) or 0)
    avg_score = summary.get("avg_score")
    avg_to_par = summary.get("avg_to_par")
    ds = summary.get("date_start")
    de = summary.get("date_end")
    player_list = summary.get("player_list", [])[:4]
    course_list = summary.get("course_list", [])[:4]

    date_txt = "—"
    if pd.notna(ds) and pd.notna(de):
        ds = pd.to_datetime(ds)
        de = pd.to_datetime(de)
        date_txt = f"{ds.month}/{ds.day}/{ds.year} → {de.month}/{de.day}/{de.year}"

    score_txt = "—" if avg_score is None or pd.isna(avg_score) else f"{avg_score:.1f}"
    to_par_txt = "—" if avg_to_par is None or pd.isna(avg_to_par) else _fmt_par_float(avg_to_par)
    players_txt = ", ".join(player_list) if player_list else "—"
    if len(summary.get("player_list", [])) > 4:
        players_txt += " …"
    courses_txt = ", ".join(course_list) if course_list else "—"
    if len(summary.get("course_list", [])) > 4:
        courses_txt += " …"

    st.markdown(f"""
    <div style="background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:12px 14px; margin:6px 0 12px 0; box-shadow:0 8px 18px rgba(0,0,0,.16);">
      <div style="font-size:13px; font-weight:800; color:#fff; margin-bottom:8px;">🧱 Baseline Summary — {label}</div>
      <div style="display:grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap:10px; margin-bottom:8px;">
        <div><div style="font-size:11px; color:#a9a9a9; font-weight:700;">Rounds</div><div style="font-size:21px; color:#fff; font-weight:900;">{rounds}</div></div>
        <div><div style="font-size:11px; color:#a9a9a9; font-weight:700;">Avg Score</div><div style="font-size:21px; color:#fff; font-weight:900;">{score_txt}</div></div>
        <div><div style="font-size:11px; color:#a9a9a9; font-weight:700;">Avg To Par</div><div style="font-size:21px; color:#fff; font-weight:900;">{to_par_txt}</div></div>
        <div><div style="font-size:11px; color:#a9a9a9; font-weight:700;">Players / Courses</div><div style="font-size:21px; color:#fff; font-weight:900;">{players} / {courses}</div></div>
      </div>
      <div style="font-size:12px; color:#d8d8d8; line-height:1.45;">
        <b>Date Range:</b> {date_txt}<br>
        <b>Players:</b> {players_txt}<br>
        <b>Courses:</b> {courses_txt}
      </div>
    </div>
    """, unsafe_allow_html=True)


def build_round_selector_df(src_df):
    work = src_df.copy()
    work["Date Played"] = pd.to_datetime(work["Date Played"], errors="coerce")
    work["Hole Score"] = pd.to_numeric(work["Hole Score"], errors="coerce").fillna(0)
    work["Par"] = pd.to_numeric(work["Par"], errors="coerce").fillna(0)

    meta = (
        work.groupby("Round Link", dropna=True)
        .agg(
            Player=("Player Name", "first"),
            Course=("Course Name", "first"),
            Date=("Date Played", "max"),
            Score=("Hole Score", "sum"),
            ParTotal=("Par", "sum"),
        )
        .reset_index()
    )

    def _fmt_round_row(r):
        if pd.notna(r["Date"]):
            date_str = r["Date"].strftime("%m/%d/%Y").lstrip("0").replace("/0", "/")
        else:
            date_str = "No Date"
        to_par = int(r["Score"] - r["ParTotal"])
        to_par_str = "E" if to_par == 0 else f"{'+' if to_par > 0 else ''}{to_par}"
        return f'{r["Player"]} — {r["Course"]} — {date_str} — {int(r["Score"])} ({to_par_str})'

    meta["Round Label"] = meta.apply(_fmt_round_row, axis=1)
    meta = meta.sort_values(["Date", "Round Label"], ascending=[False, True]).reset_index(drop=True)
    return meta

def build_approach_insights(round_dist, bench_dist, round_club):
    insights = []

    if not round_dist.empty:
        top_bucket = round_dist.sort_values(["Pct", "Attempts", "Bucket"], ascending=[False, False, True]).iloc[0]
        insights.append(f"🔥 Best approach bucket: {top_bucket['Bucket']} ({int(top_bucket['Made'])}/{int(top_bucket['Attempts'])}, {top_bucket['Pct']:.1f}%)")

    if not round_club.empty:
        top_club = round_club.sort_values(["Attempts", "Pct", "Club"], ascending=[False, False, True]).iloc[0]
        insights.append(f"🎯 Most used club: {top_club['Club']} ({int(top_club['Attempts'])} attempts, {top_club['Pct']:.1f}% GIR)")

    if (not round_dist.empty) and (not bench_dist.empty):
        cmp = pd.merge(
            round_dist[["Bucket", "Pct", "Attempts"]].rename(columns={"Pct":"RoundPct","Attempts":"RoundAtt"}),
            bench_dist[["Bucket", "Pct"]].rename(columns={"Pct":"BasePct"}),
            on="Bucket",
            how="inner"
        )
        if not cmp.empty:
            cmp["Delta"] = cmp["RoundPct"] - cmp["BasePct"]
            best_delta = cmp.sort_values(["Delta", "RoundAtt"], ascending=[False, False]).iloc[0]
            worst_delta = cmp.sort_values(["Delta", "RoundAtt"], ascending=[True, False]).iloc[0]
            insights.append(f"📈 Biggest gain vs baseline: {best_delta['Bucket']} ({best_delta['Delta']:+.1f} pts)")
            insights.append(f"📉 Biggest drop vs baseline: {worst_delta['Bucket']} ({worst_delta['Delta']:+.1f} pts)")
    return insights[:4]

def build_direction_heatmap_df(round_dir_df):
    if round_dir_df.empty:
        return pd.DataFrame(columns=["x","y","Direction","Count","Pct"])

    coords = {
        "Short Left": (-1, -1),
        "Left": (-1, 0),
        "Long Left": (-1, 1),
        "Short": (0, -1),
        "Long": (0, 1),
        "Short Right": (1, -1),
        "Right": (1, 0),
        "Long Right": (1, 1),
    }
    rows = []
    for _, r in round_dir_df.iterrows():
        direction = r["Direction"]
        if direction in coords:
            x, y = coords[direction]
            rows.append({"x": x, "y": y, "Direction": direction, "Count": r["Count"], "Pct": r["Pct"]})
    return pd.DataFrame(rows)

def render_direction_heatmap(round_dir_df):
    heat_df = build_direction_heatmap_df(round_dir_df)
    if heat_df.empty:
        st.info("No directional miss points available for heat map.")
        return

    grid = pd.DataFrame(
        [{"x": x, "y": y} for x in [-1, 0, 1] for y in [-1, 0, 1]]
    )
    labels = {
        (-1, 1): "Long Left", (0, 1): "Long", (1, 1): "Long Right",
        (-1, 0): "Left",      (0, 0): "Hole", (1, 0): "Right",
        (-1, -1): "Short Left", (0, -1): "Short", (1, -1): "Short Right",
    }
    grid["Cell"] = grid.apply(lambda r: labels[(r["x"], r["y"])], axis=1)
    grid = grid.merge(heat_df[["x","y","Count","Pct","Direction"]], on=["x","y"], how="left")
    grid["Count"] = grid["Count"].fillna(0)
    grid["Pct"] = grid["Pct"].fillna(0)
    grid["Label"] = grid.apply(
        lambda r: "HOLE" if (r["x"] == 0 and r["y"] == 0) else (f"{int(r['Count'])}\n{r['Pct']:.1f}%" if r["Count"] > 0 else ""),
        axis=1
    )

    base = alt.Chart(grid).encode(
        x=alt.X("x:O", sort=[-1,0,1], axis=alt.Axis(title=None, labels=False, ticks=False)),
        y=alt.Y("y:O", sort=[1,0,-1], axis=alt.Axis(title=None, labels=False, ticks=False)),
    )

    rects = base.mark_rect(cornerRadius=10, stroke="#666").encode(
        color=alt.Color("Count:Q", title="Miss Count"),
        tooltip=[
            alt.Tooltip("Cell:N"),
            alt.Tooltip("Count:Q"),
            alt.Tooltip("Pct:Q", format=".1f"),
        ]
    )

    txt = base.mark_text(fontWeight="bold").encode(text="Label:N")
    st.altair_chart((rects + txt).properties(height=260), use_container_width=True)


def _miss_bias_summary(dir_df):
    if dir_df is None or dir_df.empty:
        return {
            "left_count": 0, "right_count": 0, "short_count": 0, "long_count": 0,
            "diag_left_count": 0, "diag_right_count": 0,
            "left_pct": 0.0, "right_pct": 0.0, "short_pct": 0.0, "long_pct": 0.0,
            "diag_left_pct": 0.0, "diag_right_pct": 0.0,
            "top_direction": None, "top_pct": 0.0, "total": 0,
            "left_family": "Short Left + Left + Long Left",
            "right_family": "Short Right + Right + Long Right",
            "short_family": "Short Left + Short + Short Right",
            "long_family": "Long Left + Long + Long Right",
        }

    work = dir_df.copy()
    work["Direction"] = work["Direction"].astype(str)
    work["Count"] = pd.to_numeric(work.get("Count"), errors="coerce").fillna(0)
    total_count = int(work["Count"].sum())
    total = float(total_count) if total_count > 0 else 1.0

    def _contains(word):
        return work["Direction"].str.contains(word, case=False, na=False)

    left = int(work.loc[_contains("left"), "Count"].sum())
    right = int(work.loc[_contains("right"), "Count"].sum())
    short = int(work.loc[_contains("short"), "Count"].sum())
    longv = int(work.loc[_contains("long"), "Count"].sum())
    diag_left = int(work.loc[_contains("short left") | _contains("long left"), "Count"].sum())
    diag_right = int(work.loc[_contains("short right") | _contains("long right"), "Count"].sum())

    if not work.empty:
        top_row = work.sort_values(["Count", "Direction"], ascending=[False, True]).iloc[0]
        top_direction = str(top_row["Direction"])
        top_pct = float(top_row["Count"]) / total * 100.0
    else:
        top_direction = None
        top_pct = 0.0

    return {
        "left_count": left,
        "right_count": right,
        "short_count": short,
        "long_count": longv,
        "diag_left_count": diag_left,
        "diag_right_count": diag_right,
        "left_pct": left / total * 100.0,
        "right_pct": right / total * 100.0,
        "short_pct": short / total * 100.0,
        "long_pct": longv / total * 100.0,
        "diag_left_pct": diag_left / total * 100.0,
        "diag_right_pct": diag_right / total * 100.0,
        "top_direction": top_direction,
        "top_pct": top_pct,
        "total": total_count,
        "left_family": "Short Left + Left + Long Left",
        "right_family": "Short Right + Right + Long Right",
        "short_family": "Short Left + Short + Short Right",
        "long_family": "Long Left + Long + Long Right",
    }


def build_miss_direction_compare(round_dir_df, bench_dir_df):
    round_bias = _miss_bias_summary(round_dir_df)
    bench_bias = _miss_bias_summary(bench_dir_df)

    rows = []
    metrics = [
        ("Left Bias", "left_count", "left_pct", "Short Left + Left + Long Left"),
        ("Right Bias", "right_count", "right_pct", "Short Right + Right + Long Right"),
        ("Short Bias", "short_count", "short_pct", "Short Left + Short + Short Right"),
        ("Long Bias", "long_count", "long_pct", "Long Left + Long + Long Right"),
        ("Short/Long Left", "diag_left_count", "diag_left_pct", "Short Left + Long Left"),
        ("Short/Long Right", "diag_right_count", "diag_right_pct", "Short Right + Long Right"),
    ]
    for label, count_key, pct_key, family in metrics:
        r_count = int(round_bias.get(count_key, 0))
        b_count = int(bench_bias.get(count_key, 0))
        r_pct = float(round_bias.get(pct_key, 0.0))
        b_pct = float(bench_bias.get(pct_key, 0.0))
        rows.append({
            "Metric": label,
            "Includes": family,
            "Round Count": r_count,
            "Round %": round(r_pct, 1),
            "Baseline Count": b_count,
            "Baseline %": round(b_pct, 1),
            "Delta": round(r_pct - b_pct, 1),
            "Round Total Misses": int(round_bias.get("total", 0)),
            "Baseline Total Misses": int(bench_bias.get("total", 0)),
        })
    return pd.DataFrame(rows), round_bias, bench_bias


def build_grouped_miss_profile(round_bias, bench_bias, compare_label="Baseline"):
    rows = [
        {
            "Family": "Left",
            "Includes": round_bias.get("left_family", "Short Left + Left + Long Left"),
            "Round Count": int(round_bias.get("left_count", 0)),
            "Round %": float(round_bias.get("left_pct", 0.0)),
            f"{compare_label} Count": int(bench_bias.get("left_count", 0)),
            f"{compare_label} %": float(bench_bias.get("left_pct", 0.0)),
        },
        {
            "Family": "Right",
            "Includes": round_bias.get("right_family", "Short Right + Right + Long Right"),
            "Round Count": int(round_bias.get("right_count", 0)),
            "Round %": float(round_bias.get("right_pct", 0.0)),
            f"{compare_label} Count": int(bench_bias.get("right_count", 0)),
            f"{compare_label} %": float(bench_bias.get("right_pct", 0.0)),
        },
        {
            "Family": "Short",
            "Includes": round_bias.get("short_family", "Short Left + Short + Short Right"),
            "Round Count": int(round_bias.get("short_count", 0)),
            "Round %": float(round_bias.get("short_pct", 0.0)),
            f"{compare_label} Count": int(bench_bias.get("short_count", 0)),
            f"{compare_label} %": float(bench_bias.get("short_pct", 0.0)),
        },
        {
            "Family": "Long",
            "Includes": round_bias.get("long_family", "Long Left + Long + Long Right"),
            "Round Count": int(round_bias.get("long_count", 0)),
            "Round %": float(round_bias.get("long_pct", 0.0)),
            f"{compare_label} Count": int(bench_bias.get("long_count", 0)),
            f"{compare_label} %": float(bench_bias.get("long_pct", 0.0)),
        },
    ]
    out = pd.DataFrame(rows)
    out["Delta"] = (pd.to_numeric(out["Round %"], errors="coerce").fillna(0) - pd.to_numeric(out[f"{compare_label} %"], errors="coerce").fillna(0)).round(1)
    out["Round Label"] = out.apply(lambda r: f"{int(r['Round Count'])}/{int(round_bias.get('total',0))} ({float(r['Round %']):.1f}%)" if int(round_bias.get('total',0)) else "0/0 (0.0%)", axis=1)
    out[f"{compare_label} Label"] = out.apply(lambda r: f"{int(r[f'{compare_label} Count'])}/{int(bench_bias.get('total',0))} ({float(r[f'{compare_label} %']):.1f}%)" if int(bench_bias.get('total',0)) else "0/0 (0.0%)", axis=1)
    return out


def build_miss_direction_diagnosis(round_dir_df, bench_dir_df):
    compare_df, round_bias, bench_bias = build_miss_direction_compare(round_dir_df, bench_dir_df)
    insights = []

    top_dir = round_bias.get("top_direction")
    if top_dir:
        insights.append(f"Most common approach miss this round: {top_dir} ({round_bias['top_pct']:.1f}% of misses, {int(round_bias.get('total', 0))} total misses).")

    if not compare_df.empty:
        strongest = compare_df.sort_values("Delta", ascending=False).iloc[0]
        weakest = compare_df.sort_values("Delta", ascending=True).iloc[0]
        if float(strongest["Delta"]) > 4.9:
            insights.append(f"Compared to baseline, misses skewed more toward {strongest['Metric'].replace(' Bias','').lower()} ({int(strongest['Round Count'])}/{int(strongest['Round Total Misses'])}, {strongest['Delta']:+.1f} pts).")
        if float(weakest["Delta"]) < -4.9:
            insights.append(f"Compared to baseline, {weakest['Metric'].replace(' Bias','').lower()} misses were down ({int(weakest['Round Count'])}/{int(weakest['Round Total Misses'])}, {weakest['Delta']:+.1f} pts).")

    lr_gap = round_bias["left_pct"] - round_bias["right_pct"]
    sl_gap = round_bias["short_pct"] - round_bias["long_pct"]
    if abs(lr_gap) >= 10:
        insights.append("Directional control leaned left more than right." if lr_gap > 0 else "Directional control leaned right more than left.")
    if abs(sl_gap) >= 10:
        insights.append("Distance control leaned short more than long." if sl_gap > 0 else "Distance control leaned long more than short.")

    return insights[:4], compare_df, round_bias, bench_bias


def render_miss_direction_bias_cards(compare_df):
    if compare_df is None or compare_df.empty:
        st.info("No miss-bias comparison available.")
        return

    cards = st.columns(min(3, len(compare_df)))
    show_rows = compare_df.sort_values("Delta", key=lambda s: s.abs(), ascending=False).head(3).reset_index(drop=True)
    for i, (_, row) in enumerate(show_rows.iterrows()):
        with cards[i]:
            delta = float(row["Delta"])
            arrow = "🔺" if delta > 0.05 else ("🔻" if delta < -0.05 else "➡️")
            st.metric(
                row["Metric"],
                f"{int(row['Round Count'])}/{int(row['Round Total Misses'])} ({float(row['Round %']):.1f}%)",
                f"{arrow} {delta:+.1f} pts vs baseline"
            )


def render_grouped_miss_profile(group_df, compare_label="Baseline"):
    if group_df is None or group_df.empty:
        st.info("No grouped miss profile available.")
        return

    family_order = ["Left", "Right", "Short", "Long"]

    round_df = group_df[["Family", "Round Count", "Round %"]].rename(
        columns={"Round Count": "Count", "Round %": "Pct"}
    ).copy()
    base_df = group_df[["Family", f"{compare_label} Count", f"{compare_label} %"]].rename(
        columns={f"{compare_label} Count": "Count", f"{compare_label} %": "Pct"}
    ).copy()

    render_paired_compare_counts(
        round_df,
        base_df,
        key_col="Family",
        key_order=family_order,
        compare_mode=compare_label,
        title="Grouped Miss Families — Left/Right/Short/Long",
        value_label="Miss Count",
    )

    st.dataframe(group_df, use_container_width=True, hide_index=True)


def build_dispersion_points(frame):
    d = prepare_approach_frame(frame).copy()
    if d.empty:
        return pd.DataFrame(columns=["Hole", "x", "y", "Direction", "Proximity", "Club", "Bucket", "GIR"])

    direction_vectors = {
        "Short Left": (-0.7, -0.7),
        "Left": (-1.0, 0.0),
        "Long Left": (-0.7, 0.7),
        "Short": (0.0, -1.0),
        "Long": (0.0, 1.0),
        "Short Right": (0.7, -0.7),
        "Right": (1.0, 0.0),
        "Long Right": (0.7, 0.7),
    }

    def _point_xy(row):
        prox = float(row.get("Approach Proximity", 0) or 0)
        direction = row.get("Approach Miss Direction Clean", "")
        if prox <= 0:
            prox = 5.0
        if direction in direction_vectors:
            dx, dy = direction_vectors[direction]
            return pd.Series({"x": dx * prox, "y": dy * prox})
        return pd.Series({"x": 0.0, "y": 0.0})

    pts = d.join(d.apply(_point_xy, axis=1))
    pts["Hole"] = _int(_safe_col(pts, "Hole", 0))
    pts["GIR"] = pts["Approach GIR Flag"]
    return pts[["Hole", "x", "y", "Approach Miss Direction Clean", "Approach Proximity", "Approach Club", "Approach Bucket", "GIR"]].rename(
        columns={
            "Approach Miss Direction Clean": "Direction",
            "Approach Proximity": "Proximity",
            "Approach Club": "Club",
            "Approach Bucket": "Bucket",
        }
    )

def render_dispersion_plot(frame, title="Approach Dispersion"):
    pts = build_dispersion_points(frame)
    if pts.empty:
        st.info("No approach points available for dispersion plot.")
        return

    hole_pt = pd.DataFrame({"x": [0.0], "y": [0.0], "Label": ["Hole"]})

    scatter = alt.Chart(pts).mark_circle(opacity=0.8, size=110).encode(
        x=alt.X("x:Q", title="Left / Right (ft)"),
        y=alt.Y("y:Q", title="Short / Long (ft)"),
        color=alt.Color("GIR:N", title="GIR", scale=alt.Scale(domain=[0, 1], range=["#ee6c4d", "#64dfb5"])),
        tooltip=[
            alt.Tooltip("Hole:Q"),
            alt.Tooltip("Club:N"),
            alt.Tooltip("Bucket:N"),
            alt.Tooltip("Direction:N"),
            alt.Tooltip("Proximity:Q", format=".1f"),
            alt.Tooltip("GIR:N"),
        ],
    ).properties(height=360, title=title)

    hole = alt.Chart(hole_pt).mark_point(shape="diamond", size=200, filled=True).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("Label:N")]
    )
    zero_h = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(opacity=0.25).encode(y="y:Q")
    zero_v = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(opacity=0.25).encode(x="x:Q")

    st.altair_chart((zero_h + zero_v + scatter + hole).configure_view(stroke=None), use_container_width=True)

def build_distance_rank_table(round_dist, bench_dist):
    if round_dist.empty:
        return pd.DataFrame(columns=["Bucket", "Round %", "Baseline %", "Delta", "Attempts"])
    out = round_dist[["Bucket", "Pct", "Attempts"]].rename(columns={"Pct": "Round %"}).copy()
    if not bench_dist.empty:
        out = out.merge(
            bench_dist[["Bucket", "Pct"]].rename(columns={"Pct": "Baseline %"}),
            on="Bucket",
            how="left",
        )
        out["Delta"] = (out["Round %"] - out["Baseline %"]).round(1)
    else:
        out["Baseline %"] = pd.NA
        out["Delta"] = pd.NA
    return out.sort_values(["Round %", "Attempts"], ascending=[False, False]).reset_index(drop=True)

def build_round_performance_rating(round_data, benchmark_df):
    def _pct(series):
        total = len(series)
        made = pd.to_numeric(series, errors="coerce").fillna(0).sum()
        return (made / total * 100.0) if total else None

    current_score_to_par = float(
        pd.to_numeric(_safe_col(round_data, "Hole Score", 0), errors="coerce").sum()
        - pd.to_numeric(_safe_col(round_data, "Par", 0), errors="coerce").sum()
    )
    current_gir = _pct(_safe_col(round_data, "GIR", 0))
    fw_block = round_data[round_data["Par"].isin([4, 5])].copy() if "Par" in round_data else round_data.iloc[0:0].copy()
    current_fw = _pct(_safe_col(fw_block, "Fairway", 0)) if not fw_block.empty else None
    current_putts = pd.to_numeric(_safe_col(round_data, "Putts", 0), errors="coerce").fillna(0).mean() if len(round_data) else None

    base_score_to_par = float(
        pd.to_numeric(_safe_col(benchmark_df, "Hole Score", 0), errors="coerce").sum()
        - pd.to_numeric(_safe_col(benchmark_df, "Par", 0), errors="coerce").sum()
    )
    base_holes = len(benchmark_df)
    base_score_to_par_per18 = (base_score_to_par / base_holes * 18.0) if base_holes else None
    base_gir = _pct(_safe_col(benchmark_df, "GIR", 0))
    fw_base_block = benchmark_df[benchmark_df["Par"].isin([4, 5])].copy() if "Par" in benchmark_df else benchmark_df.iloc[0:0].copy()
    base_fw = _pct(_safe_col(fw_base_block, "Fairway", 0)) if not fw_base_block.empty else None
    base_putts = pd.to_numeric(_safe_col(benchmark_df, "Putts", 0), errors="coerce").fillna(0).mean() if len(benchmark_df) else None

    score_components = []
    details = []

    if base_score_to_par_per18 is not None:
        current_per18 = current_score_to_par / max(len(round_data), 1) * 18.0
        delta = base_score_to_par_per18 - current_per18
        score_components.append(delta * 3.0)
        details.append(("Score to Par /18", current_per18, base_score_to_par_per18, delta))

    if current_gir is not None and base_gir is not None:
        delta = current_gir - base_gir
        score_components.append(delta * 1.2)
        details.append(("GIR %", current_gir, base_gir, delta))

    if current_fw is not None and base_fw is not None:
        delta = current_fw - base_fw
        score_components.append(delta * 0.8)
        details.append(("FW %", current_fw, base_fw, delta))

    if current_putts is not None and base_putts is not None:
        delta = base_putts - current_putts
        score_components.append(delta * 10.0)
        details.append(("Putts / Hole", current_putts, base_putts, delta))

    total_score = sum(score_components) if score_components else 0.0

    if total_score >= 25:
        grade = "A"
    elif total_score >= 12:
        grade = "B"
    elif total_score >= 0:
        grade = "C"
    elif total_score >= -12:
        grade = "D"
    else:
        grade = "F"

    return grade, total_score, details

def build_shot_pattern_frame(full_df, player_name):
    base = full_df.copy()
    if "Player Name" in base:
        base = base[base["Player Name"] == player_name]
    return prepare_approach_frame(base)



def build_rate_lookup(summary_df, key_col):
    if summary_df.empty:
        return {}
    return {
        row[key_col]: {
            "pct": float(row["Pct"]),
            "attempts": int(row["Attempts"]),
            "made": int(row["Made"]),
        }
        for _, row in summary_df.iterrows()
        if pd.notna(row[key_col])
    }

def build_sg_style_insights(full_df, round_data, compare_mode="All Time"):
    """
    Beta / proxy version:
    - Not true PGA strokes gained
    - Converts extra makes/saves vs the player's baseline into estimated stroke values
    - Transparent and easy to refine later
    """
    benchmark_df = build_benchmark_df(full_df, round_data, compare_mode)

    # ---------- Approach ----------
    round_app = summarize_approach_by_bucket(round_data)
    bench_app = summarize_approach_by_bucket(benchmark_df)
    app_lookup = build_rate_lookup(bench_app, "Bucket")

    app_expected_made = 0.0
    app_actual_made = 0.0
    app_attempts = 0
    for _, row in round_app.iterrows():
        bucket = row["Bucket"]
        att = int(row["Attempts"])
        made = float(row["Made"])
        base_pct = app_lookup.get(bucket, {}).get("pct", 0.0)
        app_expected_made += att * base_pct / 100.0
        app_actual_made += made
        app_attempts += att
    app_extra = app_actual_made - app_expected_made
    app_sg = app_extra * 0.55  # beta conversion

    # ---------- Putting ----------
    round_putt = summarize_putting_by_bucket(round_data)
    bench_putt = summarize_putting_by_bucket(benchmark_df)
    putt_lookup = build_rate_lookup(bench_putt, "Bucket")

    putt_expected_made = 0.0
    putt_actual_made = 0.0
    putt_attempts = 0
    for _, row in round_putt.iterrows():
        bucket = row["Bucket"]
        att = int(row["Attempts"])
        made = float(row["Made"])
        base_pct = putt_lookup.get(bucket, {}).get("pct", 0.0)
        putt_expected_made += att * base_pct / 100.0
        putt_actual_made += made
        putt_attempts += att
    putt_extra = putt_actual_made - putt_expected_made
    putt_sg = putt_extra * 0.30  # beta conversion

    # ---------- Short Game ----------
    round_sg = summarize_short_game_by_bucket(round_data)
    bench_sg = summarize_short_game_by_bucket(benchmark_df)
    sg_lookup = build_rate_lookup(bench_sg, "Bucket")

    short_expected = 0.0
    short_actual = 0.0
    short_attempts = 0
    for _, row in round_sg.iterrows():
        bucket = row["Bucket"]
        att = int(row["Attempts"])
        made = float(row["Made"])
        base_pct = sg_lookup.get(bucket, {}).get("pct", 0.0)
        short_expected += att * base_pct / 100.0
        short_actual += made
        short_attempts += att
    short_extra = short_actual - short_expected
    short_sg = short_extra * 0.35  # beta conversion

    total_sg = app_sg + putt_sg + short_sg

    return {
        "compare_mode": compare_mode,
        "approach_sg": app_sg,
        "putting_sg": putt_sg,
        "short_game_sg": short_sg,
        "total_sg": total_sg,
        "approach_attempts": app_attempts,
        "putting_attempts": putt_attempts,
        "short_game_attempts": short_attempts,
        "approach_extra": app_extra,
        "putting_extra": putt_extra,
        "short_game_extra": short_extra,
    }



def build_sg_round_impact(round_data, sg_beta):
    actual_score = float(pd.to_numeric(_safe_col(round_data, "Hole Score", 0), errors="coerce").sum())
    expected_score = actual_score - float(sg_beta.get("total_sg", 0.0))
    performance_delta = actual_score - expected_score  # negative means better than expected
    return {
        "actual_score": actual_score,
        "expected_score": expected_score,
        "sg_vs_baseline": float(sg_beta.get("total_sg", 0.0)),
        "performance_delta": performance_delta,
    }

def build_club_rank_table(round_club, bench_club):
    if round_club.empty:
        return pd.DataFrame(columns=["Club", "Round Attempts", "Round GIR %", "Round Avg Prox", "Baseline GIR %", "Delta"])
    out = round_club[["Club", "Attempts", "Pct", "AvgProx"]].rename(
        columns={"Attempts": "Round Attempts", "Pct": "Round GIR %", "AvgProx": "Round Avg Prox"}
    ).copy()
    if not bench_club.empty:
        out = out.merge(
            bench_club[["Club", "Pct", "AvgProx"]].rename(columns={"Pct": "Baseline GIR %", "AvgProx": "Baseline Avg Prox"}),
            on="Club",
            how="left"
        )
        out["Delta"] = (out["Round GIR %"] - out["Baseline GIR %"]).round(1)
    else:
        out["Baseline GIR %"] = pd.NA
        out["Baseline Avg Prox"] = pd.NA
        out["Delta"] = pd.NA
    return out.sort_values(["Round GIR %", "Round Attempts"], ascending=[False, False]).reset_index(drop=True)



def build_distance_club_heatmap(frame):
    d = prepare_approach_frame(frame).copy()
    d = d[(d["Approach Bucket"].notna()) & (d["Approach Club"].astype(str).str.strip() != "")]
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Club", "Attempts", "Made", "Pct", "CellLabel", "PctLabel"])

    out = (
        d.groupby(["Approach Bucket", "Approach Club"], as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean")
         )
         .rename(columns={"Approach Bucket": "Bucket", "Approach Club": "Club"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)

    def _emoji(p):
        if p >= 50:
            return "🔥"
        elif p >= 35:
            return "🟢"
        elif p >= 20:
            return "🟡"
        else:
            return "🧊"

    out["CellLabel"] = out.apply(lambda r: f"{_emoji(r['Pct'])} {r['Pct']:.0f}%", axis=1)
    out["PctLabel"] = out["Pct"].map(lambda x: f"{x:.1f}%")
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values(["Bucket", "Club"]).reset_index(drop=True)

def render_distance_club_heatmap(frame, title="Distance vs Club Heatmap"):
    heat = build_distance_club_heatmap(frame)
    if heat.empty:
        st.info("No approach distance/club rows available for heatmap.")
        return

    club_order = (
        heat.groupby("Club", as_index=False)["Attempts"]
        .sum()
        .sort_values(["Attempts", "Club"], ascending=[False, True])["Club"]
        .tolist()
    )

    base = alt.Chart(heat).encode(
        x=alt.X("Club:N", sort=club_order, title="Club"),
        y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER, title="Distance Bucket"),
        tooltip=[
            alt.Tooltip("Bucket:N"),
            alt.Tooltip("Club:N"),
            alt.Tooltip("Attempts:Q"),
            alt.Tooltip("Made:Q"),
            alt.Tooltip("Pct:Q", format=".1f"),
            alt.Tooltip("AvgProx:Q", title="Avg Prox", format=".1f"),
        ]
    )

    rect = base.mark_rect(stroke="#666", cornerRadius=8).encode(
        color=alt.Color("Pct:Q", title="GIR %")
    )

    text = base.mark_text(fontWeight="bold").encode(
        text="CellLabel:N",
        color=alt.condition("datum.Pct >= 45", alt.value("black"), alt.value("white"))
    )

    st.altair_chart((rect + text).properties(height=max(320, len(APPROACH_BUCKET_ORDER)*26), title=title), use_container_width=True)

    pivot = heat.pivot(index="Bucket", columns="Club", values="CellLabel").reset_index()
    st.dataframe(pivot, use_container_width=True, hide_index=True)



def build_distance_performance_curve(frame):
    d = prepare_approach_frame(frame).copy()
    if d.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "AvgProx", "ToParPerHole", "Label"])

    # Score-to-par by hole for approach shots
    if "Hole Score" in d.columns and "Par" in d.columns:
        d["ScoreToParHole"] = pd.to_numeric(d["Hole Score"], errors="coerce").fillna(0) - pd.to_numeric(d["Par"], errors="coerce").fillna(0)
    else:
        d["ScoreToParHole"] = 0.0

    out = (
        d.dropna(subset=["Approach Bucket"])
         .groupby("Approach Bucket", as_index=False)
         .agg(
             Attempts=("Approach GIR Flag", "size"),
             Made=("Approach GIR Flag", "sum"),
             AvgProx=("Approach Proximity", "mean"),
             ToParPerHole=("ScoreToParHole", "mean"),
         )
         .rename(columns={"Approach Bucket": "Bucket"})
    )
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)

def render_distance_performance_curve(round_curve, bench_curve=None, compare_label="Baseline"):
    """
    Reliable SVG/HTML distance curve.
    Uses components.html so it renders consistently in Streamlit.
    """
    if round_curve is None or round_curve.empty:
        st.info("No distance performance curve data available for this round.")
        return

    round_df = round_curve.copy()
    round_df["Bucket"] = round_df["Bucket"].astype(str)
    round_df = round_df[round_df["Bucket"].isin(APPROACH_BUCKET_ORDER)].copy()
    if round_df.empty:
        st.info("No distance performance curve data available for this round.")
        return

    bucket_order = [
        b for b in APPROACH_BUCKET_ORDER
        if b in round_df["Bucket"].tolist()
        or (bench_curve is not None and not bench_curve.empty and b in bench_curve["Bucket"].astype(str).tolist())
    ]
    if not bucket_order:
        st.info("No distance performance curve data available for this round.")
        return

    round_df = round_df.set_index("Bucket").reindex(bucket_order).reset_index()
    round_df["Pct"] = pd.to_numeric(round_df["Pct"], errors="coerce").fillna(0.0)
    round_df["Made"] = pd.to_numeric(round_df["Made"], errors="coerce").fillna(0).astype(int)
    round_df["Attempts"] = pd.to_numeric(round_df["Attempts"], errors="coerce").fillna(0).astype(int)
    round_df["CurveLabel"] = round_df.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%", axis=1)

    base_df = None
    if bench_curve is not None and not bench_curve.empty:
        base_df = bench_curve.copy()
        base_df["Bucket"] = base_df["Bucket"].astype(str)
        base_df = base_df[base_df["Bucket"].isin(bucket_order)].copy()
        if not base_df.empty:
            base_df = base_df.set_index("Bucket").reindex(bucket_order).reset_index()
            base_df["Pct"] = pd.to_numeric(base_df["Pct"], errors="coerce").fillna(0.0)
            base_df["BaseLabel"] = base_df["Pct"].map(lambda x: f"{x:.0f}%")
        else:
            base_df = None

    # ---------- SVG layout ----------
    import math
    import html as _html
    import streamlit.components.v1 as components

    width = 1080
    height = 420
    left = 72
    right = 24
    top = 46
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    n = max(len(bucket_order), 1)
    step_x = plot_w / max(n - 1, 1)

    def _x(i):
        return left + (i * step_x if n > 1 else plot_w / 2)

    def _y(pct):
        pct = max(0.0, min(100.0, float(pct)))
        return top + plot_h * (1 - pct / 100.0)

    def _points(df, col="Pct"):
        pts = []
        for i, row in df.iterrows():
            pts.append((_x(i), _y(row[col])))
        return pts

    def _polyline(pts):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    round_pts = _points(round_df, "Pct")
    base_pts = _points(base_df, "Pct") if base_df is not None else []

    grid_lines = []
    for tick in [0, 25, 50, 75, 100]:
        y = _y(tick)
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>')
        grid_lines.append(f'<text x="{left-10}" y="{y+4:.1f}" fill="#bdbdbd" font-size="11" text-anchor="end">{tick}%</text>')

    x_labels = []
    for i, b in enumerate(bucket_order):
        x = _x(i)
        x_labels.append(f'<text x="{x:.1f}" y="{height-28}" fill="#d9d9d9" font-size="11" text-anchor="middle">{_html.escape(str(b))}</text>')

    round_point_elems = []
    round_label_elems = []
    for i, row in round_df.iterrows():
        x, y = round_pts[i]
        round_point_elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#4f8cff"/>')
        label = _html.escape(str(row["CurveLabel"]))
        round_label_elems.append(f'<text x="{x:.1f}" y="{max(16, y-12):.1f}" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">{label}</text>')

    base_point_elems = []
    base_label_elems = []
    if base_df is not None:
        for i, row in base_df.iterrows():
            x, y = base_pts[i]
            base_point_elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#9aa0a6"/>')
            label = _html.escape(str(row["BaseLabel"]))
            base_label_elems.append(f'<text x="{x:.1f}" y="{min(height-bottom+18, y+16):.1f}" fill="#bdbdbd" font-size="10" text-anchor="middle">{label}</text>')

    legend = f"""
    <g>
      <line x1="{left}" y1="18" x2="{left+28}" y2="18" stroke="#4f8cff" stroke-width="3"/>
      <circle cx="{left+14}" cy="18" r="4" fill="#4f8cff"/>
      <text x="{left+38}" y="22" fill="#ffffff" font-size="12">Round</text>
      <line x1="{left+120}" y1="18" x2="{left+148}" y2="18" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4"/>
      <circle cx="{left+134}" cy="18" r="3.5" fill="#9aa0a6"/>
      <text x="{left+158}" y="22" fill="#d0d0d0" font-size="12">{_html.escape(compare_label)}</text>
    </g>
    """

    svg = f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#1f1f1f" rx="12" ry="12"/>
      {''.join(grid_lines)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>
      {legend}
      <polyline points="{_polyline(round_pts)}" fill="none" stroke="#4f8cff" stroke-width="3"/>
      {'<polyline points="' + _polyline(base_pts) + '" fill="none" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4"/>' if base_pts else ''}
      {''.join(base_point_elems)}
      {''.join(round_point_elems)}
      {''.join(base_label_elems)}
      {''.join(round_label_elems)}
      {''.join(x_labels)}
      <text x="{width/2:.1f}" y="{height-8}" fill="#d9d9d9" font-size="12" text-anchor="middle">Distance Bucket</text>
      <text x="18" y="{top + plot_h/2:.1f}" fill="#d9d9d9" font-size="12" text-anchor="middle" transform="rotate(-90 18 {top + plot_h/2:.1f})">GIR %</text>
    </svg>
    """

    components.html(svg, height=height + 8, scrolling=False)



def render_dispersion_panel(frame, title="Approach Dispersion"):
    pts = _build_dispersion_points(frame)
    if pts.empty:
        st.info("No approach dispersion data available.")
        return

    zero_h = alt.Chart(pd.DataFrame({"y":[0]})).mark_rule(opacity=0.25).encode(y="y:Q")
    zero_v = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(opacity=0.25).encode(x="x:Q")

    scatter = (
        alt.Chart(pts)
        .mark_circle(size=110, opacity=0.8)
        .encode(
            x=alt.X("x:Q", title="Left / Right (ft)"),
            y=alt.Y("y:Q", title="Short / Long (ft)"),
            color=alt.Color("Approach GIR Flag:N", title="Approach GIR", scale=alt.Scale(domain=[0,1], range=["#ee6c4d", "#64dfb5"])),
            tooltip=[
                alt.Tooltip("Hole:Q"),
                alt.Tooltip("Approach Club:N", title="Club"),
                alt.Tooltip("Approach Bucket:N", title="Bucket"),
                alt.Tooltip("Approach Miss Direction Clean:N", title="Direction"),
                alt.Tooltip("Approach Proximity:Q", title="Proximity", format=".1f"),
                alt.Tooltip("Approach GIR Flag:N", title="GIR"),
            ],
        )
        .properties(height=360, title=title)
    )

    hole_df = pd.DataFrame({"x":[0.0], "y":[0.0], "Label":["Hole"]})
    hole = alt.Chart(hole_df).mark_point(shape="diamond", size=220, filled=True).encode(
        x="x:Q", y="y:Q", tooltip=[alt.Tooltip("Label:N")]
    )

    ellipse_df = build_dispersion_ellipse_df(frame)
    ellipse_layer = alt.Chart(pd.DataFrame({"x": [], "y": []})).mark_line()
    if not ellipse_df.empty:
        ellipse_layer = (
            alt.Chart(ellipse_df)
            .mark_line(strokeWidth=3, color="#f5c518", opacity=0.95)
            .encode(x="x:Q", y="y:Q")
        )

    st.altair_chart((zero_h + zero_v + scatter + hole + ellipse_layer).configure_view(stroke=None), use_container_width=True)

    if not ellipse_df.empty:
        st.caption("Yellow ellipse shows the overall shot-spread pattern for this filtered set of approach shots.")



def build_putting_round_impact(round_data, benchmark_df):
    """
    Proxy putting impact based on 1-putt performance by starting-distance bucket
    compared to the selected baseline.
    """
    round_putt_bucket = summarize_putting_by_bucket(round_data)
    bench_putt_bucket = summarize_putting_by_bucket(benchmark_df)

    if round_putt_bucket.empty:
        return {
            "attempts": 0,
            "made": 0,
            "pct": 0.0,
            "expected_makes": 0.0,
            "actual_makes": 0.0,
            "extra_makes": 0.0,
            "sg_putting": 0.0,
        }

    base_lookup = {
        str(row["Bucket"]): float(row["Pct"])
        for _, row in bench_putt_bucket.iterrows()
        if pd.notna(row["Bucket"])
    }

    expected_makes = 0.0
    actual_makes = float(round_putt_bucket["Made"].sum())
    attempts = int(round_putt_bucket["Attempts"].sum())

    for _, row in round_putt_bucket.iterrows():
        bucket = str(row["Bucket"])
        att = int(row["Attempts"])
        base_pct = base_lookup.get(bucket, 0.0)
        expected_makes += att * base_pct / 100.0

    extra_makes = actual_makes - expected_makes
    sg_putting = extra_makes * 0.30  # same beta conversion used elsewhere

    pct = (actual_makes / attempts * 100.0) if attempts else 0.0

    return {
        "attempts": attempts,
        "made": int(actual_makes),
        "pct": pct,
        "expected_makes": expected_makes,
        "actual_makes": actual_makes,
        "extra_makes": extra_makes,
        "sg_putting": sg_putting,
    }



def build_distance_sg_table(round_curve, bench_curve):
    """
    Approx SG-style per-distance proxy from GIR% deltas.
    """
    if round_curve is None or round_curve.empty:
        return pd.DataFrame(columns=["Bucket", "Round GIR %", "Baseline GIR %", "Delta", "Round Attempts", "Round Made", "Round Avg Prox", "Round To Par / Hole", "SG Proxy"])

    r = round_curve.copy()
    r["Bucket"] = r["Bucket"].astype(str)
    out = r.rename(columns={
        "Pct": "Round GIR %",
        "Attempts": "Round Attempts",
        "Made": "Round Made",
        "AvgProx": "Round Avg Prox",
        "ToParPerHole": "Round To Par / Hole",
    })[["Bucket", "Round GIR %", "Round Attempts", "Round Made", "Round Avg Prox", "Round To Par / Hole"]].copy()

    if bench_curve is not None and not bench_curve.empty:
        b = bench_curve.copy()
        b["Bucket"] = b["Bucket"].astype(str)
        out = out.merge(
            b.rename(columns={"Pct": "Baseline GIR %"})[["Bucket", "Baseline GIR %"]],
            on="Bucket",
            how="left"
        )
        out["Delta"] = (pd.to_numeric(out["Round GIR %"], errors="coerce").fillna(0) - pd.to_numeric(out["Baseline GIR %"], errors="coerce").fillna(0)).round(1)
        out["SG Proxy"] = ((out["Delta"].fillna(0) / 100.0) * pd.to_numeric(out["Round Attempts"], errors="coerce").fillna(0) * 0.55).round(2)
    else:
        out["Baseline GIR %"] = pd.NA
        out["Delta"] = pd.NA
        out["SG Proxy"] = pd.NA

    out["Bucket"] = pd.Categorical(out["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)

def render_distance_sg_heatmap(distance_sg_df):
    if distance_sg_df is None or distance_sg_df.empty:
        st.info("No distance SG-style data available.")
        return

    heat = distance_sg_df.copy()
    heat["Bucket"] = heat["Bucket"].astype(str)
    if "Metric" not in heat.columns:
        heat["Metric"] = "Approach SG"
    heat["CellLabel"] = heat.apply(
        lambda r: f"{float(r['Round GIR %']):.0f}%\nΔ {float(r['Delta']):+.0f}" if pd.notna(r.get("Delta")) else f"{float(r['Round GIR %']):.0f}%",
        axis=1
    )

    chart = (
        alt.Chart(heat)
        .mark_rect(cornerRadius=8, stroke="#666")
        .encode(
            x=alt.X("Bucket:N", sort=APPROACH_BUCKET_ORDER, title="Distance Bucket"),
            y=alt.Y("Metric:N", title=None),
            color=alt.Color("SG Proxy:Q", title="SG Proxy"),
            tooltip=[
                alt.Tooltip("Bucket:N"),
                alt.Tooltip("Round GIR %:Q", format=".1f"),
                alt.Tooltip("Baseline GIR %:Q", format=".1f"),
                alt.Tooltip("Delta:Q", format="+.1f"),
                alt.Tooltip("Round Attempts:Q"),
                alt.Tooltip("SG Proxy:Q", format="+.2f"),
            ],
        )
        .properties(height=100)
    )
    text = (
        alt.Chart(heat)
        .mark_text(fontWeight="bold")
        .encode(
            x=alt.X("Bucket:N", sort=APPROACH_BUCKET_ORDER),
            y=alt.Y("Metric:N"),
            text="CellLabel:N",
        )
    )
    st.altair_chart(chart + text, use_container_width=True)

def build_distance_improvement_tracker(full_df, round_data):
    """
    Compare this round vs prior 20 rounds for each distance bucket.
    """
    player = round_data["Player Name"].iloc[0] if ("Player Name" in round_data and not round_data.empty) else None
    if not player:
        return pd.DataFrame(columns=["Bucket", "This Round %", "Prev 20 Rounds %", "Delta"])

    hist = full_df.copy()
    hist["Date Played"] = pd.to_datetime(hist["Date Played"], errors="coerce")
    if "Player Name" in hist:
        hist = hist[hist["Player Name"] == player].copy()

    current_round_id = round_data["Round Link"].iloc[0] if ("Round Link" in round_data and not round_data.empty) else None
    hist_excl = hist[hist["Round Link"] != current_round_id].copy() if current_round_id is not None else hist.copy()

    if {"Round Link", "Date Played"} <= set(hist_excl.columns):
        prev_round_ids = (
            hist_excl[["Round Link", "Date Played"]]
            .drop_duplicates()
            .sort_values("Date Played", ascending=False)
            .head(20)["Round Link"]
            .tolist()
        )
        prev20 = hist_excl[hist_excl["Round Link"].isin(prev_round_ids)].copy()
    else:
        prev20 = hist_excl.copy()

    this_curve = build_distance_performance_curve(round_data)
    prev_curve = build_distance_performance_curve(prev20)

    if this_curve is None or this_curve.empty:
        return pd.DataFrame(columns=["Bucket", "This Round %", "Prev 20 Rounds %", "Delta"])

    out = this_curve.rename(columns={"Pct": "This Round %"})[["Bucket", "This Round %"]].copy()
    if prev_curve is not None and not prev_curve.empty:
        out = out.merge(
            prev_curve.rename(columns={"Pct": "Prev 20 Rounds %"})[["Bucket", "Prev 20 Rounds %"]],
            on="Bucket",
            how="left"
        )
        out["Delta"] = (pd.to_numeric(out["This Round %"], errors="coerce").fillna(0) - pd.to_numeric(out["Prev 20 Rounds %"], errors="coerce").fillna(0)).round(1)
    else:
        out["Prev 20 Rounds %"] = pd.NA
        out["Delta"] = pd.NA

    out["Bucket"] = pd.Categorical(out["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)



def build_approach_inside_proximity_summary(round_data, benchmark_df=None):
    """
    For each proximity/leave-distance bucket:
    - Round approaches finishing inside that range / total approach opportunities
    - Baseline approaches finishing inside that range / total approach opportunities
    This measures proximity distribution, not 1-putt %.
    """
    round_app = prepare_approach_frame(round_data).copy()
    bench_app = prepare_approach_frame(benchmark_df).copy() if benchmark_df is not None else pd.DataFrame()

    # Use putting buckets for proximity-to-hole bins
    round_app["Prox Bucket"] = round_app["Approach Proximity"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))
    if not bench_app.empty:
        bench_app["Prox Bucket"] = bench_app["Approach Proximity"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))

    def _summary(df_block, label_prefix="Round"):
        if df_block.empty:
            return pd.DataFrame(columns=["Bucket", f"{label_prefix} Inside", f"{label_prefix} Opportunities", f"{label_prefix} %"])
        total_opps = int(len(df_block))
        out = (
            df_block.dropna(subset=["Prox Bucket"])
            .groupby("Prox Bucket", as_index=False)
            .agg(Inside=("Approach GIR Flag", "size"))
            .rename(columns={"Prox Bucket": "Bucket", "Inside": f"{label_prefix} Inside"})
        )
        out[f"{label_prefix} Opportunities"] = total_opps
        out[f"{label_prefix} %"] = (
            pd.to_numeric(out[f"{label_prefix} Inside"], errors="coerce").fillna(0)
            / max(total_opps, 1) * 100
        ).round(1)
        return out

    round_sum = _summary(round_app, "Round")
    if benchmark_df is not None:
        bench_sum = _summary(bench_app, "Baseline")
        out = round_sum.merge(bench_sum, on="Bucket", how="outer")
        out["Delta"] = (
            pd.to_numeric(out["Round %"], errors="coerce").fillna(0)
            - pd.to_numeric(out["Baseline %"], errors="coerce").fillna(0)
        ).round(1)
    else:
        out = round_sum.copy()
        out["Baseline Inside"] = pd.NA
        out["Baseline Opportunities"] = pd.NA
        out["Baseline %"] = pd.NA
        out["Delta"] = pd.NA

    if not out.empty:
        out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
        out = out.sort_values("Bucket").reset_index(drop=True)
        out["DisplayLabel"] = out.apply(
            lambda r: f"{_safe_int_scalar(r.get('Round Inside', 0))}/{_safe_int_scalar(r.get('Round Opportunities', 0))} {_safe_float_scalar(r.get('Round %', 0)):.0f}%",
            axis=1
        )
    return out

def build_filtered_approach_proximity_distribution(frame):
    """
    Build a single-series distribution of approach shots by resulting proximity bucket.
    """
    app = prepare_approach_frame(frame).copy()
    if app.empty:
        return pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "DisplayLabel"])
    app["Prox Bucket"] = app["Approach Proximity"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))
    total = int(len(app))
    out = (
        app.dropna(subset=["Prox Bucket"])
        .groupby("Prox Bucket", as_index=False)
        .agg(Attempts=("Approach GIR Flag", "size"))
        .rename(columns={"Prox Bucket": "Bucket"})
    )
    out["Made"] = out["Attempts"]
    out["Pct"] = (pd.to_numeric(out["Attempts"], errors="coerce").fillna(0) / max(total, 1) * 100).round(1)
    out["DisplayLabel"] = out.apply(lambda r: f"{int(r['Attempts'])}/{total} {r['Pct']:.0f}%", axis=1)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
    return out.sort_values("Bucket").reset_index(drop=True)


def apply_approach_filters(frame, yard_buckets=None, clubs=None, courses=None, fairway_vals=None, par_vals=None, gir_mode="All"):
    d = prepare_approach_frame(frame).copy()
    if d.empty:
        return d

    if yard_buckets:
        d = d[d["Approach Bucket"].astype(str).isin([str(x) for x in yard_buckets])].copy()
    if clubs:
        d = d[d["Approach Club"].astype(str).isin([str(x) for x in clubs])].copy()
    if courses and "Course Name" in d.columns:
        d = d[d["Course Name"].astype(str).isin([str(x) for x in courses])].copy()
    if fairway_vals is not None and len(fairway_vals) > 0:
        fairway_num = pd.to_numeric(_safe_col(d, "Fairway", pd.NA), errors="coerce")
        d = d[fairway_num.isin(fairway_vals)].copy()
    if par_vals is not None and len(par_vals) > 0:
        par_num = pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")
        d = d[par_num.isin(par_vals)].copy()
    if gir_mode == "GIR Only":
        d = d[pd.to_numeric(_safe_col(d, "Approach GIR Flag", 0), errors="coerce").fillna(0).astype(int) == 1].copy()
    elif gir_mode == "Missed GIR Only":
        d = d[pd.to_numeric(_safe_col(d, "Approach GIR Flag", 0), errors="coerce").fillna(0).astype(int) == 0].copy()
    return d

def build_approach_proximity_compare(round_filtered, bench_filtered):
    """
    Compare resulting proximity distribution bucket share for Round vs baseline.
    """
    def _summary(df_block, prefix="Round"):
        app = prepare_approach_frame(df_block).copy() if "Approach Proximity" not in df_block.columns else df_block.copy()
        if app.empty:
            return pd.DataFrame(columns=["Bucket", f"{prefix} Made", f"{prefix} Attempts", f"{prefix} Pct"])
        app["Prox Bucket"] = app["Approach Proximity"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))
        total = int(len(app))
        out = (
            app.dropna(subset=["Prox Bucket"])
            .groupby("Prox Bucket", as_index=False)
            .agg(Made=("Approach GIR Flag", "size"))
            .rename(columns={"Prox Bucket": "Bucket", "Made": f"{prefix} Made"})
        )
        out[f"{prefix} Attempts"] = total
        out[f"{prefix} Pct"] = (pd.to_numeric(out[f"{prefix} Made"], errors="coerce").fillna(0) / max(total,1) * 100).round(1)
        return out

    r = _summary(round_filtered, "Round")
    b = _summary(bench_filtered, "Baseline")
    out = r.merge(b, on="Bucket", how="outer")
    if not out.empty:
        out["Bucket"] = pd.Categorical(out["Bucket"], categories=PUTT_BUCKET_ORDER, ordered=True)
        out = out.sort_values("Bucket").reset_index(drop=True)
    return out

def build_filtered_approach_metrics(frame):
    app = prepare_approach_frame(frame).copy()
    if app.empty:
        return {"attempts": 0, "gir": 0, "gir_pct": 0.0, "avg_prox": 0.0, "inside15": 0, "inside15_pct": 0.0}
    attempts = int(len(app))
    gir = int(pd.to_numeric(app["Approach GIR Flag"], errors="coerce").fillna(0).sum())
    gir_pct = (gir / attempts * 100.0) if attempts else 0.0
    avg_prox = float(pd.to_numeric(app["Approach Proximity"], errors="coerce").fillna(0).mean()) if attempts else 0.0
    inside15 = int((pd.to_numeric(app["Approach Proximity"], errors="coerce").fillna(9999) <= 15).sum())
    inside15_pct = (inside15 / attempts * 100.0) if attempts else 0.0
    return {"attempts": attempts, "gir": gir, "gir_pct": gir_pct, "avg_prox": avg_prox, "inside15": inside15, "inside15_pct": inside15_pct}


def build_short_game_inside_range_summary(round_data, benchmark_df_sg=None):
    """
    For each short-game leave-distance bucket:
    - Round chips finishing inside that range / total chip opportunities
    - Benchmark chips finishing inside that range / total chip opportunities
    This measures chipping performance into those leave distances, not 1-putt % from there.
    """
    round_sg = prepare_short_game_frame(round_data).copy()
    bench_sg = prepare_short_game_frame(benchmark_df_sg).copy() if benchmark_df_sg is not None else pd.DataFrame()

    def _summary(df_block, label_prefix="Round"):
        if df_block.empty:
            return pd.DataFrame(columns=["Bucket", f"{label_prefix} Inside", f"{label_prefix} Opportunities", f"{label_prefix} %"])
        total_opps = int(len(df_block))
        out = (
            df_block.dropna(subset=["SG Bucket"])
            .groupby("SG Bucket", as_index=False)
            .agg(Inside=("SG Attempt", "sum"))
            .rename(columns={"SG Bucket": "Bucket", "Inside": f"{label_prefix} Inside"})
        )
        out[f"{label_prefix} Opportunities"] = total_opps
        out[f"{label_prefix} %"] = (
            pd.to_numeric(out[f"{label_prefix} Inside"], errors="coerce").fillna(0)
            / max(total_opps, 1) * 100
        ).round(1)
        return out

    round_sum = _summary(round_sg, "Round")
    if benchmark_df_sg is not None:
        bench_sum = _summary(bench_sg, "Baseline")
        out = round_sum.merge(bench_sum, on="Bucket", how="outer")
        out["Delta"] = (
            pd.to_numeric(out["Round %"], errors="coerce").fillna(0)
            - pd.to_numeric(out["Baseline %"], errors="coerce").fillna(0)
        ).round(1)
    else:
        out = round_sum.copy()
        out["Baseline Inside"] = pd.NA
        out["Baseline Opportunities"] = pd.NA
        out["Baseline %"] = pd.NA
        out["Delta"] = pd.NA

    if not out.empty:
        out["Bucket"] = pd.Categorical(out["Bucket"], categories=SHORT_GAME_BUCKET_ORDER, ordered=True)
        out = out.sort_values("Bucket").reset_index(drop=True)
        out["DisplayLabel"] = out.apply(
            lambda r: f"{_safe_int_scalar(r.get('Round Inside', 0))}/{_safe_int_scalar(r.get('Round Opportunities', 0))} {_safe_float_scalar(r.get('Round %', 0)):.0f}%",
            axis=1
        )
    return out

def build_short_game_extra_stats(round_data):
    """
    Uses explicit short-game fields from the dataset:
    - Total Chips Per Hole
    - Chip Opportunity
    """
    d = round_data.copy()

    chips = pd.to_numeric(_safe_col(d, "Total Chips Per Hole", 0), errors="coerce").fillna(0)
    opportunities = pd.to_numeric(_safe_col(d, "Chip Opportunity", 0), errors="coerce").fillna(0)

    total_chips = int(chips.sum())
    total_opportunities = int(opportunities.sum())
    chips_per_hole = (total_chips / total_opportunities) if total_opportunities else 0.0
    holes_2plus = int((chips >= 2).sum())

    return {
        "opportunities": total_opportunities,
        "total_chips": total_chips,
        "chips_per_hole": chips_per_hole,
        "holes_2plus": holes_2plus,
    }




def build_us_open_par_summary(frame):
    """
    US Open Par
    Attempt:
      - Par 4 or Par 5
      - Approach GIR = Yes
    Made:
      - Attempt
      - Putts = 1
      - Score to Par = 0
    """
    d = frame.copy()
    d["ParN"] = pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")
    d["ApproachGIR"] = pd.to_numeric(_safe_col(d, "Approach GIR Value", 0), errors="coerce").fillna(0).astype(int)
    d["PuttsN"] = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0).astype(int)

    if "Score to Par" in d.columns:
        d["ScoreToParN"] = pd.to_numeric(_safe_col(d, "Score to Par", pd.NA), errors="coerce")
    else:
        d["HoleScoreN"] = pd.to_numeric(_safe_col(d, "Hole Score", pd.NA), errors="coerce")
        d["ScoreToParN"] = d["HoleScoreN"] - d["ParN"]

    d["USOpenAttempt"] = ((d["ParN"].isin([4, 5])) & (d["ApproachGIR"] == 1)).astype(int)
    d["USOpenMade"] = ((d["USOpenAttempt"] == 1) & (d["PuttsN"] == 1) & (d["ScoreToParN"] == 0)).astype(int)

    attempts = int(d["USOpenAttempt"].sum())
    made = int(d["USOpenMade"].sum())
    pct = (made / attempts * 100.0) if attempts else 0.0

    return {"attempts": attempts, "made": made, "pct": pct}




APPROACH_CLUB_ORDER = [
    "LW", "SW", "GW", "AW", "PW",
    "9I", "8I", "7I", "6I", "5I", "4I", "3I", "2I", "1I",
    "HY", "2H", "3H", "4H", "5H", "6H", "7H",
    "7W", "5W", "4W", "3W", "2W", "1W", "DRIVER"
]

def _club_sort_key(club):
    c = str(club).strip().upper()
    if c in APPROACH_CLUB_ORDER:
        return APPROACH_CLUB_ORDER.index(c)
    return len(APPROACH_CLUB_ORDER) + 100

def render_club_performance_curve(round_club, bench_club=None, compare_label="Baseline"):
    """
    Reliable SVG/HTML club performance curve.
    X-axis tapers from wedges to longer clubs.
    """
    if round_club is None or round_club.empty:
        st.info("No club performance curve data available for this round.")
        return

    round_df = round_club.copy()
    round_df["Club"] = round_df["Club"].astype(str).str.upper()
    round_df = round_df[round_df["Club"] != ""].copy()
    if round_df.empty:
        st.info("No club performance curve data available for this round.")
        return

    bench_df = bench_club.copy() if bench_club is not None and not bench_club.empty else pd.DataFrame()

    present_round = set(round_df["Club"].tolist())
    present_bench = set(bench_df["Club"].astype(str).str.upper().tolist()) if not bench_df.empty else set()
    club_order = [c for c in APPROACH_CLUB_ORDER if c in present_round or c in present_bench]

    # also include any unusual clubs after the known ordering
    extras = sorted((present_round | present_bench) - set(club_order), key=lambda x: str(x))
    club_order.extend(extras)

    if not club_order:
        st.info("No club performance curve data available for this round.")
        return

    round_df = round_df.set_index("Club").reindex(club_order).reset_index()
    round_df["Pct"] = pd.to_numeric(round_df["Pct"], errors="coerce").fillna(0.0)
    round_df["Made"] = pd.to_numeric(round_df["Made"], errors="coerce").fillna(0).astype(int)
    round_df["Attempts"] = pd.to_numeric(round_df["Attempts"], errors="coerce").fillna(0).astype(int)
    round_df["CurveLabel"] = round_df.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%", axis=1)

    base_df = None
    if not bench_df.empty:
        bench_df["Club"] = bench_df["Club"].astype(str).str.upper()
        bench_df = bench_df[bench_df["Club"].isin(club_order)].copy()
        if not bench_df.empty:
            bench_df = bench_df.set_index("Club").reindex(club_order).reset_index()
            bench_df["Pct"] = pd.to_numeric(bench_df["Pct"], errors="coerce").fillna(0.0)
            bench_df["BaseLabel"] = bench_df["Pct"].map(lambda x: f"{x:.0f}%")
            base_df = bench_df

    import html as _html
    import streamlit.components.v1 as components

    width = 1080
    height = 420
    left = 72
    right = 24
    top = 46
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    n = max(len(club_order), 1)
    step_x = plot_w / max(n - 1, 1)

    def _x(i):
        return left + (i * step_x if n > 1 else plot_w / 2)

    def _y(pct):
        pct = max(0.0, min(100.0, float(pct)))
        return top + plot_h * (1 - pct / 100.0)

    def _points(df, col="Pct"):
        pts = []
        for i, row in df.iterrows():
            pts.append((_x(i), _y(row[col])))
        return pts

    def _polyline(pts):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)

    round_pts = _points(round_df, "Pct")
    base_pts = _points(base_df, "Pct") if base_df is not None else []

    grid_lines = []
    for tick in [0, 25, 50, 75, 100]:
        y = _y(tick)
        grid_lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="rgba(255,255,255,0.12)" stroke-width="1"/>')
        grid_lines.append(f'<text x="{left-10}" y="{y+4:.1f}" fill="#bdbdbd" font-size="11" text-anchor="end">{tick}%</text>')

    x_labels = []
    for i, c in enumerate(club_order):
        x = _x(i)
        x_labels.append(f'<text x="{x:.1f}" y="{height-28}" fill="#d9d9d9" font-size="11" text-anchor="middle">{_html.escape(str(c))}</text>')

    round_point_elems = []
    round_label_elems = []
    for i, row in round_df.iterrows():
        x, y = round_pts[i]
        round_point_elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#4f8cff"/>')
        label = _html.escape(str(row["CurveLabel"]))
        round_label_elems.append(f'<text x="{x:.1f}" y="{max(16, y-12):.1f}" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle">{label}</text>')

    base_point_elems = []
    base_label_elems = []
    if base_df is not None:
        for i, row in base_df.iterrows():
            x, y = base_pts[i]
            base_point_elems.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.5" fill="#9aa0a6"/>')
            label = _html.escape(str(row["BaseLabel"]))
            base_label_elems.append(f'<text x="{x:.1f}" y="{min(height-bottom+18, y+16):.1f}" fill="#bdbdbd" font-size="10" text-anchor="middle">{label}</text>')

    legend = f"""
    <g>
      <line x1="{left}" y1="18" x2="{left+28}" y2="18" stroke="#4f8cff" stroke-width="3"/>
      <circle cx="{left+14}" cy="18" r="4" fill="#4f8cff"/>
      <text x="{left+38}" y="22" fill="#ffffff" font-size="12">Round</text>
      <line x1="{left+120}" y1="18" x2="{left+148}" y2="18" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4"/>
      <circle cx="{left+134}" cy="18" r="3.5" fill="#9aa0a6"/>
      <text x="{left+158}" y="22" fill="#d0d0d0" font-size="12">{_html.escape(compare_label)}</text>
    </g>
    """

    svg = f"""
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" xmlns="http://www.w3.org/2000/svg">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#1f1f1f" rx="12" ry="12"/>
      {''.join(grid_lines)}
      <line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>
      <line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="rgba(255,255,255,0.20)" stroke-width="1"/>
      {legend}
      <polyline points="{_polyline(round_pts)}" fill="none" stroke="#4f8cff" stroke-width="3"/>
      {'<polyline points="' + _polyline(base_pts) + '" fill="none" stroke="#9aa0a6" stroke-width="2" stroke-dasharray="6 4"/>' if base_pts else ''}
      {''.join(base_point_elems)}
      {''.join(round_point_elems)}
      {''.join(base_label_elems)}
      {''.join(round_label_elems)}
      {''.join(x_labels)}
      <text x="{width/2:.1f}" y="{height-8}" fill="#d9d9d9" font-size="12" text-anchor="middle">Approach Club</text>
      <text x="18" y="{top + plot_h/2:.1f}" fill="#d9d9d9" font-size="12" text-anchor="middle" transform="rotate(-90 18 {top + plot_h/2:.1f})">GIR %</text>
    </svg>
    """
    components.html(svg, height=height + 8, scrolling=False)




APPROACH_CLUB_DISPLAY_ORDER = [
    "LW", "SW", "GW", "AW", "PW",
    "9 Iron", "8 Iron", "7 Iron", "6 Iron", "5 Iron", "4 Iron", "3 Iron", "2 Iron", "1 Iron",
    "HY", "2H", "3H", "4H", "5H", "6H", "7H",
    "7W", "5W", "4W", "3W", "2W", "1W", "DRIVER"
]

def _club_display_sort_key(club):
    c = str(club).strip()
    if c in APPROACH_CLUB_DISPLAY_ORDER:
        return APPROACH_CLUB_DISPLAY_ORDER.index(c)
    return len(APPROACH_CLUB_DISPLAY_ORDER) + 100

def build_distance_club_compare_matrix(round_frame, bench_frame, min_attempts=1):
    round_heat = build_distance_club_heatmap(round_frame)
    bench_heat = build_distance_club_heatmap(bench_frame)

    merged = pd.merge(
        round_heat.rename(columns={
            "Attempts": "Round Attempts",
            "Made": "Round Made",
            "Pct": "Round Pct",
            "AvgProx": "Round Avg Prox"
        }),
        bench_heat.rename(columns={
            "Attempts": "Bench Attempts",
            "Made": "Bench Made",
            "Pct": "Bench Pct",
            "AvgProx": "Bench Avg Prox"
        }),
        on=["Bucket", "Club"],
        how="outer"
    )

    if merged.empty:
        return merged

    for col in ["Round Attempts", "Round Made", "Round Pct", "Round Avg Prox", "Bench Attempts", "Bench Made", "Bench Pct", "Bench Avg Prox"]:
        merged[col] = pd.to_numeric(merged.get(col), errors="coerce")

    fill_zero_cols = ["Round Attempts", "Round Made", "Round Pct", "Bench Attempts", "Bench Made", "Bench Pct"]
    merged[fill_zero_cols] = merged[fill_zero_cols].fillna(0)

    merged = merged[
        (merged["Round Attempts"] >= min_attempts) | (merged["Bench Attempts"] >= min_attempts)
    ].copy()

    if merged.empty:
        return merged

    merged["Delta"] = (merged["Round Pct"] - merged["Bench Pct"]).round(1)
    merged["CellLabel"] = merged.apply(
        lambda r: f"{int(r['Round Made'])}/{int(r['Round Attempts'])} {r['Round Pct']:.0f}%\nvs {r['Bench Pct']:.0f}%",
        axis=1
    )
    merged["Bucket"] = pd.Categorical(merged["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
    merged["_club_sort"] = merged["Club"].astype(str).apply(_club_display_sort_key)
    return merged.sort_values(["Bucket", "_club_sort", "Club"]).reset_index(drop=True)

def render_distance_club_compare_matrix(round_frame, bench_frame, compare_label="Baseline", min_attempts=1):
    heat = build_distance_club_compare_matrix(round_frame, bench_frame, min_attempts=min_attempts)
    if heat.empty:
        st.info("No usable distance × club rows available for the comparison matrix.")
        return

    club_order = heat[["Club", "_club_sort"]].drop_duplicates().sort_values(["_club_sort", "Club"])["Club"].tolist()
    bucket_order = [b for b in APPROACH_BUCKET_ORDER if b in heat["Bucket"].astype(str).tolist()]

    rect = (
        alt.Chart(heat)
        .mark_rect(stroke="#666", cornerRadius=8)
        .encode(
            x=alt.X("Club:N", sort=club_order, title="Club"),
            y=alt.Y("Bucket:N", sort=bucket_order, title="Distance Bucket"),
            color=alt.Color("Delta:Q", title="Round vs Baseline Δ", scale=alt.Scale(scheme="redyellowgreen")),
            tooltip=[
                alt.Tooltip("Bucket:N"),
                alt.Tooltip("Club:N"),
                alt.Tooltip("Round Attempts:Q"),
                alt.Tooltip("Round Made:Q"),
                alt.Tooltip("Round Pct:Q", format=".1f", title="Round GIR %"),
                alt.Tooltip("Bench Attempts:Q", title=f"{compare_label} Attempts"),
                alt.Tooltip("Bench Made:Q", title=f"{compare_label} Made"),
                alt.Tooltip("Bench Pct:Q", format=".1f", title=f"{compare_label} GIR %"),
                alt.Tooltip("Delta:Q", format="+.1f", title="Δ"),
            ]
        )
    )

    text = (
        alt.Chart(heat)
        .mark_text(fontSize=10, fontWeight="bold")
        .encode(
            x=alt.X("Club:N", sort=club_order),
            y=alt.Y("Bucket:N", sort=bucket_order),
            text="CellLabel:N",
            color=alt.condition("abs(datum.Delta) < 8", alt.value("black"), alt.value("white"))
        )
    )

    st.altair_chart(
        (rect + text)
        .properties(height=max(320, len(bucket_order) * 28), title=f"Distance × Club Matrix — Round vs {compare_label}")
        .configure_view(stroke=None),
        use_container_width=True
    )

    matrix_table = heat[[
        "Bucket", "Club", "Round Attempts", "Round Made", "Round Pct", "Bench Attempts", "Bench Made", "Bench Pct", "Delta"
    ]].copy()
    st.dataframe(matrix_table, use_container_width=True, hide_index=True)

def _build_dispersion_points(frame):
    d = prepare_approach_frame(frame).copy()
    if d.empty:
        return pd.DataFrame()

    vectors = {
        "Short Left": (-0.7, -0.7),
        "Left": (-1.0, 0.0),
        "Long Left": (-0.7, 0.7),
        "Short": (0.0, -1.0),
        "Long": (0.0, 1.0),
        "Short Right": (0.7, -0.7),
        "Right": (1.0, 0.0),
        "Long Right": (0.7, 0.7),
    }

    def _xy(row):
        prox = float(row.get("Approach Proximity", 0) or 0)
        if prox <= 0:
            prox = 5.0
        direction = row.get("Approach Miss Direction Clean", "")
        if direction in vectors:
            vx, vy = vectors[direction]
            return pd.Series({"x": vx * prox, "y": vy * prox})
        return pd.Series({"x": 0.0, "y": 0.0})

    pts = d.join(d.apply(_xy, axis=1))
    pts["Hole"] = _int(_safe_col(pts, "Hole", 0))
    return pts

def build_dispersion_ellipse_df(frame, n_std=1.5, n_points=120):
    pts = _build_dispersion_points(frame)
    if pts.empty or len(pts) < 3:
        return pd.DataFrame(columns=["x", "y"])

    arr = pts[["x", "y"]].dropna().to_numpy(dtype=float)
    if arr.shape[0] < 3:
        return pd.DataFrame(columns=["x", "y"])

    center = arr.mean(axis=0)
    cov = np.cov(arr, rowvar=False)
    if np.isnan(cov).any():
        return pd.DataFrame(columns=["x", "y"])

    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0, None)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = np.vstack([np.cos(theta), np.sin(theta)])
    scale = n_std * np.sqrt(vals)
    ellipse = (vecs @ np.diag(scale) @ circle).T + center

    return pd.DataFrame({"x": ellipse[:, 0], "y": ellipse[:, 1]})

# =========================================================
# Load + base prep
# =========================================================
df = pd.read_csv(CSV_FILE)
df["Date Played"] = pd.to_datetime(df["Date Played"], errors="coerce")
df["Month"] = df["Date Played"].dt.strftime("%B")
df["Year"] = df["Date Played"].dt.year

# =========================================================
# Sidebar Filters
# =========================================================
st.sidebar.header("🔍 Filter Rounds")
players = df["Player Name"].dropna().unique() if "Player Name" in df else []
courses = df["Course Name"].dropna().unique() if "Course Name" in df else []
months = df["Month"].dropna().unique() if "Month" in df else []
years = df["Year"].dropna().unique() if "Year" in df else []

selected_player = st.sidebar.selectbox("Player", [""] + sorted(players))
selected_course = st.sidebar.selectbox("Course", [""] + sorted(courses))
selected_month = st.sidebar.selectbox("Month", [""] + sorted(months, key=lambda x: datetime.datetime.strptime(x, "%B").month))
selected_year = st.sidebar.selectbox("Year", [""] + sorted(years, reverse=True))

filtered_df = df.copy()
if selected_player:
    filtered_df = filtered_df[filtered_df["Player Name"] == selected_player]
if selected_course:
    filtered_df = filtered_df[filtered_df["Course Name"] == selected_course]
if selected_month:
    filtered_df = filtered_df[filtered_df["Month"] == selected_month]
if selected_year:
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]

rounds = filtered_df["Round Link"].dropna().unique() if "Round Link" in filtered_df else []
if len(rounds) == 0:
    st.warning("No rounds found for selected filters.")
    st.stop()

round_selector_df = build_round_selector_df(filtered_df)
round_label_map = dict(zip(round_selector_df["Round Label"], round_selector_df["Round Link"]))
round_labels = round_selector_df["Round Label"].tolist()

if "selected_round_label" not in st.session_state or st.session_state["selected_round_label"] not in round_labels:
    st.session_state["selected_round_label"] = round_labels[0]

nav1, nav2, nav3 = st.columns([1, 3, 1])
current_idx = round_labels.index(st.session_state["selected_round_label"])

with nav1:
    if st.button("◀ Previous Round", use_container_width=True, disabled=(current_idx == len(round_labels) - 1)):
        st.session_state["selected_round_label"] = round_labels[current_idx + 1]
        st.rerun()

with nav3:
    if st.button("Next Round ▶", use_container_width=True, disabled=(current_idx == 0)):
        st.session_state["selected_round_label"] = round_labels[current_idx - 1]
        st.rerun()

with nav2:
    selected_round_label = st.selectbox("Select a Round", round_labels, index=round_labels.index(st.session_state["selected_round_label"]))
    st.session_state["selected_round_label"] = selected_round_label

selected_round = round_label_map[st.session_state["selected_round_label"]]
round_data = filtered_df[filtered_df["Round Link"] == selected_round].copy()

baseline_pool_df = build_round_selector_df(df)
baseline_pool_df = baseline_pool_df[baseline_pool_df["Round Link"] != selected_round].copy()
baseline_label_to_round = dict(zip(baseline_pool_df["Round Label"], baseline_pool_df["Round Link"]))
baseline_round_labels = baseline_pool_df["Round Label"].tolist()

with st.expander("🆚 Custom Round Baseline", expanded=False):
    st.caption("Pick one or more rounds to use as a custom comparison set. This can include rounds from other players too. Then choose 'Selected Rounds' in any comparison control below.")
    baseline_search = st.text_input("Search comparison rounds", value="", key="baseline_search_text")
    visible_baseline_labels = baseline_round_labels
    if baseline_search.strip():
        q = baseline_search.strip().lower()
        visible_baseline_labels = [lbl for lbl in baseline_round_labels if q in lbl.lower()]

    selected_baseline_labels = st.multiselect(
        "Comparison rounds",
        options=visible_baseline_labels,
        default=st.session_state.get("selected_baseline_round_labels", []),
        key="selected_baseline_round_labels",
    )
    selected_baseline_round_ids = [baseline_label_to_round[x] for x in selected_baseline_labels if x in baseline_label_to_round]
    selected_baseline_df = df[df["Round Link"].isin(selected_baseline_round_ids)].copy() if selected_baseline_round_ids else df.iloc[0:0].copy()
    if selected_baseline_labels:
        st.markdown(f"**Selected:** {len(selected_baseline_labels)} round(s)")
        render_baseline_summary_box(build_baseline_summary(selected_baseline_df, "Selected Rounds"))
    else:
        st.caption("No custom rounds selected yet.")
        selected_baseline_df = df.iloc[0:0].copy()

# =========================================================
# Selected round prep
# =========================================================
player = round_data["Player Name"].iloc[0]
course = round_data["Course Name"].iloc[0]
date = pd.to_datetime(round_data["Date Played"].iloc[0], errors="coerce").strftime("%B %d, %Y")

round_data = round_data.sort_values("Hole").copy()
round_data["Hole"] = _int(_safe_col(round_data, "Hole", 0))
round_data["Hole Score"] = _int(_safe_col(round_data, "Hole Score", 0))
round_data["Putts"] = _int(_safe_col(round_data, "Putts", 0))
round_data["Par"] = _int(_safe_col(round_data, "Par", 4))
round_data["Yards"] = _int(_safe_col(round_data, "Yards", 0))
round_data["Fairway"] = _int(_safe_col(round_data, "Fairway", 0))
round_data["GIR"] = _int(_safe_col(round_data, "GIR", 0))
round_data["Arnie"] = _int(_safe_col(round_data, "Arnie", 0))
round_data["Approach GIR Value"] = _int(_safe_col(round_data, "Approach GIR Value", 0))
round_data["Score to Par"] = _num(_safe_col(round_data, "Score to Par", 0))

fairways = round_data["Fairway"].apply(lambda x: "<span title='Fairway Hit'>🟢</span>" if x == 1 else "").tolist()
girs = round_data["GIR"].apply(lambda x: "<span title='Green in Regulation'>🟢</span>" if x == 1 else "").tolist()
arnies = round_data["Arnie"].apply(lambda x: "<span title='Arnie (Par w/o FW or GIR)'>🅰️</span>" if x == 1 else "").tolist()
approach_gir = round_data["Approach GIR Value"].apply(lambda x: "<span title='Approach GIR'>🟡</span>" if x == 1 else "").tolist()

lost_balls = (
    _int(_safe_col(round_data, "Lost Ball Tee Shot Quantity", 0))
    + _int(_safe_col(round_data, "Lost Ball Approach Shot Quantity", 0))
).astype(int).tolist()

approach_clubs = _safe_col(round_data, "Approach Shot Club Used", "").fillna("").tolist()
approach_yards = _num(_safe_col(round_data, "Approach Shot Distance (how far you had to the hole)", 0)).round(0).astype(int).tolist()
prox_to_hole = _num(_safe_col(round_data, "Proximity to Hole - How far is your First Putt (FT)", 0)).round(0).astype(int).tolist()
putt_made_ft = _safe_col(round_data, "Feet of Putt Made (How far was the putt you made)", "").fillna("").tolist()
approach_miss_dir = _safe_col(round_data, "Approach Shot Direction Miss", "").fillna("").tolist()

holes = round_data["Hole"].tolist()
pars = round_data["Par"].tolist()
scores = round_data["Hole Score"].tolist()
putts = round_data["Putts"].tolist()
yards = round_data["Yards"].tolist()

par_row = insert_segment_sums(pars)
score_row = insert_segment_sums(scores)
putts_row = insert_segment_sums(putts)
fw_row = insert_icon_sums(fairways, "🟢")
gir_row = insert_icon_sums(girs, "🟢")
arnie_row = insert_icon_sums(arnies, "🅰️")
yards_row = insert_segment_sums(yards)
lost_ball_row = insert_segment_sums(lost_balls)
approach_yards_row = approach_yards[:9] + [""] + approach_yards[9:18] + ["", ""]
approach_gir_row = insert_icon_sums(approach_gir, "🟡")
prox_to_hole_row = prox_to_hole[:9] + [""] + prox_to_hole[9:18] + ["", ""]
approach_clubs_row = approach_clubs[:9] + [""] + approach_clubs[9:18] + ["", ""]
approach_miss_dir_row = approach_miss_dir[:9] + [""] + approach_miss_dir[9:18] + ["", ""]

putt_made_ft_numeric = _num(_safe_col(round_data, "Feet of Putt Made (How far was the putt you made)", 0))
out_ft = int(putt_made_ft_numeric.iloc[:9].sum())
in_ft = int(putt_made_ft_numeric.iloc[9:18].sum())
total_ft = out_ft + in_ft
putt_made_ft_row = putt_made_ft[:9] + [out_ft] + putt_made_ft[9:18] + [in_ft, total_ft]

hole_nums = holes[:9] + ["Out"] + holes[9:18] + ["In", "Total"]

# =========================================================
# Summary stats
# =========================================================
total_score = sum(scores)
total_putts = sum(putts)
holes_played = len(round_data)

fw_total = int(round_data.loc[round_data["Par"].isin([4, 5]), "Fairway"].sum())
fw_attempts = int(round_data.loc[round_data["Par"].isin([4, 5]), "Fairway"].count())
fw_pct_num = (fw_total / fw_attempts * 100) if fw_attempts else 0.0

gir_total = int(round_data["GIR"].sum())
gir_attempts = int(len(round_data))
gir_pct_num = (gir_total / gir_attempts * 100) if gir_attempts else 0.0

avg_par3 = round(round_data.loc[round_data["Par"] == 3, "Hole Score"].mean(), 2) if not round_data.loc[round_data["Par"] == 3].empty else 0
avg_par4 = round(round_data.loc[round_data["Par"] == 4, "Hole Score"].mean(), 2) if not round_data.loc[round_data["Par"] == 4].empty else 0
avg_par5 = round(round_data.loc[round_data["Par"] == 5, "Hole Score"].mean(), 2) if not round_data.loc[round_data["Par"] == 5].empty else 0

total_1_putts = int((round_data["Putts"] == 1).sum())
total_3_plus_putts = int((round_data["Putts"] >= 3).sum())
total_3_putt_bogeys = int(_num(_safe_col(round_data, "3 Putt Bogey", 0)).sum())

putts_made_ft_numeric = _num(_safe_col(round_data, "Feet of Putt Made (How far was the putt you made)", 0)).fillna(0)
longest_putt_made = float(putts_made_ft_numeric.max()) if not putts_made_ft_numeric.empty else 0.0
putts_made_5_or_less = int(((putts_made_ft_numeric > 0) & (putts_made_ft_numeric <= 5)).sum())
putts_made_5_to_10 = int(((putts_made_ft_numeric > 5) & (putts_made_ft_numeric <= 10)).sum())
putts_made_over_10 = int((putts_made_ft_numeric > 10).sum())

pro_par_total = int(_num(_safe_col(round_data, "Pro Par", 0)).sum())
pro_birdie_total = int(_num(_safe_col(round_data, "Pro Birdie", 0)).sum())
pro_eagle_total = int(_num(_safe_col(round_data, "Pro Eagle+", 0)).sum())
pro_pars_total = pro_par_total + pro_birdie_total + pro_eagle_total

total_scrambles = int(_num(_safe_col(round_data, "Scramble", 0)).sum())
total_scramble_ops = int(_num(_safe_col(round_data, "Scramble Opportunity", 0)).sum())
scrambles_display = _fmt_frac_pct(total_scrambles, total_scramble_ops)

_gir = _int(_safe_col(round_data, "GIR", 0))
_putts_clean = _int(_safe_col(round_data, "Putts", 0))
total_updowns = int(((_gir == 0) & (_putts_clean == 1)).sum())
total_updown_ops = int(_num(_safe_col(round_data, "Scramble Opportunity", 0)).sum())
updowns_display = _fmt_frac_pct(total_updowns, total_updown_ops)

lost_ball_tee = int(_num(_safe_col(round_data, "Lost Ball Tee Shot Quantity", 0)).sum())
lost_ball_appr = int(_num(_safe_col(round_data, "Lost Ball Approach Shot Quantity", 0)).sum())
total_lost_balls = lost_ball_tee + lost_ball_appr
lost_balls_display = f"Tee {lost_ball_tee} / Approach {lost_ball_appr} / Total {total_lost_balls}"

score_type_counts = round_data["Score Label"].value_counts() if "Score Label" in round_data else pd.Series(dtype=int)

def categorize_score_label(score_label):
    if score_label in ["Birdie", "Eagle", "Albatross", "Par"]:
        return "Par or Better"
    elif score_label == "Bogey":
        return "Bogey"
    return "Double+"

categories = round_data["Score Label"].apply(categorize_score_label) if "Score Label" in round_data else pd.Series([""] * len(round_data))
cat_counts = categories.value_counts()

gir3_m, gir3_t, gir3_pct = _made_total_pct_by_par(round_data, "GIR", 3)
gir4_m, gir4_t, gir4_pct = _made_total_pct_by_par(round_data, "GIR", 4)
gir5_m, gir5_t, gir5_pct = _made_total_pct_by_par(round_data, "GIR", 5)

fw4_m, fw4_t, fw4_pct = _made_total_pct_by_par(round_data, "Fairway", 4)
fw5_m, fw5_t, fw5_pct = _made_total_pct_by_par(round_data, "Fairway", 5)

seves_total = int(_num(_safe_col(round_data, "Seve", 0)).sum())
hole_outs_total = build_short_game_hole_out_stats(round_data)["count"]
arnies_total = int(_num(_safe_col(round_data, "Arnie", 0)).sum())

score_to_par_total = int(_num(_safe_col(round_data, "Score to Par", total_score - sum(pars))).sum())
score_to_par_str = _fmt_to_par(score_to_par_total)
putts_per_hole = (total_putts / holes_played) if holes_played else 0.0

# Benchmarks
player_df = df[df["Player Name"] == player].copy()
player_df["Date Played"] = pd.to_datetime(player_df["Date Played"], errors="coerce")

def _round_over_par(src_df, round_id):
    block = src_df[src_df["Round Link"] == round_id]
    hole_sum = pd.to_numeric(block["Hole Score"], errors="coerce").sum()
    par_sum = pd.to_numeric(block["Par"], errors="coerce").sum()
    return int(hole_sum - par_sum)

if {"Date Played", "Round Link"} <= set(filtered_df.columns):
    last5_round_ids_filtered = (
        filtered_df[["Round Link", "Date Played"]]
        .drop_duplicates()
        .sort_values("Date Played")
        .tail(5)["Round Link"]
        .tolist()
    )
else:
    last5_round_ids_filtered = []

last5_over_par_values = [_round_over_par(filtered_df, rid) for rid in last5_round_ids_filtered]
last5_avg_over_par = (sum(last5_over_par_values) / len(last5_over_par_values)) if last5_over_par_values else 0.0

last100 = filtered_df.sort_values("Date Played").tail(100)
last100_total_over_par = int(
    pd.to_numeric(last100["Hole Score"], errors="coerce").sum()
    - pd.to_numeric(last100["Par"], errors="coerce").sum()
) if not last100.empty else 0
n_last100_holes = int(last100.shape[0]) if last100 is not None else 0
last100_per18 = (last100_total_over_par * 18 / n_last100_holes) if n_last100_holes else 0.0

round_order = player_df.groupby("Round Link")["Date Played"].max().sort_values(ascending=False)
last5_round_ids = round_order.index[:5].tolist()
last5_df = player_df[player_df["Round Link"].isin(last5_round_ids)]
last100_df = player_df.sort_values("Date Played", ascending=False).head(100)

def ud_stats(frame):
    gir0 = (_int(_safe_col(frame, "GIR", 0)) == 0)
    putt1 = (_int(_safe_col(frame, "Putts", 0)) == 1)
    up_made = int((gir0 & putt1).sum())
    up_ops = int(_num(_safe_col(frame, "Scramble Opportunity", 0)).sum())
    pct = (up_made / up_ops * 100) if up_ops else 0.0
    return up_made, up_ops, pct

def scramble_stats(frame):
    made = int(_num(_safe_col(frame, "Scramble", 0)).sum())
    ops = int(_num(_safe_col(frame, "Scramble Opportunity", 0)).sum())
    pct = (made / ops * 100) if ops else 0.0
    return made, ops, pct

def lostball_stats(frame):
    tee = int(_num(_safe_col(frame, "Lost Ball Tee Shot Quantity", 0)).sum())
    appr = int(_num(_safe_col(frame, "Lost Ball Approach Shot Quantity", 0)).sum())
    total = tee + appr
    rounds_n = max(1, int(frame["Round Link"].nunique())) if "Round Link" in frame else 1
    holes_n = int(frame.shape[0])
    per_round = total / rounds_n
    per_18 = (total / (holes_n / 18.0)) if holes_n else 0.0
    return tee, appr, total, per_round, per_18

ud5_m, ud5_o, ud5_pct = ud_stats(last5_df)
sc5_m, sc5_o, sc5_pct = scramble_stats(last5_df)
lb5_t, lb5_a, lb5_tot, lb5_per_round, lb5_per18 = lostball_stats(last5_df)

ud100_m, ud100_o, ud100_pct = ud_stats(last100_df)
sc100_m, sc100_o, sc100_pct = scramble_stats(last100_df)
lb100_t, lb100_a, lb100_tot, lb100_per_round, lb100_per18 = lostball_stats(last100_df)

_hist = df[df["Player Name"] == player].copy()
_hist["Date Played"] = pd.to_datetime(_hist["Date Played"], errors="coerce")
_hist_excl = _hist[_hist["Round Link"] != selected_round].copy()

def _gir_pct(dataframe):
    if dataframe.empty or "GIR" not in dataframe:
        return None
    made = pd.to_numeric(dataframe["GIR"], errors="coerce").fillna(0).sum()
    total = len(dataframe)
    return (made / total * 100.0) if total else None

def _fw_pct_p45(dataframe):
    if dataframe.empty or "Fairway" not in dataframe or "Par" not in dataframe:
        return None
    block = dataframe[dataframe["Par"].isin([4, 5])]
    if block.empty:
        return None
    made = pd.to_numeric(block["Fairway"], errors="coerce").fillna(0).sum()
    total = block["Fairway"].count()
    return (made / total * 100.0) if total else None

def _putts_per_hole(dataframe):
    if dataframe.empty or "Putts" not in dataframe:
        return None
    return pd.to_numeric(dataframe["Putts"], errors="coerce").fillna(0).mean()

def _putts_per_round_mean(dataframe):
    if dataframe.empty or "Putts" not in dataframe or "Round Link" not in dataframe:
        return None
    per_round = dataframe.assign(P=pd.to_numeric(dataframe["Putts"], errors="coerce").fillna(0)).groupby("Round Link")["P"].sum()
    return per_round.mean() if not per_round.empty else None

def _delta_str(curr, ref, suffix=""):
    if curr is None or ref is None:
        return "—"
    diff = curr - ref
    arrow = "🔺" if diff > 0 else ("🔻" if diff < 0 else "—")
    return f"{arrow} {diff:+.1f}{suffix}"

def _fmt_val(v, suffix=""):
    if v is None:
        return "n/a"
    return f"{v:.1f}{suffix}"

_prev5_round_ids = (
    _hist_excl.sort_values("Date Played")
    .dropna(subset=["Round Link"])
    .drop_duplicates(subset=["Round Link"], keep="last")
    .sort_values("Date Played", ascending=False)["Round Link"]
    .head(5)
    .tolist()
)
_prev5_df = _hist_excl[_hist_excl["Round Link"].isin(_prev5_round_ids)].copy()
_last100_df = _hist_excl.sort_values(["Date Played", "Hole"], ascending=[False, True]).head(100).copy()

prev5_gir = _gir_pct(_prev5_df)
last100_gir = _gir_pct(_last100_df)
prev5_fw = _fw_pct_p45(_prev5_df)
last100_fw = _fw_pct_p45(_last100_df)
prev5_pph = _putts_per_hole(_prev5_df)
last100_pph = _putts_per_hole(_last100_df)
prev5_ppr = _putts_per_round_mean(_prev5_df)
last100_ppr_equiv = (last100_pph * 18.0) if last100_pph is not None else None

curr_pph = putts_per_hole
curr_ppr = float(total_putts)
curr_gir = gir_pct_num
curr_fw = fw_pct_num

# =========================================================
# Header
# =========================================================
st.markdown("🏌️ *“Arnie steps to the tee with precision in mind. Seve follows, carving creativity from the rough.”*", unsafe_allow_html=True)


def build_overview_metric_rows(round_overview, base_overview, compare_label):
    rows = [
        {"Metric": "Avg Score / 18", "Round": round_overview["score_per18"], "Baseline": base_overview["score_per18"], "Better": "Lower", "Fmt": "{:.1f}"},
        {"Metric": "Putts / 18", "Round": round_overview["putts_per18"], "Baseline": base_overview["putts_per18"], "Better": "Lower", "Fmt": "{:.1f}"},
        {"Metric": "GIR %", "Round": round_overview["gir_pct"], "Baseline": base_overview["gir_pct"], "Better": "Higher", "Fmt": "{:.1f}%"},
        {"Metric": "Fairway %", "Round": round_overview["fw_pct"], "Baseline": base_overview["fw_pct"], "Better": "Higher", "Fmt": "{:.1f}%"},
        {"Metric": "Scramble %", "Round": round_overview["scramble_pct"], "Baseline": base_overview["scramble_pct"], "Better": "Higher", "Fmt": "{:.1f}%"},
        {"Metric": "Up & Down %", "Round": round_overview["updown_pct"], "Baseline": base_overview["updown_pct"], "Better": "Higher", "Fmt": "{:.1f}%"},
        {"Metric": "Arnies / 18", "Round": round_overview["arnies_per18"], "Baseline": base_overview["arnies_per18"], "Better": "Higher", "Fmt": "{:.2f}"},
        {"Metric": "Seves / 18", "Round": round_overview["seves_per18"], "Baseline": base_overview["seves_per18"], "Better": "Higher", "Fmt": "{:.2f}"},
        {"Metric": "Avg First-Putt Proximity", "Round": round_overview["avg_prox"], "Baseline": base_overview["avg_prox"], "Better": "Lower", "Fmt": "{:.1f} ft"},
        {"Metric": "Avg Prox if GIR", "Round": round_overview["avg_prox_gir"], "Baseline": base_overview["avg_prox_gir"], "Better": "Lower", "Fmt": "{:.1f} ft"},
        {"Metric": "Avg Prox if No GIR", "Round": round_overview["avg_prox_no_gir"], "Baseline": base_overview["avg_prox_no_gir"], "Better": "Lower", "Fmt": "{:.1f} ft"},
        {"Metric": "Lost Balls / 18", "Round": round_overview["lost_balls_per18"], "Baseline": base_overview["lost_balls_per18"], "Better": "Lower", "Fmt": "{:.2f}"},
    ]
    out = pd.DataFrame(rows)
    out["Compare Label"] = compare_label
    out["Delta"] = out["Round"] - out["Baseline"]
    out["Arrow"] = out.apply(lambda r: _delta_arrow(r["Delta"], higher_better=(r["Better"] == "Higher"), tol=0.05), axis=1)
    out["GoodDelta"] = out.apply(lambda r: (r["Baseline"] - r["Round"]) if r["Better"] == "Lower" else (r["Round"] - r["Baseline"]), axis=1)
    out["Round Display"] = out.apply(lambda r: r["Fmt"].format(r["Round"]), axis=1)
    out["Baseline Display"] = out.apply(lambda r: r["Fmt"].format(r["Baseline"]), axis=1)
    out["Delta Display"] = out.apply(
        lambda r: (f'{r["Arrow"]} {r["Delta"]:+.1f}%' if "%" in r["Fmt"] else (f'{r["Arrow"]} {r["Delta"]:+.1f} ft' if "ft" in r["Fmt"] else f'{r["Arrow"]} {r["Delta"]:+.2f}' if ".2f" in r["Fmt"] else f'{r["Arrow"]} {r["Delta"]:+.1f}')),
        axis=1
    )
    return out


def render_overview_cards(metric_df, compare_label):
    import streamlit.components.v1 as components

    html = """
    <style>
      .ov-wrap {
        display:grid;
        grid-template-columns:repeat(4, minmax(180px,1fr));
        gap:12px;
        margin:4px 0 10px 0;
        font-family:Segoe UI, Roboto, Arial, sans-serif;
      }
      .ov-card {
        background:linear-gradient(180deg,#2b2b2b 0%, #222 100%);
        border:1px solid rgba(255,255,255,.08);
        border-radius:14px;
        padding:12px 14px;
        box-shadow:0 8px 18px rgba(0,0,0,.18);
      }
      .ov-k {
        font-size:12px;
        color:#a9a9a9;
        margin-bottom:6px;
        font-weight:700;
      }
      .ov-v {
        font-size:26px;
        color:#fff;
        font-weight:800;
        line-height:1.05;
        margin-bottom:6px;
      }
      .ov-b {
        font-size:12px;
        color:#bdbdbd;
        margin-bottom:4px;
      }
      .ov-d {
        font-size:12px;
        font-weight:800;
      }
      .ov-good {color:#64dfb5;}
      .ov-bad {color:#ee6c4d;}
      .ov-flat {color:#bdbdbd;}
    </style>
    <div class="ov-wrap">
    """

    for _, r in metric_df.iterrows():
        cls = "ov-good" if r["GoodDelta"] > 0.05 else ("ov-bad" if r["GoodDelta"] < -0.05 else "ov-flat")
        html += f"""
        <div class="ov-card">
          <div class="ov-k">{r["Metric"]}</div>
          <div class="ov-v">{r["Round Display"]}</div>
          <div class="ov-b">vs {compare_label}: {r["Baseline Display"]}</div>
          <div class="ov-d {cls}">{r["Delta Display"]}</div>
        </div>
        """

    html += "</div>"

    n_cards = max(len(metric_df), 1)
    n_rows = (n_cards + 3) // 4
    height_px = 115 * n_rows + 20

    components.html(html, height=height_px, scrolling=False)


def render_overview_delta_chart(metric_df, compare_label):
    plot_df = metric_df.copy()
    plot_df["ColorKey"] = plot_df["GoodDelta"].apply(lambda x: "Better" if x > 0.05 else ("Worse" if x < -0.05 else "Flat"))
    plot_df["SignedLabel"] = plot_df["GoodDelta"].apply(lambda x: f"{x:+.1f}")
    chart = alt.Chart(plot_df).mark_bar(cornerRadius=8).encode(
        y=alt.Y("Metric:N", sort=alt.EncodingSortField(field="GoodDelta", op="sum", order="descending"), title=None),
        x=alt.X("GoodDelta:Q", title=f"Improvement vs {compare_label}"),
        color=alt.Color("ColorKey:N", scale=alt.Scale(domain=["Better","Flat","Worse"], range=["#64dfb5","#9aa0a6","#ee6c4d"]), legend=None),
        tooltip=[
            alt.Tooltip("Metric:N"),
            alt.Tooltip("Round Display:N", title="Round"),
            alt.Tooltip("Baseline Display:N", title=compare_label),
            alt.Tooltip("Delta Display:N", title="Change"),
        ],
    )
    labels = alt.Chart(plot_df).mark_text(align="left", dx=6, fontWeight="bold", color="white").encode(
        y=alt.Y("Metric:N", sort=alt.EncodingSortField(field="GoodDelta", op="sum", order="descending")),
        x=alt.X("GoodDelta:Q"),
        text="SignedLabel:N",
    )
    st.altair_chart((chart + labels).properties(height=340).configure_view(strokeOpacity=0), use_container_width=True)



def build_score_mix_summary(frame):
    d = frame.copy()
    if d.empty or "Score Label" not in d.columns:
        return {
            "score_df": pd.DataFrame(columns=["Category", "Count", "Percent"]),
            "cat_df": pd.DataFrame(columns=["Category", "Count", "Percent"]),
        }

    score_order = ["Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]
    score_counts = d["Score Label"].value_counts()

    score_df = pd.DataFrame({
        "Category": score_order,
        "Count": [int(score_counts.get(k, 0)) for k in score_order],
    })
    total = max(int(score_df["Count"].sum()), 1)
    score_df["Percent"] = score_df["Count"] / total * 100.0

    def _score_bucket(x):
        if x in ["Albatross", "Eagle", "Birdie", "Par"]:
            return "Par or Better"
        if x == "Bogey":
            return "Bogey"
        return "Double+"

    cat_counts = d["Score Label"].fillna("").apply(_score_bucket).value_counts()
    cat_order = ["Par or Better", "Bogey", "Double+"]
    cat_df = pd.DataFrame({
        "Category": cat_order,
        "Count": [int(cat_counts.get(k, 0)) for k in cat_order],
    })
    cat_total = max(int(cat_df["Count"].sum()), 1)
    cat_df["Percent"] = cat_df["Count"] / cat_total * 100.0

    return {"score_df": score_df, "cat_df": cat_df}


def _build_segment_row_html(series_name, df_block, color_map, total_count):
    if df_block is None or df_block.empty:
        return f"""
        <div class="mix-row">
          <div class="mix-series">{series_name}</div>
          <div class="mix-bar-wrap"><div class="mix-bar-bg"></div></div>
          <div class="mix-total">0 holes</div>
        </div>
        """

    parts_html = ""
    legend_html = ""
    for _, r in df_block.iterrows():
        cat = str(r["Category"])
        count = int(pd.to_numeric(r["Count"], errors="coerce") or 0)
        pct = float(pd.to_numeric(r["Percent"], errors="coerce") or 0.0)
        color = color_map.get(cat, "#777")
        label = f"{cat}: {count} ({pct:.1f}%)"
        parts_html += f'<div class="mix-seg" style="width:{pct:.4f}%; background:{color};" title="{label}"></div>'
        legend_html += f'<div class="mix-legend-item"><span class="mix-dot" style="background:{color};"></span>{cat}: {count} ({pct:.1f}%)</div>'

    return f"""
    <div class="mix-row">
      <div class="mix-series">{series_name}</div>
      <div class="mix-bar-wrap">
        <div class="mix-bar-bg">
          {parts_html}
        </div>
      </div>
      <div class="mix-total">{int(total_count)} holes</div>
    </div>
    <div class="mix-legend">{legend_html}</div>
    """

def render_segmented_score_mix_compare(round_df, base_df, compare_label):
    color_map = {
        "Eagle": "#71c7ec",
        "Birdie": "#64dfb5",
        "Par": "#c7c7c7",
        "Bogey": "#f2c14e",
        "Double Bogey": "#ee6c4d",
        "Triple Bogey +": "#b23a48",
    }

    round_total = int(pd.to_numeric(round_df.get("Count", 0), errors="coerce").fillna(0).sum()) if round_df is not None and not round_df.empty else 0
    base_total = int(pd.to_numeric(base_df.get("Count", 0), errors="coerce").fillna(0).sum()) if base_df is not None and not base_df.empty else 0

    html = """
    <style>
      .mix-card {background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px 14px 10px 14px; margin-bottom:10px;}
      .mix-row {display:grid; grid-template-columns:110px 1fr 80px; gap:10px; align-items:center; margin-bottom:8px;}
      .mix-series {font-size:13px; font-weight:800; color:#fff;}
      .mix-bar-wrap {position:relative;}
      .mix-bar-bg {position:relative; display:flex; width:100%; height:28px; overflow:hidden; border-radius:999px; background:#3a3a3a; box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);}
      .mix-seg {height:100%;}
      .mix-total {font-size:12px; color:#bdbdbd; text-align:right; font-variant-numeric:tabular-nums;}
      .mix-legend {display:flex; flex-wrap:wrap; gap:6px 12px; margin:8px 0 10px 120px;}
      .mix-legend-item {font-size:11px; color:#d8d8d8; white-space:nowrap; line-height:1.3;}
      .mix-dot {display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; vertical-align:middle;}
      .mix-seg-label {display:none !important;}
      @media (min-width: 300px) {
        .mix-seg-label {display:none !important;}
      }
    </style>
    <div class="mix-card">
    """
    html += _build_segment_row_html("This Round", round_df, color_map, round_total)
    html += _build_segment_row_html(compare_label, base_df, color_map, base_total)
    html += "</div>"
    import streamlit.components.v1 as components
    components.html(html, height=190, scrolling=False)

def render_segmented_category_mix_compare(round_df, base_df, compare_label):
    color_map = {
        "Par or Better": "#64dfb5",
        "Bogey": "#f2c14e",
        "Double+": "#ee6c4d",
    }

    round_total = int(pd.to_numeric(round_df.get("Count", 0), errors="coerce").fillna(0).sum()) if round_df is not None and not round_df.empty else 0
    base_total = int(pd.to_numeric(base_df.get("Count", 0), errors="coerce").fillna(0).sum()) if base_df is not None and not base_df.empty else 0

    html = """
    <style>
      .mix-card {background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px 14px 10px 14px; margin-bottom:10px;}
      .mix-row {display:grid; grid-template-columns:110px 1fr 80px; gap:10px; align-items:center; margin-bottom:8px;}
      .mix-series {font-size:13px; font-weight:800; color:#fff;}
      .mix-bar-wrap {position:relative;}
      .mix-bar-bg {position:relative; display:flex; width:100%; height:28px; overflow:hidden; border-radius:999px; background:#3a3a3a; box-shadow: inset 0 0 0 1px rgba(255,255,255,.06);}
      .mix-seg {height:100%;}
      .mix-total {font-size:12px; color:#bdbdbd; text-align:right; font-variant-numeric:tabular-nums;}
      .mix-legend {display:flex; flex-wrap:wrap; gap:6px 12px; margin:8px 0 10px 120px;}
      .mix-legend-item {font-size:11px; color:#d8d8d8; white-space:nowrap; line-height:1.3;}
      .mix-dot {display:inline-block; width:10px; height:10px; border-radius:999px; margin-right:6px; vertical-align:middle;}
    </style>
    <div class="mix-card">
    """
    html += _build_segment_row_html("This Round", round_df, color_map, round_total)
    html += _build_segment_row_html(compare_label, base_df, color_map, base_total)
    html += "</div>"
    import streamlit.components.v1 as components
    components.html(html, height=210, scrolling=False)




def render_scorecard_summary_panel(
    player,
    course,
    date,
    total_score,
    score_to_par_str,
    holes_played,
    putts_per_hole,
    gir_total,
    gir_pct_num,
    fw_total,
    fw_attempts,
    fw_pct_num,
    scrambles_display,
    updowns_display,
    us_open_made,
    us_open_attempts,
    us_open_pct,
    avg_par3,
    avg_par4,
    avg_par5,
    total_1_putts,
    total_3_plus_putts,
    total_3_putt_bogeys,
    longest_putt_made,
    putts_made_5_or_less,
    putts_made_5_to_10,
    putts_made_over_10,
    pro_pars_total,
    arnies_total,
    seves_total,
    hole_outs_total,
    lost_balls_display,
    gir3_m, gir3_t, gir3_pct,
    gir4_m, gir4_t, gir4_pct,
    gir5_m, gir5_t, gir5_pct,
    fw4_m, fw4_t, fw4_pct,
    fw5_m, fw5_t, fw5_pct
):
    import streamlit.components.v1 as components

    html = f"""
    <style>
      .scsum-wrap {{
        margin-top: 10px;
        font-family: Segoe UI, Roboto, Arial, sans-serif;
      }}
      .scsum-head {{
        background: linear-gradient(180deg, #2b2b2b 0%, #222 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 12px 14px;
        margin-bottom: 10px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
      }}
      .scsum-title {{
        font-size: 18px;
        font-weight: 800;
        color: #fff;
        margin-bottom: 2px;
      }}
      .scsum-sub {{
        font-size: 12px;
        color: #bdbdbd;
      }}
      .scsum-grid {{
        display:grid;
        grid-template-columns: repeat(3, minmax(220px, 1fr));
        gap: 12px;
      }}
      .scsum-card {{
        background: linear-gradient(180deg, #2a2a2a 0%, #202020 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 12px 14px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
      }}
      .scsum-h {{
        font-size: 13px;
        font-weight: 800;
        color: #fff;
        margin-bottom: 10px;
      }}
      .scsum-row {{
        display:flex;
        justify-content:space-between;
        gap: 10px;
        align-items:flex-start;
        margin: 6px 0;
        font-size: 12px;
        line-height: 1.35;
      }}
      .scsum-l {{
        color:#d7d7d7;
        font-weight:600;
      }}
      .scsum-r {{
        color:#fff;
        font-weight:800;
        text-align:right;
        font-variant-numeric: tabular-nums;
      }}
      .scsum-big {{
        font-size: 26px;
        font-weight: 900;
        color: #fff;
        margin-bottom: 2px;
      }}
      .scsum-muted {{
        color:#a9a9a9;
        font-weight:600;
      }}
    </style>

    <div class="scsum-wrap">
      <div class="scsum-head">
        <div class="scsum-title">{player} • {course}</div>
        <div class="scsum-sub">{date} • {holes_played} holes</div>
      </div>

      <div class="scsum-grid">
        <div class="scsum-card">
          <div class="scsum-h">🏌️ Round Snapshot</div>
          <div class="scsum-big">{total_score} <span class="scsum-muted">({score_to_par_str})</span></div>
          <div class="scsum-row"><div class="scsum-l">Putts / Hole</div><div class="scsum-r">{putts_per_hole:.2f}</div></div>
          <div class="scsum-row"><div class="scsum-l">GIR</div><div class="scsum-r">{gir_total}/{holes_played} ({gir_pct_num:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">Fairways</div><div class="scsum-r">{fw_total}/{fw_attempts} ({fw_pct_num:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">Scrambles</div><div class="scsum-r">{scrambles_display}</div></div>
          <div class="scsum-row"><div class="scsum-l">Up & Downs</div><div class="scsum-r">{updowns_display}</div></div>
          <div class="scsum-row"><div class="scsum-l">US Open Pars</div><div class="scsum-r">{us_open_made}/{us_open_attempts} ({us_open_pct:.1f}%)</div></div>
        </div>

        <div class="scsum-card">
          <div class="scsum-h">💡 Advanced Insights</div>
          <div class="scsum-row"><div class="scsum-l">1-Putts</div><div class="scsum-r">{total_1_putts}</div></div>
          <div class="scsum-row"><div class="scsum-l">3+ Putts</div><div class="scsum-r">{total_3_plus_putts}</div></div>
          <div class="scsum-row"><div class="scsum-l">3-Putt Bogeys</div><div class="scsum-r">{total_3_putt_bogeys}</div></div>
          <div class="scsum-row"><div class="scsum-l">Longest Putt Made</div><div class="scsum-r">{longest_putt_made:.1f} ft</div></div>
          <div class="scsum-row"><div class="scsum-l">Putts Made ≤ 5 ft</div><div class="scsum-r">{putts_made_5_or_less}</div></div>
          <div class="scsum-row"><div class="scsum-l">Putts Made 5–10 ft</div><div class="scsum-r">{putts_made_5_to_10}</div></div>
          <div class="scsum-row"><div class="scsum-l">Putts Made > 10 ft</div><div class="scsum-r">{putts_made_over_10}</div></div>
          <div class="scsum-row"><div class="scsum-l">Pro Pars+</div><div class="scsum-r">{pro_pars_total}</div></div>
        </div>

        <div class="scsum-card">
          <div class="scsum-h">🎯 Ball Striking + Creativity</div>
          <div class="scsum-row"><div class="scsum-l">GIR — Par 3</div><div class="scsum-r">{gir3_m}/{gir3_t} ({gir3_pct:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">GIR — Par 4</div><div class="scsum-r">{gir4_m}/{gir4_t} ({gir4_pct:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">GIR — Par 5</div><div class="scsum-r">{gir5_m}/{gir5_t} ({gir5_pct:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">FW — Par 4</div><div class="scsum-r">{fw4_m}/{fw4_t} ({fw4_pct:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">FW — Par 5</div><div class="scsum-r">{fw5_m}/{fw5_t} ({fw5_pct:.1f}%)</div></div>
          <div class="scsum-row"><div class="scsum-l">Arnies / Seves</div><div class="scsum-r">{arnies_total} / {seves_total}</div></div>
          <div class="scsum-row"><div class="scsum-l">Hole Outs / Lost Balls</div><div class="scsum-r">{hole_outs_total} / {lost_balls_display}</div></div>
        </div>
      </div>
    </div>
    """
    components.html(html, height=360, scrolling=False)


def render_overview_category_trends(round_cat_df, base_cat_df):
    if round_cat_df is None or round_cat_df.empty or base_cat_df is None or base_cat_df.empty:
        st.info("No category trend data available.")
        return

    merged = pd.merge(
        round_cat_df[["Category", "Count", "Percent"]].rename(columns={"Count": "Round Count", "Percent": "Round %"}),
        base_cat_df[["Category", "Count", "Percent"]].rename(columns={"Count": "Base Count", "Percent": "Base %"}),
        on="Category",
        how="outer"
    ).fillna(0)

    rows_html = ""
    for _, r in merged.iterrows():
        cat = str(r["Category"])
        round_pct = float(pd.to_numeric(r["Round %"], errors="coerce") or 0.0)
        base_pct = float(pd.to_numeric(r["Base %"], errors="coerce") or 0.0)
        delta = round_pct - base_pct

        if cat == "Par or Better":
            good = delta > 0.05
        else:
            good = delta < -0.05

        if abs(delta) <= 0.05:
            arrow = "➡️"
            cls = "oct-flat"
        else:
            arrow = "🔼" if good else "🔽"
            cls = "oct-good" if good else "oct-bad"

        rows_html += f"""
        <div class="oct-row">
          <div class="oct-l">{cat}</div>
          <div class="oct-mid">{round_pct:.1f}%</div>
          <div class="oct-r {cls}">{arrow} {delta:+.1f}%</div>
        </div>
        """

    html = f"""
    <style>
      .oct-wrap {{
        background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%);
        border:1px solid rgba(255,255,255,.08);
        border-radius:14px;
        padding:12px 14px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
      }}
      .oct-head {{
        font-size:13px;
        font-weight:800;
        color:#fff;
        margin-bottom:8px;
      }}
      .oct-sub {{
        font-size:11px;
        color:#aaa;
        margin-bottom:8px;
      }}
      .oct-row {{
        display:grid;
        grid-template-columns: 1fr 80px 92px;
        gap:10px;
        align-items:center;
        margin:7px 0;
      }}
      .oct-l {{ color:#e0e0e0; font-weight:700; }}
      .oct-mid {{ color:#fff; font-weight:800; text-align:right; font-variant-numeric:tabular-nums; }}
      .oct-r {{ text-align:right; font-weight:800; font-variant-numeric:tabular-nums; }}
      .oct-good {{ color:#64dfb5; }}
      .oct-bad {{ color:#ee6c4d; }}
      .oct-flat {{ color:#bdbdbd; }}
    </style>
    <div class="oct-wrap">
      <div class="oct-head">📊 Round Shape vs Baseline</div>
      <div class="oct-sub">How this round’s scoring categories compare to your selected baseline.</div>
      {rows_html}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def build_trends_round_frame(frame):
    d = frame.copy()
    round_col = _resolve_round_col(d)
    if round_col is None or round_col not in d.columns or d.empty:
        return pd.DataFrame(columns=["Round", "Date", "ScorePer18", "GIRPct", "FWPct", "PuttsPer18", "BirdieRate", "ScramblePct", "UpDownPct"])

    work = d.copy()
    work["Date Played"] = pd.to_datetime(_safe_col(work, "Date Played", pd.NaT), errors="coerce")
    work["Hole Score"] = pd.to_numeric(_safe_col(work, "Hole Score", 0), errors="coerce").fillna(0)
    work["Par"] = pd.to_numeric(_safe_col(work, "Par", 0), errors="coerce").fillna(0)
    work["Putts"] = pd.to_numeric(_safe_col(work, "Putts", 0), errors="coerce").fillna(0)
    work["GIR"] = pd.to_numeric(_safe_col(work, "GIR", 0), errors="coerce").fillna(0)
    work["Fairway"] = pd.to_numeric(_safe_col(work, "Fairway", 0), errors="coerce").fillna(0)
    work["Scramble"] = pd.to_numeric(_safe_col(work, "Scramble", 0), errors="coerce").fillna(0)
    work["Scramble Opportunity"] = pd.to_numeric(_safe_col(work, "Scramble Opportunity", 0), errors="coerce").fillna(0)
    work["BirdieFlag"] = pd.to_numeric(_safe_col(work, "Birdie", 0), errors="coerce").fillna(0)
    if "Score Label" in work.columns:
        work["BirdieFlag"] = ((work["Score Label"].astype(str) == "Birdie") | (work["Score Label"].astype(str) == "Eagle") | (work["Score Label"].astype(str) == "Albatross")).astype(int)

    def _agg(block):
        holes = len(block)
        fw_block = block[block["Par"].isin([4, 5])].copy()
        gir0 = (pd.to_numeric(block["GIR"], errors="coerce").fillna(0) == 0)
        putt1 = (pd.to_numeric(block["Putts"], errors="coerce").fillna(0) == 1)

        score_per18 = _standardized_score_per18(block)

        return pd.Series({
            "Date": pd.to_datetime(block["Date Played"], errors="coerce").max(),
            "ScorePer18": score_per18,
            "GIRPct": (block["GIR"].sum() / holes * 100.0) if holes else 0.0,
            "FWPct": (fw_block["Fairway"].sum() / len(fw_block) * 100.0) if len(fw_block) else 0.0,
            "PuttsPer18": (block["Putts"].sum() / holes * 18.0) if holes else 0.0,
            "BirdieRate": (block["BirdieFlag"].sum() / holes * 100.0) if holes else 0.0,
            "ScramblePct": (block["Scramble"].sum() / block["Scramble Opportunity"].sum() * 100.0) if block["Scramble Opportunity"].sum() else 0.0,
            "UpDownPct": ((gir0 & putt1).sum() / block["Scramble Opportunity"].sum() * 100.0) if block["Scramble Opportunity"].sum() else 0.0,
        })

    out = work.groupby(round_col, dropna=True).apply(_agg).reset_index().rename(columns={round_col: "Round"})
    out = out.sort_values(["Date", "Round"]).reset_index(drop=True)
    out["RoundNumber"] = range(1, len(out) + 1)
    for c in ["ScorePer18", "GIRPct", "FWPct", "PuttsPer18", "BirdieRate", "ScramblePct", "UpDownPct"]:
        out[f"{c}_Roll5"] = out[c].rolling(5, min_periods=1).mean()
    return out

def render_trend_line_chart(trend_df, value_col, roll_col, title, y_title, lower_better=False):
    if trend_df is None or trend_df.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    plot_df = trend_df.copy()
    plot_df["RoundNumber"] = pd.to_numeric(plot_df["RoundNumber"], errors="coerce")
    plot_df[value_col] = pd.to_numeric(plot_df[value_col], errors="coerce")
    plot_df[roll_col] = pd.to_numeric(plot_df[roll_col], errors="coerce")
    plot_df["DateLabel"] = pd.to_datetime(plot_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    plot_df = plot_df.dropna(subset=["RoundNumber", value_col, roll_col]).copy()

    if plot_df.empty:
        st.info(f"No data available for {title.lower()}.")
        return

    base_val = float(plot_df[value_col].mean()) if len(plot_df) else 0.0
    base_df = pd.DataFrame({"Baseline": [base_val]})

    line_actual = alt.Chart(plot_df).mark_line(point=True, strokeWidth=2.8).encode(
        x=alt.X("RoundNumber:Q", title="Round"),
        y=alt.Y(f"{value_col}:Q", title=y_title),
        tooltip=[
            alt.Tooltip("RoundNumber:Q", title="Round #"),
            alt.Tooltip("DateLabel:N", title="Date"),
            alt.Tooltip(f"{value_col}:Q", title=y_title, format=".1f"),
            alt.Tooltip(f"{roll_col}:Q", title="Rolling 5", format=".1f"),
        ],
        color=alt.value("#4f8cff")
    )

    line_roll = alt.Chart(plot_df).mark_line(strokeWidth=3.2, opacity=0.95).encode(
        x=alt.X("RoundNumber:Q"),
        y=alt.Y(f"{roll_col}:Q"),
        color=alt.value("#64dfb5")
    )

    base_rule = alt.Chart(base_df).mark_rule(strokeDash=[6, 4], opacity=0.8, color="#f2c14e").encode(
        y=alt.Y("Baseline:Q")
    )

    st.altair_chart(
        (base_rule + line_actual + line_roll).properties(height=280, title=title).configure_view(strokeOpacity=0),
        use_container_width=True
    )


def build_trends_strengths(trend_df):
    if trend_df is None or trend_df.empty:
        return ["No round trend data available yet."]

    last_row = trend_df.iloc[-1]
    metric_cols = [c for c in ["ScorePer18", "GIRPct", "FWPct", "PuttsPer18", "BirdieRate", "ScramblePct", "UpDownPct"] if c in trend_df.columns]
    if not metric_cols:
        return ["No round trend data available yet."]

    all_time = trend_df[metric_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)

    metrics = [
        ("Score / 18", "ScorePer18", True, ""),
        ("GIR %", "GIRPct", False, "%"),
        ("FW %", "FWPct", False, "%"),
        ("Putts / 18", "PuttsPer18", True, ""),
        ("Birdie Rate", "BirdieRate", False, "%"),
        ("Scramble %", "ScramblePct", False, "%"),
        ("Up & Down %", "UpDownPct", False, "%"),
    ]

    strengths = []
    for label, col, lower_better, unit in metrics:
        if col not in trend_df.columns or col not in all_time.index:
            continue
        cur = float(pd.to_numeric(last_row.get(col), errors="coerce"))
        base = float(pd.to_numeric(all_time.get(col), errors="coerce"))
        raw_delta = cur - base
        good_delta = (-raw_delta) if lower_better else raw_delta
        strengths.append((label, cur, base, raw_delta, good_delta, unit))

    strengths = sorted(strengths, key=lambda x: x[4], reverse=True)

    out = []
    for label, cur, base, raw_delta, good_delta, unit in strengths[:3]:
        if good_delta > 0.05:
            out.append(f"{label}: {cur:.1f}{unit} vs all-time {base:.1f}{unit} ({raw_delta:+.1f}{unit})")

    if not out:
        out.append("No clear strengths bubble above baseline yet.")
    return out

def build_trends_insights(trend_df):
    if trend_df is None or trend_df.empty or len(trend_df) < 2:
        return {
            "good": ["Need at least 2 rounds to generate trend insights."],
            "watch": ["Add a few more rounds to unlock trend comparisons."],
        }

    last5 = trend_df.tail(5)
    prior20 = trend_df.iloc[:-5].tail(20) if len(trend_df) > 5 else trend_df.iloc[:-1]

    metrics = [
        ("Score / 18", "ScorePer18", True),
        ("GIR %", "GIRPct", False),
        ("FW %", "FWPct", False),
        ("Putts / 18", "PuttsPer18", True),
        ("Birdie Rate", "BirdieRate", False),
        ("Scramble %", "ScramblePct", False),
        ("Up & Down %", "UpDownPct", False),
    ]

    scored = []
    for label, col, lower_better in metrics:
        if col not in trend_df.columns:
            continue
        recent = float(pd.to_numeric(last5[col], errors="coerce").fillna(0).mean()) if len(last5) else 0.0
        prior = float(pd.to_numeric(prior20[col], errors="coerce").fillna(0).mean()) if len(prior20) else recent
        raw_delta = recent - prior
        good_delta = (-raw_delta) if lower_better else raw_delta
        scored.append((label, recent, prior, raw_delta, good_delta, lower_better))

    if not scored:
        return {
            "good": ["No trend metrics available yet."],
            "watch": ["No trend metrics available yet."],
        }

    good_sorted = sorted(scored, key=lambda x: x[4], reverse=True)
    bad_sorted = sorted(scored, key=lambda x: x[4])

    good = []
    for label, recent, prior, raw_delta, good_delta, lower_better in good_sorted[:3]:
        if good_delta > 0.05:
            unit = "%" if "Rate" in label or "%" in label else ""
            good.append(f"{label}: {recent:.1f}{unit} vs {prior:.1f}{unit} ({raw_delta:+.1f}{unit})")

    watch = []
    for label, recent, prior, raw_delta, good_delta, lower_better in bad_sorted[:3]:
        if good_delta < -0.05:
            unit = "%" if "Rate" in label or "%" in label else ""
            watch.append(f"{label}: {recent:.1f}{unit} vs {prior:.1f}{unit} ({raw_delta:+.1f}{unit})")

    if not good:
        good.append("No major positive trend spikes yet versus the prior sample.")
    if not watch:
        watch.append("No major negative trends stood out versus the prior sample.")

    return {"good": good, "watch": watch}

def render_trends_summary_cards(trend_df):
    if trend_df is None or trend_df.empty:
        st.info("No round trend data available.")
        return

    import streamlit.components.v1 as components

    last_row = trend_df.iloc[-1]
    metric_cols = [c for c in ["ScorePer18", "GIRPct", "FWPct", "PuttsPer18", "BirdieRate"] if c in trend_df.columns]
    if not metric_cols:
        st.info("No round trend data available.")
        return
    all_time = trend_df[metric_cols].apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)

    cards = [
        ("Score / 18", "ScorePer18", True, "{:.1f}"),
        ("GIR %", "GIRPct", False, "{:.1f}%"),
        ("FW %", "FWPct", False, "{:.1f}%"),
        ("Putts / 18", "PuttsPer18", True, "{:.1f}"),
        ("Birdie Rate", "BirdieRate", False, "{:.1f}%"),
    ]

    html = """
    <style>
      .tr-wrap {display:grid; grid-template-columns:repeat(5, minmax(190px,1fr)); gap:12px; margin:8px 0 14px 0; font-family:Segoe UI, Roboto, Arial, sans-serif;}
      .tr-card {background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:14px 16px; box-shadow:0 8px 18px rgba(0,0,0,.16); min-height:132px;}
      .tr-k {font-size:13px; color:#c8c8c8; font-weight:800; margin-bottom:8px; line-height:1.28; min-height:36px; letter-spacing:.1px;}
      .tr-v {font-size:25px; color:#fff; font-weight:900; margin-bottom:6px; line-height:1.05;}
      .tr-b {font-size:12px; color:#bdbdbd;}
      .tr-d {font-size:12px; font-weight:800; margin-top:4px;}
      .tr-good {color:#64dfb5;}
      .tr-bad {color:#ee6c4d;}
      .tr-flat {color:#bdbdbd;}
    </style>
    <div class="tr-wrap">
    """
    for label, col, lower_better, fmt in cards:
        if col not in trend_df.columns or col not in all_time.index:
            continue
        cur = float(pd.to_numeric(last_row.get(col), errors="coerce"))
        base = float(pd.to_numeric(all_time.get(col), errors="coerce"))
        diff = cur - base
        good_delta = -diff if lower_better else diff
        cls = "tr-good" if good_delta > 0.05 else ("tr-bad" if good_delta < -0.05 else "tr-flat")
        arrow = "🔼" if good_delta > 0.05 else ("🔽" if good_delta < -0.05 else "➡️")
        html += f"""
        <div class="tr-card">
          <div class="tr-k">{label}</div>
          <div class="tr-v">{fmt.format(cur)}</div>
          <div class="tr-b">All-time avg: {fmt.format(base)}</div>
          <div class="tr-d {cls}">{arrow} {diff:+.1f}{'%' if '%' in fmt else ''}</div>
        </div>
        """
    html += "</div>"
    components.html(html, height=150, scrolling=False)




def build_round_leaderboard_frame(frame):
    d = frame.copy()
    round_col = _resolve_round_col(d)
    if round_col is None or round_col not in d.columns or d.empty:
        return pd.DataFrame(columns=[
            "Round", "Date", "Player", "Course", "Holes", "TotalScore", "PlusMinus", "ScoreToPar", "ScorePer18", "ScoreDisplay",
            "GIRPct", "GIRDisplay", "FWPct", "FWDisplay", "PuttsPer18", "PuttsDisplay",
            "BirdieRate", "BirdieDisplay", "ScramblePct", "ScrambleDisplay", "UpDownPct", "UpDownDisplay",
            "Arnies", "ArniesDisplay", "Seves", "SevesDisplay",
            "OnePutts", "OnePuttsDisplay", "ThreePuttPlus", "ThreePuttPlusDisplay", "ThreePuttBogeys", "ThreePuttBogeysDisplay",
            "ParOrBetterStreak", "BirdieBetterStreak",
            "ParCount", "BogeyCount", "DoublePlusCount",
            "ParDisplay", "BogeyDisplay", "DoublePlusDisplay",
            "LostBalls", "LostBallsDisplay"
        ])

    d["Date Played"] = pd.to_datetime(_safe_col(d, "Date Played", pd.NaT), errors="coerce")
    d["Hole Score"] = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
    d["Par"] = pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)
    d["Putts"] = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0)
    d["GIR"] = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0)
    d["Fairway"] = pd.to_numeric(_safe_col(d, "Fairway", 0), errors="coerce").fillna(0)
    d["Scramble"] = pd.to_numeric(_safe_col(d, "Scramble", 0), errors="coerce").fillna(0)
    d["Scramble Opportunity"] = pd.to_numeric(_safe_col(d, "Scramble Opportunity", 0), errors="coerce").fillna(0)
    d["Arnie"] = pd.to_numeric(_safe_col(d, "Arnie", 0), errors="coerce").fillna(0)
    d["Seve"] = pd.to_numeric(_safe_col(d, "Seve", 0), errors="coerce").fillna(0)
    d["LostBallTee"] = pd.to_numeric(_safe_col(d, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0)
    d["LostBallAppr"] = pd.to_numeric(_safe_col(d, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0)
    d["ThreePuttBogeyField"] = pd.to_numeric(_safe_col(d, "3 Putt Bogey", 0), errors="coerce").fillna(0)
    if "Hole" in d.columns:
        d["Hole"] = pd.to_numeric(_safe_col(d, "Hole", 0), errors="coerce").fillna(0)

    if "Score Label" in d.columns:
        d["ScoreLabelClean"] = d["Score Label"].fillna("").astype(str)
        d["BirdieFlag"] = d["ScoreLabelClean"].isin(["Birdie", "Eagle", "Albatross"]).astype(int)
    else:
        d["ScoreLabelClean"] = ""
        d["BirdieFlag"] = pd.to_numeric(_safe_col(d, "Birdie", 0), errors="coerce").fillna(0)

    def _longest_streak(bool_series):
        best = 0
        cur = 0
        for v in list(bool_series):
            if bool(v):
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return int(best)

    def _agg(block):
        if "Hole" in block.columns:
            block = block.sort_values("Hole").copy()
        holes = len(block)
        fw_block = block[block["Par"].isin([4, 5])].copy()
        gir0 = (pd.to_numeric(block["GIR"], errors="coerce").fillna(0) == 0)
        putt1 = (pd.to_numeric(block["Putts"], errors="coerce").fillna(0) == 1)
        putts_clean = pd.to_numeric(block["Putts"], errors="coerce").fillna(0)

        par3_scores = pd.to_numeric(block.loc[block["Par"] == 3, "Hole Score"], errors="coerce").dropna()
        par4_scores = pd.to_numeric(block.loc[block["Par"] == 4, "Hole Score"], errors="coerce").dropna()
        par5_scores = pd.to_numeric(block.loc[block["Par"] == 5, "Hole Score"], errors="coerce").dropna()
        avg_par3 = float(par3_scores.mean()) if len(par3_scores) else 0.0
        avg_par4 = float(par4_scores.mean()) if len(par4_scores) else 0.0
        avg_par5 = float(par5_scores.mean()) if len(par5_scores) else 0.0

        total_score = float(pd.to_numeric(block["Hole Score"], errors="coerce").fillna(0).sum())
        total_par = float(pd.to_numeric(block["Par"], errors="coerce").fillna(0).sum())
        score_to_par = total_score - total_par
        plus_minus = f"{score_to_par:+.0f}" if abs(score_to_par) >= 1e-9 else "E"

        score_to_par_series = pd.to_numeric(block["Hole Score"], errors="coerce").fillna(0) - pd.to_numeric(block["Par"], errors="coerce").fillna(0)
        par_or_better = (score_to_par_series <= 0)
        birdie_or_better = (score_to_par_series <= -1)

        gir_made = int(pd.to_numeric(block["GIR"], errors="coerce").fillna(0).sum())
        fw_made = int(pd.to_numeric(fw_block["Fairway"], errors="coerce").fillna(0).sum()) if len(fw_block) else 0
        putts_total = float(putts_clean.sum())
        birdies_total = int(pd.to_numeric(block["BirdieFlag"], errors="coerce").fillna(0).sum())
        scramble_made = int(pd.to_numeric(block["Scramble"], errors="coerce").fillna(0).sum())
        scramble_ops = int(pd.to_numeric(block["Scramble Opportunity"], errors="coerce").fillna(0).sum())
        updown_made = int((gir0 & putt1).sum())
        arnies = int(pd.to_numeric(block["Arnie"], errors="coerce").fillna(0).sum())
        seves = int(pd.to_numeric(block["Seve"], errors="coerce").fillna(0).sum())
        lost_balls = int(pd.to_numeric(block["LostBallTee"], errors="coerce").fillna(0).sum() + pd.to_numeric(block["LostBallAppr"], errors="coerce").fillna(0).sum())
        one_putts = int((putts_clean == 1).sum())
        three_putt_plus = int((putts_clean >= 3).sum())
        three_putt_bogeys = int(pd.to_numeric(block["ThreePuttBogeyField"], errors="coerce").fillna(0).sum())

        score_label_clean = block["ScoreLabelClean"].astype(str)
        par_count = int(score_label_clean.isin(["Par"]).sum())
        bogey_count = int(score_label_clean.isin(["Bogey"]).sum())
        double_plus_count = int(score_label_clean.isin(["Double Bogey", "Triple Bogey +"]).sum())

        gir_pct = (gir_made / holes * 100.0) if holes else 0.0
        fw_pct = (fw_made / len(fw_block) * 100.0) if len(fw_block) else 0.0
        putts_per18 = (putts_total / holes * 18.0) if holes else 0.0
        birdie_rate = (birdies_total / holes * 100.0) if holes else 0.0
        scramble_pct = (scramble_made / scramble_ops * 100.0) if scramble_ops else 0.0
        updown_pct = (updown_made / scramble_ops * 100.0) if scramble_ops else 0.0

        return pd.Series({
            "Date": pd.to_datetime(block["Date Played"], errors="coerce").max(),
            "Player": block["Player Name"].iloc[0] if "Player Name" in block.columns and len(block) else "",
            "Course": block["Course Name"].iloc[0] if "Course Name" in block.columns and len(block) else "",
            "Holes": holes,
            "TotalScore": total_score,
            "PlusMinus": plus_minus,
            "ScoreToPar": score_to_par,
            "ScorePer18": (4 * avg_par3) + (10 * avg_par4) + (4 * avg_par5),
            "ScoreDisplay": f"{int(round(total_score))} ({plus_minus})",
            "GIRPct": gir_pct,
            "GIRDisplay": f"{gir_made}/{holes} ({gir_pct:.1f}%)",
            "FWPct": fw_pct,
            "FWDisplay": f"{fw_made}/{len(fw_block)} ({fw_pct:.1f}%)" if len(fw_block) else "0/0 (-)",
            "PuttsPer18": putts_per18,
            "PuttsDisplay": f"{int(round(putts_total))}/{holes} ({putts_per18:.1f}/18)",
            "BirdieRate": birdie_rate,
            "BirdieDisplay": f"{birdies_total}/{holes} ({birdie_rate:.1f}%)",
            "ScramblePct": scramble_pct,
            "ScrambleDisplay": f"{scramble_made}/{scramble_ops} ({scramble_pct:.1f}%)" if scramble_ops else "0/0 (-)",
            "UpDownPct": updown_pct,
            "UpDownDisplay": f"{updown_made}/{scramble_ops} ({updown_pct:.1f}%)" if scramble_ops else "0/0 (-)",
            "Arnies": arnies,
            "ArniesDisplay": f"{arnies}/{holes} ({(arnies / holes * 18.0):.1f}/18)" if holes else "0/0",
            "Seves": seves,
            "SevesDisplay": f"{seves}/{holes} ({(seves / holes * 18.0):.1f}/18)" if holes else "0/0",
            "OnePutts": one_putts,
            "OnePuttsDisplay": f"{one_putts}/{holes} ({(one_putts / holes * 100.0):.1f}%)",
            "ThreePuttPlus": three_putt_plus,
            "ThreePuttPlusDisplay": f"{three_putt_plus}/{holes} ({(three_putt_plus / holes * 100.0):.1f}%)",
            "ThreePuttBogeys": three_putt_bogeys,
            "ThreePuttBogeysDisplay": f"{three_putt_bogeys}/{holes} ({(three_putt_bogeys / holes * 100.0):.1f}%)",
            "ParOrBetterStreak": _longest_streak(par_or_better),
            "BirdieBetterStreak": _longest_streak(birdie_or_better),
            "ParCount": par_count,
            "BogeyCount": bogey_count,
            "DoublePlusCount": double_plus_count,
            "ParDisplay": f"{par_count}/{holes} ({(par_count / holes * 100.0):.1f}%)",
            "BogeyDisplay": f"{bogey_count}/{holes} ({(bogey_count / holes * 100.0):.1f}%)",
            "DoublePlusDisplay": f"{double_plus_count}/{holes} ({(double_plus_count / holes * 100.0):.1f}%)",
            "LostBalls": lost_balls,
            "LostBallsDisplay": f"{lost_balls}/{holes} ({(lost_balls / holes * 18.0):.1f}/18)" if holes else "0/0",
        })

    out = d.groupby(round_col, dropna=True).apply(_agg).reset_index().rename(columns={round_col: "Round"})
    out = out[out["Holes"] == 18].copy()
    return out.sort_values(["Date", "Round"]).reset_index(drop=True)

def get_top_rounds(df_rounds, metric_col, n=5, lower_better=False, display_col=None):
    if df_rounds is None or df_rounds.empty or metric_col not in df_rounds.columns:
        return pd.DataFrame(columns=["Date", "Player", "Course", "Value", "Score"])
    out = df_rounds.copy()
    out[metric_col] = pd.to_numeric(out[metric_col], errors="coerce")
    out = out.dropna(subset=[metric_col]).copy()
    out = out.sort_values(metric_col, ascending=lower_better).head(n).copy()

    if display_col and display_col in out.columns:
        out["Value"] = out[display_col].astype(str)
    else:
        if metric_col in ["ScoreToPar", "ParOrBetterStreak", "BirdieBetterStreak", "ParCount", "BogeyCount", "DoublePlusCount", "LostBalls", "Arnies", "Seves"]:
            out["Value"] = pd.to_numeric(out[metric_col], errors="coerce").fillna(0).astype(int).astype(str)
        else:
            out["Value"] = pd.to_numeric(out[metric_col], errors="coerce").map(lambda x: f"{x:.1f}" if pd.notna(x) else "—")

    out["Score"] = out["ScoreDisplay"].astype(str) if "ScoreDisplay" in out.columns else out.get("ScoreToParDisplay","")
    return out[["Date", "Player", "Course", "Value", "Score"]].reset_index(drop=True)


def _style_bestof_table(df_table, title=""):
    out = df_table.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce").dt.strftime("%Y-%m-%d")

    def _clean_text(v):
        s = str(v)
        s = s.replace(".0)", ")")
        s = s.replace("(+0)", "(E)")
        s = s.replace("(-0)", "(E)")
        s = s.replace(" 0.0%", " 0%")
        s = s.replace(" 100.0%", " 100%")
        return s

    if "Value" in out.columns:
        out["Value"] = out["Value"].map(_clean_text)
    if "Score" in out.columns:
        out["Score"] = out["Score"].map(_clean_text)

    return out


def build_overview_round_summary(metric_df):


    if metric_df is None or metric_df.empty:
        return {"good": [], "lock": []}

    # Exclude raw total score from "what went well / lock in" since score-per-18 is the cleaner comparison metric
    work_df = metric_df[metric_df["Metric"] != "Total Score"].copy()

    good_df = work_df.sort_values(["GoodDelta", "Metric"], ascending=[False, True]).copy()
    lock_df = work_df.sort_values(["GoodDelta", "Metric"], ascending=[True, True]).copy()

    good = []
    for _, r in good_df.head(3).iterrows():
        if r["GoodDelta"] > 0.05:
            good.append(f'{r["Metric"]}: {r["Round Display"]} ({r["Delta Display"]})')

    lock = []
    for _, r in lock_df.head(3).iterrows():
        if r["GoodDelta"] < -0.05:
            lock.append(f'{r["Metric"]}: {r["Round Display"]} ({r["Delta Display"]})')

    if not good:
        good.append("No major spike areas stood out versus this comparison set.")
    if not lock:
        lock.append("No major leak areas stood out versus this comparison set.")

    return {"good": good, "lock": lock}

def build_round_diagnosis(metric_df):
    if metric_df is None or metric_df.empty:
        return pd.DataFrame(columns=["Metric", "Area", "Stroke Impact", "Round", "Baseline", "Delta"])

    impact_map = {
        "Putts / 18": ("Putting", 1.00),
        "GIR %": ("Approach Play", 0.12),
        "Fairway %": ("Tee Ball", 0.05),
        "Scramble %": ("Short Game", 0.04),
        "Up & Down %": ("Short Game", 0.03),
        "Avg First-Putt Proximity": ("Approach Proximity", 0.03),
        "Lost Balls / 18": ("Penalty Avoidance", 1.00),
        "Arnies / 18": ("Recovery / Creativity", 0.40),
        "Seves / 18": ("Recovery / Creativity", 0.50),
    }

    rows = []
    for _, r in metric_df.iterrows():
        metric = str(r.get("Metric", ""))
        if metric not in impact_map:
            continue
        area, weight = impact_map[metric]
        good_delta = float(pd.to_numeric(r.get("GoodDelta", 0), errors="coerce") or 0.0)
        impact = good_delta * weight
        rows.append({
            "Metric": metric,
            "Area": area,
            "Stroke Impact": impact,
            "Round": r.get("Round Display", "—"),
            "Baseline": r.get("Baseline Display", "—"),
            "Delta": r.get("DeltaDisplay", r.get("Delta Display", "—")),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.sort_values(["Stroke Impact", "Metric"], ascending=[False, True]).reset_index(drop=True)
    return out


def build_round_impact_summary(metric_df):
    diag = build_round_diagnosis(metric_df)
    if diag.empty:
        return {"total": 0.0, "good": pd.DataFrame(), "bad": pd.DataFrame(), "areas": pd.DataFrame()}

    area_df = diag.groupby("Area", as_index=False)["Stroke Impact"].sum().sort_values("Stroke Impact", ascending=False)
    return {
        "total": float(diag["Stroke Impact"].sum()),
        "good": diag[diag["Stroke Impact"] > 0.05].head(3).copy(),
        "bad": diag[diag["Stroke Impact"] < -0.05].sort_values("Stroke Impact", ascending=True).head(3).copy(),
        "areas": area_df,
    }


def render_round_impact_cards(metric_df, compare_label):
    import streamlit.components.v1 as components

    summary = build_round_impact_summary(metric_df)
    good = summary["good"]
    bad = summary["bad"]
    total = summary["total"]

    def _rows(block, positive=True):
        if block is None or block.empty:
            return "<div class='rid-row'><div class='rid-m'>No major drivers stood out.</div><div class='rid-v'>—</div></div>"
        html = ""
        for _, r in block.iterrows():
            sign = "+" if float(r["Stroke Impact"]) > 0 else ""
            html += f"<div class='rid-row'><div class='rid-m'>{r['Area']}<div class='rid-sub'>{r['Metric']}</div></div><div class='rid-v'>{sign}{float(r['Stroke Impact']):.2f}</div></div>"
        return html

    total_cls = "rid-good" if total > 0.05 else ("rid-bad" if total < -0.05 else "rid-flat")
    total_label = f"{total:+.2f} est." if abs(total) > 0.01 else "Even"

    html = f"""
    <style>
      .rid-grid {{display:grid; grid-template-columns: 1.1fr 1fr 1fr; gap:12px; margin:6px 0 12px 0; font-family:Segoe UI, Roboto, Arial, sans-serif;}}
      .rid-card {{background:linear-gradient(180deg,#2a2a2a 0%, #202020 100%); border:1px solid rgba(255,255,255,.08); border-radius:14px; padding:12px 14px; box-shadow:0 8px 18px rgba(0,0,0,.16);}}
      .rid-h {{font-size:13px; font-weight:800; color:#fff; margin-bottom:8px;}}
      .rid-big {{font-size:28px; font-weight:900; color:#fff; margin-bottom:4px;}}
      .rid-subtxt {{font-size:12px; color:#bdbdbd; line-height:1.45;}}
      .rid-row {{display:flex; justify-content:space-between; gap:10px; align-items:flex-start; padding:6px 0; border-top:1px solid rgba(255,255,255,.06);}}
      .rid-row:first-child {{border-top:none; padding-top:2px;}}
      .rid-m {{color:#fff; font-size:12px; font-weight:800;}}
      .rid-sub {{color:#bdbdbd; font-size:11px; font-weight:600; margin-top:2px;}}
      .rid-v {{color:#fff; font-size:12px; font-weight:900; white-space:nowrap; font-variant-numeric:tabular-nums;}}
      .rid-good {{color:#64dfb5;}}
      .rid-bad {{color:#ee6c4d;}}
      .rid-flat {{color:#bdbdbd;}}
    </style>
    <div class="rid-grid">
      <div class="rid-card">
        <div class="rid-h">🧮 Estimated Round Impact vs {compare_label}</div>
        <div class="rid-big {total_cls}">{total_label}</div>
        <div class="rid-subtxt">Positive values suggest you saved strokes versus the selected baseline. Negative values suggest where strokes leaked out.</div>
      </div>
      <div class="rid-card">
        <div class="rid-h">🔺 Biggest Gains</div>
        {_rows(good, True)}
      </div>
      <div class="rid-card">
        <div class="rid-h">🔻 Biggest Leaks</div>
        {_rows(bad, False)}
      </div>
    </div>
    """
    components.html(html, height=220, scrolling=False)


def build_why_round_happened(round_data, metric_df, compare_label):
    summary = build_round_impact_summary(metric_df)
    good = []
    watch = []

    if not summary["good"].empty:
        top = summary["good"].iloc[0]
        good.append(f"Strongest edge vs {compare_label}: {top['Area']} ({float(top['Stroke Impact']):+.2f} est.).")
    if len(summary["good"]) > 1:
        top2 = summary["good"].iloc[1]
        good.append(f"Secondary lift: {top2['Area']} ({float(top2['Stroke Impact']):+.2f} est.).")

    if not summary["bad"].empty:
        leak = summary["bad"].iloc[0]
        watch.append(f"Largest leak vs {compare_label}: {leak['Area']} ({float(leak['Stroke Impact']):+.2f} est.).")
    if len(summary["bad"]) > 1:
        leak2 = summary["bad"].iloc[1]
        watch.append(f"Next biggest issue: {leak2['Area']} ({float(leak2['Stroke Impact']):+.2f} est.).")

    score_to_par = pd.to_numeric(_safe_col(round_data, "Score to Par", pd.NA), errors="coerce")
    if score_to_par.isna().all():
        score_to_par = pd.to_numeric(_safe_col(round_data, "Hole Score", 0), errors="coerce") - pd.to_numeric(_safe_col(round_data, "Par", 0), errors="coerce")
    doubles = int((score_to_par >= 2).sum())
    if doubles:
        watch.append(f"Blow-up holes mattered: {doubles} hole(s) at double bogey or worse.")

    three_putt_bogeys = int(pd.to_numeric(_safe_col(round_data, "3 Putt Bogey", 0), errors="coerce").fillna(0).sum())
    if three_putt_bogeys:
        watch.append(f"Three-putt bogeys cost momentum: {three_putt_bogeys} this round.")

    lost_balls = int(pd.to_numeric(_safe_col(round_data, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0).sum() + pd.to_numeric(_safe_col(round_data, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0).sum())
    if lost_balls:
        watch.append(f"Penalty pressure showed up: {lost_balls} lost ball(s).")

    if not good:
        good.append(f"No single category clearly outperformed the {compare_label.lower()} sample.")
    if not watch:
        watch.append(f"No single category clearly underperformed the {compare_label.lower()} sample.")

    return {"good": good[:3], "watch": watch[:4]}


def build_trend_vs_baseline_df(trend_source, baseline_df):
    round_col = _resolve_round_col(trend_source)
    if round_col is None or round_col not in trend_source.columns or trend_source.empty or baseline_df is None or baseline_df.empty:
        return pd.DataFrame(columns=["Round","Date","ScoreDelta","GIRDelta","FWDelta","PuttsDelta","OverallEdge"])

    base_overview = build_overview_snapshot(baseline_df)
    rows = []
    for rid, block in trend_source.groupby(round_col, dropna=True):
        metric_df = build_overview_metric_rows(build_overview_snapshot(block), base_overview, "Baseline")
        impact_summary = build_round_impact_summary(metric_df)
        def _gd(metric):
            sub = metric_df[metric_df["Metric"] == metric]
            return float(pd.to_numeric(sub["GoodDelta"].iloc[0], errors="coerce")) if not sub.empty else 0.0
        rows.append({
            "Round": rid,
            "Date": pd.to_datetime(_safe_col(block, "Date Played", pd.NaT), errors="coerce").max(),
            "ScoreDelta": _gd("Avg Score / 18"),
            "GIRDelta": _gd("GIR %"),
            "FWDelta": _gd("Fairway %"),
            "PuttsDelta": _gd("Putts / 18"),
            "OverallEdge": float(impact_summary["total"]),
        })
    out = pd.DataFrame(rows).sort_values(["Date", "Round"]).reset_index(drop=True)
    if out.empty:
        return out
    out["RoundNumber"] = range(1, len(out) + 1)
    for c in ["ScoreDelta", "GIRDelta", "FWDelta", "PuttsDelta", "OverallEdge"]:
        out[f"{c}_Roll5"] = pd.to_numeric(out[c], errors="coerce").rolling(5, min_periods=1).mean()
    return out


def render_trend_vs_baseline_chart(trend_df, value_col, roll_col, title, y_title):
    if trend_df is None or trend_df.empty or value_col not in trend_df.columns:
        st.info(f"No data available for {title.lower()}.")
        return
    plot_df = trend_df.copy()
    plot_df["DateLabel"] = pd.to_datetime(plot_df["Date"], errors="coerce").dt.strftime("%Y-%m-%d")
    base_rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(strokeDash=[6,4], color="#f2c14e", opacity=0.8).encode(y="y:Q")
    actual = alt.Chart(plot_df).mark_line(point=True, strokeWidth=2.8).encode(
        x=alt.X("RoundNumber:Q", title="Round"),
        y=alt.Y(f"{value_col}:Q", title=y_title),
        tooltip=[alt.Tooltip("RoundNumber:Q", title="Round #"), alt.Tooltip("DateLabel:N", title="Date"), alt.Tooltip(f"{value_col}:Q", title=y_title, format=".2f"), alt.Tooltip(f"{roll_col}:Q", title="Rolling 5", format=".2f")],
        color=alt.value("#4f8cff")
    )
    roll = alt.Chart(plot_df).mark_line(strokeWidth=3.2, opacity=0.95).encode(
        x=alt.X("RoundNumber:Q"), y=alt.Y(f"{roll_col}:Q"), color=alt.value("#64dfb5")
    )
    st.altair_chart((base_rule + actual + roll).properties(height=280, title=title).configure_view(strokeOpacity=0), use_container_width=True)





# =========================================================
# Estimated Handicap Helpers
# =========================================================
HANDICAP_DEFAULT_RATING_18 = 72.0
HANDICAP_DEFAULT_SLOPE = 113.0

def _first_existing_col(frame, candidates):
    for col in candidates:
        if col in frame.columns:
            return col
    return None

def _score_differential_count_to_use(n):
    """WHS-style count of differentials to average.
    This is an estimate for dashboard use, not an official handicap service.
    """
    n = int(n or 0)
    if n <= 0:
        return 0
    if n <= 3:
        return 1
    if n == 4:
        return 1
    if n == 5:
        return 1
    if n == 6:
        return 2
    if n in [7, 8]:
        return 2
    if n in [9, 10, 11]:
        return 3
    if n in [12, 13, 14]:
        return 4
    if n in [15, 16]:
        return 5
    if n in [17, 18]:
        return 6
    if n == 19:
        return 7
    return 8

def _score_differential_adjustment(n):
    """Small low-sample adjustment approximating WHS early-record handling."""
    n = int(n or 0)
    if n <= 3:
        return -2.0
    if n == 4:
        return -1.0
    if n == 6:
        return -1.0
    return 0.0

def _round_differentials_for_frame(frame):
    """Build normalized 18-hole score differentials by round.

    If Course Rating / Slope Rating columns exist, they are used.
    If not, the fallback assumes a standard course: 72.0 rating and 113 slope.
    Partial rounds are scaled to 18 holes so 9-hole and other partial samples can contribute.
    """
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["Round Link", "Date Played", "ScoreDiff", "Holes", "Score18", "Rating18", "Slope", "Source"])

    d = frame.copy()
    d["Date Played"] = pd.to_datetime(_safe_col(d, "Date Played", pd.NaT), errors="coerce")
    d["Hole Score"] = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
    d["Par"] = pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)

    round_col = _resolve_round_col(d)
    if not round_col or round_col not in d.columns:
        d["__RoundKey"] = "Single Sample"
        round_col = "__RoundKey"

    rating_col = _first_existing_col(d, [
        "Course Rating", "CourseRating", "Course Rating (18)", "Rating",
        "course_rating", "course rating"
    ])
    slope_col = _first_existing_col(d, [
        "Slope Rating", "SlopeRating", "Course Slope", "Slope",
        "slope_rating", "slope rating"
    ])

    rows = []
    for rid, block in d.groupby(round_col, dropna=True):
        holes = int(len(block))
        if holes <= 0:
            continue
        score = float(pd.to_numeric(block["Hole Score"], errors="coerce").fillna(0).sum())
        par_total = float(pd.to_numeric(block["Par"], errors="coerce").fillna(0).sum())
        date_played = pd.to_datetime(block["Date Played"], errors="coerce").max()

        # Normalize played score to a standard par-72 18-hole equivalent.
        # This avoids skew from 9-hole rounds, par-3 courses, and unusual par mixes.
        score18 = _standardized_score_per18(block)
        par18 = 72.0

        rating18 = None
        source = "default rating/slope"
        if rating_col:
            rating_vals = pd.to_numeric(block[rating_col], errors="coerce").dropna()
            if len(rating_vals):
                rating_raw = float(rating_vals.iloc[0])
                # If a 9-hole/partial rating appears to be supplied, scale it to 18.
                rating18 = rating_raw * (18.0 / holes) if holes < 18 and rating_raw < 55 else rating_raw
                source = "course rating/slope"

        if rating18 is None or rating18 <= 0:
            # Neutral fallback: standard 72 rating, adjusted only if the sample is clearly not a par-72 equivalent.
            rating18 = HANDICAP_DEFAULT_RATING_18 if abs(par18 - 72.0) <= 4.0 else par18

        slope = HANDICAP_DEFAULT_SLOPE
        if slope_col:
            slope_vals = pd.to_numeric(block[slope_col], errors="coerce").dropna()
            if len(slope_vals) and float(slope_vals.iloc[0]) > 0:
                slope = float(slope_vals.iloc[0])

        score_diff = (score18 - rating18) * 113.0 / slope
        rows.append({
            "Round Link": rid,
            "Date Played": date_played,
            "ScoreDiff": round(float(score_diff), 1),
            "Holes": holes,
            "Score18": round(float(score18), 1),
            "Rating18": round(float(rating18), 1),
            "Slope": round(float(slope), 0),
            "Source": source,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("Date Played", ascending=False).reset_index(drop=True)

def estimate_player_handicap(frame):
    diffs = _round_differentials_for_frame(frame)
    if diffs.empty:
        return {"handicap": pd.NA, "rounds": 0, "holes": 0, "used": 0, "source": "No rounds", "differentials": diffs}

    # Use the most recent 20 round-level differentials, similar to WHS.
    recent = diffs.sort_values("Date Played", ascending=False).head(20).copy()
    n = int(len(recent))
    k = _score_differential_count_to_use(n)
    if k <= 0:
        return {"handicap": pd.NA, "rounds": n, "holes": int(recent["Holes"].sum()), "used": 0, "source": "No usable differentials", "differentials": recent}

    used = recent.sort_values("ScoreDiff", ascending=True).head(k).copy()
    estimate = float(used["ScoreDiff"].mean()) + _score_differential_adjustment(n)
    estimate = max(0.0, estimate)

    return {
        "handicap": round(estimate, 1),
        "rounds": n,
        "holes": int(recent["Holes"].sum()),
        "used": k,
        "source": "estimated from recent score differentials",
        "differentials": recent,
    }

def _format_handicap_value(x):
    if pd.isna(x):
        return "—"
    return f"{float(x):.1f}"

def render_handicap_estimate_box(frame, title="Estimated Handicap"):
    est = estimate_player_handicap(frame)
    handicap = est.get("handicap", pd.NA)
    rounds = int(est.get("rounds", 0) or 0)
    holes = int(est.get("holes", 0) or 0)
    used = int(est.get("used", 0) or 0)
    sample_note = "Good sample" if holes >= 360 else ("Limited sample" if holes >= 54 else "Very limited sample")

    st.markdown(
        f"""
        <div style="background:linear-gradient(180deg,#1f2937 0%,#171717 100%); border:1px solid rgba(255,255,255,.11); border-radius:16px; padding:14px 16px; margin:8px 0 14px 0; box-shadow:0 8px 18px rgba(0,0,0,.18);">
          <div style="font-size:12px; color:#bdbdbd; font-weight:800; text-transform:uppercase; letter-spacing:.05em;">{title}</div>
          <div style="display:flex; align-items:flex-end; gap:14px; margin-top:4px;">
            <div style="font-size:38px; color:#fff; font-weight:950; line-height:1;">{_format_handicap_value(handicap)}</div>
            <div style="font-size:13px; color:#d7d7d7; padding-bottom:4px;">{sample_note}<br>{rounds} recent round(s) • {holes} hole(s) • best {used} differential(s)</div>
          </div>
          <div style="font-size:12px; color:#a9a9a9; margin-top:8px; line-height:1.35;">Estimate only. Uses score differential logic, normalizes partial rounds to 18 holes, and assumes 72.0 / 113 when rating or slope is missing.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    return est

# =========================================================
# Compare Tab Helpers
# =========================================================
COMPARE_METRIC_DEFS = [
    ("Avg Score / 18", "score_per18", "lower", "{:.1f}"),
    ("Score to Par / 18", "to_par_per18", "lower", "{:+.1f}"),
    ("Putts / 18", "putts_per18", "lower", "{:.1f}"),
    ("GIR %", "gir_pct", "higher", "{:.1f}%"),
    ("Fairway %", "fw_pct", "higher", "{:.1f}%"),
    ("Scramble %", "scramble_pct", "higher", "{:.1f}%"),
    ("Up & Down %", "updown_pct", "higher", "{:.1f}%"),
    ("1-Putt %", "one_putt_pct", "higher", "{:.1f}%"),
    ("3+ Putt %", "three_plus_putt_pct", "lower", "{:.1f}%"),
    ("Birdie Conversion % (GIR)", "birdie_conv_gir_pct", "higher", "{:.1f}%"),
    ("Avg First-Putt Proximity", "avg_first_putt_prox", "lower", "{:.1f} ft"),
    ("Avg Prox if GIR", "avg_prox_gir", "lower", "{:.1f} ft"),
    ("Avg Prox if No GIR", "avg_prox_no_gir", "lower", "{:.1f} ft"),
    ("Arnies / 18", "arnies_per18", "higher", "{:.2f}"),
    ("Seves / 18", "seves_per18", "higher", "{:.2f}"),
    ("Lost Balls / 18", "lost_balls_per18", "lower", "{:.2f}"),
    ("Pro Pars+ / 18", "pro_pars_plus_per18", "higher", "{:.2f}"),
    ("Hole Outs / 18", "hole_outs_per18", "higher", "{:.2f}"),
    ("Estimated Handicap", "estimated_handicap", "lower", "{:.1f}"),
]

COMPARE_METRIC_LABELS = [m[0] for m in COMPARE_METRIC_DEFS]
COMPARE_METRIC_BY_LABEL = {m[0]: m for m in COMPARE_METRIC_DEFS}


def _compare_score_to_par_series(d):
    if "Score to Par" in d.columns:
        return pd.to_numeric(_safe_col(d, "Score to Par", 0), errors="coerce").fillna(0)
    return (
        pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
        - pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)
    )


def _compare_one_player_summary(player_name, frame):
    d = frame.copy()
    holes = int(len(d))
    round_col = _resolve_round_col(d)
    rounds = int(d[round_col].dropna().nunique()) if round_col and round_col in d.columns else (1 if holes else 0)

    score = pd.to_numeric(_safe_col(d, "Hole Score", 0), errors="coerce").fillna(0)
    par = pd.to_numeric(_safe_col(d, "Par", 0), errors="coerce").fillna(0)
    putts = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0)
    gir = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0)

    fw_block = d[par.isin([4, 5])].copy() if holes else d.iloc[0:0].copy()
    fw = pd.to_numeric(_safe_col(fw_block, "Fairway", 0), errors="coerce").fillna(0) if not fw_block.empty else pd.Series(dtype=float)

    scramble_made = pd.to_numeric(_safe_col(d, "Scramble", 0), errors="coerce").fillna(0).sum()
    scramble_ops = pd.to_numeric(_safe_col(d, "Scramble Opportunity", 0), errors="coerce").fillna(0).sum()

    one_putts = int((putts == 1).sum())
    three_plus_putts = int((putts >= 3).sum())

    score_to_par = _compare_score_to_par_series(d)
    birdie_or_better_on_gir = int(((gir == 1) & (score_to_par < 0)).sum())
    gir_made = int(gir.sum())

    first_putt_prox = pd.to_numeric(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", pd.NA), errors="coerce").replace(0, pd.NA)
    avg_first_putt_prox = float(first_putt_prox.mean()) if first_putt_prox.notna().any() else 0.0

    prox_gir = first_putt_prox[gir == 1]
    prox_no_gir = first_putt_prox[gir == 0]

    arnies = pd.to_numeric(_safe_col(d, "Arnie", 0), errors="coerce").fillna(0).sum()
    seves = pd.to_numeric(_safe_col(d, "Seve", 0), errors="coerce").fillna(0).sum()
    lost_balls = (
        pd.to_numeric(_safe_col(d, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(_safe_col(d, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0).sum()
    )

    pro_pars_plus = (
        pd.to_numeric(_safe_col(d, "Pro Par", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(_safe_col(d, "Pro Birdie", 0), errors="coerce").fillna(0).sum()
        + pd.to_numeric(_safe_col(d, "Pro Eagle+", 0), errors="coerce").fillna(0).sum()
    )
    hole_outs = build_short_game_hole_out_stats(d)["count"] if holes else 0
    handicap_estimate = estimate_player_handicap(d)["handicap"] if holes else pd.NA

    return {
        "Player": str(player_name),
        "Rounds": rounds,
        "Holes": holes,
        "score_per18": _standardized_score_per18(d),
        "to_par_per18": _standardized_to_par_per18(d),
        "putts_per18": _per18(float(putts.sum()), holes),
        "gir_pct": (float(gir.sum()) / holes * 100.0) if holes else 0.0,
        "fw_pct": (float(fw.sum()) / len(fw) * 100.0) if len(fw) else 0.0,
        "scramble_pct": (float(scramble_made) / float(scramble_ops) * 100.0) if scramble_ops else 0.0,
        "updown_pct": (float(((gir == 0) & (putts == 1)).sum()) / float(scramble_ops) * 100.0) if scramble_ops else 0.0,
        "one_putt_pct": (one_putts / holes * 100.0) if holes else 0.0,
        "three_plus_putt_pct": (three_plus_putts / holes * 100.0) if holes else 0.0,
        "birdie_conv_gir_pct": (birdie_or_better_on_gir / gir_made * 100.0) if gir_made else 0.0,
        "avg_first_putt_prox": avg_first_putt_prox,
        "avg_prox_gir": float(prox_gir.mean()) if prox_gir.notna().any() else 0.0,
        "avg_prox_no_gir": float(prox_no_gir.mean()) if prox_no_gir.notna().any() else 0.0,
        "arnies_per18": _per18(float(arnies), holes),
        "seves_per18": _per18(float(seves), holes),
        "lost_balls_per18": _per18(float(lost_balls), holes),
        "pro_pars_plus_per18": _per18(float(pro_pars_plus), holes),
        "hole_outs_per18": _per18(float(hole_outs), holes),
        "estimated_handicap": handicap_estimate,
    }


def build_compare_player_summary(compare_df):
    if compare_df is None or compare_df.empty or "Player Name" not in compare_df.columns:
        return pd.DataFrame()

    rows = []
    for player_name, block in compare_df.groupby("Player Name", dropna=True):
        rows.append(_compare_one_player_summary(player_name, block))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Holes", "Player"], ascending=[False, True]).reset_index(drop=True)


def build_compare_year_summary(compare_df, label_mode="year"):
    """Build Compare-tab summaries grouped by Year or Player-Year.

    label_mode:
      - "year": one row per year, label shown as 2025 / 2026
      - "player_year": one row per player-year, label shown as Jake 2025
    """
    if compare_df is None or compare_df.empty or "Year" not in compare_df.columns:
        return pd.DataFrame()

    d = compare_df.copy()
    d["Year"] = pd.to_numeric(_safe_col(d, "Year", pd.NA), errors="coerce")
    d = d.dropna(subset=["Year"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["Year"] = d["Year"].astype(int)

    rows = []
    if label_mode == "player_year" and "Player Name" in d.columns:
        group_cols = ["Player Name", "Year"]
        for (player_name, yr), block in d.groupby(group_cols, dropna=True):
            row = _compare_one_player_summary(str(player_name), block)
            row["Player"] = f"{player_name} {int(yr)}"
            row["Base Player"] = str(player_name)
            row["Year"] = int(yr)
            rows.append(row)
    else:
        for yr, block in d.groupby("Year", dropna=True):
            label = str(int(yr))
            row = _compare_one_player_summary(label, block)
            row["Player"] = label
            row["Year"] = int(yr)
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = [c for c in ["Year", "Base Player", "Player"] if c in out.columns]
    return out.sort_values(sort_cols).reset_index(drop=True)


def build_compare_yoy_variance_table(summary_df, metric_labels, base_label, current_label):
    """Build year-over-year variance table with stat-aware trend arrows."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), {}

    base_rows = summary_df[summary_df["Player"].astype(str) == str(base_label)]
    curr_rows = summary_df[summary_df["Player"].astype(str) == str(current_label)]
    if base_rows.empty or curr_rows.empty:
        return pd.DataFrame(), {}

    base = base_rows.iloc[0]
    curr = curr_rows.iloc[0]
    rows = []
    improved = declined = flat = 0
    best_gain = None
    worst_drop = None

    for metric_label in metric_labels:
        _label, key, better, _fmt = COMPARE_METRIC_BY_LABEL[metric_label]
        old_val = pd.to_numeric(base.get(key, pd.NA), errors="coerce")
        new_val = pd.to_numeric(curr.get(key, pd.NA), errors="coerce")
        if pd.isna(old_val) or pd.isna(new_val):
            continue
        delta = float(new_val) - float(old_val)
        good_delta = -delta if better == "lower" else delta
        if abs(good_delta) <= 0.05:
            trend = "➡️ Flat"
            flat += 1
        elif good_delta > 0:
            trend = "🔼 Better"
            improved += 1
        else:
            trend = "🔽 Worse"
            declined += 1

        if best_gain is None or good_delta > best_gain["good_delta"]:
            best_gain = {"metric": metric_label, "good_delta": good_delta, "delta": delta}
        if worst_drop is None or good_delta < worst_drop["good_delta"]:
            worst_drop = {"metric": metric_label, "good_delta": good_delta, "delta": delta}

        rows.append({
            "Metric": metric_label,
            str(base_label): _format_compare_value(metric_label, old_val),
            str(current_label): _format_compare_value(metric_label, new_val),
            "Change": _format_yoy_delta(metric_label, delta),
            "Trend": trend,
            "GoodDelta": good_delta,
        })

    table = pd.DataFrame(rows)
    summary = {
        "improved": improved,
        "declined": declined,
        "flat": flat,
        "best_gain": best_gain,
        "worst_drop": worst_drop,
    }
    return table, summary


def _format_yoy_delta(metric_label, delta):
    if metric_label not in COMPARE_METRIC_BY_LABEL:
        return f"{delta:+.1f}"
    _label, _key, _better, fmt = COMPARE_METRIC_BY_LABEL[metric_label]
    if "%" in fmt:
        return f"{delta:+.1f} pts"
    if "ft" in fmt:
        return f"{delta:+.1f} ft"
    if ".2f" in fmt:
        return f"{delta:+.1f}"
    return f"{delta:+.1f}"


def render_compare_yoy_summary_cards(variance_summary, base_label, current_label):
    improved = int(variance_summary.get("improved", 0) or 0)
    declined = int(variance_summary.get("declined", 0) or 0)
    flat = int(variance_summary.get("flat", 0) or 0)
    best_gain = variance_summary.get("best_gain")
    worst_drop = variance_summary.get("worst_drop")

    best_txt = "—"
    if best_gain:
        best_txt = f"{best_gain['metric']} ({_format_yoy_delta(best_gain['metric'], best_gain['delta'])})"
    worst_txt = "—"
    if worst_drop:
        worst_txt = f"{worst_drop['metric']} ({_format_yoy_delta(worst_drop['metric'], worst_drop['delta'])})"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Improved Metrics", improved)
    c2.metric("Declined Metrics", declined)
    c3.metric("Flat Metrics", flat)
    c4.metric("Comparison", f"{base_label} → {current_label}")

    st.markdown(
        f"""
        <div style="background:linear-gradient(180deg,#202a25 0%,#1f1f1f 100%); border:1px solid rgba(100,223,181,.14); border-radius:14px; padding:12px 14px; margin:8px 0 12px 0; line-height:1.55;">
          <b>✅ Biggest gain:</b> {best_txt}<br>
          <b>👀 Biggest drop:</b> {worst_txt}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_compare_yoy_trend_chart(summary_df, metric_label):
    if summary_df is None or summary_df.empty or "Year" not in summary_df.columns:
        st.info("No year trend data available.")
        return
    _label, key, better, _fmt = COMPARE_METRIC_BY_LABEL[metric_label]
    plot_df = summary_df[["Player", "Year", key]].copy()
    plot_df = plot_df.rename(columns={key: "Value"})
    plot_df["Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Value", "Year"]).copy()
    if plot_df.empty:
        st.info("No usable values for this year trend metric.")
        return
    plot_df["Display"] = plot_df["Value"].apply(lambda x: _format_compare_value(metric_label, x))
    chart = (
        alt.Chart(plot_df)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("Year:O", title="Year"),
            y=alt.Y("Value:Q", title=metric_label),
            color=alt.Color("Player:N", title="Series"),
            tooltip=[
                alt.Tooltip("Player:N", title="Series"),
                alt.Tooltip("Year:O"),
                alt.Tooltip("Display:N", title=metric_label),
            ],
        )
        .properties(height=330, title=f"{metric_label} — Year Trend")
        .configure_view(strokeOpacity=0)
    )
    st.altair_chart(chart, use_container_width=True)


def _format_compare_value(metric_label, value):
    if pd.isna(value):
        return "—"
    _, _, _, fmt = COMPARE_METRIC_BY_LABEL[metric_label]
    try:
        return fmt.format(float(value))
    except Exception:
        return "—"


def _style_compare_detail_table(detail_df):
    """Format the Compare tab detail table with clean decimals and percent signs.
    Percent metrics are already stored as 0-100 values, so this adds a percent sign
    without rescaling.
    """
    if detail_df is None or detail_df.empty:
        return detail_df

    fmt = {
        "Rounds": lambda x: "—" if pd.isna(x) else f"{int(round(float(x))):,}",
        "Holes": lambda x: "—" if pd.isna(x) else f"{int(round(float(x))):,}",
    }

    for label, _key, _better, metric_fmt in COMPARE_METRIC_DEFS:
        if label not in detail_df.columns:
            continue
        if "%" in metric_fmt:
            fmt[label] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f}%"
        elif "ft" in metric_fmt:
            fmt[label] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f} ft"
        elif metric_fmt.startswith("{:+"):
            fmt[label] = lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}"
        elif ".2f" in metric_fmt:
            # Keep per-18 event stats readable without excessive digits.
            fmt[label] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f}"
        else:
            fmt[label] = lambda x: "—" if pd.isna(x) else f"{float(x):.1f}"

    try:
        return detail_df.style.format(fmt, na_rep="—")
    except Exception:
        return detail_df


def build_compare_metric_matrix(summary_df, metric_labels):
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    rows = []
    for metric_label in metric_labels:
        _, key, better, _ = COMPARE_METRIC_BY_LABEL[metric_label]
        row = {"Metric": metric_label, "Better": "Lower" if better == "lower" else "Higher"}
        vals = []
        for _, player_row in summary_df.iterrows():
            player_name = str(player_row["Player"])
            raw_val = player_row.get(key, pd.NA)
            vals.append((player_name, raw_val))
            row[player_name] = _format_compare_value(metric_label, raw_val)

        numeric_vals = [(p, float(v)) for p, v in vals if pd.notna(v)]
        if numeric_vals:
            if better == "lower":
                best_val = min(v for _, v in numeric_vals)
            else:
                best_val = max(v for _, v in numeric_vals)
            winners = [p for p, v in numeric_vals if abs(v - best_val) < 1e-9]
            row["Leader"] = ", ".join(winners)
        else:
            row["Leader"] = "—"
        rows.append(row)

    return pd.DataFrame(rows)


def render_compare_metric_chart(summary_df, metric_label):
    if summary_df is None or summary_df.empty:
        st.info("No comparison data available for the selected filters.")
        return

    _, key, better, _ = COMPARE_METRIC_BY_LABEL[metric_label]
    plot_df = summary_df[["Player", "Holes", key]].copy()
    plot_df = plot_df.rename(columns={key: "Value"})
    plot_df["Value"] = pd.to_numeric(plot_df["Value"], errors="coerce")
    plot_df = plot_df.dropna(subset=["Value"]).copy()
    if plot_df.empty:
        st.info("No usable values for this selected metric.")
        return

    sort_order = "ascending" if better == "lower" else "descending"
    plot_df["Display"] = plot_df["Value"].apply(lambda x: _format_compare_value(metric_label, x))

    base = alt.Chart(plot_df)

    chart = (
        base
        .mark_bar(cornerRadiusTopRight=8, cornerRadiusBottomRight=8)
        .encode(
            y=alt.Y("Player:N", sort=alt.EncodingSortField(field="Value", op="sum", order=sort_order), title=None),
            x=alt.X("Value:Q", title=metric_label),
            tooltip=[
                alt.Tooltip("Player:N"),
                alt.Tooltip("Display:N", title=metric_label),
                alt.Tooltip("Holes:Q", title="Holes", format=",.0f"),
            ],
        )
    )

    labels = (
        base
        .mark_text(align="left", dx=6, fontWeight="bold", color="white")
        .encode(
            y=alt.Y("Player:N", sort=alt.EncodingSortField(field="Value", op="sum", order=sort_order)),
            x=alt.X("Value:Q"),
            text="Display:N",
        )
    )

    combined = (
        alt.layer(chart, labels)
        .properties(height=max(260, 42 * len(plot_df)), title=f"{metric_label} — Player Compare")
        .configure_view(strokeOpacity=0, clip=False)
        .configure_axis(
            labelColor="white",
            titleColor="white",
            gridColor="rgba(255,255,255,0.10)",
            tickColor="rgba(255,255,255,0.20)",
            domainColor="rgba(255,255,255,0.20)",
        )
    )
    st.altair_chart(combined, use_container_width=True)


def render_compare_leader_cards(summary_df):
    if summary_df is None or summary_df.empty:
        return

    card_defs = [
        ("Best Scoring", "Avg Score / 18"),
        ("Lowest Est. Handicap", "Estimated Handicap"),
        ("Best GIR", "GIR %"),
        ("Best Putting", "Putts / 18"),
        ("Best Birdie Conversion", "Birdie Conversion % (GIR)"),
    ]

    cols = st.columns(len(card_defs))
    for col, (card_title, metric_label) in zip(cols, card_defs):
        _label, key, better, _fmt = COMPARE_METRIC_BY_LABEL[metric_label]
        vals = summary_df[["Player", key, "Holes"]].copy()
        vals[key] = pd.to_numeric(vals[key], errors="coerce")
        vals["Holes"] = pd.to_numeric(vals["Holes"], errors="coerce").fillna(0).astype(int)
        vals = vals.dropna(subset=[key])
        if vals.empty:
            with col:
                st.metric(card_title, "—")
            continue

        vals = vals.sort_values(
            [key, "Holes", "Player"],
            ascending=[True if better == "lower" else False, False, True]
        ).reset_index(drop=True)
        vals["Rank"] = vals.index + 1
        best = vals.iloc[0]

        rank_rows = []
        for _, r in vals.head(5).iterrows():
            rank_rows.append(
                f"""
                <div style="display:flex; justify-content:space-between; gap:8px; padding:3px 0; border-top:1px solid rgba(255,255,255,.07);">
                  <span style="white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">
                    <b>#{int(r['Rank'])}</b> {r['Player']}
                  </span>
                  <span style="font-variant-numeric:tabular-nums; font-weight:800; white-space:nowrap;">
                    {_format_compare_value(metric_label, r[key])}
                  </span>
                </div>
                """
            )

        with col:
            st.markdown(
                f"""
                <div style="background:linear-gradient(180deg,#2b2b2b 0%,#1f1f1f 100%); border:1px solid rgba(255,255,255,.10); border-radius:16px; padding:13px 14px; min-height:170px; box-shadow:0 8px 18px rgba(0,0,0,.18);">
                  <div style="font-size:12px; color:#bdbdbd; font-weight:800; text-transform:uppercase; letter-spacing:.04em; margin-bottom:5px;">{card_title}</div>
                  <div style="font-size:23px; color:#fff; font-weight:950; line-height:1.15; margin-bottom:2px;">#{int(best['Rank'])} {best['Player']}</div>
                  <div style="font-size:15px; color:#64dfb5; font-weight:900; margin-bottom:8px;">{_format_compare_value(metric_label, best[key])}</div>
                  <div style="font-size:12px; color:#e7e7e7; line-height:1.3;">{''.join(rank_rows)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_compare_tab(full_df):
    st.markdown("### 🆚 Compare")
    st.caption("Compare players across the same headline metrics from the Overview table, with independent filters for players, years, courses, months, trips, and date range.")

    compare_view = st.radio(
        "Compare View",
        ["Player vs Player", "Year vs Year", "Player-Year Matrix"],
        horizontal=True,
        key="compare_view_mode",
        help="Year vs Year is best for Jake 2025 vs Jake 2026. Player-Year Matrix lets you compare Jake 2025, Jake 2026, Brent 2025, Brent 2026, etc.",
    )

    work = full_df.copy()
    work["Date Played"] = pd.to_datetime(_safe_col(work, "Date Played", pd.NaT), errors="coerce")
    if "Year" not in work.columns:
        work["Year"] = work["Date Played"].dt.year
    if "Month" not in work.columns:
        work["Month"] = work["Date Played"].dt.strftime("%B")

    all_players = sorted([str(x) for x in _safe_col(work, "Player Name", "").dropna().astype(str).unique().tolist() if str(x).strip()])
    all_courses = sorted([str(x) for x in _safe_col(work, "Course Name", "").dropna().astype(str).unique().tolist() if str(x).strip()])
    all_years = sorted([int(x) for x in pd.to_numeric(_safe_col(work, "Year", pd.NA), errors="coerce").dropna().unique().tolist()], reverse=True)
    month_order = list(pd.Series(pd.date_range("2026-01-01", periods=12, freq="MS")).dt.strftime("%B"))
    all_months = [m for m in month_order if m in set(_safe_col(work, "Month", "").dropna().astype(str).tolist())]

    default_players = []
    if "player" in globals() and player in all_players:
        default_players.append(player)
    default_players += [p for p in all_players if p not in default_players][:3]

    with st.container():
        c1, c2, c3 = st.columns([1.35, 1, 1])
        with c1:
            cmp_players = st.multiselect("Players", options=all_players, default=default_players[:4], key="compare_players")
        with c2:
            cmp_years = st.multiselect("Years", options=all_years, default=all_years[:1] if all_years else [], key="compare_years")
        with c3:
            cmp_courses = st.multiselect("Courses", options=all_courses, default=[], key="compare_courses")

        c4, c5, c6 = st.columns([1, 1, 1])
        with c4:
            cmp_months = st.multiselect("Months", options=all_months, default=[], key="compare_months")
        with c5:
            min_date = work["Date Played"].min()
            max_date = work["Date Played"].max()
            if pd.notna(min_date) and pd.notna(max_date):
                cmp_date_range = st.date_input(
                    "Date Range",
                    value=(min_date.date(), max_date.date()),
                    min_value=min_date.date(),
                    max_value=max_date.date(),
                    key="compare_date_range",
                )
            else:
                cmp_date_range = None
        with c6:
            min_holes = st.number_input("Minimum holes per player", min_value=1, max_value=5000, value=18, step=1, key="compare_min_holes")

        trip_options = []
        if "Golf Trip" in work.columns:
            trip_options = sorted([str(x) for x in work["Golf Trip"].dropna().astype(str).unique().tolist() if str(x).strip()])
        if trip_options:
            cmp_trips = st.multiselect("Golf Trip", options=trip_options, default=[], key="compare_trips")
        else:
            cmp_trips = []

    compare_df = work.copy()
    if cmp_players:
        compare_df = compare_df[compare_df["Player Name"].astype(str).isin(cmp_players)].copy()
    if compare_view == "Year vs Year" and len(cmp_players) != 1:
        st.caption("Year vs Year works best with one selected player. If multiple players are selected, the years will be combined across those players. Use Player-Year Matrix for side-by-side player/year comparisons.")
    if cmp_years:
        compare_df = compare_df[pd.to_numeric(compare_df["Year"], errors="coerce").isin(cmp_years)].copy()
    if cmp_courses:
        compare_df = compare_df[compare_df["Course Name"].astype(str).isin(cmp_courses)].copy()
    if cmp_months:
        compare_df = compare_df[compare_df["Month"].astype(str).isin(cmp_months)].copy()
    if cmp_trips and "Golf Trip" in compare_df.columns:
        compare_df = compare_df[compare_df["Golf Trip"].astype(str).isin(cmp_trips)].copy()
    if cmp_date_range and isinstance(cmp_date_range, tuple) and len(cmp_date_range) == 2:
        start_date, end_date = cmp_date_range
        compare_df = compare_df[
            (compare_df["Date Played"].dt.date >= start_date)
            & (compare_df["Date Played"].dt.date <= end_date)
        ].copy()

    if compare_view == "Player-Year Matrix":
        summary_df = build_compare_year_summary(compare_df, label_mode="player_year")
        empty_msg = "No player-year groups meet the selected Compare filters."
    elif compare_view == "Year vs Year":
        summary_df = build_compare_year_summary(compare_df, label_mode="year")
        empty_msg = "No year groups meet the selected Compare filters."
    else:
        summary_df = build_compare_player_summary(compare_df)
        empty_msg = "No players meet the selected Compare filters."

    if not summary_df.empty:
        summary_df = summary_df[pd.to_numeric(summary_df["Holes"], errors="coerce").fillna(0) >= int(min_holes)].copy()

    if summary_df.empty:
        st.info(empty_msg)
        return

    if compare_view == "Player vs Player":
        render_compare_leader_cards(summary_df)
    else:
        st.caption(f"Compare groups shown: {len(summary_df)}")

    metric_group = st.radio(
        "Metric Set",
        ["Core", "Scoring", "Ball Striking", "Putting", "Short Game", "All"],
        horizontal=True,
        key="compare_metric_group",
    )

    group_map = {
        "Core": ["Estimated Handicap", "Avg Score / 18", "Score to Par / 18", "GIR %", "Fairway %", "Putts / 18", "Scramble %", "Birdie Conversion % (GIR)"],
        "Scoring": ["Estimated Handicap", "Avg Score / 18", "Score to Par / 18", "Birdie Conversion % (GIR)", "Pro Pars+ / 18", "Arnies / 18", "Seves / 18", "Lost Balls / 18"],
        "Ball Striking": ["GIR %", "Fairway %", "Avg First-Putt Proximity", "Avg Prox if GIR", "Avg Prox if No GIR", "Pro Pars+ / 18"],
        "Putting": ["Putts / 18", "1-Putt %", "3+ Putt %", "Birdie Conversion % (GIR)", "Avg First-Putt Proximity"],
        "Short Game": ["Scramble %", "Up & Down %", "Avg Prox if No GIR", "Arnies / 18", "Seves / 18", "Hole Outs / 18"],
        "All": COMPARE_METRIC_LABELS,
    }
    selected_metric_labels = group_map.get(metric_group, COMPARE_METRIC_LABELS)

    if compare_view == "Year vs Year":
        year_labels = summary_df["Player"].astype(str).tolist()
        if len(year_labels) >= 2:
            default_base_idx = 0
            default_current_idx = len(year_labels) - 1
            yc1, yc2 = st.columns(2)
            with yc1:
                yoy_base = st.selectbox("Baseline Year", options=year_labels, index=default_base_idx, key="compare_yoy_base_year")
            with yc2:
                yoy_current = st.selectbox("Current Year", options=year_labels, index=default_current_idx, key="compare_yoy_current_year")

            if yoy_base == yoy_current:
                st.info("Pick two different years to show variance and trend.")
            else:
                variance_table, variance_summary = build_compare_yoy_variance_table(summary_df, selected_metric_labels, yoy_base, yoy_current)
                st.markdown("#### Year-over-Year Trend Summary")
                render_compare_yoy_summary_cards(variance_summary, yoy_base, yoy_current)
                st.markdown("#### Year-over-Year Variance Table")
                show_table = variance_table.drop(columns=["GoodDelta"], errors="ignore")
                st.dataframe(show_table, use_container_width=True, hide_index=True)

                st.markdown("#### Year Trend Chart")
                yoy_chart_metric = st.selectbox("Year Trend Metric", options=selected_metric_labels, index=0, key="compare_yoy_chart_metric")
                render_compare_yoy_trend_chart(summary_df, yoy_chart_metric)
        else:
            st.info("Select at least two years to show year-over-year variance and trend.")

    st.markdown("#### Comparison Matrix")
    matrix = build_compare_metric_matrix(summary_df, selected_metric_labels)
    st.dataframe(matrix, use_container_width=True, hide_index=True)

    st.markdown("#### Visual Metric Compare")
    chart_metric = st.selectbox("Chart Metric", options=selected_metric_labels, index=0, key="compare_chart_metric")
    render_compare_metric_chart(summary_df, chart_metric)

    st.markdown("#### Player Summary Detail")
    detail_cols = ["Player", "Rounds", "Holes"] + [m[1] for m in COMPARE_METRIC_DEFS]
    detail = summary_df[[c for c in detail_cols if c in summary_df.columns]].copy()
    rename_map = {key: label for label, key, _, _ in COMPARE_METRIC_DEFS}
    detail = detail.rename(columns=rename_map)
    st.dataframe(_style_compare_detail_table(detail), use_container_width=True, hide_index=True)

    with st.expander("Compare notes", expanded=False):
        st.markdown("""
        - **Avg Score / 18** and **Score to Par / 18** are normalized so 9-hole and 18-hole samples can be compared.
        - **Birdie Conversion % (GIR)** is birdie-or-better holes divided by GIR made.
        - **Avg Prox if GIR** shows approach quality on greens hit; **Avg Prox if No GIR** shows how playable misses are.
        - **Estimated Handicap** is not official. It uses score differential logic, normalizes partial rounds to 18 holes, and falls back to a neutral 72.0 / 113 course assumption when rating or slope is missing.
        - The minimum-holes filter helps avoid misleading leaders from tiny samples.
        """)

# =========================================================
# Tabs
# =========================================================
tab_scorecard, tab_overview, tab_compare, tab_analysis, tab_putting, tab_trends, tab_bestof, tab_shortgame = st.tabs(["Scorecard", "Overview", "Compare", "Ball Striking", "Putting", "Trends", "Best Of", "Short Game"])

with tab_overview:
    st.markdown("### 📋 High-Level Overview")
    st.caption("Quick top-line round review with a cleaner comparison view.")
    compare_mode_overview = st.radio("Compare this round against:", _compare_mode_options(), horizontal=True, key="overview_compare_mode")
    compare_label_overview = _compare_mode_display_label(compare_mode_overview)
    benchmark_df = build_benchmark_df(df, round_data, compare_mode_overview, selected_baseline_round_ids)
    if compare_mode_overview == "Selected Rounds" and not selected_baseline_round_ids:
        st.info("Pick one or more rounds in 'Custom Round Baseline' above to compare this round against a custom sample.")

    round_overview = build_overview_snapshot(round_data)
    base_overview = build_overview_snapshot(benchmark_df)
    metric_df = build_overview_metric_rows(round_overview, base_overview, compare_label_overview)

    # Keep these detailed proximity splits in the comparison table, but remove
    # them from the Overview box/card sections so the top layout does not compress.
    overview_box_exclude = ["Avg Prox if GIR", "Avg Prox if No GIR"]
    metric_df_boxes = metric_df[~metric_df["Metric"].isin(overview_box_exclude)].copy()

    handicap_left, handicap_right = st.columns([1, 1])
    with handicap_left:
        current_handicap_est = render_handicap_estimate_box(player_df, title=f"Estimated Handicap — {player}")
    with handicap_right:
        if benchmark_df is not None and not benchmark_df.empty:
            render_handicap_estimate_box(benchmark_df, title=f"Estimated Handicap — {compare_label_overview}")
        else:
            st.info("No baseline rounds available for handicap estimate.")

    render_baseline_summary_box(build_baseline_summary(benchmark_df, compare_mode_overview))
    render_overview_cards(metric_df_boxes, compare_label_overview)
    render_round_impact_cards(metric_df_boxes, compare_label_overview)

    why_summary = build_why_round_happened(round_data, metric_df_boxes, compare_label_overview)
    why_left, why_right = st.columns(2)
    with why_left:
        why_good_html = "<br>".join([f"• {x}" for x in why_summary["good"]])
        st.markdown(
            f"""
            <div style="background:#1f2c25; border:1px solid rgba(100,223,181,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
              <b>🧠 Why this round worked</b><br>
              {why_good_html}
            </div>
            """,
            unsafe_allow_html=True
        )
    with why_right:
        why_watch_html = "<br>".join([f"• {x}" for x in why_summary["watch"]])
        st.markdown(
            f"""
            <div style="background:#2d2323; border:1px solid rgba(238,108,77,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
              <b>🔍 Why the score landed where it did</b><br>
              {why_watch_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    top_left, top_right = st.columns([1.25, 1.0])

    with top_left:
        st.markdown("#### Comparison Table")
        overview_table = metric_df[["Metric", "Round Display", "Baseline Display", "Delta Display", "Better"]].rename(columns={
            "Round Display": "Round",
            "Baseline Display": compare_label_overview,
            "Delta Display": "Change",
        }).copy()
        st.dataframe(overview_table, use_container_width=True, hide_index=True)

    with top_right:
        st.markdown("#### Improvement Snapshot")
        render_overview_delta_chart(metric_df_boxes, compare_label_overview)

    quick1, quick2, quick3, quick4 = st.columns(4)
    with quick1:
        st.markdown(
            f"""
            <div style="background:#242424; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; line-height:1.55;">
              <b>🎯 Ball Striking</b><br>
              GIR: <span style="font-weight:800;">{round_overview['gir_pct']:.1f}%</span><br>
              Fairway: <span style="font-weight:800;">{round_overview['fw_pct']:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with quick2:
        st.markdown(
            f"""
            <div style="background:#242424; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; line-height:1.55;">
              <b>⛳ Scoring Support</b><br>
              Scramble: <span style="font-weight:800;">{round_overview['scramble_pct']:.1f}%</span><br>
              Up & Down: <span style="font-weight:800;">{round_overview['updown_pct']:.1f}%</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with quick3:
        st.markdown(
            f"""
            <div style="background:#242424; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; line-height:1.55;">
              <b>🪄 Creativity Stats</b><br>
              Arnies / 18: <span style="font-weight:800;">{round_overview['arnies_per18']:.2f}</span><br>
              Seves / 18: <span style="font-weight:800;">{round_overview['seves_per18']:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with quick4:
        st.markdown(
            f"""
            <div style="background:#242424; border:1px solid rgba(255,255,255,.08); border-radius:12px; padding:12px 14px; line-height:1.55;">
              <b>🧠 Round Shape</b><br>
              Avg First-Putt Prox: <span style="font-weight:800;">{round_overview['avg_prox']:.1f} ft</span><br>
              Lost Balls / 18: <span style="font-weight:800;">{round_overview['lost_balls_per18']:.2f}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("### Overall Scoring Breakdown")

    round_mix = build_score_mix_summary(round_data)
    base_mix = build_score_mix_summary(benchmark_df)

    mix_left, mix_right = st.columns(2)
    with mix_left:
        st.markdown("##### Score Type Mix")
        render_segmented_score_mix_compare(round_mix["score_df"], base_mix["score_df"], compare_label_overview)
    with mix_right:
        st.markdown("##### Score Category Mix")
        render_segmented_category_mix_compare(round_mix["cat_df"], base_mix["cat_df"], compare_label_overview)

    ov_summary = build_overview_round_summary(metric_df)

    st.markdown("#### Round Summary")
    sum_left, sum_right = st.columns(2)
    with sum_left:
        good_html = "<br>".join([f"• {x}" for x in ov_summary["good"]])
        st.markdown(
            f"""
            <div style="background:#1f2c25; border:1px solid rgba(100,223,181,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
              <b>✅ What went well</b><br>
              {good_html}
            </div>
            """,
            unsafe_allow_html=True
        )
    with sum_right:
        lock_html = "<br>".join([f"• {x}" for x in ov_summary["lock"]])
        st.markdown(
            f"""
            <div style="background:#2d2323; border:1px solid rgba(238,108,77,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
              <b>🔒 What needs to be locked in</b><br>
              {lock_html}
            </div>
            """,
            unsafe_allow_html=True
        )

    render_trivia_block("overview", "Overview trivia")

with tab_scorecard:
    table_html = f"""
    <style>
      .sc-wrap {{
        background:linear-gradient(180deg,#262626 0%, #202020 100%);
        padding:10px; border-radius:14px;
        box-shadow: 0 10px 22px rgba(0,0,0,.18);
        border:1px solid rgba(255,255,255,.06);
      }}
      .sc-table {{
        width:100%; border-collapse:separate; border-spacing:0;
        font-size:12.2px; line-height:1.22; color:#fff;
        overflow:hidden;
      }}
      .sc-table thead th {{
        position: sticky; top: 0; z-index: 2;
        background:#3a3a3a; color:#fff; text-align:center;
        padding:5px 6px; font-weight:800; border-bottom:1px solid rgba(255,255,255,.08);
      }}
      .sc-table thead th:first-child {{
        border-top-left-radius:10px;
      }}
      .sc-table thead th:last-child {{
        border-top-right-radius:10px;
      }}
      .sc-table tbody td, .sc-table tbody th {{
        padding:4px 6px; border-bottom:1px solid rgba(255,255,255,.05);
        text-align:center;
      }}
      .sc-table tbody tr:nth-child(odd)  {{ background:#333333; }}
      .sc-table tbody tr:nth-child(even) {{ background:#2c2c2c; }}
      .sc-table tbody tr:hover {{ background:#3b3b3b; }}
      .sc-label {{
        text-align:left; font-weight:800; color:#fff; white-space:nowrap;
        background:rgba(255,255,255,.02);
      }}
      .sc-score {{
        font-size:18px; font-weight:900; letter-spacing:.15px;
      }}
    </style>

    <div class="sc-wrap">
      <table class="sc-table">
        <thead>
          <tr>
            <th style="text-align:left;">Hole</th>
            {''.join(f"<th>{col}</th>" for col in hole_nums)}
          </tr>
        </thead>
        <tbody>
    """

    stat_rows = [
        ("Par", par_row),
        ("Yards", yards_row),
        ("Score", score_row),
        ("Putts", putts_row),
        ("Fairway", fw_row),
        ("GIR", gir_row),
        ("Appr Miss Dir", approach_miss_dir_row),
        ("Arnie", arnie_row),
        ("Lost Balls", lost_ball_row),
        ("Appr Club", approach_clubs_row),
        ("Appr Yards", approach_yards_row),
        ("Appr GIR", approach_gir_row),
        ("Prox (FT)", prox_to_hole_row),
        ("FT Made", putt_made_ft_row),
    ]

    for label, row in stat_rows:
        table_html += f"<tr><td class='sc-label'>{label}</td>"
        for j, val in enumerate(row):
            if label == "Score":
                color = "#ffffff"
                if j not in [9, 19, 20] and isinstance(val, int):
                    par_val = par_row[j]
                    if isinstance(par_val, int):
                        if val <= par_val - 1:
                            color = "#f5c518"
                        elif val == par_val + 1:
                            color = "#ff9999"
                        elif val == par_val + 2:
                            color = "#ff6666"
                        elif val >= par_val + 3:
                            color = "#cc0000"
                table_html += f"<td class='sc-score' style='color:{color};'>{val}</td>"
            elif label == "Lost Balls":
                is_total_col = j in [9, 19, 20]
                is_zero_num = isinstance(val, (int, float)) and val == 0
                display_val = "" if (not is_total_col and is_zero_num) else val
                style = "font-variant-numeric: tabular-nums;" if isinstance(val, (int, float)) else ""
                table_html += f"<td style='{style}'>{display_val}</td>"
            else:
                is_num = isinstance(val, (int, float))
                style = "font-variant-numeric: tabular-nums;" if is_num else ""
                table_html += f"<td style='{style}'>{val}</td>"
        table_html += "</tr>"

    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

    summary_header_html = f"""
    🏌️ {player} | {course} | {date}<br><br>
    <b>📊 Round Totals — {holes_played} Holes</b>
    """

    us_open_summary = build_us_open_par_summary(round_data)
    us_open_made = int(us_open_summary["made"])
    us_open_attempts = int(us_open_summary["attempts"])
    us_open_pct = float(us_open_summary["pct"])

    cards_html = f"""
    <div style="display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 4px 0;">
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">Score</div>
        <div style="font-size:22px;font-weight:700;">{total_score} <span style="font-size:14px;color:#bbb;">({score_to_par_str})</span></div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">Putts / Hole</div>
        <div style="font-size:22px;font-weight:700;">{putts_per_hole:.2f}</div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">GIR</div>
        <div style="font-size:22px;font-weight:700;">{gir_total}/{holes_played} <span style="font-size:14px;color:#bbb;">({gir_pct_num:.1f}%)</span></div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">Fairways</div>
        <div style="font-size:22px;font-weight:700;">{fw_total}/{fw_attempts} <span style="font-size:14px;color:#bbb;">({fw_pct_num:.1f}%)</span></div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">Scrambles</div>
        <div style="font-size:22px;font-weight:700;">{scrambles_display}</div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">Up & Downs</div>
        <div style="font-size:22px;font-weight:700;">{updowns_display}</div>
      </div>
      <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:10px;">
        <div style="font-size:12px;color:#aaa;">US Open Pars</div>
        <div style="font-size:22px;font-weight:700;">{us_open_made}/{us_open_attempts} <span style="font-size:14px;color:#bbb;">({us_open_pct:.1f}%)</span></div>
      </div>
    </div>
    """

    benchmarks_html = f"""
    <div style="margin-top:6px; line-height:1.5;">
      <b>Score to Par — Benchmarks</b><br>
      Current Round: {_fmt_to_par(int(total_score - sum(pars)))}<br>
      Last 5 Rounds (avg): {_fmt_par_float(last5_avg_over_par)}<br>
      Last 100 Holes — Total: {_fmt_to_par(last100_total_over_par)} | Per-18: {_fmt_par_float(last100_per18)}
    </div>
    """

    summary_details_html = f"""
    <br>
    <b>💡 Advanced Insights</b><br>
    Total 1 Putts: {total_1_putts}<br>
    Total 3+ Putts: {total_3_plus_putts}<br>
    3-Putt Bogeys: {total_3_putt_bogeys}<br>
    Longest Putt Made: {longest_putt_made:.1f} ft<br>
    Putts Made ≤ 5 ft: {putts_made_5_or_less}<br>
    Putts Made 5–10 ft: {putts_made_5_to_10}<br>
    Putts Made > 10 ft: {putts_made_over_10}<br>
    Pro Pars+: {pro_pars_total}<br>
    Arnies: {arnies_total}<br>
    Scrambles: {scrambles_display}<br>
    Up & Downs: {updowns_display}<br>
    GIR — Par 3: {gir3_m}/{gir3_t} {gir3_pct:.1f}% {get_emoji(gir3_pct)} |
    Par 4: {gir4_m}/{gir4_t} {gir4_pct:.1f}% {get_emoji(gir4_pct)} |
    Par 5: {gir5_m}/{gir5_t} {gir5_pct:.1f}% {get_emoji(gir5_pct)}<br>
    Fairways — Par 4: {fw4_m}/{fw4_t} {fw4_pct:.1f}% {get_emoji(fw4_pct)} |
    Par 5: {fw5_m}/{fw5_t} {fw5_pct:.1f}% {get_emoji(fw5_pct)}<br>
    GIR Overall: {gir_pct_num:.1f}% {get_emoji(gir_pct_num)}<br>
    Seves: {seves_total} | Hole Outs: {hole_outs_total} | Lost Balls: {lost_balls_display}<br><br>

    <b>📈 Scoring Averages</b><br>
    Par 3 Avg: {avg_par3:.1f}<br>
    Par 4 Avg: {avg_par4:.1f}<br>
    Par 5 Avg: {avg_par5:.1f}<br><br>

    <b>🎯 Score Breakdown</b><br>
    Birdie: {score_type_counts.get("Birdie", 0)} |
    Par: {score_type_counts.get("Par", 0)} |
    Bogey: {score_type_counts.get("Bogey", 0)} |
    Double Bogey: {score_type_counts.get("Double Bogey", 0)} |
    Triple Bogey +: {score_type_counts.get("Triple Bogey +", 0)}<br>
    Par or Better: {cat_counts.get("Par or Better", 0)} ({round(cat_counts.get("Par or Better", 0)/max(len(round_data),1)*100,1)}%) |
    Bogey: {cat_counts.get("Bogey", 0)} ({round(cat_counts.get("Bogey", 0)/max(len(round_data),1)*100,1)}%) |
    Double+: {cat_counts.get("Double+", 0)} ({round(cat_counts.get("Double+", 0)/max(len(round_data),1)*100,1)}%)<br>
    """

    render_scorecard_summary_panel(
        player=player,
        course=course,
        date=date,
        total_score=total_score,
        score_to_par_str=score_to_par_str,
        holes_played=holes_played,
        putts_per_hole=putts_per_hole,
        gir_total=gir_total,
        gir_pct_num=gir_pct_num,
        fw_total=fw_total,
        fw_attempts=fw_attempts,
        fw_pct_num=fw_pct_num,
        scrambles_display=scrambles_display,
        updowns_display=updowns_display,
        us_open_made=us_open_made,
        us_open_attempts=us_open_attempts,
        us_open_pct=us_open_pct,
        avg_par3=avg_par3,
        avg_par4=avg_par4,
        avg_par5=avg_par5,
        total_1_putts=total_1_putts,
        total_3_plus_putts=total_3_plus_putts,
        total_3_putt_bogeys=total_3_putt_bogeys,
        longest_putt_made=longest_putt_made,
        putts_made_5_or_less=putts_made_5_or_less,
        putts_made_5_to_10=putts_made_5_to_10,
        putts_made_over_10=putts_made_over_10,
        pro_pars_total=pro_pars_total,
        arnies_total=arnies_total,
        seves_total=seves_total,
        hole_outs_total=hole_outs_total,
        lost_balls_display=lost_balls_display,
        gir3_m=gir3_m,
        gir3_t=gir3_t,
        gir3_pct=gir3_pct,
        gir4_m=gir4_m,
        gir4_t=gir4_t,
        gir4_pct=gir4_pct,
        gir5_m=gir5_m,
        gir5_t=gir5_t,
        gir5_pct=gir5_pct,
        fw4_m=fw4_m,
        fw4_t=fw4_t,
        fw4_pct=fw4_pct,
        fw5_m=fw5_m,
        fw5_t=fw5_t,
        fw5_pct=fw5_pct
    )

    perf_grade, perf_score, perf_details = build_round_performance_rating(round_data, build_benchmark_df(df, round_data, "All Time"))
    detail_lines = []
    for name, cur, base, delta in perf_details:
        if name == "Putts / Hole":
            detail_lines.append(f"{name}: {cur:.2f} vs {base:.2f} ({delta:+.2f})")
        else:
            detail_lines.append(f"{name}: {cur:.1f} vs {base:.1f} ({delta:+.1f})")
    st.markdown(
        f"""
        <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
          <b>🏅 Round Performance Rating</b><br>
          Grade: <span style="font-size:20px; font-weight:800;">{perf_grade}</span>
          <span style="color:#aaa;">(composite score {perf_score:+.1f} vs all-time baseline)</span><br>
          {'<br>'.join(detail_lines[:4])}
        </div>
        """,
        unsafe_allow_html=True,
    )

    sg_beta = build_sg_style_insights(df, round_data, compare_mode="All Time")
    sg_color = "#64dfb5" if sg_beta["total_sg"] >= 0 else "#ee6c4d"
    st.markdown(
        f"""
        <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
          <b>📈 Strokes Gained Style (Beta)</b>
          <span style="color:#aaa;">— proxy vs your all-time baseline, not a PGA SG model</span><br>
          Total: <span style="font-size:20px; font-weight:800; color:{sg_color};">{sg_beta['total_sg']:+.2f}</span><br>
          Approach: {sg_beta['approach_sg']:+.2f}
          <span style="color:#aaa;">({sg_beta['approach_extra']:+.2f} makes vs expected)</span><br>
          Putting: {sg_beta['putting_sg']:+.2f}
          <span style="color:#aaa;">({sg_beta['putting_extra']:+.2f} makes vs expected)</span><br>
          Short Game: {sg_beta['short_game_sg']:+.2f}
          <span style="color:#aaa;">({sg_beta['short_game_extra']:+.2f} saves vs expected)</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    sg_impact = build_sg_round_impact(round_data, sg_beta)
    impact_color = "#64dfb5" if sg_impact["performance_delta"] < 0 else "#ee6c4d"
    impact_phrase = f"{abs(sg_impact['performance_delta']):.1f} strokes {'better' if sg_impact['performance_delta'] < 0 else 'worse'} than expected"
    st.markdown(
        f"""
        <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
          <b>🎯 Round Impact vs Baseline</b><br>
          SG vs Baseline: <span style="font-weight:800;">{sg_impact['sg_vs_baseline']:+.2f}</span><br>
          Expected Score: {sg_impact['expected_score']:.1f}<br>
          Actual Score: {sg_impact['actual_score']:.0f}<br>
          Performance: <span style="font-weight:800; color:{impact_color};">{impact_phrase}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    comparisons_html = f"""
    <div style="margin-top:6px; padding:10px; background:#262626; border-radius:10px; line-height:1.5;">
      <b>📊 Benchmarks (Recent)</b><br>
      <b>Up &amp; Downs</b> — Last 5: {ud5_m}/{ud5_o} ({ud5_pct:.1f}%) | Last 100: {ud100_m}/{ud100_o} ({ud100_pct:.1f}%)<br>
      <b>Scrambles</b> — Last 5: {sc5_m}/{sc5_o} ({sc5_pct:.1f}%) | Last 100: {sc100_m}/{sc100_o} ({sc100_pct:.1f}%)<br>
      <b>Lost Balls</b> — Last 5: Tee {lb5_t} / Appr {lb5_a} / Total {lb5_tot} <span style="color:#aaa;">(avg {lb5_per_round:.2f}/rnd, {lb5_per18:.2f}/18)</span><br>
      <b>Lost Balls</b> — Last 100: Tee {lb100_t} / Appr {lb100_a} / Total {lb100_tot} <span style="color:#aaa;">(avg {lb100_per_round:.2f}/rnd, {lb100_per18:.2f}/18)</span>
    </div>
    """
    st.markdown(comparisons_html, unsafe_allow_html=True)

    comp_html = f"""
    <div style="margin-top:8px; padding:10px; background:#2a2a2a; border-radius:10px;">
      <div style="font-weight:700; margin-bottom:6px;">📊 Quick Comparisons</div>
      <table style="width:100%; border-collapse:collapse; font-size:12.5px;">
        <thead>
          <tr style="text-align:left; background:#333;">
            <th style="padding:6px;">Metric</th>
            <th style="padding:6px; text-align:center;">Current</th>
            <th style="padding:6px; text-align:center;">Prev 5 (value)</th>
            <th style="padding:6px; text-align:center;">Δ vs Prev 5</th>
            <th style="padding:6px; text-align:center;">Last 100 (value)</th>
            <th style="padding:6px; text-align:center;">Δ vs Last 100</th>
          </tr>
        </thead>
        <tbody>
          <tr style="background:#2f2f2f;">
            <td style="padding:6px; font-weight:600;">Putts / Hole</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(curr_pph)}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(prev5_pph)}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_pph, prev5_pph)}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(last100_pph)}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_pph, last100_pph)}</td>
          </tr>
          <tr style="background:#282828;">
            <td style="padding:6px; font-weight:600;">Putts / Round</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(curr_ppr)}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(prev5_ppr)}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_ppr, prev5_ppr)}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(last100_ppr_equiv)}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_ppr, last100_ppr_equiv)}</td>
          </tr>
          <tr style="background:#2f2f2f;">
            <td style="padding:6px; font-weight:600;">GIR %</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(curr_gir, '%')}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(prev5_gir, '%')}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_gir, prev5_gir, '%')}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(last100_gir, '%')}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_gir, last100_gir, '%')}</td>
          </tr>
          <tr style="background:#282828;">
            <td style="padding:6px; font-weight:600;">Fairway % (P4/P5)</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(curr_fw, '%')}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(prev5_fw, '%')}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_fw, prev5_fw, '%')}</td>
            <td style="text-align:center; padding:6px;">{_fmt_val(last100_fw, '%')}</td>
            <td style="text-align:center; padding:6px;">{_delta_str(curr_fw, last100_fw, '%')}</td>
          </tr>
        </tbody>
      </table>
      <div style="margin-top:4px; font-size:12px; color:#aaa;">
        Putts / Round for “Last 100” is a round-equivalent = 18 × (putts/hole over last 100 holes).
      </div>
    </div>
    """
    st.markdown(comp_html, unsafe_allow_html=True)

    with st.expander("🔎 Debug: Strokes Gained Style (Beta)", expanded=False):
        sg_dbg = build_sg_style_insights(df, round_data, compare_mode="All Time")
        st.write({
            "compare_mode": sg_dbg["compare_mode"],
            "approach_sg": round(sg_dbg["approach_sg"], 3),
            "putting_sg": round(sg_dbg["putting_sg"], 3),
            "short_game_sg": round(sg_dbg["short_game_sg"], 3),
            "total_sg": round(sg_dbg["total_sg"], 3),
            "approach_attempts": sg_dbg["approach_attempts"],
            "putting_attempts": sg_dbg["putting_attempts"],
            "short_game_attempts": sg_dbg["short_game_attempts"],
            "approach_extra": round(sg_dbg["approach_extra"], 3),
            "putting_extra": round(sg_dbg["putting_extra"], 3),
            "short_game_extra": round(sg_dbg["short_game_extra"], 3),
        })

    _delta = pd.to_numeric(round_data["Hole Score"], errors="coerce") - pd.to_numeric(round_data["Par"], errors="coerce")
    _holes = round_data["Hole"].astype(int)
    _labels = round_data.get("Score Label", pd.Series([""] * len(round_data), index=round_data.index))

    def _fmt_delta(n: float) -> str:
        n = int(n)
        return "E" if n == 0 else (f"+{n}" if n > 0 else f"{n}")

    best_idx = _delta.idxmin()
    worst_idx = _delta.idxmax()
    best_hole_num = int(round_data.loc[best_idx, "Hole"])
    best_delta_str = _fmt_delta(_delta.loc[best_idx])
    best_label = str(_labels.loc[best_idx])

    worst_hole_num = int(round_data.loc[worst_idx, "Hole"])
    worst_delta_str = _fmt_delta(_delta.loc[worst_idx])
    worst_label = str(_labels.loc[worst_idx])

    par_or_better = (_delta <= 0).tolist()
    hole_seq = _holes.tolist()

    max_len = cur_len = 0
    best_start = best_end = cur_start = None
    for ok, h in zip(par_or_better, hole_seq):
        if ok:
            cur_len += 1
            if cur_len == 1:
                cur_start = h
            if cur_len > max_len:
                max_len, best_start, best_end = cur_len, cur_start, h
        else:
            cur_len = 0

    streak_text = f"{max_len} holes (H{best_start}–H{best_end})" if max_len else "—"

    callouts_html = f"""
    <div style="margin-top:8px; padding:10px 12px; background:#2a2a2a; border-radius:10px; line-height:1.5;">
      <b>⭐ Best Hole:</b> H{best_hole_num} ({best_delta_str}) — {best_label}<br>
      <b>⚠️ Worst Hole:</b> H{worst_hole_num} ({worst_delta_str}) — {worst_label}<br>
      <b>🔗 Longest Par-or-Better Streak:</b> {streak_text}
    </div>
    """
    st.markdown(callouts_html, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 📊 Score Mix")
    order = ["Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]
    counts = [int(score_type_counts.get(k, 0)) for k in order]
    total = sum(counts) or 1

    df_mix = pd.DataFrame({
        "Category": order,
        "Count": counts,
        "Percent": [c / total * 100 for c in counts],
        "Group": ["All Holes"] * len(order)
    })

    df_plot = df_mix[df_mix["Count"] > 0].copy()
    if df_plot.empty:
        df_plot = df_mix.copy()

    color_scale = alt.Scale(domain=order, range=["#71c7ec", "#64dfb5", "#bdbdbd", "#f2c14e", "#ee6c4d", "#b23a48"])

    base = (
        alt.Chart(df_plot)
        .transform_calculate(pct='round(datum.Percent * 10) / 10')
        .transform_stack(stack='Count', as_=['start', 'end'], groupby=['Group'])
        .transform_calculate(mid='(datum.start + datum.end) / 2')
    )

    bar = base.mark_bar(height=34).encode(
        y=alt.Y("Group:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("end:Q", stack=None, axis=None),
        x2="start:Q",
        color=alt.Color("Category:N", scale=color_scale, legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("Category:N"), alt.Tooltip("Count:Q", title="Holes"), alt.Tooltip("pct:Q", title="% of Round", format=".1f")]
    )

    text = (
        base.mark_text(baseline="middle", dy=0, fontWeight="bold")
        .encode(y="Group:N", x="mid:Q", text=alt.Text("label:N"), opacity=alt.condition("datum.Percent < 8", alt.value(0), alt.value(1)))
        .transform_calculate(label='datum.Category + " " + (format(datum.pct, ".1f")) + "%"')
    )

    st.altair_chart((bar + text).configure_view(stroke=None).configure_axis(grid=False, domain=False), use_container_width=True)
    counts_line = " • ".join(f"{row.Category}: {row.Count} ({row.Percent:.1f}%)" for _, row in df_mix.iterrows())
    st.caption(counts_line)
    st.markdown(summary_details_html, unsafe_allow_html=True)

    st.markdown(benchmarks_html, unsafe_allow_html=True)

    st.markdown("#### Hole-by-Hole (Score vs Par)")
    df_line = pd.DataFrame({
        "Hole": round_data["Hole"].astype(int),
        "Delta": (pd.to_numeric(round_data["Hole Score"], errors="coerce") - pd.to_numeric(round_data["Par"], errors="coerce")).astype(int)
    }).sort_values("Hole")

    delta_min = int(df_line["Delta"].min())
    delta_max = int(df_line["Delta"].max())
    if delta_min == delta_max:
        delta_min -= 1
        delta_max += 1
    tick_vals = list(range(delta_min, delta_max + 1))

    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(opacity=0.4).encode(y="y:Q")
    line = alt.Chart(df_line).mark_line().encode(
        x=alt.X("Hole:O", sort=None, axis=alt.Axis(title=None)),
        y=alt.Y("Delta:Q", scale=alt.Scale(domain=[delta_min, delta_max], nice=False, clamp=True), axis=alt.Axis(title="To Par", values=tick_vals, format="d", tickCount=len(tick_vals)))
    )
    pts = alt.Chart(df_line).mark_point(size=64).encode(
        x="Hole:O", y="Delta:Q",
        color=alt.condition("datum.Delta <= 0", alt.value("#64dfb5"), alt.value("#ee6c4d")),
        tooltip=[alt.Tooltip("Hole:O"), alt.Tooltip("Delta:Q", title="To Par", format="d")]
    )
    st.altair_chart(zero + line + pts, use_container_width=True)

    with st.expander("🔎 Debug: Score-to-Par audit (current vs last 5 rounds / last 100 holes)", expanded=False):
        audit = round_data.copy()
        audit["ParN"] = pd.to_numeric(audit["Par"], errors="coerce").fillna(0)
        audit["ScoreN"] = pd.to_numeric(audit["Hole Score"], errors="coerce").fillna(0)
        audit["Delta"] = audit["ScoreN"] - audit["ParN"]
        st.dataframe(
            audit.sort_values("Hole")[["Hole", "ParN", "ScoreN", "Delta"]].rename(columns={"ParN": "Par", "ScoreN": "Score"}),
            use_container_width=True
        )
        st.write(f"**Current totals** — Score: {int(audit['ScoreN'].sum())} | Par: {int(audit['ParN'].sum())} | Δ (Score−Par): {int(audit['Delta'].sum()):+d}")

    render_trivia_block("scorecard", "Scorecard trivia")

    download_html = f"""
    <!doctype html>
    <html>
    <head>
    <meta charset="utf-8">
    <title>{player} — {course} — {date}</title>
    </head>
    <body style="background:#1e1e1e;color:#eaeaea;font-family:Segoe UI, Roboto, Arial,sans-serif;">
    <h2>{player} &middot; {course} &middot; {date}</h2>
    {table_html}
    <div style="margin-top:12px">
      {summary_header_html}
      {cards_html}
      {summary_details_html}
    </div>
    </body>
    </html>
    """.strip()

    st.download_button(
        "⬇️ Download Round (HTML)",
        data=download_html.encode("utf-8"),
        file_name=f"{player}_{course}_{date.replace(',', '')}_scorecard.html".replace(" ", "_"),
        mime="text/html"
    )

with tab_compare:
    render_compare_tab(df)


with tab_analysis:
    st.markdown("### 🎯 Ball Striking")
    st.caption("Cleaned-up approach analysis: player comparisons first, deeper filter tools below.")

    analysis_base_player_df = df[df["Player Name"] == player].copy() if "Player Name" in df else df.copy()

    st.markdown("#### Player Compare — Approach by Distance")
    ac1, ac2 = st.columns(2)
    with ac1:
        show_approach_round_line = st.checkbox("Show selected round line", value=True, key="show_approach_round_line")
    with ac2:
        show_approach_baseline_line = st.checkbox("Show baseline line", value=True, key="show_approach_baseline_line")

    approach_player_options = sorted([x for x in df["Player Name"].dropna().unique().tolist() if str(x).strip() != ""]) if "Player Name" in df.columns else []
    default_approach_compare_players = [player] if str(player) in [str(x) for x in approach_player_options] else []
    selected_approach_compare_players = st.multiselect(
        "Compare player approach profiles",
        options=approach_player_options,
        default=default_approach_compare_players,
        key="selected_approach_compare_players",
        help="Select 1–4 players. Uncheck 'Show selected round line' for a cleaner player-vs-player view.",
    )
    if len(selected_approach_compare_players) > 4:
        st.warning("Showing the first 4 comparison players to keep the approach charts readable.")
        selected_approach_compare_players = selected_approach_compare_players[:4]

    compare_mode_analysis = st.radio("Approach baseline:", _compare_mode_options(), horizontal=True, key="analysis_compare_mode")
    compare_label_analysis = _compare_mode_display_label(compare_mode_analysis)
    benchmark_df_analysis = build_benchmark_df(df, round_data, compare_mode_analysis, selected_baseline_round_ids)
    if compare_mode_analysis == "Selected Rounds" and not selected_baseline_round_ids:
        st.info("Pick one or more rounds in 'Custom Round Baseline' above to activate the Selected Rounds baseline.")

    st.markdown("#### Player Compare Filters")
    fpc1, fpc2, fpc3, fpc4 = st.columns(4)
    with fpc1:
        analysis_compare_par_filter = st.selectbox(
            "Par filter",
            ["All", "Par 3", "Par 4", "Par 5"],
            index=0,
            key="analysis_compare_par_filter",
        )
    with fpc2:
        fairway_disabled = analysis_compare_par_filter == "Par 3"
        analysis_compare_fairway_filter = st.selectbox(
            "Fairway filter",
            ["All", "Fairway Yes", "Fairway No"],
            index=0,
            key="analysis_compare_fairway_filter",
            disabled=fairway_disabled,
            help="Fairway filter is ignored for Par 3s.",
        )
    with fpc3:
        approach_compare_min_attempts = st.slider(
            "Minimum attempts",
            1, 20, 3, 1,
            key="analysis_player_compare_min_attempts",
        )
    with fpc4:
        approach_compare_max_clubs = st.slider(
            "Max clubs to show",
            4, 20, 12, 1,
            key="analysis_player_compare_max_clubs",
        )

    analysis_compare_par_vals = None
    if analysis_compare_par_filter == "Par 3":
        analysis_compare_par_vals = [3]
    elif analysis_compare_par_filter == "Par 4":
        analysis_compare_par_vals = [4]
    elif analysis_compare_par_filter == "Par 5":
        analysis_compare_par_vals = [5]

    analysis_compare_fairway_vals = None
    if analysis_compare_par_filter != "Par 3":
        if analysis_compare_fairway_filter == "Fairway Yes":
            analysis_compare_fairway_vals = [1]
            if analysis_compare_par_vals is None:
                analysis_compare_par_vals = [4, 5]
        elif analysis_compare_fairway_filter == "Fairway No":
            analysis_compare_fairway_vals = [0]
            if analysis_compare_par_vals is None:
                analysis_compare_par_vals = [4, 5]

    if analysis_compare_par_filter == "Par 3":
        st.caption("Par 3 selected: fairway filter is ignored because Par 3s do not have fairway attempts.")
    elif analysis_compare_fairway_filter != "All" and analysis_compare_par_filter == "All":
        st.caption("Fairway filter selected: comparison is limited to Par 4 and Par 5 approach shots.")

    def _analysis_compare_filtered_frame(frame):
        return apply_approach_filters(
            frame,
            par_vals=analysis_compare_par_vals,
            fairway_vals=analysis_compare_fairway_vals,
        )

    round_analysis_compare_frame = _analysis_compare_filtered_frame(round_data)
    baseline_analysis_compare_frame = _analysis_compare_filtered_frame(benchmark_df_analysis)

    approach_overlay_summaries = []
    for compare_player in selected_approach_compare_players:
        player_frame = df[df["Player Name"] == compare_player].copy() if "Player Name" in df.columns else pd.DataFrame()
        player_filtered_frame = _analysis_compare_filtered_frame(player_frame)
        approach_overlay_summaries.append((str(compare_player), summarize_approach_bucket_for_chart(player_filtered_frame)))

    round_approach_distance_summary = summarize_approach_bucket_for_chart(round_analysis_compare_frame)
    baseline_approach_distance_summary = summarize_approach_bucket_for_chart(baseline_analysis_compare_frame)

    render_approach_distance_player_compare(
        round_approach_distance_summary,
        baseline_approach_distance_summary,
        APPROACH_BUCKET_ORDER,
        compare_label=compare_label_analysis,
        overlay_summaries=approach_overlay_summaries,
        show_round=show_approach_round_line,
        show_baseline=show_approach_baseline_line,
        title="Approach Distance Control — Player Comparison",
    )

    st.markdown("#### Player Compare — GIR % Charts")
    approach_distance_series_defs = []
    approach_club_series_defs = []
    if show_approach_round_line:
        approach_distance_series_defs.append(("This Round", round_approach_distance_summary))
        approach_club_series_defs.append(("This Round", summarize_approach_club_for_chart(round_analysis_compare_frame, min_attempts=1)))
    if show_approach_baseline_line:
        approach_distance_series_defs.append((compare_label_analysis, baseline_approach_distance_summary))
        approach_club_series_defs.append((compare_label_analysis, summarize_approach_club_for_chart(baseline_analysis_compare_frame, min_attempts=1)))
    for compare_player in selected_approach_compare_players:
        player_frame = df[df["Player Name"] == compare_player].copy() if "Player Name" in df.columns else pd.DataFrame()
        player_filtered_frame = _analysis_compare_filtered_frame(player_frame)
        approach_distance_series_defs.append((str(compare_player), summarize_approach_bucket_for_chart(player_filtered_frame)))
        approach_club_series_defs.append((str(compare_player), summarize_approach_club_for_chart(player_filtered_frame, min_attempts=1)))

    render_approach_gir_player_compare_bars(
        approach_distance_series_defs,
        approach_club_series_defs,
        APPROACH_BUCKET_ORDER,
        min_attempts=approach_compare_min_attempts,
        max_clubs=approach_compare_max_clubs,
    )

    st.markdown("#### Ball-Striking Filter View (Club / Yardage / Course)")
    filter_options_base = build_shot_pattern_frame(df, player)
    if not filter_options_base.empty:
        fa1, fa2, fa3 = st.columns(3)
        with fa1:
            yard_options = [b for b in APPROACH_BUCKET_ORDER if b in filter_options_base["Approach Bucket"].dropna().astype(str).tolist()]
            analysis_yards = st.multiselect("Yardage Buckets", yard_options, default=[], key="analysis_filter_yards_multi")
        with fa2:
            club_options = sorted([c for c in filter_options_base["Approach Club"].dropna().unique().tolist() if str(c).strip() != ""])
            analysis_clubs = st.multiselect("Clubs", club_options, default=[], key="analysis_filter_clubs_multi")
        with fa3:
            course_options = sorted([c for c in filter_options_base["Course Name"].dropna().unique().tolist()]) if "Course Name" in filter_options_base else []
            analysis_courses = st.multiselect("Courses", course_options, default=[], key="analysis_filter_courses_multi")

        fa4, fa5, fa6 = st.columns(3)
        with fa4:
            fairway_labels = {1: "Fairway Hit", 0: "Fairway Miss"}
            analysis_fairway_labels = st.multiselect("Fairway", list(fairway_labels.values()), default=[], key="analysis_filter_fairway_multi")
            analysis_fairway_vals = [k for k, v in fairway_labels.items() if v in analysis_fairway_labels]
        with fa5:
            analysis_pars = st.multiselect("Hole Par", [3, 4, 5], default=[], key="analysis_filter_par_multi")
        with fa6:
            analysis_gir_mode = st.selectbox("GIR Filter", ["All", "GIR Only", "Missed GIR Only"], index=0, key="analysis_filter_gir_mode")

        round_filter_view = apply_approach_filters(
            round_data,
            yard_buckets=analysis_yards,
            clubs=analysis_clubs,
            courses=analysis_courses,
            fairway_vals=analysis_fairway_vals,
            par_vals=analysis_pars,
            gir_mode=analysis_gir_mode,
        )
        bench_filter_view = apply_approach_filters(
            benchmark_df_analysis,
            yard_buckets=analysis_yards,
            clubs=analysis_clubs,
            courses=analysis_courses,
            fairway_vals=analysis_fairway_vals,
            par_vals=analysis_pars,
            gir_mode=analysis_gir_mode,
        )
        analysis_all_player_view = apply_approach_filters(
            analysis_base_player_df,
            yard_buckets=analysis_yards,
            clubs=analysis_clubs,
            courses=analysis_courses,
            fairway_vals=analysis_fairway_vals,
            par_vals=analysis_pars,
            gir_mode=analysis_gir_mode,
        )

        filt_metrics = build_filtered_approach_metrics(round_filter_view)
        bench_metrics = build_filtered_approach_metrics(bench_filter_view)

        fb1, fb2, fb3, fb4 = st.columns(4)
        with fb1:
            st.metric("Filtered Attempts", filt_metrics["attempts"], f'{filt_metrics["attempts"] - bench_metrics["attempts"]:+d}')
        with fb2:
            st.metric("Filtered GIR", f'{filt_metrics["gir"]}/{filt_metrics["attempts"]}', f'{filt_metrics["gir_pct"] - bench_metrics["gir_pct"]:+.1f}%')
        with fb3:
            st.metric("Avg Proximity", f'{filt_metrics["avg_prox"]:.1f} ft', f'{bench_metrics["avg_prox"] - filt_metrics["avg_prox"]:+.1f} ft')
        with fb4:
            st.metric("Inside 15 ft", f'{filt_metrics["inside15"]}/{filt_metrics["attempts"]}', f'{filt_metrics["inside15_pct"] - bench_metrics["inside15_pct"]:+.1f}%')

        prox_compare = build_approach_proximity_compare(round_filter_view, bench_filter_view)
        if not prox_compare.empty:
            prox_long = pd.DataFrame({
                "Bucket": list(prox_compare["Bucket"].astype(str)) * 2,
                "Series": ["Round"] * len(prox_compare) + [compare_label_analysis] * len(prox_compare),
                "Made": list(pd.to_numeric(prox_compare["Round Made"], errors="coerce").fillna(0).astype(int))
                        + list(pd.to_numeric(prox_compare["Baseline Made"], errors="coerce").fillna(0).astype(int)),
                "Attempts": list(pd.to_numeric(prox_compare["Round Attempts"], errors="coerce").fillna(0).astype(int))
                            + list(pd.to_numeric(prox_compare["Baseline Attempts"], errors="coerce").fillna(0).astype(int)),
                "Pct": list(pd.to_numeric(prox_compare["Round Pct"], errors="coerce").fillna(0.0))
                       + list(pd.to_numeric(prox_compare["Baseline Pct"], errors="coerce").fillna(0.0)),
                "Label": [
                    f"{_safe_int_scalar(r.get('Round Made', 0))}/{_safe_int_scalar(r.get('Round Attempts', 0))} {_safe_float_scalar(r.get('Round Pct', 0)):.0f}%"
                    for _, r in prox_compare.iterrows()
                ] + [
                    f"{_safe_int_scalar(r.get('Baseline Made', 0))}/{_safe_int_scalar(r.get('Baseline Attempts', 0))} {_safe_float_scalar(r.get('Baseline Pct', 0)):.0f}%"
                    for _, r in prox_compare.iterrows()
                ],
            })
            st.markdown("##### Filtered Proximity Distribution")
            render_paired_compare_bars(prox_long, "Bucket", PUTT_BUCKET_ORDER, compare_label_analysis, "Proximity Bucket", "Share %")
            st.dataframe(prox_compare, use_container_width=True, hide_index=True)
        else:
            st.info("No filtered proximity distribution available.")

        st.markdown("##### Best Clubs")
        best_club_min_attempts = st.slider("Minimum attempts for Best Clubs", 1, 20, 3, 1, key="analysis_best_club_min")
        best_clubs_df = summarize_approach_by_club(analysis_all_player_view, min_attempts=best_club_min_attempts)
        if not best_clubs_df.empty:
            best_clubs_df = best_clubs_df.rename(columns={"Pct": "GIR %", "AvgProx": "Avg Prox"})
            st.dataframe(best_clubs_df, use_container_width=True, hide_index=True)
        else:
            st.info("No club rows qualify for Best Clubs with the current filters.")

        st.markdown("##### Best Yardage Windows")
        best_yard_min_attempts = st.slider("Minimum attempts for Best Yardage Windows", 1, 20, 3, 1, key="analysis_best_yard_min")
        best_yard_df = summarize_approach_by_bucket(analysis_all_player_view)
        if not best_yard_df.empty:
            best_yard_df = best_yard_df[pd.to_numeric(best_yard_df["Attempts"], errors="coerce").fillna(0) >= best_yard_min_attempts].copy()
            best_yard_df = best_yard_df.sort_values(["Pct", "Attempts", "Bucket"], ascending=[False, False, True]).rename(columns={"Pct": "GIR %", "AvgProx": "Avg Prox"})
            st.dataframe(best_yard_df, use_container_width=True, hide_index=True)
        else:
            st.info("No yardage rows qualify for Best Yardage Windows with the current filters.")
    else:
        st.info("No analysis filter-view data available.")

    st.markdown("#### Shot Pattern Dashboard")
    shot_options_base = build_shot_pattern_frame(df, player)
    if not shot_options_base.empty:
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            shot_bucket_options = [b for b in APPROACH_BUCKET_ORDER if b in shot_options_base["Approach Bucket"].dropna().astype(str).tolist()]
            shot_buckets = st.multiselect("Distance Bucket Filter", shot_bucket_options, default=[], key="analysis_shot_bucket_filter_multi")
        with sp2:
            club_vals = sorted([c for c in shot_options_base["Approach Club"].dropna().unique().tolist() if str(c).strip() != ""])
            shot_clubs = st.multiselect("Club Filter", club_vals, default=[], key="analysis_shot_club_filter_multi")
        with sp3:
            course_vals = sorted([c for c in shot_options_base["Course Name"].dropna().unique().tolist()]) if "Course Name" in shot_options_base else []
            shot_courses = st.multiselect("Course Filter", course_vals, default=[], key="analysis_shot_course_filter_multi")

        sp4, sp5, sp6 = st.columns(3)
        with sp4:
            fairway_labels = {1: "Fairway Hit", 0: "Fairway Miss"}
            shot_fairway_labels = st.multiselect("Fairway Filter", list(fairway_labels.values()), default=[], key="analysis_shot_fairway_filter_multi")
            shot_fairway_vals = [k for k, v in fairway_labels.items() if v in shot_fairway_labels]
        with sp5:
            shot_pars = st.multiselect("Hole Par Filter", [3,4,5], default=[], key="analysis_shot_par_filter_multi")
        with sp6:
            shot_gir_mode = st.selectbox("GIR Filter", ["All", "GIR Only", "Missed GIR Only"], index=0, key="analysis_shot_gir_mode")

        round_shot_view = apply_approach_filters(
            round_data,
            yard_buckets=shot_buckets,
            clubs=shot_clubs,
            courses=shot_courses,
            fairway_vals=shot_fairway_vals,
            par_vals=shot_pars,
            gir_mode=shot_gir_mode,
        )
        bench_shot_view = apply_approach_filters(
            benchmark_df_analysis,
            yard_buckets=shot_buckets,
            clubs=shot_clubs,
            courses=shot_courses,
            fairway_vals=shot_fairway_vals,
            par_vals=shot_pars,
            gir_mode=shot_gir_mode,
        )

        if not round_shot_view.empty or not bench_shot_view.empty:
            shot_metrics = build_filtered_approach_metrics(round_shot_view)
            shot_bench_metrics = build_filtered_approach_metrics(bench_shot_view)
            a1, a2, a3, a4 = st.columns(4)
            with a1:
                st.metric("Filtered Attempts", shot_metrics["attempts"], f'{shot_metrics["attempts"] - shot_bench_metrics["attempts"]:+d}')
            with a2:
                st.metric("Filtered GIR", f'{shot_metrics["gir"]}/{shot_metrics["attempts"]}', f'{shot_metrics["gir_pct"] - shot_bench_metrics["gir_pct"]:+.1f}%')
            with a3:
                st.metric("Avg Proximity", f'{shot_metrics["avg_prox"]:.1f} ft', f'{shot_bench_metrics["avg_prox"] - shot_metrics["avg_prox"]:+.1f} ft')
            with a4:
                st.metric("Inside 15 ft", f'{shot_metrics["inside15"]}/{shot_metrics["attempts"]}', f'{shot_metrics["inside15_pct"] - shot_bench_metrics["inside15_pct"]:+.1f}%')

            round_shot_summary = (
                round_shot_view.groupby(["Approach Bucket"], as_index=False)
                .agg(Attempts=("Approach GIR Flag", "size"), Made=("Approach GIR Flag", "sum"), AvgProx=("Approach Proximity", "mean"))
                .rename(columns={"Approach Bucket": "Bucket"})
            ) if not round_shot_view.empty else pd.DataFrame(columns=["Bucket","Attempts","Made","AvgProx"])
            if not round_shot_summary.empty:
                round_shot_summary["Pct"] = (round_shot_summary["Made"] / round_shot_summary["Attempts"] * 100).round(1)
                round_shot_summary["Label"] = round_shot_summary.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%", axis=1)

            bench_shot_summary = (
                bench_shot_view.groupby(["Approach Bucket"], as_index=False)
                .agg(Attempts=("Approach GIR Flag", "size"), Made=("Approach GIR Flag", "sum"), AvgProx=("Approach Proximity", "mean"))
                .rename(columns={"Approach Bucket": "Bucket"})
            ) if not bench_shot_view.empty else pd.DataFrame(columns=["Bucket","Attempts","Made","AvgProx"])
            if not bench_shot_summary.empty:
                bench_shot_summary["Pct"] = (bench_shot_summary["Made"] / bench_shot_summary["Attempts"] * 100).round(1)
                bench_shot_summary["Label"] = bench_shot_summary.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} {r['Pct']:.0f}%", axis=1)

            shot_long = build_compare_long(round_shot_summary, bench_shot_summary, "Bucket", round_label="Round", bench_label=compare_label_analysis)
            if not shot_long.empty:
                render_paired_compare_bars(shot_long, "Bucket", APPROACH_BUCKET_ORDER, compare_label_analysis, "Distance Bucket", "GIR %")
            else:
                st.info("No shot pattern comparison bars available.")


            st.dataframe(
                round_shot_view.sort_values(["Date Played", "Hole"], ascending=[False, True])[[
                    "Date Played", "Course Name", "Hole", "Approach Club", "Approach Distance", "Approach Bucket", "Approach GIR Flag", "Approach Miss Direction Clean", "Approach Proximity"
                ]].rename(columns={
                    "Approach Club": "Club",
                    "Approach Distance": "Distance",
                    "Approach Bucket": "Bucket",
                    "Approach GIR Flag": "GIR",
                    "Approach Miss Direction Clean": "Miss Dir",
                    "Approach Proximity": "Prox",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No shot pattern rows match the current filters.")
    else:
        st.info("No shot pattern data available for this player.")

    render_trivia_block("analysis", "Ball Striking tab trivia")

with tab_putting:
    st.markdown("### 🏌️ Putting Breakdown")
    compare_mode_putt = st.radio("Compare this round against:", _compare_mode_options(), horizontal=True, key="putting_compare_mode")
    benchmark_df_putt = build_benchmark_df(df, round_data, compare_mode_putt, selected_baseline_round_ids)
    if compare_mode_putt == "Selected Rounds" and not selected_baseline_round_ids:
        st.info("Pick one or more rounds in 'Custom Round Baseline' above to activate the Selected Rounds baseline.")

    st.markdown("##### Putting chart display")
    putt_display_c1, putt_display_c2 = st.columns(2)
    with putt_display_c1:
        show_putting_round_line = st.checkbox("Show selected round line", value=True, key="show_putting_round_line")
    with putt_display_c2:
        show_putting_baseline_line = st.checkbox("Show baseline line", value=True, key="show_putting_baseline_line")

    putt_player_options = sorted([x for x in df["Player Name"].dropna().unique().tolist() if str(x).strip() != ""]) if "Player Name" in df.columns else []
    default_putt_compare_players = []
    if str(player) in [str(x) for x in putt_player_options]:
        default_putt_compare_players = [player]

    selected_putting_compare_players = st.multiselect(
        "Compare player putting profiles",
        options=putt_player_options,
        default=default_putt_compare_players,
        key="selected_putting_compare_players",
        help="Select 1–4 players. Uncheck 'Show selected round line' above for a cleaner player-vs-player view.",
    )

    if len(selected_putting_compare_players) > 4:
        st.warning("Showing the first 4 comparison players to keep the putting charts readable.")
        selected_putting_compare_players = selected_putting_compare_players[:4]

    overlay_fingerprint_items = []
    for compare_player in selected_putting_compare_players:
        player_frame = df[df["Player Name"] == compare_player].copy() if "Player Name" in df.columns else pd.DataFrame()
        overlay_fingerprint_items.append((str(compare_player), build_putting_skill_fingerprint(player_frame)))

    round_putt = prepare_putting_frame(round_data)
    bench_putt = prepare_putting_frame(benchmark_df_putt)

    round_attempts = int(len(round_putt))
    round_made = int(round_putt["Putt Made Flag"].sum()) if not round_putt.empty else 0
    round_pct = (round_made / round_attempts * 100) if round_attempts else 0.0
    bench_attempts = int(len(bench_putt))
    bench_made = int(bench_putt["Putt Made Flag"].sum()) if not bench_putt.empty else 0
    bench_pct = (bench_made / bench_attempts * 100) if bench_attempts else 0.0

    zone_stats = build_putting_zone_stats(round_data)
    bench_zone_stats = build_putting_zone_stats(benchmark_df_putt)
    finger_insights = build_putting_fingerprint_insights(round_data)
    baseline_putt = build_putting_baseline_snapshot(benchmark_df_putt)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Round Putt Attempts", round_attempts)
    with c2:
        st.metric("Round 1-Putts", f"{round_made}/{round_attempts}", f"{round_pct:.1f}%")
    with c3:
        st.metric(f"{compare_mode_putt} Attempts", bench_attempts)
    with c4:
        st.metric(f"{compare_mode_putt} 1-Putts", f"{bench_made}/{bench_attempts}", f"{bench_pct:.1f}%")

    z1, z2, z3, z4 = st.columns(4)
    with z1:
        st.metric("0–6 ft", f"{zone_stats['inside6']['made']}/{zone_stats['inside6']['attempts']}", f"{zone_stats['inside6']['pct']:.1f}%")
    with z2:
        st.metric("6–10 ft", f"{zone_stats['six10']['made']}/{zone_stats['six10']['attempts']}", f"{zone_stats['six10']['pct']:.1f}%")
    with z3:
        st.metric("10–20 ft", f"{zone_stats['ten20']['made']}/{zone_stats['ten20']['attempts']}", f"{zone_stats['ten20']['pct']:.1f}%")
    with z4:
        st.metric("Lag Putts 20+ ft", f"{zone_stats['lag20']['attempts']} tries", f"3P {zone_stats['lag20']['three_putt_pct']:.1f}%")

    putt_impact = build_putting_round_impact(round_data, benchmark_df_putt)
    total_putts_round = int(pd.to_numeric(_safe_col(round_data, "Putts", 0), errors="coerce").fillna(0).sum())
    three_putts_round = int((pd.to_numeric(_safe_col(round_data, "Putts", 0), errors="coerce").fillna(0) >= 3).sum())
    round_holes_played = int(len(round_data)) if round_data is not None else 0
    three_putt_pct_round = (three_putts_round / round_holes_played * 100.0) if round_holes_played else 0.0
    sg_color = "#64dfb5" if putt_impact["sg_putting"] >= 0 else "#ee6c4d"

    compare_label = _baseline_label(compare_mode_putt)
    compare_label_per18 = _baseline_label(compare_mode_putt, per18=True)

    total_putts_delta = total_putts_round - baseline_putt["total_putts_per18"]
    one_putt_delta = putt_impact["pct"] - baseline_putt["one_putt_pct"]
    three_putts_delta = three_putts_round - baseline_putt["three_putts_per18"]
    total_putts_arrow = _delta_arrow(total_putts_delta, higher_better=False, tol=0.1)
    one_putt_arrow = _delta_arrow(one_putt_delta, higher_better=True, tol=0.25)
    three_putts_arrow = _delta_arrow(three_putts_delta, higher_better=False, tol=0.1)

    st.markdown(
        f"""
        <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.6;">
          <b>🧾 Putting Round Summary</b><br>
          Total Putts: {total_putts_round} <span style="color:#aaa;">vs {baseline_putt['total_putts_per18']:.1f} {compare_label_per18}</span> {total_putts_arrow}<br>
          1-Putts: {putt_impact['made']}/{putt_impact['attempts']} ({putt_impact['pct']:.1f}%) <span style="color:#aaa;">vs {baseline_putt['one_putt_pct']:.1f}% {compare_label}</span> {one_putt_arrow}<br>
          3-Putts: {three_putts_round}/{round_holes_played} ({three_putt_pct_round:.1f}%) <span style="color:#aaa;">vs {baseline_putt['three_putts_per18']:.1f} {compare_label_per18} • {baseline_putt['three_putt_pct']:.1f}% {compare_label}</span> {three_putts_arrow}<br>
          Putting SG vs {compare_label}: <span style="font-weight:800; color:{sg_color};">{putt_impact['sg_putting']:+.2f}</span>
          <span style="color:#aaa;">(Expected Makes: {putt_impact['expected_makes']:.1f} | Actual Makes: {putt_impact['actual_makes']:.0f})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    fingerprint_df = build_putting_skill_fingerprint(round_data)
    bench_fingerprint_df = build_putting_skill_fingerprint(benchmark_df_putt)

    render_putting_distance_comparison_dashboard(
        fingerprint_df,
        bench_fingerprint_df,
        PUTT_BUCKET_ORDER,
        compare_label=compare_mode_putt,
        overlay_fps=overlay_fingerprint_items,
        show_round=show_putting_round_line,
        show_baseline=show_putting_baseline_line,
        title="Putting Distance Control — Player Comparison",
    )

    st.markdown("#### Putting Skill Fingerprint")
    if not fingerprint_df.empty:
        fingerprint_long = build_compare_long(
            fingerprint_df.rename(columns={"OnePutt": "Made", "MakePct": "Pct", "DisplayLabel": "Label"})[["Bucket", "Attempts", "Made", "Pct", "Label"]],
            bench_fingerprint_df.rename(columns={"OnePutt": "Made", "MakePct": "Pct", "DisplayLabel": "Label"})[["Bucket", "Attempts", "Made", "Pct", "Label"]] if not bench_fingerprint_df.empty else pd.DataFrame(columns=["Bucket", "Attempts", "Made", "Pct", "Label"]),
            "Bucket", round_label="Round", bench_label=compare_mode_putt
        )
        render_paired_compare_bars(fingerprint_long, "Bucket", PUTT_BUCKET_ORDER, compare_mode_putt, "Distance Range", "1-Putt %")

        fp_table = pd.merge(
            fingerprint_df[["Bucket", "Attempts", "OnePutt", "TwoPutt", "ThreePuttPlus", "MakePct", "TwoPuttPct", "ThreePuttPct", "AvgStartFt"]],
            bench_fingerprint_df[["Bucket", "MakePct", "ThreePuttPct"]].rename(columns={"MakePct": f"{compare_mode_putt} 1-Putt %", "ThreePuttPct": f"{compare_mode_putt} 3-Putt %"}),
            on="Bucket",
            how="left"
        )
        fp_table["Δ 1-Putt %"] = (pd.to_numeric(fp_table["MakePct"], errors="coerce").fillna(0) - pd.to_numeric(fp_table[f"{compare_mode_putt} 1-Putt %"], errors="coerce").fillna(0)).round(1)
        fp_table["1P Arrow"] = fp_table["Δ 1-Putt %"].apply(lambda x: _delta_arrow(x, higher_better=True, tol=0.25))
        fp_table["Δ 3-Putt %"] = (pd.to_numeric(fp_table["ThreePuttPct"], errors="coerce").fillna(0) - pd.to_numeric(fp_table[f"{compare_mode_putt} 3-Putt %"], errors="coerce").fillna(0)).round(1)
        fp_table["3P Arrow"] = fp_table["Δ 3-Putt %"].apply(lambda x: _delta_arrow(x, higher_better=False, tol=0.25))
        _render_golf_dataframe(
            fp_table.rename(columns={
                "OnePutt": "1-Putts",
                "TwoPutt": "2-Putts",
                "ThreePuttPlus": "3-Putts+",
                "MakePct": "Round 1-Putt %",
                "TwoPuttPct": "Round 2-Putt %",
                "ThreePuttPct": "Round 3-Putt %",
                "AvgStartFt": "Avg Start Ft",
            }),
            use_container_width=True,
            hide_index=True,
        )
        if finger_insights:
            st.markdown(
                f"""
                <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
                  <b>🧬 Putting Fingerprint Notes</b><br>
                  {'<br>'.join(finger_insights)}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No usable putting fingerprint data found for this round.")

    st.markdown("#### Lag Putting (20+ ft)")
    lag = zone_stats["lag20"]
    lag_attempts_delta = lag["attempts"] - baseline_putt["lag_attempts_avg"]
    lag_start_delta = lag["avg_start_ft"] - baseline_putt["lag_avg_start_ft"]
    lag_two_putt_delta = lag["two_putt_or_better_pct"] - baseline_putt["lag_two_putt_or_better_pct"]
    lag_three_putt_delta = lag["three_putt_pct"] - baseline_putt["lag_three_putt_pct"]
    st.markdown(
        f"""
        <div style="margin-top:4px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.6;">
          <b>Lag Putting Snapshot</b><br>
          Attempts: {lag['attempts']} <span style="color:#aaa;">vs {baseline_putt['lag_attempts_avg']:.1f} {compare_label}</span> {_delta_arrow(lag_attempts_delta, higher_better=False, tol=0.1)}<br>
          Avg start length: {lag['avg_start_ft']:.1f} ft <span style="color:#aaa;">vs {baseline_putt['lag_avg_start_ft']:.1f} ft {compare_label}</span> {_delta_arrow(lag_start_delta, higher_better=False, tol=0.25)}<br>
          2-putt or better: {lag['two_putt_or_better_pct']:.1f}% <span style="color:#aaa;">vs {baseline_putt['lag_two_putt_or_better_pct']:.1f}% {compare_label}</span> {_delta_arrow(lag_two_putt_delta, higher_better=True, tol=0.25)}<br>
          3-putt rate: {lag['three_putt_pct']:.1f}% <span style="color:#aaa;">vs {baseline_putt['lag_three_putt_pct']:.1f}% {compare_label}</span> {_delta_arrow(lag_three_putt_delta, higher_better=False, tol=0.25)}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### 3-Putts by Starting Distance")
    round_three_putt_bucket = summarize_three_putt_by_bucket(round_data)
    bench_three_putt_bucket = summarize_three_putt_by_bucket(benchmark_df_putt)
    three_putt_long = build_compare_long(round_three_putt_bucket, bench_three_putt_bucket, "Bucket", round_label="Round", bench_label=compare_mode_putt)
    if not three_putt_long.empty:
        render_paired_compare_bars(three_putt_long, "Bucket", PUTT_BUCKET_ORDER, compare_mode_putt, "Putt Range", "3-Putt %")
        three_putt_table = pd.merge(
            round_three_putt_bucket.rename(columns={"Attempts": "Round Attempts", "Made": "Round 3-Putts", "Pct": "Round 3-Putt %"}),
            bench_three_putt_bucket.rename(columns={"Attempts": f"{compare_mode_putt} Attempts", "Made": f"{compare_mode_putt} 3-Putts", "Pct": f"{compare_mode_putt} 3-Putt %"}),
            on="Bucket",
            how="outer"
        ).sort_values("Bucket")
        three_putt_table["Δ 3-Putt %"] = (pd.to_numeric(three_putt_table["Round 3-Putt %"], errors="coerce").fillna(0) - pd.to_numeric(three_putt_table[f"{compare_mode_putt} 3-Putt %"], errors="coerce").fillna(0)).round(1)
        three_putt_table["Arrow"] = three_putt_table["Δ 3-Putt %"].apply(lambda x: _delta_arrow(x, higher_better=False, tol=0.25))
        st.dataframe(three_putt_table, use_container_width=True, hide_index=True)
    else:
        st.info("No usable 3-putt distance data found for this round / comparison group.")

    st.markdown("#### Putts by Starting Distance (1-Putt %)")
    round_putt_bucket = summarize_putting_by_bucket(round_data)
    bench_putt_bucket = summarize_putting_by_bucket(benchmark_df_putt)

    putt_long = build_compare_long(round_putt_bucket, bench_putt_bucket, "Bucket", round_label="Round", bench_label=compare_mode_putt)
    if not putt_long.empty:
        render_paired_compare_bars(putt_long, "Bucket", PUTT_BUCKET_ORDER, compare_mode_putt, "Putt Range", "1-Putt %")

        putt_table = pd.merge(
            round_putt_bucket.rename(columns={"Attempts": "Round Attempts", "Made": "Round 1-Putts", "Pct": "Round 1-Putt %"}),
            bench_putt_bucket.rename(columns={"Attempts": f"{compare_mode_putt} Attempts", "Made": f"{compare_mode_putt} 1-Putts", "Pct": f"{compare_mode_putt} 1-Putt %"}),
            on="Bucket",
            how="outer"
        ).sort_values("Bucket")
        putt_table["Δ 1-Putt %"] = (pd.to_numeric(putt_table["Round 1-Putt %"], errors="coerce").fillna(0) - pd.to_numeric(putt_table[f"{compare_mode_putt} 1-Putt %"], errors="coerce").fillna(0)).round(1)
        putt_table["Arrow"] = putt_table["Δ 1-Putt %"].apply(lambda x: _delta_arrow(x, higher_better=True, tol=0.25))
        st.dataframe(putt_table, use_container_width=True, hide_index=True)
    else:
        st.info("No usable putting bucket data found for this round / comparison group.")

    st.markdown("#### Make % by Distance")
    if not round_putt_bucket.empty:
        render_putting_distance_line_chart(
            round_putt_bucket,
            PUTT_BUCKET_ORDER,
            title="Attempts by Distance + Make %",
            value_col="Pct",
            made_col="Made",
            attempts_col="Attempts",
            value_label="1-Putt %",
            good_metric=True,
        )
    else:
        st.info("No usable make-by-distance data found for this round.")

    st.markdown("#### 3-Putt % by Distance")
    if not round_three_putt_bucket.empty:
        render_putting_distance_line_chart(
            round_three_putt_bucket,
            PUTT_BUCKET_ORDER,
            title="Attempts by Distance + 3-Putt %",
            value_col="Pct",
            made_col="Made",
            attempts_col="Attempts",
            value_label="3-Putt %",
            good_metric=False,
        )
    else:
        st.info("No usable 3-putt-by-distance data found for this round.")

    with st.expander("🔎 Debug: Putting SG vs Baseline", expanded=False):
        st.write({
            "compare_mode": compare_mode_putt,
            "attempts": putt_impact["attempts"],
            "actual_makes": round(putt_impact["actual_makes"], 3),
            "expected_makes": round(putt_impact["expected_makes"], 3),
            "extra_makes": round(putt_impact["extra_makes"], 3),
            "sg_putting": round(putt_impact["sg_putting"], 3),
        })

    putting_debug = prepare_putting_frame(round_data).copy()
    if not putting_debug.empty:
        putting_debug = putting_debug.sort_values("Hole")[[
            "Hole", "First Putt Distance", "Putt Bucket", "Putt Made Feet", "Putt Made Flag", "Putts Clean"
        ]].rename(columns={
            "First Putt Distance": "Start Ft",
            "Putt Bucket": "Bucket",
            "Putt Made Feet": "Made Ft",
            "Putt Made Flag": "1-Putt Flag",
            "Putts Clean": "Putts"
        })
    render_debug_section("🔎 Debug: Putting rows used in calculations", putting_debug)

    render_trivia_block("putting", "Putting tab trivia")


with tab_trends:
    st.markdown("### 📈 Player Trends")
    st.caption("Long-term performance view across rounds, with rolling averages and quick trend callouts. Filters can be combined — for example, one course inside one month inside one year.")

    trends_player = player if "player" in locals() else (selected_player if "selected_player" in locals() and selected_player else None)
    if trends_player and "Player Name" in df.columns:
        trends_source = df[df["Player Name"] == trends_player].copy()
    else:
        trends_source = filtered_df.copy() if "filtered_df" in locals() else df.copy()

    # Combined filters for trend analysis
    trends_source["Date Played"] = pd.to_datetime(_safe_col(trends_source, "Date Played", pd.NaT), errors="coerce")
    if "Month" not in trends_source.columns:
        trends_source["Month"] = trends_source["Date Played"].dt.strftime("%B")
    if "Year" not in trends_source.columns:
        trends_source["Year"] = trends_source["Date Played"].dt.year

    tf1, tf2, tf3 = st.columns(3)
    with tf1:
        trend_year_options = sorted([int(x) for x in trends_source["Year"].dropna().unique().tolist()]) if "Year" in trends_source.columns else []
        trend_years = st.multiselect("Year Filter", trend_year_options, default=[], key="trend_years_filter")
    with tf2:
        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        trend_month_options = [m for m in month_order if m in trends_source["Month"].dropna().astype(str).unique().tolist()] if "Month" in trends_source.columns else []
        trend_months = st.multiselect("Month Filter", trend_month_options, default=[], key="trend_months_filter")
    with tf3:
        trend_course_options = sorted([str(x) for x in trends_source["Course Name"].dropna().unique().tolist()]) if "Course Name" in trends_source.columns else []
        trend_courses = st.multiselect("Course Filter", trend_course_options, default=[], key="trend_courses_filter")

    if trend_years:
        trends_source = trends_source[trends_source["Year"].isin(trend_years)].copy()
    if trend_months:
        trends_source = trends_source[trends_source["Month"].astype(str).isin(trend_months)].copy()
    if trend_courses:
        trends_source = trends_source[trends_source["Course Name"].astype(str).isin(trend_courses)].copy()

    trend_df = build_trends_round_frame(trends_source)

    st.markdown("#### Trend vs Baseline Over Time")
    trend_compare_mode = st.radio("Trend baseline:", _compare_mode_options(), horizontal=True, key="trend_compare_mode")
    trend_compare_label = _compare_mode_display_label(trend_compare_mode)
    trend_baseline_df = build_benchmark_df(df, round_data, trend_compare_mode, selected_baseline_round_ids)
    if trend_compare_mode == "Selected Rounds" and not selected_baseline_round_ids:
        st.info("Pick one or more rounds in 'Custom Round Baseline' above to activate the Selected Rounds trend baseline.")
    else:
        render_baseline_summary_box(build_baseline_summary(trend_baseline_df, trend_compare_mode))
        trend_vs_base_df = build_trend_vs_baseline_df(trends_source, trend_baseline_df)
        if not trend_vs_base_df.empty:
            tvb1, tvb2 = st.columns(2)
            with tvb1:
                render_trend_vs_baseline_chart(trend_vs_base_df, "OverallEdge", "OverallEdge_Roll5", f"Estimated Edge vs {trend_compare_label}", "Estimated strokes vs baseline")
            with tvb2:
                render_trend_vs_baseline_chart(trend_vs_base_df, "ScoreDelta", "ScoreDelta_Roll5", f"Score / 18 Delta vs {trend_compare_label}", "Positive = better")
            tvb3, tvb4 = st.columns(2)
            with tvb3:
                render_trend_vs_baseline_chart(trend_vs_base_df, "GIRDelta", "GIRDelta_Roll5", f"GIR Delta vs {trend_compare_label}", "Pts better than baseline")
            with tvb4:
                render_trend_vs_baseline_chart(trend_vs_base_df, "PuttsDelta", "PuttsDelta_Roll5", f"Putts / 18 Delta vs {trend_compare_label}", "Positive = better")

    if trend_df.empty:
        st.info("No round-based trend data available yet.")
    else:
        render_trends_summary_cards(trend_df)
        st.caption(f"Rounds in trend view: {len(trend_df)}")
        st.caption("Current strengths compare the latest filtered round against the average of all currently filtered rounds.")

        t1, t2 = st.columns(2)
        with t1:
            render_trend_line_chart(trend_df, "ScorePer18", "ScorePer18_Roll5", "Score / 18 Trend", "Score / 18", lower_better=True)
        with t2:
            render_trend_line_chart(trend_df, "PuttsPer18", "PuttsPer18_Roll5", "Putts / 18 Trend", "Putts / 18", lower_better=True)

        t3, t4 = st.columns(2)
        with t3:
            render_trend_line_chart(trend_df, "GIRPct", "GIRPct_Roll5", "GIR % Trend", "GIR %", lower_better=False)
        with t4:
            render_trend_line_chart(trend_df, "FWPct", "FWPct_Roll5", "Fairway % Trend", "Fairway %", lower_better=False)

        t5, t6 = st.columns(2)
        with t5:
            render_trend_line_chart(trend_df, "BirdieRate", "BirdieRate_Roll5", "Birdie Rate Trend", "Birdie Rate %", lower_better=False)
        with t6:
            render_trend_line_chart(trend_df, "ScramblePct", "ScramblePct_Roll5", "Scramble % Trend", "Scramble %", lower_better=False)

        trend_insights = build_trends_insights(trend_df)
        trend_strengths = build_trends_strengths(trend_df)

        s1, s2, s3 = st.columns(3)
        with s1:
            strength_html = "<br>".join([f"• {x}" for x in trend_strengths])
            st.markdown(
                f"""
                <div style="background:#1e2930; border:1px solid rgba(113,199,236,.20); border-radius:12px; padding:12px 14px; line-height:1.6;">
                  <b>💪 Current strengths</b><br>
                  {strength_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        with s2:
            good_html = "<br>".join([f"• {x}" for x in trend_insights["good"]])
            st.markdown(
                f"""
                <div style="background:#1f2c25; border:1px solid rgba(100,223,181,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
                  <b>✅ Most improved lately</b><br>
                  {good_html}
                </div>
                """,
                unsafe_allow_html=True
            )
        with s3:
            watch_html = "<br>".join([f"• {x}" for x in trend_insights["watch"]])
            st.markdown(
                f"""
                <div style="background:#2d2323; border:1px solid rgba(238,108,77,.18); border-radius:12px; padding:12px 14px; line-height:1.6;">
                  <b>👀 Needs attention</b><br>
                  {watch_html}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("#### Trend Data")
        trend_table = trend_df[["Date", "Round", "ScorePer18", "PuttsPer18", "GIRPct", "FWPct", "BirdieRate", "ScramblePct", "UpDownPct"]].copy()
        trend_table = trend_table.rename(columns={
            "ScorePer18": "Score / 18",
            "PuttsPer18": "Putts / 18",
            "GIRPct": "GIR %",
            "FWPct": "FW %",
            "BirdieRate": "Birdie Rate %",
            "ScramblePct": "Scramble %",
            "UpDownPct": "Up & Down %",
        })
        st.dataframe(trend_table, use_container_width=True, hide_index=True)

    render_trivia_block("trends", "Trends tab trivia")







def build_bestof_counts_summary(frame):
    d = frame.copy()
    if d is None or d.empty:
        return pd.DataFrame(columns=["Metric", "Count"])

    score_labels = _safe_col(d, "Score Label", "").fillna("").astype(str)
    putts = pd.to_numeric(_safe_col(d, "Putts", 0), errors="coerce").fillna(0)
    gir = pd.to_numeric(_safe_col(d, "GIR", 0), errors="coerce").fillna(0)
    fw = pd.to_numeric(_safe_col(d, "Fairway", 0), errors="coerce").fillna(0)
    par = pd.to_numeric(_safe_col(d, "Par", pd.NA), errors="coerce")
    scramble = pd.to_numeric(_safe_col(d, "Scramble", 0), errors="coerce").fillna(0)
    seve = pd.to_numeric(_safe_col(d, "Seve", 0), errors="coerce").fillna(0)
    arnie = pd.to_numeric(_safe_col(d, "Arnie", 0), errors="coerce").fillna(0)
    lost_balls = (
        pd.to_numeric(_safe_col(d, "Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0)
        + pd.to_numeric(_safe_col(d, "Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0)
    )

    fw_block = d[par.isin([4, 5])] if "Par" in d.columns else d.iloc[0:0]
    hole_outs = build_short_game_hole_out_stats(d).get("count", 0)
    three_putt_bogeys = int(pd.to_numeric(_safe_col(d, "3 Putt Bogey", 0), errors="coerce").fillna(0).sum())

    rows = [
        ("Albatross", int(score_labels.eq("Albatross").sum())),
        ("Eagles", int(score_labels.eq("Eagle").sum())),
        ("Birdies", int(score_labels.eq("Birdie").sum())),
        ("Pars", int(score_labels.eq("Par").sum())),
        ("Bogeys", int(score_labels.eq("Bogey").sum())),
        ("Double+", int(score_labels.isin(["Double Bogey", "Triple Bogey +"]).sum())),
        ("GIR", int(gir.sum())),
        ("Fairways", int(pd.to_numeric(_safe_col(fw_block, "Fairway", 0), errors="coerce").fillna(0).sum()) if len(fw_block) else 0),
        ("1 Putts", int((putts == 1).sum())),
        ("3+ Putts", int((putts >= 3).sum())),
        ("3 Putt Bogeys", three_putt_bogeys),
        ("Scrambles", int(scramble.sum())),
        ("Arnies", int(arnie.sum())),
        ("Seves", int(seve.sum())),
        ("Hole Outs", int(hole_outs)),
        ("Lost Balls", int(lost_balls.sum())),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Count"])

def render_bestof_counts_summary(df_counts, title="Filtered Summary Counts"):
    import streamlit.components.v1 as components

    if df_counts is None or df_counts.empty:
        st.info("No count summary available.")
        return

    left = df_counts.iloc[: (len(df_counts)+1)//2].copy()
    right = df_counts.iloc[(len(df_counts)+1)//2 :].copy()

    def _rows(block):
        html = ""
        for _, r in block.iterrows():
            html += f"""
            <div class="boc-row">
              <div class="boc-m">{r['Metric']}</div>
              <div class="boc-c">{int(r['Count'])}</div>
            </div>
            """
        return html

    html = f"""
    <style>
      .boc-card {{
        background: linear-gradient(180deg,#2a2a2a 0%, #202020 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 10px 12px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
        margin-bottom: 10px;
        font-family: Segoe UI, Roboto, Arial, sans-serif;
      }}
      .boc-title {{
        font-size: 14px;
        font-weight: 800;
        color: #fff;
        margin-bottom: 8px;
      }}
      .boc-grid {{
        display:grid;
        grid-template-columns: 1fr 1fr;
        gap: 14px;
      }}
      .boc-row {{
        display:flex;
        justify-content:space-between;
        gap:8px;
        padding: 5px 0;
        border-top:1px solid rgba(255,255,255,.05);
      }}
      .boc-row:first-child {{ border-top:none; }}
      .boc-m {{
        color:#d7d7d7;
        font-size:11.5px;
        font-weight:700;
      }}
      .boc-c {{
        color:#fff;
        font-size:11.5px;
        font-weight:800;
        font-variant-numeric: tabular-nums;
      }}
    </style>
    <div class="boc-card">
      <div class="boc-title">{title}</div>
      <div class="boc-grid">
        <div>{_rows(left)}</div>
        <div>{_rows(right)}</div>
      </div>
    </div>
    """
    height = 60 + max(len(left), len(right)) * 28
    components.html(html, height=height, scrolling=False)

def render_bestof_compact_table(title, df_table):
    import streamlit.components.v1 as components

    if df_table is None or df_table.empty:
        st.info("No rows for this metric.")
        return

    show_score = "Score" in df_table.columns and title != "Best Scoring Rounds"
    rows_html = ""
    for i, row in df_table.reset_index(drop=True).iterrows():
        rank = i + 1
        date = str(row.get("Date", ""))
        player = str(row.get("Player", ""))
        course = str(row.get("Course", ""))
        stat = str(row.get("Stat", row.get("Round Score", row.get("Value", ""))))
        score = str(row.get("Score", ""))

        meta = f"{date} • {player} • {course}"
        if title == "Best Scoring Rounds":
            meta = f"{date} • {player} • {course} • {stat}"
            stat = ""
        rows_html += f"""
        <div class="bo-row">
          <div class="bo-rank">{rank}</div>
          <div class="bo-main">
            {"<div class='bo-stat'>" + stat + "</div>" if stat else ""}
            <div class="bo-meta">{meta}</div>
          </div>
          {"<div class='bo-score'>" + score + "</div>" if show_score else ""}
        </div>
        """

    html = f"""
    <style>
      .bo-card {{
        background: linear-gradient(180deg,#2a2a2a 0%, #202020 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 14px;
        padding: 10px 10px 6px 10px;
        box-shadow: 0 8px 18px rgba(0,0,0,.16);
        margin-bottom: 10px;
        font-family: Segoe UI, Roboto, Arial, sans-serif;
      }}
      .bo-title {{
        font-size: 14px;
        font-weight: 800;
        color: #fff;
        margin-bottom: 6px;
      }}
      .bo-row {{
        display:grid;
        grid-template-columns: 28px 1fr 88px;
        gap: 8px;
        align-items:center;
        padding: 6px 0;
        border-top: 1px solid rgba(255,255,255,.06);
      }}
      .bo-row:first-of-type {{ border-top:none; padding-top:2px; }}
      .bo-rank {{
        width: 22px; height: 22px; border-radius: 999px;
        background:#3a3a3a; color:#fff; font-size:11px; font-weight:800;
        display:flex; align-items:center; justify-content:center;
      }}
      .bo-main {{ min-width: 0; }}
      .bo-stat {{
        color:#fff; font-size:12px; font-weight:800; line-height:1.15;
        margin-bottom:2px;
      }}
      .bo-meta {{
        color:#bdbdbd; font-size:10.5px; line-height:1.2;
        white-space:normal;
      }}
      .bo-score {{
        color:#fff; font-size:11px; font-weight:800;
        text-align:right; white-space:nowrap;
        font-variant-numeric: tabular-nums;
      }}
    </style>
    <div class="bo-card">
      <div class="bo-title">{title}</div>
      {rows_html}
    </div>
    """
    height = 48 + (len(df_table) * 46)
    components.html(html, height=height, scrolling=False)

with tab_bestof:
    st.markdown("### 🏆 Best Of")
    st.caption("Top round performances for key stats. Only 18-hole rounds are included so comparisons stay clean. Filters can be combined by player, year, month, and course.")

    lb_source = df.copy()
    lb_source["Date Played"] = pd.to_datetime(_safe_col(lb_source, "Date Played", pd.NaT), errors="coerce")
    if "Month" not in lb_source.columns:
        lb_source["Month"] = lb_source["Date Played"].dt.strftime("%B")
    if "Year" not in lb_source.columns:
        lb_source["Year"] = lb_source["Date Played"].dt.year

    l1, l2, l3, l4 = st.columns(4)
    with l1:
        lb_players = sorted([str(x) for x in lb_source["Player Name"].dropna().unique().tolist()]) if "Player Name" in lb_source.columns else []
        lb_player_filter = st.multiselect("Player Filter", lb_players, default=[player] if "player" in locals() and player in lb_players else [], key="lb_player_filter")
    with l2:
        lb_years = sorted([int(x) for x in lb_source["Year"].dropna().unique().tolist()]) if "Year" in lb_source.columns else []
        lb_year_filter = st.multiselect("Year Filter", lb_years, default=[], key="lb_year_filter")
    with l3:
        month_order = ["January","February","March","April","May","June","July","August","September","October","November","December"]
        lb_months = [m for m in month_order if m in lb_source["Month"].dropna().astype(str).unique().tolist()] if "Month" in lb_source.columns else []
        lb_month_filter = st.multiselect("Month Filter", lb_months, default=[], key="lb_month_filter")
    with l4:
        lb_courses = sorted([str(x) for x in lb_source["Course Name"].dropna().unique().tolist()]) if "Course Name" in lb_source.columns else []
        lb_course_filter = st.multiselect("Course Filter", lb_courses, default=[], key="lb_course_filter")

    if lb_player_filter:
        lb_source = lb_source[lb_source["Player Name"].astype(str).isin(lb_player_filter)].copy()
    if lb_year_filter:
        lb_source = lb_source[lb_source["Year"].isin(lb_year_filter)].copy()
    if lb_month_filter:
        lb_source = lb_source[lb_source["Month"].astype(str).isin(lb_month_filter)].copy()
    if lb_course_filter:
        lb_source = lb_source[lb_source["Course Name"].astype(str).isin(lb_course_filter)].copy()

    lb_rounds = build_round_leaderboard_frame(lb_source)

    if lb_rounds.empty:
        st.info("No 18-hole round data available for the current filters.")
    else:
        st.caption(f"18-hole rounds in Best Of view: {len(lb_rounds)}")

        summary_counts_df = build_bestof_counts_summary(lb_source)
        render_bestof_counts_summary(summary_counts_df, title="Filtered Summary Counts")

        metrics = [
            ("Best Scoring Rounds", "ScoreToPar", True, "ScoreDisplay"),
            ("Best GIR", "GIRPct", False, "GIRDisplay"),
            ("Best Fairway", "FWPct", False, "FWDisplay"),
            ("Best Putts / 18", "PuttsPer18", True, "PuttsDisplay"),
            ("Most 1 Putts", "OnePutts", False, "OnePuttsDisplay"),
            ("Most 3+ Putts", "ThreePuttPlus", False, "ThreePuttPlusDisplay"),
            ("Most 3 Putt + Bogeys", "ThreePuttBogeys", False, "ThreePuttBogeysDisplay"),
            ("Best Birdie Rate", "BirdieRate", False, "BirdieDisplay"),
            ("Best Scramble", "ScramblePct", False, "ScrambleDisplay"),
            ("Best Up & Down", "UpDownPct", False, "UpDownDisplay"),
            ("Most Arnies", "Arnies", False, "ArniesDisplay"),
            ("Most Seves", "Seves", False, "SevesDisplay"),
            ("Most Hole Outs", "HoleOuts", False, "HoleOutsDisplay"),
            ("Longest Par-or-Better Streak", "ParOrBetterStreak", False, None),
            ("Longest Birdie-or-Better Streak", "BirdieBetterStreak", False, None),
            ("Most Pars", "ParCount", False, "ParDisplay"),
            ("Most Bogeys", "BogeyCount", False, "BogeyDisplay"),
            ("Most Double+", "DoublePlusCount", False, "DoublePlusDisplay"),
            ("Most Lost Balls", "LostBalls", False, "LostBallsDisplay"),
        ]

        cols = st.columns(2)
        
        for idx, (title, col, lower_better, display_col) in enumerate(metrics):
            with cols[idx % 2]:
                st.markdown(f"#### {title}")
                top_df = get_top_rounds(lb_rounds, col, n=5, lower_better=lower_better, display_col=display_col).copy()
                if not top_df.empty:
                    top_df = _style_bestof_table(top_df, title=title)

                    if title == "Best Scoring Rounds":
                        top_df = top_df.drop(columns=["Score"], errors="ignore")
                        top_df = top_df.rename(columns={"Value": "Round Score"})
                        display_cols = [c for c in ["Date", "Player", "Course", "Round Score"] if c in top_df.columns]
                    else:
                        top_df = top_df.rename(columns={"Value": "Stat", "Score": "Score"})
                        if "Score" not in top_df.columns:
                            top_df["Score"] = ""
                        display_cols = [c for c in ["Date", "Player", "Course", "Stat", "Score"] if c in top_df.columns]

                    top_df = top_df.loc[:, display_cols].copy()

                    render_bestof_compact_table(title, top_df)
                else:
                    st.info("No rows for this metric.")

        st.markdown("### 👎 Worst Of")
        worst_metrics = [
            ("Worst Scoring Rounds", "ScoreToPar", False, "ScoreDisplay"),
            ("Worst GIR", "GIRPct", True, "GIRDisplay"),
            ("Worst Fairway", "FWPct", True, "FWDisplay"),
            ("Worst Putts / 18", "PuttsPer18", False, "PuttsDisplay"),
            ("Most 3+ Putts", "ThreePuttPlus", False, "ThreePuttPlusDisplay"),
            ("Most 3 Putt Bogeys", "ThreePuttBogeys", False, "ThreePuttBogeysDisplay"),
            ("Most Lost Balls", "LostBalls", False, "LostBallsDisplay"),
        ]

        wcols = st.columns(2)
        for widx, (title, col, lower_better, display_col) in enumerate(worst_metrics):
            with wcols[widx % 2]:
                top_df = get_top_rounds(lb_rounds, col, n=5, lower_better=lower_better, display_col=display_col).copy()
                if not top_df.empty:
                    top_df = _style_bestof_table(top_df, title=title)
                    if title == "Worst Scoring Rounds":
                        top_df = top_df.drop(columns=["Score"], errors="ignore")
                        top_df = top_df.rename(columns={"Value": "Round Score"})
                        display_cols = [c for c in ["Date", "Player", "Course", "Round Score"] if c in top_df.columns]
                    else:
                        top_df = top_df.rename(columns={"Value": "Stat", "Score": "Score"})
                        if "Score" not in top_df.columns:
                            top_df["Score"] = ""
                        display_cols = [c for c in ["Date", "Player", "Course", "Stat", "Score"] if c in top_df.columns]
                    top_df = top_df.loc[:, display_cols].copy()
                    render_bestof_compact_table(title, top_df)
                else:
                    st.info("No rows for this metric.")


    render_trivia_block("bestof", "Best Of tab trivia")

with tab_shortgame:


    st.markdown("### ⛳ Short Game / Chipping Breakdown")
    compare_mode_sg = st.radio("Compare this round against:", _compare_mode_options(), horizontal=True, key="shortgame_compare_mode")
    benchmark_df_sg = build_benchmark_df(df, round_data, compare_mode_sg, selected_baseline_round_ids)
    if compare_mode_sg == "Selected Rounds" and not selected_baseline_round_ids:
        st.info("Pick one or more rounds in 'Custom Round Baseline' above to activate the Selected Rounds baseline.")

    round_sg = prepare_short_game_frame(round_data)
    bench_sg = prepare_short_game_frame(benchmark_df_sg)

    round_attempts = int(len(round_sg))
    round_made = int(round_sg["SG OnePutt"].sum()) if not round_sg.empty else 0
    round_pct = (round_made / round_attempts * 100) if round_attempts else 0.0
    bench_attempts = int(len(bench_sg))
    bench_made = int(bench_sg["SG OnePutt"].sum()) if not bench_sg.empty else 0
    bench_pct = (bench_made / bench_attempts * 100) if bench_attempts else 0.0

    sg_extra = build_short_game_extra_stats(round_data)
    sg_bench = build_short_game_extra_stats(benchmark_df_sg) if "benchmark_df_sg" in locals() else {"holes_2plus":0,"opportunities":0}
    sg_prox = build_short_game_proximity_stats(round_data)
    sg_bogey = build_bogey_avoidance_stats(round_data)
    sg_insights = build_short_game_insights(round_data)

    round_2plus_pct = (sg_extra["holes_2plus"] / sg_extra["opportunities"]) if sg_extra["opportunities"] else 0
    bench_2plus_pct = (sg_bench["holes_2plus"] / sg_bench["opportunities"]) if sg_bench["opportunities"] else 0
    delta_2plus = (round_2plus_pct - bench_2plus_pct) * 100

    updowns_made = int(pd.to_numeric(_safe_col(round_data, "Up and Down", 0), errors="coerce").fillna(0).sum()) if "Up and Down" in round_data else int(pd.to_numeric(_safe_col(round_data, "Up & Down", 0), errors="coerce").fillna(0).sum()) if "Up & Down" in round_data else 0
    updowns_ops = int(pd.to_numeric(_safe_col(round_data, "Up and Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up and Down Opportunity" in round_data else int(pd.to_numeric(_safe_col(round_data, "Up & Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up & Down Opportunity" in round_data else round_attempts
    updowns_pct = (updowns_made / updowns_ops * 100.0) if updowns_ops else 0.0
    bench_updowns_made = int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up and Down", 0), errors="coerce").fillna(0).sum()) if "Up and Down" in benchmark_df_sg else int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up & Down", 0), errors="coerce").fillna(0).sum()) if "Up & Down" in benchmark_df_sg else 0
    bench_updowns_ops = int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up and Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up and Down Opportunity" in benchmark_df_sg else int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up & Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up & Down Opportunity" in benchmark_df_sg else bench_attempts
    bench_updowns_pct = (bench_updowns_made / bench_updowns_ops * 100.0) if bench_updowns_ops else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Round SG Attempts", round_attempts)
    with c2:
        st.metric("Round 1-Putt Saves", f"{round_made}/{round_attempts}", f"{round_pct:.1f}%")
    with c3:
        st.metric(f"{compare_mode_sg} SG Attempts", bench_attempts)
    with c4:
        st.metric(f"{compare_mode_sg} 1-Putt Saves", f"{bench_made}/{bench_attempts}", f"{bench_pct:.1f}%")

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Avg Leave Distance", f"{sg_prox['avg_leave_ft']:.1f} ft")
    with c6:
        st.metric("Inside 6 ft", f"{sg_prox['inside6']}/{sg_prox['attempts']}", f"{sg_prox['inside6_pct']:.1f}%")
    with c7:
        st.metric("Inside 10 ft Conversion", f"{sg_prox['convert_inside10']}/{sg_prox['inside10']}", f"{sg_prox['convert_inside10_pct']:.1f}%")
    with c8:
        st.metric("Bogey Avoidance", f"{sg_bogey['par_or_better']}/{sg_bogey['attempts']}", f"{sg_bogey['par_or_better_pct']:.1f}%")

    c9, c10, c11, c12 = st.columns(4)
    with c9:
        st.metric("Total Chips", sg_extra["total_chips"])
    with c10:
        st.metric("Chips / Holes Available", f'{sg_extra["total_chips"]}/{sg_extra["opportunities"]}', f'{sg_extra["chips_per_hole"]:.1f}')
    with c11:
        st.metric("Holes w/ 2+ Chips", f'{sg_extra["holes_2plus"]}/{sg_extra["opportunities"]} ({round_2plus_pct*100:.1f}%)', f'{-delta_2plus:+.1f}% vs avg')
    with c12:
        st.metric("Up & Down %", f"{updowns_made}/{updowns_ops} ({updowns_pct:.1f}%)", f"{(updowns_pct - bench_updowns_pct):+.1f}% vs avg")

    if sg_insights:
        st.markdown(
            f"""
            <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
              <b>🧠 Short Game Notes</b><br>
              {'<br>'.join(sg_insights)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Chipping Inside Leave Distances")
    chip_inside_df = build_short_game_inside_range_summary(round_data, benchmark_df_sg)
    if not chip_inside_df.empty:
        chip_long = pd.DataFrame({
            "Bucket": list(chip_inside_df["Bucket"].astype(str)) * 2,
            "Series": ["Round"] * len(chip_inside_df) + [compare_mode_sg] * len(chip_inside_df),
            "Made": list(pd.to_numeric(chip_inside_df["Round Inside"], errors="coerce").fillna(0).astype(int))
                    + list(pd.to_numeric(chip_inside_df["Baseline Inside"], errors="coerce").fillna(0).astype(int)),
            "Attempts": list(pd.to_numeric(chip_inside_df["Round Opportunities"], errors="coerce").fillna(0).astype(int))
                        + list(pd.to_numeric(chip_inside_df["Baseline Opportunities"], errors="coerce").fillna(0).astype(int)),
            "Pct": list(pd.to_numeric(chip_inside_df["Round %"], errors="coerce").fillna(0.0))
                   + list(pd.to_numeric(chip_inside_df["Baseline %"], errors="coerce").fillna(0.0)),
            "Label": list(chip_inside_df["DisplayLabel"])
                     + [
                        f"{_safe_int_scalar(r.get('Baseline Inside', 0))}/{_safe_int_scalar(r.get('Baseline Opportunities', 0))} {_safe_float_scalar(r.get('Baseline %', 0)):.0f}%"
                        for _, r in chip_inside_df.iterrows()
                     ],
        })
        render_paired_compare_bars(chip_long, "Bucket", SHORT_GAME_BUCKET_ORDER, compare_mode_sg, "Leave Distance", "Chip-Inside %")
        st.dataframe(chip_inside_df, use_container_width=True, hide_index=True)
    else:
        st.info("No chip-inside leave-distance data found for this round / comparison group.")

    st.markdown("#### Short Game Leave Distance → 1-Putt %")
    round_sg_bucket = summarize_short_game_by_bucket(round_data)
    bench_sg_bucket = summarize_short_game_by_bucket(benchmark_df_sg)

    sg_long = build_compare_long(round_sg_bucket, bench_sg_bucket, "Bucket", round_label="Round", bench_label=compare_mode_sg)
    if not sg_long.empty:
        render_paired_compare_bars(sg_long, "Bucket", SHORT_GAME_BUCKET_ORDER, compare_mode_sg, "Leave Distance", "1-Putt %")

        sg_table = pd.merge(
            round_sg_bucket.rename(columns={"Attempts": "Round Attempts", "Made": "Round 1-Putt Saves", "Pct": "Round 1-Putt %"}),
            bench_sg_bucket.rename(columns={"Attempts": f"{compare_mode_sg} Attempts", "Made": f"{compare_mode_sg} 1-Putt Saves", "Pct": f"{compare_mode_sg} 1-Putt %"}),
            on="Bucket",
            how="outer"
        ).sort_values("Bucket")
        st.dataframe(sg_table, use_container_width=True, hide_index=True)
    else:
        st.info("No usable short game bucket data found for this round / comparison group.")

    short_game_debug = prepare_short_game_frame(round_data).copy()
    if not short_game_debug.empty:
        short_game_debug = short_game_debug.sort_values("Hole")[[
            "Hole", "SG GIR", "SG Proximity", "SG Bucket", "SG Putts", "SG OnePutt"
        ]].rename(columns={
            "SG GIR": "GIR",
            "SG Proximity": "Leave Ft",
            "SG Bucket": "Bucket",
            "SG Putts": "Putts",
            "SG OnePutt": "1-Putt Save"
        })
    render_debug_section("🔎 Debug: Short game rows used in calculations", short_game_debug)

    render_trivia_block("shortgame", "Short game tab trivia")
