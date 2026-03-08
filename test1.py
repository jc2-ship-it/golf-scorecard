
import pandas as pd
import streamlit as st
import datetime
import altair as alt
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

def build_benchmark_df(full_df, round_df, mode):
    player = round_df["Player Name"].iloc[0] if "Player Name" in round_df else None
    course = round_df["Course Name"].iloc[0] if "Course Name" in round_df else None
    round_date = pd.to_datetime(round_df["Date Played"].iloc[0], errors="coerce")
    year = int(round_date.year) if pd.notna(round_date) else None
    month = int(round_date.month) if pd.notna(round_date) else None

    base = full_df.copy()
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
    out["Pct"] = (out["Made"] / out["Attempts"] * 100).round(1)
    out["Label"] = out.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)
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
      .pair-wrap {background:#222; padding:12px; border-radius:12px; margin-bottom:10px;}
      .pair-cat {margin:10px 0 14px 0; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,.08);}
      .pair-cat:last-child {border-bottom:none; margin-bottom:0; padding-bottom:0;}
      .pair-cat-title {font-weight:700; margin-bottom:6px; color:#fff;}
      .pair-row {
        display:grid;
        grid-template-columns: 90px minmax(140px, 1fr) 120px 58px;
        gap:10px;
        align-items:center;
        margin:4px 0;
      }
      .pair-series {font-size:12px; color:#ddd; font-weight:700;}
      .pair-bar-bg {
        width:100%;
        background:#3a3a3a;
        border-radius:999px;
        height:18px;
        overflow:hidden;
        position:relative;
      }
      .pair-bar-fill-round {
        height:18px;
        border-radius:999px;
        background:#4f8cff;
      }
      .pair-bar-fill-base {
        height:18px;
        border-radius:999px;
        background:#8a8a8a;
      }
      .pair-value {
        font-size:12px;
        color:#fff;
        font-variant-numeric: tabular-nums;
        text-align:left;
        white-space:nowrap;
        font-weight:700;
      }
      .pair-delta {
        font-size:11px;
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
    height_px = max(220, 42 + len(ordered_categories) * 82)
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
      .pair-wrap {background:#222; padding:12px; border-radius:12px; margin-bottom:10px;}
      .pair-cat {margin:10px 0 14px 0; padding-bottom:10px; border-bottom:1px solid rgba(255,255,255,.08);}
      .pair-cat:last-child {border-bottom:none; margin-bottom:0; padding-bottom:0;}
      .pair-cat-title {font-weight:700; margin-bottom:6px; color:#fff;}
      .pair-row {
        display:grid;
        grid-template-columns: 90px minmax(140px, 1fr) 120px 58px;
        gap:10px;
        align-items:center;
        margin:4px 0;
      }
      .pair-series {font-size:12px; color:#ddd; font-weight:700;}
      .pair-bar-bg {
        width:100%;
        background:#3a3a3a;
        border-radius:999px;
        height:18px;
        overflow:hidden;
        position:relative;
      }
      .pair-bar-fill-round {
        height:18px;
        border-radius:999px;
        background:#4f8cff;
      }
      .pair-bar-fill-base {
        height:18px;
        border-radius:999px;
        background:#8a8a8a;
      }
      .pair-value {
        font-size:12px;
        color:#fff;
        font-variant-numeric: tabular-nums;
        text-align:left;
        white-space:nowrap;
        font-weight:700;
      }
      .pair-delta {
        font-size:11px;
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
    height_px = max(220, 42 + len(ordered_categories) * 82)
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

    txt = base.mark_text(fontWeight="bold").encode(text="DisplayLabel:N")
    st.altair_chart((rects + txt).properties(height=260), use_container_width=True)



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
    d = prepare_approach_frame(frame).copy()
    if d.empty:
        st.info("No approach dispersion data available.")
        return

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
        # center GIR / no-direction shots
        return pd.Series({"x": 0.0, "y": 0.0})

    pts = d.join(d.apply(_xy, axis=1))
    pts["Hole"] = _int(_safe_col(pts, "Hole", 0))

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

    st.altair_chart((zero_h + zero_v + scatter + hole).configure_view(stroke=None), use_container_width=True)



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


def apply_approach_filters(frame, yard_buckets=None, clubs=None, courses=None, fairway_vals=None, par_vals=None):
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
hole_outs_total = int(_num(_safe_col(round_data, "Hole Out", 0)).sum())
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

# =========================================================
# Tabs
# =========================================================
tab_scorecard, tab_approach, tab_putting, tab_shortgame = st.tabs(["Scorecard", "Approach", "Putting", "Short Game"])

with tab_scorecard:
    table_html = f"""
    <style>
      .sc-wrap {{
        background:#2a2a2a; padding:10px; border-radius:10px;
        box-shadow: 0 6px 14px rgba(0,0,0,.15);
      }}
      .sc-table {{
        width:100%; border-collapse:separate; border-spacing:0;
        font-size:12.5px; line-height:1.28; color:#fff;
      }}
      .sc-table thead th {{
        position: sticky; top: 0; z-index: 2;
        background:#3a3a3a; color:#fff; text-align:center;
        padding:4px 6px; font-weight:700; border-bottom:1px solid rgba(255,255,255,.08);
      }}
      .sc-table tbody td, .sc-table tbody th {{
        padding:3px 6px; border-bottom:1px solid rgba(255,255,255,.06);
        text-align:center;
      }}
      .sc-table tbody tr:nth-child(odd)  {{ background:#353535; }}
      .sc-table tbody tr:nth-child(even) {{ background:#2f2f2f; }}
      .sc-label {{
        text-align:left; font-weight:700; color:#fff; white-space:nowrap;
      }}
      .sc-score {{
        font-size:18px; font-weight:800; letter-spacing:.2px;
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
    Double+: {cat_counts.get("Double+", 0)} ({round(cat_counts.get("Double+", 0)/max(len(round_data),1)*100,1)}%)<br><br>

    <b>💡 Advanced Insights</b><br>
    Total 1 Putts: {total_1_putts}<br>
    Total 3+ Putts: {total_3_plus_putts}<br>
    3-Putt Bogeys: {total_3_putt_bogeys}<br>
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
    Seves: {seves_total} | Hole Outs: {hole_outs_total} | Lost Balls: {lost_balls_display}
    """

    st.markdown(summary_header_html, unsafe_allow_html=True)
    st.markdown(cards_html, unsafe_allow_html=True)
    st.markdown(benchmarks_html, unsafe_allow_html=True)

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

    random_fact = random.choice(load_fun_facts())
    st.markdown(f"<br><b>💡 Fun Fact:</b> {random_fact}", unsafe_allow_html=True)

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

with tab_approach:
    st.markdown("### 🎯 Approach Breakdown")
    compare_mode = st.radio("Compare this round against:", ["All Time", "Same Year", "Same Month", "Same Course"], horizontal=True, key="approach_compare_mode")
    benchmark_df = build_benchmark_df(df, round_data, compare_mode)

    round_approach = prepare_approach_frame(round_data)
    bench_approach = prepare_approach_frame(benchmark_df)

    round_attempts = int(len(round_approach))
    round_gir_made = int(round_approach["Approach GIR Flag"].sum()) if not round_approach.empty else 0
    round_gir_pct = (round_gir_made / round_attempts * 100) if round_attempts else 0.0
    bench_attempts = int(len(bench_approach))
    bench_gir_made = int(bench_approach["Approach GIR Flag"].sum()) if not bench_approach.empty else 0
    bench_gir_pct = (bench_gir_made / bench_attempts * 100) if bench_attempts else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.metric("Round Approach Attempts", round_attempts)
    with k2:
        st.metric("Round Approach GIR", f"{round_gir_made}/{round_attempts}", f"{round_gir_pct:.1f}%")
    with k3:
        st.metric(f"{compare_mode} Attempts", bench_attempts)
    with k4:
        st.metric(f"{compare_mode} GIR", f"{bench_gir_made}/{bench_attempts}", f"{bench_gir_pct:.1f}%")

    round_us_open = build_us_open_par_summary(round_data)
    bench_us_open = build_us_open_par_summary(benchmark_df)

    k5 = st.columns(1)[0]
    with k5:
        st.metric(
            "US Open Pars",
            f'{round_us_open["made"]}/{round_us_open["attempts"]} ({round_us_open["pct"]:.1f}%)',
            f'{round_us_open["pct"] - bench_us_open["pct"]:+.1f}% vs {compare_mode}'
        )

    st.markdown("#### Approach GIR by Distance Bucket")
    round_dist = summarize_approach_by_bucket(round_data)
    bench_dist = summarize_approach_by_bucket(benchmark_df)

    dist_long = build_compare_long(round_dist, bench_dist, "Bucket", round_label="Round", bench_label=compare_mode)
    if not dist_long.empty:
        render_paired_compare_bars(dist_long, "Bucket", APPROACH_BUCKET_ORDER, compare_mode, "Distance Bucket", "GIR %")

        dist_table = pd.merge(
            round_dist.rename(columns={"Attempts":"Round Attempts","Made":"Round Made","Pct":"Round %","AvgProx":"Round Avg Prox"}),
            bench_dist.rename(columns={"Attempts":f"{compare_mode} Attempts","Made":f"{compare_mode} Made","Pct":f"{compare_mode} %","AvgProx":f"{compare_mode} Avg Prox"}),
            on="Bucket",
            how="outer",
        ).sort_values("Bucket")
        st.dataframe(dist_table, use_container_width=True, hide_index=True)

        insights = build_approach_insights(round_dist, bench_dist, summarize_approach_by_club(round_data, min_attempts=1))
        if insights:
            insight_html = "<br>".join(insights)
            st.markdown(
                f"""
                <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
                  <b>💡 Approach Insights</b><br>
                  {insight_html}
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.info("No usable distance bucket data found for this round / comparison group.")

    st.markdown("#### Approach Proximity Ranges")
    prox_inside_df = build_approach_inside_proximity_summary(round_data, benchmark_df)
    if not prox_inside_df.empty:
        prox_long = pd.DataFrame({
            "Bucket": list(prox_inside_df["Bucket"].astype(str)) * 2,
            "Series": ["Round"] * len(prox_inside_df) + [compare_mode] * len(prox_inside_df),
            "Made": list(pd.to_numeric(prox_inside_df["Round Inside"], errors="coerce").fillna(0).astype(int))
                    + list(pd.to_numeric(prox_inside_df["Baseline Inside"], errors="coerce").fillna(0).astype(int)),
            "Attempts": list(pd.to_numeric(prox_inside_df["Round Opportunities"], errors="coerce").fillna(0).astype(int))
                        + list(pd.to_numeric(prox_inside_df["Baseline Opportunities"], errors="coerce").fillna(0).astype(int)),
            "Pct": list(pd.to_numeric(prox_inside_df["Round %"], errors="coerce").fillna(0.0))
                   + list(pd.to_numeric(prox_inside_df["Baseline %"], errors="coerce").fillna(0.0)),
            "Label": list(prox_inside_df["DisplayLabel"])
                     + [
                        f"{_safe_int_scalar(r.get('Baseline Inside', 0))}/{_safe_int_scalar(r.get('Baseline Opportunities', 0))} {_safe_float_scalar(r.get('Baseline %', 0)):.0f}%"
                        for _, r in prox_inside_df.iterrows()
                     ],
        })
        render_paired_compare_bars(prox_long, "Bucket", PUTT_BUCKET_ORDER, compare_mode, "Proximity Bucket", "Approach-Inside %")
        st.dataframe(prox_inside_df, use_container_width=True, hide_index=True)
    else:
        st.info("No approach proximity-range data found for this round / comparison group.")

    st.markdown("#### Distance Performance Curve")
    round_curve = build_distance_performance_curve(round_data)
    bench_curve = build_distance_performance_curve(benchmark_df)
    render_distance_performance_curve(round_curve, bench_curve, compare_label=compare_mode)

    if not round_curve.empty:
        curve_table = round_curve.rename(columns={
            "Attempts": "Round Attempts",
            "Made": "Round Made",
            "Pct": "Round GIR %",
            "AvgProx": "Round Avg Prox",
            "ToParPerHole": "Round To Par / Hole",
        })
        st.dataframe(curve_table, use_container_width=True, hide_index=True)

        st.markdown("#### Distance SG Heatmap")
        dist_sg_df = build_distance_sg_table(round_curve, bench_curve)
        if not dist_sg_df.empty:
            dist_sg_df["Metric"] = "Approach SG"
            render_distance_sg_heatmap(dist_sg_df.copy())
            st.dataframe(dist_sg_df, use_container_width=True, hide_index=True)
        else:
            st.info("No distance SG-style data available.")

        st.markdown("#### Distance Improvement Tracker")
        improvement_df = build_distance_improvement_tracker(df, round_data)
        if not improvement_df.empty:
            st.dataframe(improvement_df, use_container_width=True, hide_index=True)
        else:
            st.info("No distance improvement tracker data available.")

    st.markdown("#### Approach Dispersion Plot")
    render_dispersion_panel(round_data, title="Round Approach Dispersion")

    st.markdown("#### Approach GIR by Club")
    min_club_attempts = st.slider("Minimum attempts for club chart (applies to round and comparison display)", 1, 10, 1, 1)
    round_club = summarize_approach_by_club(round_data, min_attempts=1)
    bench_club = summarize_approach_by_club(benchmark_df, min_attempts=1)

    club_keys = sorted(set(round_club["Club"].tolist()) | set(bench_club["Club"].tolist()))
    if club_keys:
        merged_counts = pd.merge(
            round_club[["Club", "Attempts"]],
            bench_club[["Club", "Attempts"]],
            on="Club",
            how="outer",
            suffixes=("_Round", "_Bench")
        ).fillna(0)
        keep_clubs = merged_counts[(merged_counts["Attempts_Round"] >= min_club_attempts) | (merged_counts["Attempts_Bench"] >= min_club_attempts)]["Club"].tolist()
        round_club = round_club[round_club["Club"].isin(keep_clubs)].copy()
        bench_club = bench_club[bench_club["Club"].isin(keep_clubs)].copy()

    club_long = build_compare_long(round_club, bench_club, "Club", round_label="Round", bench_label=compare_mode)
    if not club_long.empty:
        # Only keep clubs actually used in the selected round for the chart + table
        round_used_clubs = round_club["Club"].dropna().astype(str).tolist()
        club_long = club_long[club_long["Club"].astype(str).isin(round_used_clubs)].copy()

        if not club_long.empty:
            club_order = (
                round_club[round_club["Club"].astype(str).isin(round_used_clubs)]
                .sort_values(["Attempts", "Pct", "Club"], ascending=[False, False, True])["Club"]
                .astype(str)
                .tolist()
            )
            render_paired_compare_bars(club_long, "Club", club_order, compare_mode, "Club", "GIR %")

        club_table = pd.merge(
            round_club.rename(columns={"Attempts": "Round Attempts", "Made": "Round Made", "Pct": "Round GIR %", "AvgProx": "Round Avg Prox"}),
            bench_club.rename(columns={"Attempts": f"{compare_mode} Attempts", "Made": f"{compare_mode} Made", "Pct": f"{compare_mode} GIR %", "AvgProx": f"{compare_mode} Avg Prox"}),
            on="Club",
            how="left"
        )
        club_table = club_table[club_table["Club"].astype(str).isin(round_used_clubs)].copy()
        club_table["_club_sort"] = club_table["Club"].astype(str).apply(_club_sort_key)
        st.dataframe(club_table.sort_values(["_club_sort", "Round Attempts"], ascending=[True, False]).drop(columns=["_club_sort"]), use_container_width=True, hide_index=True)

        st.markdown("#### Club Performance Curve")
        render_club_performance_curve(round_club, bench_club, compare_label=compare_mode)

        st.markdown("#### Club Performance Ranking")
        club_rank = build_club_rank_table(round_club, bench_club)
        club_rank = club_rank[club_rank["Club"].astype(str).isin(round_used_clubs)].copy()
        st.dataframe(club_rank, use_container_width=True, hide_index=True)
    else:
        st.info("No usable approach club data found for this round / comparison group.")

    st.markdown("#### Approach Miss Direction")
    round_dir = summarize_approach_miss_direction(round_data)
    bench_dir = summarize_approach_miss_direction(benchmark_df)
    if not round_dir.empty or not bench_dir.empty:
        dir_order = (
            pd.concat([round_dir[["Direction", "Count"]], bench_dir[["Direction", "Count"]]], ignore_index=True)
            .groupby("Direction", as_index=False)["Count"]
            .sum()
            .sort_values(["Count", "Direction"], ascending=[False, True])["Direction"]
            .tolist()
        )

        render_paired_compare_counts(round_dir, bench_dir, "Direction", dir_order, compare_mode, "Direction", "Count")

        dir_table = pd.merge(
            round_dir.rename(columns={"Count": "Round Count", "Pct": "Round %"}),
            bench_dir.rename(columns={"Count": f"{compare_mode} Count", "Pct": f"{compare_mode} %"}),
            on="Direction",
            how="outer"
        ).sort_values("Direction")
        st.dataframe(dir_table, use_container_width=True, hide_index=True)
    else:
        st.info("No approach miss direction data found.")

    st.markdown("#### Approach Miss Heat Map")
    render_direction_heatmap(round_dir)

    st.markdown("#### Distance Performance Ranking")
    rank_df = build_distance_rank_table(round_dist, bench_dist)
    if not rank_df.empty:
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    else:
        st.info("No distance ranking rows available.")

    st.markdown("#### Distance vs Club Heatmap")
    render_distance_club_heatmap(round_data, title="Round Distance vs Club Heatmap")

    st.markdown("#### Approach Filter View (Club / Yardage / Course)")
    filter_options_base = build_shot_pattern_frame(df, player)
    if not filter_options_base.empty:
        fa1, fa2, fa3 = st.columns(3)
        with fa1:
            yard_options = [b for b in APPROACH_BUCKET_ORDER if b in filter_options_base["Approach Bucket"].dropna().astype(str).tolist()]
            approach_yards = st.multiselect("Yardage Buckets", yard_options, default=[], key="approach_filter_yards_multi")
        with fa2:
            club_options = sorted([c for c in filter_options_base["Approach Club"].dropna().unique().tolist() if str(c).strip() != ""])
            approach_clubs = st.multiselect("Clubs", club_options, default=[], key="approach_filter_clubs_multi")
        with fa3:
            course_options = sorted([c for c in filter_options_base["Course Name"].dropna().unique().tolist()]) if "Course Name" in filter_options_base else []
            approach_courses = st.multiselect("Courses", course_options, default=[], key="approach_filter_courses_multi")

        fa4, fa5 = st.columns(2)
        with fa4:
            fairway_labels = {1: "Fairway Hit", 0: "Fairway Miss"}
            approach_fairway_labels = st.multiselect("Fairway", list(fairway_labels.values()), default=[], key="approach_filter_fairway_multi")
            approach_fairway_vals = [k for k, v in fairway_labels.items() if v in approach_fairway_labels]
        with fa5:
            par_options = [3, 4, 5]
            approach_pars = st.multiselect("Hole Par", par_options, default=[], key="approach_filter_par_multi")

        round_filter_view = apply_approach_filters(
            round_data,
            yard_buckets=approach_yards,
            clubs=approach_clubs,
            courses=approach_courses,
            fairway_vals=approach_fairway_vals,
            par_vals=approach_pars,
        )
        bench_filter_view = apply_approach_filters(
            benchmark_df,
            yard_buckets=approach_yards,
            clubs=approach_clubs,
            courses=approach_courses,
            fairway_vals=approach_fairway_vals,
            par_vals=approach_pars,
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
                "Series": ["Round"] * len(prox_compare) + [compare_mode] * len(prox_compare),
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
            render_paired_compare_bars(prox_long, "Bucket", PUTT_BUCKET_ORDER, compare_mode, "Proximity Bucket", "Share %")
            st.dataframe(prox_compare, use_container_width=True, hide_index=True)
        else:
            st.info("No filtered proximity distribution available.")
    else:
        st.info("No approach filter-view data available.")

    st.markdown("#### Shot Pattern Dashboard")
    shot_options_base = build_shot_pattern_frame(df, player)
    if not shot_options_base.empty:
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            shot_bucket_options = [b for b in APPROACH_BUCKET_ORDER if b in shot_options_base["Approach Bucket"].dropna().astype(str).tolist()]
            shot_buckets = st.multiselect("Distance Bucket Filter", shot_bucket_options, default=[], key="shot_bucket_filter_multi")
        with sp2:
            club_vals = sorted([c for c in shot_options_base["Approach Club"].dropna().unique().tolist() if str(c).strip() != ""])
            shot_clubs = st.multiselect("Club Filter", club_vals, default=[], key="shot_club_filter_multi")
        with sp3:
            course_vals = sorted([c for c in shot_options_base["Course Name"].dropna().unique().tolist()]) if "Course Name" in shot_options_base else []
            shot_courses = st.multiselect("Course Filter", course_vals, default=[], key="shot_course_filter_multi")

        sp4, sp5 = st.columns(2)
        with sp4:
            fairway_labels = {1: "Fairway Hit", 0: "Fairway Miss"}
            shot_fairway_labels = st.multiselect("Fairway Filter", list(fairway_labels.values()), default=[], key="shot_fairway_filter_multi")
            shot_fairway_vals = [k for k, v in fairway_labels.items() if v in shot_fairway_labels]
        with sp5:
            shot_pars = st.multiselect("Hole Par Filter", [3,4,5], default=[], key="shot_par_filter_multi")

        round_shot_view = apply_approach_filters(
            round_data,
            yard_buckets=shot_buckets,
            clubs=shot_clubs,
            courses=shot_courses,
            fairway_vals=shot_fairway_vals,
            par_vals=shot_pars,
        )
        bench_shot_view = apply_approach_filters(
            benchmark_df,
            yard_buckets=shot_buckets,
            clubs=shot_clubs,
            courses=shot_courses,
            fairway_vals=shot_fairway_vals,
            par_vals=shot_pars,
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

            shot_long = build_compare_long(round_shot_summary, bench_shot_summary, "Bucket", round_label="Round", bench_label=compare_mode)
            if not shot_long.empty:
                render_paired_compare_bars(shot_long, "Bucket", APPROACH_BUCKET_ORDER, compare_mode, "Distance Bucket", "GIR %")
            else:
                st.info("No shot pattern comparison bars available.")

            st.markdown("##### Shot Pattern Dispersion")
            render_dispersion_panel(round_shot_view, title="Filtered Shot Pattern Dispersion")

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

    approach_debug = prepare_approach_frame(round_data).copy()
    if not approach_debug.empty:
        approach_debug = approach_debug.sort_values("Hole")[[
            "Hole", "Approach Club", "Approach Distance", "Approach Bucket",
            "Approach GIR Flag", "Approach Miss Direction Clean", "Approach Proximity"
        ]].rename(columns={
            "Approach Club": "Club",
            "Approach Distance": "Distance",
            "Approach Bucket": "Bucket",
            "Approach GIR Flag": "Approach GIR",
            "Approach Miss Direction Clean": "Miss Dir",
            "Approach Proximity": "Prox"
        })
    render_debug_section("🔎 Debug: Approach rows used in calculations", approach_debug)

with tab_putting:
    st.markdown("### 🏌️ Putting Breakdown")
    compare_mode_putt = st.radio("Compare this round against:", ["All Time", "Same Year", "Same Month", "Same Course"], horizontal=True, key="putting_compare_mode")
    benchmark_df_putt = build_benchmark_df(df, round_data, compare_mode_putt)

    round_putt = prepare_putting_frame(round_data)
    bench_putt = prepare_putting_frame(benchmark_df_putt)

    round_attempts = int(len(round_putt))
    round_made = int(round_putt["Putt Made Flag"].sum()) if not round_putt.empty else 0
    round_pct = (round_made / round_attempts * 100) if round_attempts else 0.0
    bench_attempts = int(len(bench_putt))
    bench_made = int(bench_putt["Putt Made Flag"].sum()) if not bench_putt.empty else 0
    bench_pct = (bench_made / bench_attempts * 100) if bench_attempts else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Round Putt Attempts", round_attempts)
    with c2:
        st.metric("Round 1-Putts", f"{round_made}/{round_attempts}", f"{round_pct:.1f}%")
    with c3:
        st.metric(f"{compare_mode_putt} Attempts", bench_attempts)
    with c4:
        st.metric(f"{compare_mode_putt} 1-Putts", f"{bench_made}/{bench_attempts}", f"{bench_pct:.1f}%")

    putt_impact = build_putting_round_impact(round_data, benchmark_df_putt)
    total_putts_round = int(pd.to_numeric(_safe_col(round_data, "Putts", 0), errors="coerce").fillna(0).sum())
    three_putts_round = int((pd.to_numeric(_safe_col(round_data, "Putts", 0), errors="coerce").fillna(0) >= 3).sum())
    sg_color = "#64dfb5" if putt_impact["sg_putting"] >= 0 else "#ee6c4d"

    st.markdown(
        f"""
        <div style="margin-top:8px; padding:10px 12px; background:#242424; border-radius:10px; line-height:1.55;">
          <b>🧾 Putting Round Summary</b><br>
          Total Putts: {total_putts_round}<br>
          1-Putts: {putt_impact['made']}/{putt_impact['attempts']} ({putt_impact['pct']:.1f}%)<br>
          3-Putts: {three_putts_round}<br>
          Putting SG vs {compare_mode_putt}: <span style="font-weight:800; color:{sg_color};">{putt_impact['sg_putting']:+.2f}</span>
          <span style="color:#aaa;">(Expected Makes: {putt_impact['expected_makes']:.1f} | Actual Makes: {putt_impact['actual_makes']:.0f})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        st.dataframe(putt_table, use_container_width=True, hide_index=True)
    else:
        st.info("No usable putting bucket data found for this round / comparison group.")

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

with tab_shortgame:
    st.markdown("### ⛳ Short Game / Chipping Breakdown")
    compare_mode_sg = st.radio("Compare this round against:", ["All Time", "Same Year", "Same Month", "Same Course"], horizontal=True, key="shortgame_compare_mode")
    benchmark_df_sg = build_benchmark_df(df, round_data, compare_mode_sg)

    round_sg = prepare_short_game_frame(round_data)
    bench_sg = prepare_short_game_frame(benchmark_df_sg)

    round_attempts = int(len(round_sg))
    round_made = int(round_sg["SG OnePutt"].sum()) if not round_sg.empty else 0
    round_pct = (round_made / round_attempts * 100) if round_attempts else 0.0
    bench_attempts = int(len(bench_sg))
    bench_made = int(bench_sg["SG OnePutt"].sum()) if not bench_sg.empty else 0
    bench_pct = (bench_made / bench_attempts * 100) if bench_attempts else 0.0

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Round SG Attempts", round_attempts)
    with c2:
        st.metric("Round 1-Putt Saves", f"{round_made}/{round_attempts}", f"{round_pct:.1f}%")
    with c3:
        st.metric(f"{compare_mode_sg} SG Attempts", bench_attempts)
    with c4:
        st.metric(f"{compare_mode_sg} 1-Putt Saves", f"{bench_made}/{bench_attempts}", f"{bench_pct:.1f}%")

    sg_extra = build_short_game_extra_stats(round_data)
    sg_bench = build_short_game_extra_stats(benchmark_df_sg) if "benchmark_df_sg" in locals() else {"holes_2plus":0,"opportunities":0}

    round_2plus_pct = (sg_extra["holes_2plus"] / sg_extra["opportunities"]) if sg_extra["opportunities"] else 0
    bench_2plus_pct = (sg_bench["holes_2plus"] / sg_bench["opportunities"]) if sg_bench["opportunities"] else 0
    delta_2plus = (round_2plus_pct - bench_2plus_pct) * 100

    updowns_made = int(pd.to_numeric(_safe_col(round_data, "Up and Down", 0), errors="coerce").fillna(0).sum()) if "Up and Down" in round_data else int(pd.to_numeric(_safe_col(round_data, "Up & Down", 0), errors="coerce").fillna(0).sum()) if "Up & Down" in round_data else 0
    updowns_ops = int(pd.to_numeric(_safe_col(round_data, "Up and Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up and Down Opportunity" in round_data else int(pd.to_numeric(_safe_col(round_data, "Up & Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up & Down Opportunity" in round_data else round_attempts
    updowns_pct = (updowns_made / updowns_ops * 100.0) if updowns_ops else 0.0
    bench_updowns_made = int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up and Down", 0), errors="coerce").fillna(0).sum()) if "Up and Down" in benchmark_df_sg else int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up & Down", 0), errors="coerce").fillna(0).sum()) if "Up & Down" in benchmark_df_sg else 0
    bench_updowns_ops = int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up and Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up and Down Opportunity" in benchmark_df_sg else int(pd.to_numeric(_safe_col(benchmark_df_sg, "Up & Down Opportunity", 0), errors="coerce").fillna(0).sum()) if "Up & Down Opportunity" in benchmark_df_sg else bench_attempts
    bench_updowns_pct = (bench_updowns_made / bench_updowns_ops * 100.0) if bench_updowns_ops else 0.0

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        st.metric("Total Chips", sg_extra["total_chips"])
    with c6:
        st.metric(
            "Chips / Holes Available",
            f'{sg_extra["total_chips"]}/{sg_extra["opportunities"]}',
            f'{sg_extra["chips_per_hole"]:.1f}'
        )
    with c7:
        st.metric(
            "Holes w/ 2+ Chips",
            f'{sg_extra["holes_2plus"]}/{sg_extra["opportunities"]} ({round_2plus_pct*100:.1f}%)',
            f'{-delta_2plus:+.1f}% vs avg'
        )
    with c8:
        st.metric(
            "Up & Down %",
            f"{updowns_made}/{updowns_ops} ({updowns_pct:.1f}%)",
            f"{(updowns_pct - bench_updowns_pct):+.1f}% vs avg"
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
