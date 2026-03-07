
import pandas as pd
import streamlit as st
import datetime
import altair as alt
import os, json, random
from pathlib import Path

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
    mapping = {
        "P WEDGE": "PW",
        "PWEDGE": "PW",
        "PITCHING WEDGE": "PW",
        "A WEDGE": "AW",
        "APPROACH WEDGE": "AW",
        "G WEDGE": "GW",
        "GAP WEDGE": "GW",
        "S WEDGE": "SW",
        "SAND WEDGE": "SW",
        "L WEDGE": "LW",
        "LOB WEDGE": "LW",
        "9I": "9 Iron",
        "8I": "8 Iron",
        "7I": "7 Iron",
        "6I": "6 Iron",
        "5I": "5 Iron",
        "4I": "4 Iron",
        "3I": "3 Iron",
    }
    key = s.upper()
    return mapping.get(key, s)

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
    d["First Putt Distance"] = _num(_safe_col(d, "Proximity to Hole - How far is your First Putt (FT)", 0))
    d["Putt Made Feet"] = _num(_safe_col(d, "Feet of Putt Made (How far was the putt you made)", 0))
    d["Putts Clean"] = _int(_safe_col(d, "Putts", 0))
    d["Putt Bucket"] = d["First Putt Distance"].apply(lambda x: _bucket_value(x, PUTT_BUCKETS))
    d["Putt Attempt"] = (d["First Putt Distance"] > 0).astype(int)
    d["Putt Made Flag"] = (d["Putt Made Feet"] > 0).astype(int)
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

    chart = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            y=alt.Y(f"{key_col}:N", sort=key_order, title=title),
            x=alt.X("Pct:Q", title=x_title),
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

    round_labels = long_df[long_df["Series"] == "Round"].copy()
    label_chart = (
        alt.Chart(round_labels)
        .mark_text(align="left", dx=6)
        .encode(
            y=alt.Y(f"{key_col}:N", sort=key_order),
            x=alt.X("Pct:Q"),
            text="Label:N",
        )
    )

    st.altair_chart(chart + label_chart, use_container_width=True)

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

    st.markdown("#### Approach GIR by Distance Bucket")
    round_dist = summarize_approach_by_bucket(round_data)
    bench_dist = summarize_approach_by_bucket(benchmark_df)

    dist_long = build_compare_long(round_dist, bench_dist, "Bucket", round_label="Round", bench_label=compare_mode)
    if not dist_long.empty:
        dist_long["Bucket"] = pd.Categorical(dist_long["Bucket"], categories=APPROACH_BUCKET_ORDER, ordered=True)
        dist_long = dist_long.sort_values(["Bucket", "Series"]).copy()

        chart = (
            alt.Chart(dist_long)
            .mark_bar()
            .encode(
                y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER, title="Distance Bucket"),
                x=alt.X("Pct:Q", title="GIR %"),
                color=alt.Color("Series:N", title="Series"),
                xOffset="Series:N",
                tooltip=[
                    alt.Tooltip("Bucket:N"),
                    alt.Tooltip("Series:N"),
                    alt.Tooltip("Made:Q"),
                    alt.Tooltip("Attempts:Q"),
                    alt.Tooltip("Pct:Q", format=".1f"),
                    alt.Tooltip("Label:N", title="Summary"),
                ],
            )
            .properties(height=max(320, len(APPROACH_BUCKET_ORDER) * 28))
        )

        round_labels_df = dist_long[dist_long["Series"] == "Round"].copy()
        label_chart = (
            alt.Chart(round_labels_df)
            .mark_text(align="left", dx=6)
            .encode(
                y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER),
                x=alt.X("Pct:Q"),
                text="Label:N",
            )
        )

        marker_src = bench_dist.copy()
        if not marker_src.empty:
            marker_chart = (
                alt.Chart(marker_src)
                .mark_tick(thickness=3, size=22, color="white")
                .encode(
                    y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER),
                    x=alt.X("Pct:Q"),
                    tooltip=[
                        alt.Tooltip("Bucket:N"),
                        alt.Tooltip("Pct:Q", title=f"{compare_mode} GIR %", format=".1f"),
                        alt.Tooltip("Label:N", title=compare_mode),
                    ],
                )
            )
            st.altair_chart(chart + label_chart + marker_chart, use_container_width=True)
        else:
            st.altair_chart(chart + label_chart, use_container_width=True)

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
        club_order = club_long.groupby("Club", as_index=False)["Attempts"].sum().sort_values(["Attempts", "Club"], ascending=[False, True])["Club"].tolist()
        club_chart = (
            alt.Chart(club_long)
            .mark_bar()
            .encode(
                y=alt.Y("Club:N", sort=club_order, title="Club"),
                x=alt.X("Pct:Q", title="GIR %"),
                color=alt.Color("Series:N", title="Series"),
                xOffset="Series:N",
                tooltip=[alt.Tooltip("Club:N"), alt.Tooltip("Series:N"), alt.Tooltip("Made:Q"), alt.Tooltip("Attempts:Q"), alt.Tooltip("Pct:Q", format=".1f"), alt.Tooltip("Label:N", title="Summary")]
            )
            .properties(height=max(320, len(club_order) * 28))
        )
        round_club_labels = club_long[club_long["Series"] == "Round"].copy()
        club_label_chart = alt.Chart(round_club_labels).mark_text(align="left", dx=6).encode(y=alt.Y("Club:N", sort=club_order), x=alt.X("Pct:Q"), text="Label:N")
        st.altair_chart(club_chart + club_label_chart, use_container_width=True)

        club_table = pd.merge(
            round_club.rename(columns={"Attempts": "Round Attempts", "Made": "Round Made", "Pct": "Round GIR %", "AvgProx": "Round Avg Prox"}),
            bench_club.rename(columns={"Attempts": f"{compare_mode} Attempts", "Made": f"{compare_mode} Made", "Pct": f"{compare_mode} GIR %", "AvgProx": f"{compare_mode} Avg Prox"}),
            on="Club",
            how="outer"
        )
        st.dataframe(club_table.sort_values(["Round Attempts", f"{compare_mode} Attempts"], ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("No usable approach club data found for this round / comparison group.")

    st.markdown("#### Approach Miss Direction")
    round_dir = summarize_approach_miss_direction(round_data)
    bench_dir = summarize_approach_miss_direction(benchmark_df)
    if not round_dir.empty or not bench_dir.empty:
        dir_long = pd.concat([round_dir.assign(Series="Round"), bench_dir.assign(Series=compare_mode)], ignore_index=True)
        dir_order = dir_long.groupby("Direction", as_index=False)["Count"].sum().sort_values(["Count", "Direction"], ascending=[False, True])["Direction"].tolist()
        dir_chart = (
            alt.Chart(dir_long)
            .mark_bar()
            .encode(
                y=alt.Y("Direction:N", sort=dir_order, title="Direction"),
                x=alt.X("Count:Q", title="Miss Count"),
                color=alt.Color("Series:N", title="Series"),
                xOffset="Series:N",
                tooltip=[alt.Tooltip("Direction:N"), alt.Tooltip("Series:N"), alt.Tooltip("Count:Q"), alt.Tooltip("Pct:Q", format=".1f")]
            )
            .properties(height=max(280, len(dir_order) * 28))
        )
        st.altair_chart(dir_chart, use_container_width=True)

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

    st.markdown("#### Approach Dispersion Plot")
    render_dispersion_plot(round_data, title="Round Approach Dispersion")

    st.markdown("#### Distance Performance Ranking")
    rank_df = build_distance_rank_table(round_dist, bench_dist)
    if not rank_df.empty:
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    else:
        st.info("No distance ranking rows available.")

    st.markdown("#### Shot Pattern Dashboard")
    shot_base = build_shot_pattern_frame(df, player)
    if not shot_base.empty:
        sp1, sp2, sp3 = st.columns(3)
        with sp1:
            shot_bucket_options = ["All"] + [b for b in APPROACH_BUCKET_ORDER if b in shot_base["Approach Bucket"].dropna().astype(str).tolist()]
            shot_bucket = st.selectbox("Distance Bucket Filter", shot_bucket_options, key="shot_bucket_filter")
        with sp2:
            club_vals = sorted([c for c in shot_base["Approach Club"].dropna().unique().tolist() if str(c).strip() != ""])
            shot_club = st.selectbox("Club Filter", ["All"] + club_vals, key="shot_club_filter")
        with sp3:
            course_vals = sorted([c for c in shot_base["Course Name"].dropna().unique().tolist()]) if "Course Name" in shot_base else []
            shot_course = st.selectbox("Course Filter", ["All"] + course_vals, key="shot_course_filter")

        shot_view = shot_base.copy()
        if shot_bucket != "All":
            shot_view = shot_view[shot_view["Approach Bucket"].astype(str) == shot_bucket]
        if shot_club != "All":
            shot_view = shot_view[shot_view["Approach Club"] == shot_club]
        if shot_course != "All":
            shot_view = shot_view[shot_view["Course Name"] == shot_course]

        if not shot_view.empty:
            shot_attempts = len(shot_view)
            shot_gir = int(shot_view["Approach GIR Flag"].sum())
            shot_pct = shot_gir / shot_attempts * 100 if shot_attempts else 0
            shot_prox = float(shot_view["Approach Proximity"].mean()) if shot_attempts else 0.0
            a1, a2, a3 = st.columns(3)
            with a1:
                st.metric("Filtered Attempts", shot_attempts)
            with a2:
                st.metric("Filtered GIR", f"{shot_gir}/{shot_attempts}", f"{shot_pct:.1f}%")
            with a3:
                st.metric("Avg Proximity", f"{shot_prox:.1f} ft")

            shot_summary = (
                shot_view.groupby(["Approach Bucket"], as_index=False)
                .agg(Attempts=("Approach GIR Flag", "size"), Made=("Approach GIR Flag", "sum"), AvgProx=("Approach Proximity", "mean"))
                .rename(columns={"Approach Bucket": "Bucket"})
            )
            shot_summary["Pct"] = (shot_summary["Made"] / shot_summary["Attempts"] * 100).round(1)
            shot_summary["Label"] = shot_summary.apply(lambda r: f"{int(r['Made'])}/{int(r['Attempts'])} • {r['Pct']:.1f}%", axis=1)

            dash = alt.Chart(shot_summary).mark_bar().encode(
                y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER, title="Bucket"),
                x=alt.X("Pct:Q", title="GIR %"),
                tooltip=[alt.Tooltip("Bucket:N"), alt.Tooltip("Attempts:Q"), alt.Tooltip("Made:Q"), alt.Tooltip("Pct:Q", format=".1f"), alt.Tooltip("AvgProx:Q", format=".1f")]
            ).properties(height=max(220, len(shot_summary) * 28))
            labels = alt.Chart(shot_summary).mark_text(align="left", dx=6).encode(y=alt.Y("Bucket:N", sort=APPROACH_BUCKET_ORDER), x="Pct:Q", text="Label:N")
            st.altair_chart(dash + labels, use_container_width=True)
            st.dataframe(
                shot_view.sort_values(["Date Played", "Hole"], ascending=[False, True])[[
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
        st.metric("Round Makes", f"{round_made}/{round_attempts}", f"{round_pct:.1f}%")
    with c3:
        st.metric(f"{compare_mode_putt} Attempts", bench_attempts)
    with c4:
        st.metric(f"{compare_mode_putt} Makes", f"{bench_made}/{bench_attempts}", f"{bench_pct:.1f}%")

    st.markdown("#### Putts by Starting Distance")
    round_putt_bucket = summarize_putting_by_bucket(round_data)
    bench_putt_bucket = summarize_putting_by_bucket(benchmark_df_putt)
    render_bucket_compare_tab(round_putt_bucket, bench_putt_bucket, "Bucket", PUTT_BUCKET_ORDER, compare_mode_putt, "Putt Range", "Make %")

    putting_debug = prepare_putting_frame(round_data).copy()
    if not putting_debug.empty:
        putting_debug = putting_debug.sort_values("Hole")[[
            "Hole", "First Putt Distance", "Putt Bucket", "Putt Made Feet", "Putt Made Flag", "Putts Clean"
        ]].rename(columns={
            "First Putt Distance": "Start Ft",
            "Putt Bucket": "Bucket",
            "Putt Made Feet": "Made Ft",
            "Putt Made Flag": "Made Flag",
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

    st.markdown("#### Short Game Leave Distance → 1-Putt %")
    round_sg_bucket = summarize_short_game_by_bucket(round_data)
    bench_sg_bucket = summarize_short_game_by_bucket(benchmark_df_sg)
    render_bucket_compare_tab(round_sg_bucket, bench_sg_bucket, "Bucket", SHORT_GAME_BUCKET_ORDER, compare_mode_sg, "Leave Distance", "1-Putt %")

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
