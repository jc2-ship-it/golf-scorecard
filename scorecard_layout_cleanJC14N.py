import pandas as pd
import streamlit as st
import datetime
import altair as alt
import os, json, random
from pathlib import Path



# Load CSV
df = pd.read_csv("Hole Data-Grid view (18).csv")

# Convert date column
df["Date Played"] = pd.to_datetime(df["Date Played"], errors="coerce")
df["Month"] = df["Date Played"].dt.strftime("%B")
df["Year"] = df["Date Played"].dt.year

# Set page layout
st.set_page_config(layout="wide")

# Sidebar Filters
st.sidebar.header("🔍 Filter Rounds")
players = df["Player Name"].dropna().unique()
courses = df["Course Name"].dropna().unique()
months = df["Month"].dropna().unique()
years = df["Year"].dropna().unique()

selected_player = st.sidebar.selectbox("Player", [""] + sorted(players))
selected_course = st.sidebar.selectbox("Course", [""] + sorted(courses))
selected_month = st.sidebar.selectbox("Month", [""] + sorted(months, key=lambda x: datetime.datetime.strptime(x, "%B").month))
selected_year = st.sidebar.selectbox("Year", [""] + sorted(years, reverse=True))

# Apply filters
filtered_df = df.copy()
if selected_player:
    filtered_df = filtered_df[filtered_df["Player Name"] == selected_player]
if selected_course:
    filtered_df = filtered_df[filtered_df["Course Name"] == selected_course]
if selected_month:
    filtered_df = filtered_df[filtered_df["Month"] == selected_month]
if selected_year:
    filtered_df = filtered_df[filtered_df["Year"] == selected_year]

# Group by round
rounds = filtered_df["Round Link"].unique()
if len(rounds) == 0:
    st.warning("No rounds found for selected filters.")
    st.stop()

selected_round = st.selectbox("Select a Round", rounds)
round_data = filtered_df[filtered_df["Round Link"] == selected_round]

# Metadata
player = round_data["Player Name"].iloc[0]
course = round_data["Course Name"].iloc[0]
date = round_data["Date Played"].iloc[0].strftime("%B %d, %Y")

# Cleanup
round_data = round_data.sort_values("Hole")
round_data["Hole"] = round_data["Hole"].astype(int)
round_data["Hole Score"] = round_data["Hole Score"].astype(int)
round_data["Putts"] = pd.to_numeric(round_data["Putts"], errors="coerce").fillna(0).astype(int)
round_data["Par"] = pd.to_numeric(round_data["Par"], errors="coerce").fillna(4).astype(int)
round_data["Yards"] = pd.to_numeric(round_data["Yards"], errors="coerce").fillna(0).astype(int)

# Emojis
fairways = round_data["Fairway"].apply(lambda x: "<span title='Fairway Hit'>🟢</span>" if x == 1 else "").tolist()
girs = round_data["GIR"].apply(lambda x: "<span title='Green in Regulation'>🟢</span>" if x == 1 else "").tolist()
arnies = round_data["Arnie"].apply(lambda x: "<span title='Arnie (Par w/o FW or GIR)'>🅰️</span>" if x == 1 else "").tolist()
approach_gir = round_data["Approach GIR Value"].apply(lambda x: "<span title='Approach GIR'>🟡</span>" if x == 1 else "").tolist()

# Other fields
lost_balls = (round_data["Lost Ball Tee Shot Quantity"].fillna(0) + round_data["Lost Ball Approach Shot Quantity"].fillna(0)).astype(int).tolist()
approach_clubs = round_data["Approach Shot Club Used"].fillna("").tolist()
approach_yards = round_data["Approach Shot Distance (how far you had to the hole)"].fillna(0).round(0).astype(int).tolist()
prox_to_hole = round_data["Proximity to Hole - How far is your First Putt (FT)"].fillna(0).round(0).astype(int).tolist()
putt_made_ft = round_data["Feet of Putt Made (How far was the putt you made)"].fillna("").tolist()

# Base stats
holes = round_data["Hole"].tolist()
pars = round_data["Par"].tolist()
scores = round_data["Hole Score"].tolist()
putts = round_data["Putts"].tolist()
yards = round_data["Yards"].tolist()

# Helpers
def segment_total(vals, start, end): return sum(vals[start:end])
def icon_total(icon_row, start, end, symbol): return sum(symbol in str(cell) for cell in icon_row[start:end])
def insert_segment_sums(row, skip_total=False):
    out, inn = segment_total(row, 0, 9), segment_total(row, 9, 18)
    return row[:9] + [out] + row[9:18] + [inn] + ([""] if skip_total else [out + inn])
def insert_icon_sums(row, symbol):
    out, inn = icon_total(row, 0, 9, symbol), icon_total(row, 9, 18, symbol)
    return row[:9] + [out] + row[9:18] + [inn, out + inn]

# Build rows
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

# Putting Made Row
putt_made_ft_numeric = pd.to_numeric(round_data["Feet of Putt Made (How far was the putt you made)"], errors="coerce").fillna(0)
out_ft = int(putt_made_ft_numeric.iloc[:9].sum())
in_ft = int(putt_made_ft_numeric.iloc[9:18].sum())
total_ft = out_ft + in_ft
putt_made_ft_row = putt_made_ft[:9] + [out_ft] + putt_made_ft[9:18] + [in_ft, total_ft]

# Header
hole_nums = holes[:9] + ["Out"] + holes[9:18] + ["In", "Total"]

# Summary stats
total_score = sum(scores)
total_putts = sum(putts)
fw_hit = fairways.count("<span title='Fairway Hit'>🟢</span>")
fw_att = sum(round_data["Par"].isin([4, 5]))
gir_hit = girs.count("<span title='Green in Regulation'>🟢</span>")
gir_att = len(girs)
arnie_total = arnies.count("<span title='Arnie (Par w/o FW or GIR)'>🅰️</span>")
fw_pct = f"{round(100 * fw_hit / fw_att, 1)}%" if fw_att else "-"
gir_pct = f"{round(100 * gir_hit / gir_att, 1)}%" if gir_att else "-"
avg_par3 = round(sum(s for s, p in zip(scores, pars) if p == 3) / sum(p == 3 for p in pars), 2) if 3 in pars else "-"
avg_par4 = round(sum(s for s, p in zip(scores, pars) if p == 4) / sum(p == 4 for p in pars), 2) if 4 in pars else "-"
avg_par5 = round(sum(s for s, p in zip(scores, pars) if p == 5) / sum(p == 5 for p in pars), 2) if 5 in pars else "-"
total_1_putts = (round_data["Putts"] == 1).sum()
total_3_plus_putts = (round_data["Putts"] >= 3).sum()
total_3_putt_bogeys = round_data["3 Putt Bogey"].sum() if "3 Putt Bogey" in round_data else 0

# Quote
st.markdown(
    "🏌️ *“Arnie steps to the tee with precision in mind. Seve follows, carving creativity from the rough.”*",
    unsafe_allow_html=True
)

# Scorecard Table
table_html = f"""
<div style="background:#333; padding:12px; border-radius:10px;">
  <table style="width:100%; border-collapse:collapse; font-size:13px;">
    <thead>
      <tr style="background-color:#444;">
        <th style="padding:6px; color:#fff; text-align:left;">Hole</th>
        {''.join(f"<th style='padding:6px; color:#fff; text-align:center;'>{col}</th>" for col in hole_nums)}
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
    ("Arnie", arnie_row),
    ("Lost Balls", lost_ball_row),
    ("Appr Club", approach_clubs_row),
    ("Appr Yards", approach_yards_row),
    ("Appr GIR", approach_gir_row),
    ("Prox (FT)", prox_to_hole_row),
    ("FT Made", putt_made_ft_row),
]

row_colors = ["#3a3a3a", "#2c2c2c"]
for i, (label, row) in enumerate(stat_rows):
    bg = row_colors[i % 2]
    table_html += f"<tr style='background-color:{bg};'><td style='padding:6px; color:#fff; font-weight:bold;'>{label}</td>"
    for j, val in enumerate(row):
        if label == "Score":
            color = "#ffffff"
            if j not in [9, 19, 20] and isinstance(val, int):
                par = par_row[j]
                if val <= par - 1:
                    color = "#f5c518"
                elif val == par + 1:
                    color = "#ff9999"
                elif val == par + 2:
                    color = "#ff6666"
                elif val >= par + 3:
                    color = "#cc0000"
            table_html += f"<td style='padding:4px; text-align:center; color:{color}; font-size:20px; font-weight:bold;'>{val}</td>"
        else:
            style = "font-size:13px; color:#fff;" if isinstance(val, (int, float)) else "color:#fff;"
            table_html += f"<td style='padding:4px; text-align:center; {style}'>{val}</td>"
    table_html += "</tr>"
table_html += "</tbody></table></div>"
# Make the table header sticky while scrolling
st.markdown("""
<style>
table thead th {
  position: sticky; top: 0; z-index: 2;
  background: #444 !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(table_html, unsafe_allow_html=True)

# Summary shown below the table

# --- New Summary Stats ---
# Calculate Pro Pars+ = Pro Par + Pro Birdie + Pro Eagle+
pro_pars_plus = round_data[["Pro Par", "Pro Birdie", "Pro Eagle+"]].sum(axis=1).sum()

# Scrambles and Up & Downs
total_scrambles = round_data["Scramble"].sum() if "Scramble" in round_data else 0
total_scramble_ops = round_data["Scramble Opportunity"].sum() if "Scramble Opportunity" in round_data else 0
scramble_pct = f"{round(100 * total_scrambles / total_scramble_ops, 1)}%" if total_scramble_ops else "-"

# Up & Downs: no GIR + 1 putt; opportunities = Scramble Opportunity
_gir   = pd.to_numeric(round_data.get("GIR", 0),   errors="coerce").fillna(0).astype(int)
_putts = pd.to_numeric(round_data.get("Putts", 0), errors="coerce").fillna(0).astype(int)

total_updowns     = int(((_gir == 0) & (_putts == 1)).sum())
total_updown_ops  = int(pd.to_numeric(round_data.get("Scramble Opportunity", 0), errors="coerce").fillna(0).sum())
updown_pct        = f"{(100*total_updowns/total_updown_ops):.1f}%" if total_updown_ops else "-"

# Lost Balls
lost_ball_tee = pd.to_numeric(round_data["Lost Ball Tee Shot Quantity"], errors="coerce").fillna(0).astype(int).sum()
lost_ball_appr = pd.to_numeric(round_data["Lost Ball Approach Shot Quantity"], errors="coerce").fillna(0).astype(int).sum()
total_lost_balls = int(lost_ball_tee + lost_ball_appr)
lost_balls_display = f"Tee {lost_ball_tee} / Approach {lost_ball_appr} / Total {total_lost_balls}"

# Score Type Breakdown
score_types = round_data["Score Label"].value_counts().to_dict()
score_type_summary = " | ".join(f"{label}: {count}" for label, count in score_types.items())

# Score Category Breakdown
def categorize(score, par):
    if score <= par:
        return "Par or Better"
    elif score == par + 1:
        return "Bogey"
    else:
        return "Double Bogey+"

categories = round_data["Hole Score"].combine(round_data["Par"], categorize)


par = round_data["Par"]
score = round_data["Hole Score"]
categories = score.combine(par, categorize)
category_counts = categories.value_counts()
category_total = category_counts.sum()
score_cat_summary = " | ".join(
    f"{cat}: {category_counts.get(cat, 0)} ({round(100 * category_counts.get(cat, 0) / category_total, 1)}%)"
    for cat in ["Par or Better", "Bogey", "Double Bogey+"]
)

# GIR by hole type
def gir_pct_by_par(par_val):
    holes = round_data[round_data["Par"] == par_val]
    total = len(holes)
    girs = holes["GIR"].sum()
    return f"{round(100 * girs / total, 1)}%" if total else "-"

gir_par3 = gir_pct_by_par(3)
gir_par4 = gir_pct_by_par(4)
gir_par5 = gir_pct_by_par(5)

# Fairway % by Par 4 and 5 only
def fw_pct_by_par(par_val):
    holes = round_data[round_data["Par"] == par_val]
    total = len(holes)
    fws = holes["Fairway"].sum()
    return f"{round(100 * fws / total, 1)}%" if total else "-"

fw_par4 = fw_pct_by_par(4)
fw_par5 = fw_pct_by_par(5)

# Seves and Hole Outs
seves_total = round_data["Seve"].sum() if "Seve" in round_data else 0
hole_outs_total = round_data["Hole Out"].sum() if "Hole Out" in round_data else 0
pro_par_total = round_data["Pro Par"].sum() if "Pro Par" in round_data else 0
pro_birdie_total = round_data["Pro Birdie"].sum() if "Pro Birdie" in round_data else 0
pro_eagle_total = round_data["Pro Eagle+"].sum() if "Pro Eagle+" in round_data else 0

pro_pars_total = pro_par_total + pro_birdie_total + pro_eagle_total

total_holes = len(round_data)


# ---- FINAL HTML OUTPUT ----

# Total Holes
total_holes = len(round_data)

# Score Type Counts
score_type_counts = round_data["Score Label"].value_counts()

# Score Category Breakdown
def categorize(score_label):
    if score_label in ["Birdie", "Eagle", "Albatross", "Par"]:
        return "Par or Better"
    elif score_label == "Bogey":
        return "Bogey"
    else:
        return "Double+"

categories = round_data["Score Label"].apply(categorize)
cat_counts = categories.value_counts()

# Scoring Averages
par3_scores = round_data[round_data["Par"] == 3]["Hole Score"]
par4_scores = round_data[round_data["Par"] == 4]["Hole Score"]
par5_scores = round_data[round_data["Par"] == 5]["Hole Score"]

par3_avg = par3_scores.mean() if not par3_scores.empty else 0
par4_avg = par4_scores.mean() if not par4_scores.empty else 0
par5_avg = par5_scores.mean() if not par5_scores.empty else 0

# Pro Pars+
pro_pars_total = round_data[["Pro Par", "Pro Birdie", "Pro Eagle+"]].sum().sum()

# GIR by Hole Type
gir_par3_pct = round_data[round_data["Par"] == 3]["GIR"].mean() * 100 if not round_data[round_data["Par"] == 3].empty else 0
gir_par4_pct = round_data[round_data["Par"] == 4]["GIR"].mean() * 100 if not round_data[round_data["Par"] == 4].empty else 0
gir_par5_pct = round_data[round_data["Par"] == 5]["GIR"].mean() * 100 if not round_data[round_data["Par"] == 5].empty else 0

# FW by Hole Type (using column "Fairway")
fw_par4_made = round_data[(round_data["Par"] == 4) & (round_data["Fairway"] == 1)].shape[0]
fw_par4_total = round_data[round_data["Par"] == 4]["Fairway"].count()
fw_par4_pct = (fw_par4_made / fw_par4_total) * 100 if fw_par4_total > 0 else 0

fw_par5_made = round_data[(round_data["Par"] == 5) & (round_data["Fairway"] == 1)].shape[0]
fw_par5_total = round_data[round_data["Par"] == 5]["Fairway"].count()
fw_par5_pct = (fw_par5_made / fw_par5_total) * 100 if fw_par5_total > 0 else 0

# --- GIR & Fairway: made/total/% by par (for display like 0/4 0.0%) ---
def _made_total_pct_by_par(df, metric_col, par_value):
    if metric_col not in df or "Par" not in df:
        return 0, 0, 0.0
    block = df[df["Par"] == par_value]
    total = int(block.shape[0])
    made = int(pd.to_numeric(block[metric_col], errors="coerce").fillna(0).sum())
    pct = (made / total * 100.0) if total else 0.0
    return made, total, pct

# GIR by Par 3/4/5
gir3_m, gir3_t, gir3_pct = _made_total_pct_by_par(round_data, "GIR", 3)
gir4_m, gir4_t, gir4_pct = _made_total_pct_by_par(round_data, "GIR", 4)
gir5_m, gir5_t, gir5_pct = _made_total_pct_by_par(round_data, "GIR", 5)

# Fairways by Par 4/5
fw4_m, fw4_t, fw4_pct = _made_total_pct_by_par(round_data, "Fairway", 4)
fw5_m, fw5_t, fw5_pct = _made_total_pct_by_par(round_data, "Fairway", 5)

# Seves and Hole Outs
seves_total = round_data["Seve"].sum() if "Seve" in round_data else 0
hole_outs_total = round_data["Hole Out"].sum() if "Hole Out" in round_data else 0

# Lost Balls
lost_ball_tee = round_data["Lost Ball Tee Shot Quantity"].sum()
lost_ball_appr = round_data["Lost Ball Approach Shot Quantity"].sum()
total_lost_balls = lost_ball_tee + lost_ball_appr

# Total 1 Putts, 3+ Putts, 3-Putt Bogeys
one_putts = (round_data["Putts"] == 1).sum()
three_plus_putts = (round_data["Putts"] >= 3).sum()
three_putt_bogeys = round_data[(round_data["Putts"] >= 3) & (round_data["Score Label"] == "Bogey")].shape[0]

# ---- FINAL HTML OUTPUT ----
def get_emoji(pct):
    if pct >= 50:
        return "🔥"
    elif pct < 25:
        return "❄️"
    else:
        return ""
# Fairway total, attempts, and percentage (Par 4 & 5 only)
if "Fairway" in round_data:
    _fw_series = pd.to_numeric(
        round_data.loc[round_data["Par"].isin([4, 5]), "Fairway"],
        errors="coerce"
    ).fillna(0)
    fw_total = int(_fw_series.sum())
    fw_attempts = int(_fw_series.count())   # only P4/P5 holes
    fw_pct = (fw_total / fw_attempts * 100) if fw_attempts else 0
else:
    fw_total = 0
    fw_attempts = 0
    fw_pct = 0

# GIR total and percentage
gir_total = round_data["GIR"].sum() if "GIR" in round_data else 0
holes_played = round_data.shape[0]
gir_pct = (gir_total / holes_played * 100) if holes_played > 0 else 0
arnies_total = round_data["Arnie"].sum() if "Arnie" in round_data else 0# --- score-to-par + putts/hole helpers ---
if "Score to Par" in round_data:
    _stp = pd.to_numeric(round_data["Score to Par"], errors="coerce").fillna(0)
    score_to_par_total = int(_stp.sum())
else:
    score_to_par_total = int(total_score - sum(pars))

def _fmt_to_par(n: int) -> str:
    return "E" if n == 0 else f"{'+' if n > 0 else ''}{n}"

score_to_par_str = _fmt_to_par(score_to_par_total)
putts_per_hole = (total_putts / holes_played) if holes_played else 0.0
# -- Scrambles / Up & Downs display strings --
if total_scramble_ops:
    scrambles_display = f"{int(total_scrambles)}/{int(total_scramble_ops)} ({(total_scrambles/total_scramble_ops*100):.1f}%)"
else:
    scrambles_display = "0/0 (-)"

if total_updown_ops:
    updowns_display = f"{int(total_updowns)}/{int(total_updown_ops)} ({(total_updowns/total_updown_ops*100):.1f}%)"
else:
    updowns_display = "0/0 (-)"

# --- Score-to-par benchmarks (current, last 5 rounds avg, last 100 holes total & per-18) ---

# Current round over par
current_over_par = int(total_score - sum(pars))

# Helper: compute round-level "over par" for a given Round Link inside the (already filtered) dataframe
def _round_over_par(df, round_id):
    block = df[df["Round Link"] == round_id]
    hole_sum = pd.to_numeric(block["Hole Score"], errors="coerce").sum()
    par_sum  = pd.to_numeric(block["Par"],        errors="coerce").sum()
    return int(hole_sum - par_sum)

# Last 5 complete rounds by date (within current sidebar filters)
if {"Date Played", "Round Link"} <= set(filtered_df.columns):
    last5_round_ids = (
        filtered_df[["Round Link", "Date Played"]]
        .drop_duplicates()
        .sort_values("Date Played")
        .tail(5)["Round Link"]
        .tolist()
    )
else:
    last5_round_ids = []

last5_over_par_values = [_round_over_par(filtered_df, rid) for rid in last5_round_ids]
last5_avg_over_par = (sum(last5_over_par_values) / len(last5_over_par_values)) if last5_over_par_values else 0.0

# Last 100 holes (by played date) — total Δ and per-18 normalization
last100 = filtered_df.sort_values("Date Played").tail(100)
last100_total_over_par = int(
    pd.to_numeric(last100["Hole Score"], errors="coerce").sum()
    - pd.to_numeric(last100["Par"],        errors="coerce").sum()
)
n_last100_holes = int(last100.shape[0]) if last100 is not None else 0
last100_per18 = (last100_total_over_par * 18 / n_last100_holes) if n_last100_holes else 0.0

# Formatter for floating deltas (keeps one decimal and adds + sign)
def _fmt_par_float(x: float) -> str:
    if abs(x) < 1e-9:
        return "E"
    return f"{'+' if x > 0 else ''}{x:.1f}"

# Strings for display (use your existing _fmt_to_par for integers)
bench_current_str        = _fmt_to_par(current_over_par)                 # e.g. "+6"
bench_last5_avg_str      = _fmt_par_float(last5_avg_over_par)            # e.g. "+10.2"
bench_last100_total_str  = _fmt_to_par(last100_total_over_par)           # e.g. "+73"
bench_last100_per18_str  = _fmt_par_float(last100_per18)                 # e.g. "+13.1"

benchmarks_html = f"""
<div style="margin-top:6px; line-height:1.6;">
  <b>Score to Par — Benchmarks</b><br>
  Current Round: {bench_current_str}<br>
  Last 5 Rounds (avg): {bench_last5_avg_str}<br>
  Last 100 Holes — Total: {bench_last100_total_str} | Per-18: {bench_last100_per18_str}
</div>
"""

# --- Header (round info), Bubble cards, and Details (text blocks) ---
# --- Benchmarks: last 5 rounds / last 100 holes (Up&Downs, Scrambles, Lost Balls) ---
player_df = df[df["Player Name"] == player].copy()
player_df["Date Played"] = pd.to_datetime(player_df["Date Played"], errors="coerce")

# Per-round ordering (most recent round date first)
round_order = (
    player_df.groupby("Round Link")["Date Played"].max()
    .sort_values(ascending=False)
)

last5_round_ids = round_order.index[:5].tolist()
last5_df = player_df[player_df["Round Link"].isin(last5_round_ids)]

# Last 100 holes for this player (most recent rows first)
last100_df = player_df.sort_values("Date Played", ascending=False).head(100)

def _safe_int(series):
    return pd.to_numeric(series, errors="coerce").fillna(0).astype(int)

def ud_stats(frame: pd.DataFrame):
    gir0 = (_safe_int(frame.get("GIR", 0)) == 0)
    putt1 = (_safe_int(frame.get("Putts", 0)) == 1)
    up_made = int((gir0 & putt1).sum())
    up_ops = int(_safe_int(frame.get("Scramble Opportunity", 0)).sum())
    pct = (up_made / up_ops * 100) if up_ops else 0.0
    return up_made, up_ops, pct

def scramble_stats(frame: pd.DataFrame):
    made = int(_safe_int(frame.get("Scramble", 0)).sum())
    ops  = int(_safe_int(frame.get("Scramble Opportunity", 0)).sum())
    pct = (made / ops * 100) if ops else 0.0
    return made, ops, pct

def lostball_stats(frame: pd.DataFrame):
    tee  = int(_safe_int(frame.get("Lost Ball Tee Shot Quantity", 0)).sum())
    appr = int(_safe_int(frame.get("Lost Ball Approach Shot Quantity", 0)).sum())
    total = tee + appr
    rounds_n = max(1, int(frame["Round Link"].nunique())) if "Round Link" in frame else 1
    holes_n = int(frame.shape[0])
    per_round = total / rounds_n
    per_18 = (total / (holes_n / 18.0)) if holes_n else 0.0
    return tee, appr, total, per_round, per_18

# Compute for last 5 rounds
ud5_m, ud5_o, ud5_pct = ud_stats(last5_df)
sc5_m, sc5_o, sc5_pct = scramble_stats(last5_df)
lb5_t, lb5_a, lb5_tot, lb5_per_round, lb5_per18 = lostball_stats(last5_df)

# Compute for last 100 holes
ud100_m, ud100_o, ud100_pct = ud_stats(last100_df)
sc100_m, sc100_o, sc100_pct = scramble_stats(last100_df)
lb100_t, lb100_a, lb100_tot, lb100_per_round, lb100_per18 = lostball_stats(last100_df)

# Compact comparisons block (separate HTML so we don't alter existing summary blocks)
comparisons_html = f"""
<div style="margin-top:6px; padding:10px; background:#262626; border-radius:10px; line-height:1.6;">
  <b>📊 Benchmarks (Recent)</b><br>
  <b>Up &amp; Downs</b> — Last 5: {ud5_m}/{ud5_o} ({ud5_pct:.1f}%) | Last 100: {ud100_m}/{ud100_o} ({ud100_pct:.1f}%)<br>
  <b>Scrambles</b> — Last 5: {sc5_m}/{sc5_o} ({sc5_pct:.1f}%) | Last 100: {sc100_m}/{sc100_o} ({sc100_pct:.1f}%)<br>
  <b>Lost Balls</b> — Last 5: Tee {lb5_t} / Appr {lb5_a} / Total {lb5_tot} <span style="color:#aaa;">(avg {lb5_per_round:.2f}/rnd, {lb5_per18:.2f}/18)</span><br>
  <b>Lost Balls</b> — Last 100: Tee {lb100_t} / Appr {lb100_a} / Total {lb100_tot} <span style="color:#aaa;">(avg {lb100_per_round:.2f}/rnd, {lb100_per18:.2f}/18)</span>
</div>
"""

summary_header_html = f"""
🏌️ {player} | {course} | {date}<br><br>
<b>📊 Round Totals — {holes_played} Holes</b>
"""

cards_html = f"""
<div style="display:flex; gap:12px; flex-wrap:wrap; margin:8px 0 4px 0;">
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">Score</div>
    <div style="font-size:22px;font-weight:700;">{total_score} <span style="font-size:14px;color:#bbb;">({score_to_par_str})</span></div>
  </div>
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">Putts / Hole</div>
    <div style="font-size:22px;font-weight:700;">{putts_per_hole:.2f}</div>
  </div>
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">GIR</div>
    <div style="font-size:22px;font-weight:700;">{gir_total}/{holes_played} <span style="font-size:14px;color:#bbb;">({gir_pct:.1f}%)</span></div>
  </div>
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">Fairways</div>
    <div style="font-size:22px;font-weight:700;">{fw_total}/{fw_attempts} <span style="font-size:14px;color:#bbb;">({fw_pct:.1f}%)</span></div>
  </div>
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">Scrambles</div>
    <div style="font-size:22px;font-weight:700;">{scrambles_display}</div>
  </div>
  <div style="flex:1; min-width:160px; background:#2a2a2a; border-radius:12px; padding:12px;">
    <div style="font-size:12px;color:#aaa;">Up & Downs</div>
    <div style="font-size:22px;font-weight:700;">{updowns_display}</div>
  </div>
</div>
"""

summary_details_html = f"""
<br>

<b>📈 Scoring Averages</b><br>
Par 3 Avg: {par3_avg:.1f}<br>
Par 4 Avg: {par4_avg:.1f}<br>
Par 5 Avg: {par5_avg:.1f}<br><br>

<b>🎯 Score Breakdown</b><br>
Birdie: {score_type_counts.get("Birdie", 0)} | Par: {score_type_counts.get("Par", 0)} | Bogey: {score_type_counts.get("Bogey", 0)} | Double Bogey: {score_type_counts.get("Double Bogey", 0)} | Triple Bogey +: {score_type_counts.get("Triple Bogey +", 0)}<br>
Par or Better: {cat_counts.get("Par or Better", 0)} ({round(cat_counts.get("Par or Better", 0)/total_holes*100,1)}%) |
Bogey: {cat_counts.get("Bogey", 0)} ({round(cat_counts.get("Bogey", 0)/total_holes*100,1)}%) |
Double+: {cat_counts.get("Double+", 0)} ({round(cat_counts.get("Double+", 0)/total_holes*100,1)}%)<br><br>

<b>💡 Advanced Insights</b><br>
Total 1 Putts: {one_putts}<br>
Total 3+ Putts: {three_plus_putts}<br>
3-Putt Bogeys: {three_putt_bogeys}<br>
Pro Pars+: {pro_pars_total}<br>
Scrambles: {scrambles_display}<br>
Up & Downs: {updowns_display}<br>
GIR — Par 3: {gir3_m}/{gir3_t} {gir3_pct:.1f}% {get_emoji(gir3_pct)} | Par 4: {gir4_m}/{gir4_t} {gir4_pct:.1f}% {get_emoji(gir4_pct)} | Par 5: {gir5_m}/{gir5_t} {gir5_pct:.1f}% {get_emoji(gir5_pct)}<br>
Fairways — Par 4: {fw4_m}/{fw4_t} {fw4_pct:.1f}% {get_emoji(fw4_pct)} | Par 5: {fw5_m}/{fw5_t} {fw5_pct:.1f}% {get_emoji(fw5_pct)}<br>
GIR Overall: {gir_pct:.1f}% {get_emoji(gir_pct)}<br>
Seves: {seves_total} | Hole Outs: {hole_outs_total} | Lost Balls: {lost_balls_display}
"""

# Render header + bubbles right under the scorecard (before visuals)
st.markdown(summary_header_html, unsafe_allow_html=True)
st.markdown(cards_html, unsafe_allow_html=True)
# Benchmarks block (score to par)
st.markdown(benchmarks_html, unsafe_allow_html=True)

# --- Score to Par: Current vs Last 5 Rounds vs Last 100 Holes ---
# (Safe, stand-alone, does not overwrite existing variables)

# Build a player-only dataframe in chronological order
_player_df = df[df["Player Name"] == player].copy()
_player_df["Date Played"] = pd.to_datetime(_player_df["Date Played"], errors="coerce")
_player_df["Hole Score"]  = pd.to_numeric(_player_df["Hole Score"], errors="coerce")
_player_df["Par"]         = pd.to_numeric(_player_df["Par"], errors="coerce")
_player_df = _player_df.sort_values(["Date Played", "Round Link", "Hole"])

# Current round score-to-par (you already calculate total_score & pars above)
_stp_current = int(total_score - sum(pars))  # already protected above

# Last 5 rounds: average score-to-par per round
_round_stp = (
    _player_df
    .assign(delta=_player_df["Hole Score"] - _player_df["Par"])
    .groupby("Round Link", dropna=True)["delta"]
    .sum()
    .sort_index()  # order by Round Link grouping (we already sorted rows by date)
)
_last5 = _round_stp.tail(5)
_stp_last5_avg = float(_last5.mean()) if len(_last5) else float("nan")

# Last 100 holes: total score-to-par, plus scaled per-18
_last100 = _player_df.tail(100)
_stp_last100_total = int((_last100["Hole Score"] - _last100["Par"]).sum()) if not _last100.empty else 0
_stp_last100_per18 = (_stp_last100_total / 100.0 * 18.0) if len(_last100) == 100 else float("nan")

def _fmt_stp(n):
    # n can be int or float('nan')
    try:
        if pd.isna(n):
            return "—"
        n = int(round(n))
        return "E" if n == 0 else (f"+{n}" if n > 0 else f"{n}")
    except Exception:
        return "—"

# Small comparison panel (HTML)
stp_panel_html = f"""
<div style="margin-top:10px; background:#2a2a2a; border-radius:12px; padding:12px;">
  <div style="font-weight:700; margin-bottom:6px;">Score to Par — Benchmarks</div>
  <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:8px;">
    <div style="background:#232323; border-radius:10px; padding:10px;">
      <div style="font-size:12px; color:#aaa;">Current Round</div>
      <div style="font-size:22px; font-weight:800;">{_fmt_stp(_stp_current)}</div>
    </div>
    <div style="background:#232323; border-radius:10px; padding:10px;">
      <div style="font-size:12px; color:#aaa;">Last 5 Rounds (avg)</div>
      <div style="font-size:22px; font-weight:800;">{_fmt_stp(_stp_last5_avg)}</div>
    </div>
    <div style="background:#232323; border-radius:10px; padding:10px;">
      <div style="font-size:12px; color:#aaa;">Last 100 Holes</div>
      <div style="font-size:16px; font-weight:800; line-height:1.2;">
        Total: {_fmt_stp(_stp_last100_total)}<br>
        <span style="font-size:12px; color:#bbb;">Per-18: {_fmt_stp(_stp_last100_per18)}</span>
      </div>
    </div>
  </div>
</div>
"""

st.markdown(stp_panel_html, unsafe_allow_html=True)

st.markdown(comparisons_html, unsafe_allow_html=True)
# ===================== YEARLY SUMMARY (2025) — per selected player =====================
# Build a 2025-only dataset for the selected player (ignores sidebar year filter)
y_player = str(player)
y_df = df[(df["Player Name"] == y_player) & (df["Year"] == 2025)].copy()

if not y_df.empty:
    # Coerce numerics safely
    y_df["Par"] = pd.to_numeric(y_df["Par"], errors="coerce").fillna(0)
    y_df["Hole Score"] = pd.to_numeric(y_df["Hole Score"], errors="coerce").fillna(0)
    y_df["Putts"] = pd.to_numeric(y_df["Putts"], errors="coerce").fillna(0)
    y_df["GIR"] = pd.to_numeric(y_df.get("GIR", 0), errors="coerce").fillna(0)
    y_df["Fairway"] = pd.to_numeric(y_df.get("Fairway", 0), errors="coerce").fillna(0)
    y_df["Approach GIR Value"] = pd.to_numeric(y_df.get("Approach GIR Value", 0), errors="coerce").fillna(0)
    y_df["Scramble"] = pd.to_numeric(y_df.get("Scramble", 0), errors="coerce").fillna(0)
    y_df["Scramble Opportunity"] = pd.to_numeric(y_df.get("Scramble Opportunity", 0), errors="coerce").fillna(0)
    y_df["Seve"] = pd.to_numeric(y_df.get("Seve", 0), errors="coerce").fillna(0)
    y_df["Arnie"] = pd.to_numeric(y_df.get("Arnie", 0), errors="coerce").fillna(0)
    y_df["Pro Par"] = pd.to_numeric(y_df.get("Pro Par", 0), errors="coerce").fillna(0)
    y_df["Pro Birdie"] = pd.to_numeric(y_df.get("Pro Birdie", 0), errors="coerce").fillna(0)
    y_df["Pro Eagle+"] = pd.to_numeric(y_df.get("Pro Eagle+", 0), errors="coerce").fillna(0)
    y_df["Lost Ball Tee Shot Quantity"] = pd.to_numeric(y_df.get("Lost Ball Tee Shot Quantity", 0), errors="coerce").fillna(0)
    y_df["Lost Ball Approach Shot Quantity"] = pd.to_numeric(y_df.get("Lost Ball Approach Shot Quantity", 0), errors="coerce").fillna(0)
    y_df["Proximity to Hole - How far is your First Putt (FT)"] = pd.to_numeric(
        y_df.get("Proximity to Hole - How far is your First Putt (FT)", 0), errors="coerce"
    ).fillna(0)

    # Totals / deltas
    y_holes = int(y_df.shape[0])
    y_score_sum = int(y_df["Hole Score"].sum())
    y_par_sum = int(y_df["Par"].sum())
    y_delta = int(y_score_sum - y_par_sum)

    # Averages (score & putts)
    y_score_per_hole = (y_score_sum / y_holes) if y_holes else 0.0
    y_avg9 = round(y_score_per_hole * 9)
    y_avg18 = round(y_score_per_hole * 18)

    y_putts_total = int(y_df["Putts"].sum())
    y_putts_per_hole = (y_putts_total / y_holes) if y_holes else 0.0
    y_putts_per18 = y_putts_per_hole * 18

    # Fairways (Par 4/5 only)
    y_p45 = y_df[y_df["Par"].isin([4, 5])]
    y_fw_made = int(y_p45["Fairway"].sum())
    y_fw_attempts = int(y_p45.shape[0])
    y_fw_pct = (y_fw_made / y_fw_attempts * 100.0) if y_fw_attempts else 0.0

    # GIR (official + "unofficial")
    y_gir_made = int(y_df["GIR"].sum())
    y_gir_pct = (y_gir_made / y_holes * 100.0) if y_holes else 0.0
    y_ugir_made = int(y_df["Approach GIR Value"].sum())
    y_ugir_pct = (y_ugir_made / y_holes * 100.0) if y_holes else 0.0

    # Birdie putt attempts within distance (GIR holes are birdie putts)
    y_first_putt_ft = y_df["Proximity to Hole - How far is your First Putt (FT)"]
    y_is_birdie_putt = (y_df["GIR"] == 1)
    y_bpa_20 = int(((y_is_birdie_putt) & (y_first_putt_ft <= 20)).sum())
    y_bpa_10 = int(((y_is_birdie_putt) & (y_first_putt_ft <= 10)).sum())

    # Scoring averages by hole par
    def _avg_score_for_par(par_val):
        block = y_df[y_df["Par"] == par_val]["Hole Score"]
        return float(block.mean()) if not block.empty else 0.0

    y_p3_avg = _avg_score_for_par(3)
    y_p4_avg = _avg_score_for_par(4)
    y_p5_avg = _avg_score_for_par(5)

    # GIR% by par & FW% by par
    def _pct(col, frame):
        total = frame.shape[0]
        made = frame[col].sum()
        return (made / total * 100.0) if total else 0.0

    y_gir_p3 = _pct("GIR", y_df[y_df["Par"] == 3])
    y_gir_p4 = _pct("GIR", y_df[y_df["Par"] == 4])
    y_gir_p5 = _pct("GIR", y_df[y_df["Par"] == 5])

    y_fw_p4 = _pct("Fairway", y_df[y_df["Par"] == 4])
    y_fw_p5 = _pct("Fairway", y_df[y_df["Par"] == 5])

    # Score breakdown (labels)
    lbl = y_df.get("Score Label", pd.Series([""] * y_df.shape[0], index=y_df.index))
    def _cnt(name): return int((lbl == name).sum())
    brk_eagle = _cnt("Eagle")
    brk_birdie = _cnt("Birdie")
    brk_par = _cnt("Par")
    brk_bogey = _cnt("Bogey")
    brk_double = _cnt("Double Bogey")
    brk_triplep = _cnt("Triple Bogey +")
    # percents
    def _pct_of_total(n): return (n / y_holes * 100.0) if y_holes else 0.0
    p_eagle = _pct_of_total(brk_eagle)
    p_birdie = _pct_of_total(brk_birdie)
    p_par = _pct_of_total(brk_par)
    p_bogey = _pct_of_total(brk_bogey)
    p_double = _pct_of_total(brk_double)
    p_triplep = _pct_of_total(brk_triplep)

    # Scramble% and Up&Down%
    y_scr_made = int(y_df["Scramble"].sum())
    y_scr_ops = int(y_df["Scramble Opportunity"].sum())
    y_scr_pct = (y_scr_made / y_scr_ops * 100.0) if y_scr_ops else 0.0

    y_up_made = int(((y_df["GIR"] == 0) & (y_df["Putts"] == 1)).sum())
    y_up_ops = int(y_df["Scramble Opportunity"].sum())
    y_up_pct = (y_up_made / y_up_ops * 100.0) if y_up_ops else 0.0

    # Sand Save% (only if present)
    if ("Sand Save" in y_df.columns) and ("Sand Save Opportunity" in y_df.columns):
        y_df["Sand Save"] = pd.to_numeric(y_df["Sand Save"], errors="coerce").fillna(0)
        y_df["Sand Save Opportunity"] = pd.to_numeric(y_df["Sand Save Opportunity"], errors="coerce").fillna(0)
        y_ss_made = int(y_df["Sand Save"].sum())
        y_ss_ops = int(y_df["Sand Save Opportunity"].sum())
        y_ss_pct = (y_ss_made / y_ss_ops * 100.0) if y_ss_ops else 0.0
    else:
        y_ss_pct = None  # not available

    # Putting details
    y_one_putts = int((y_df["Putts"] == 1).sum())
    y_three_plus = int((y_df["Putts"] >= 3).sum())
    y_bogey_rows = (lbl == "Bogey")
    y_three_putt_bogeys = int(((y_df["Putts"] >= 3) & y_bogey_rows).sum())
    y_three_putt_bogey_pct = (y_three_putt_bogeys / y_three_plus * 100.0) if y_three_plus else 0.0

    # Other tallies
    y_arnies = int(y_df["Arnie"].sum())
    y_seves = int(y_df["Seve"].sum())
    y_propars_plus = int(y_df["Pro Par"].sum() + y_df["Pro Birdie"].sum() + y_df["Pro Eagle+"].sum())

    # Lost balls
    y_lb_tee = int(y_df["Lost Ball Tee Shot Quantity"].sum())
    y_lb_appr = int(y_df["Lost Ball Approach Shot Quantity"].sum())
    y_lb_total = int(y_lb_tee + y_lb_appr)

    # Little badge helpers
    def _ice_if_low(pct, low=35):  # for % where higher is good
        return " 🧊" if pct < low else ""
    def _moon_if_ok(pct, good=33): # gentle badge if ≥ threshold
        return " 🌕" if pct >= good else ""

    # Build HTML
    yearly_html = f"""
    <div style="margin-top:10px; padding:12px; background:#262626; border-radius:12px; line-height:1.6;">
      <div style="font-weight:800; font-size:16px;">📅 Yearly Summary ➤ 2025 | {y_player}</div>
      <div>⛳ <b>Total Holes Played:</b> {y_holes}</div>
      <div>🎯 <b>Score:</b> {y_score_sum} ({'+' if y_delta>0 else ''}{y_delta})</div>
      <div>📌 <b>Avg Score ➤ 9H:</b> {y_avg9} | <b>18H:</b> {y_avg18}</div>
      <div>🟢 <b>Putts:</b> {y_putts_total} ({y_putts_per_hole:.2f}/hole | {y_putts_per18:.1f}/18{_ice_if_low(100 - min(y_putts_per18,100))})</div>
      <div>🏹 <b>Fairways Hit:</b> {y_fw_made} / {y_fw_attempts} ({y_fw_pct:.1f}%){_ice_if_low(y_fw_pct)}</div>
      <div>⛳ <b>GIR:</b> {y_gir_made} / {y_holes} ({y_gir_pct:.1f}%)</div>
      <div>🟡 <b>Unofficial GIR:</b> {y_ugir_made} / {y_holes} ({y_ugir_pct:.1f}%)</div>
      <div>🎯 <b>Birdie Putt Attempts ➤ ≤20ft:</b> {y_bpa_20} | ≤10ft: {y_bpa_10}</div>

      <div>📈 <b>Scoring Averages ➤</b> Par 3: {y_p3_avg:.2f} | Par 4: {y_p4_avg:.2f} | Par 5: {y_p5_avg:.2f}</div>
      <div>🎯 <b>GIR % ➤</b> Par 3: {y_gir_p3:.1f}%{_moon_if_ok(y_gir_p3)} | Par 4: {y_gir_p4:.1f}%{_moon_if_ok(y_gir_p4)} | Par 5: {y_gir_p5:.1f}%{_moon_if_ok(y_gir_p5)}</div>
      <div>🏹 <b>FW % ➤</b> Par 4: {y_fw_p4:.1f}%{_ice_if_low(y_fw_p4)} | Par 5: {y_fw_p5:.1f}%{_ice_if_low(y_fw_p5)}</div>

      <div>📊 <b>Score Breakdown ➤</b>
        🦅 E-{brk_eagle} | 🐥 B-{brk_birdie} | 🟩 P-{brk_par} | 🟠 Bgy-{brk_bogey} | 🔴 Dbl-{brk_double} | 💀 Trp+ {brk_triplep}<br>
        <span style="color:#aaa;">({p_eagle:.1f}% | {p_birdie:.1f}% | {p_par:.1f}% | {p_bogey:.1f}% | {p_double:.1f}% | {p_triplep:.1f}%)</span>
      </div>

      <div>🔁 <b>Scramble %:</b> {y_scr_pct:.1f}% | <b>U&amp;D %:</b> {y_up_pct:.1f}% | <b>Sand Save %:</b> {f"{y_ss_pct:.1f}%" if y_ss_pct is not None else "—"}</div>
      <div>🥄 <b>Putting ➤</b> ✅ 1P: {y_one_putts} | 🔴 3P: {y_three_plus} | 💀 3P Bgy: {y_three_putt_bogeys} ({y_three_putt_bogey_pct:.1f}% 💀)</div>
      <div>🧱 <b>Arnies:</b> {y_arnies} | 🎨 <b>Seves:</b> {y_seves} | 💪 <b>Pro Pars +:</b> {y_propars_plus}</div>
      <div>📦 <b>Lost Balls ➤</b> 🚀 Tee: {y_lb_tee} | 🎯 Approach: {y_lb_appr} | 📦 Total: {y_lb_total}</div>
    </div>
    """

    st.markdown(yearly_html, unsafe_allow_html=True)
else:
    st.info("No 2025 data for this player yet.")
# =================== END YEARLY SUMMARY (2025) =============================================


# =============== QUICK COMPARISONS (Current vs Prev 5 Rounds / Last 100 Holes) ===============
# Build a player history excluding the current round
_hist = df[df["Player Name"] == player].copy()
_hist["Date Played"] = pd.to_datetime(_hist["Date Played"], errors="coerce")
_current_round_id = selected_round
_current_round_date = pd.to_datetime(round_data["Date Played"].iloc[0], errors="coerce")

# Exclude current round rows from history
_hist_excl = _hist[_hist["Round Link"] != _current_round_id].copy()

# --- Helper functions ---
def _safe_num(x):
    try:
        return float(x)
    except Exception:
        return 0.0

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
    # Average total putts per round, over the rounds present in `dataframe`
    if dataframe.empty or "Putts" not in dataframe or "Round Link" not in dataframe:
        return None
    per_round = (
        dataframe.assign(P=pd.to_numeric(dataframe["Putts"], errors="coerce").fillna(0))
        .groupby("Round Link")["P"]
        .sum()
    )
    return per_round.mean() if not per_round.empty else None

def _delta_str(curr, ref, suffix=""):
    if curr is None or ref is None:
        return "—"
    diff = curr - ref
    arrow = "🔺" if diff > 0 else ("🔻" if diff < 0 else "—")
    # show absolute difference; signed with +/-
    return f"{arrow} {diff:+.1f}{suffix}"

def _fmt_val(v, suffix=""):
    if v is None:
        return "n/a"
    return f"{v:.1f}{suffix}"

# --- Windows: Prev 5 rounds, Last 100 holes ---
# Prev 5 rounds = 5 most recent unique Round Link values BEFORE the current round date (or simply excluding current)
_prev5_round_ids = (
    _hist_excl.sort_values("Date Played")
              .dropna(subset=["Round Link"])
              .drop_duplicates(subset=["Round Link"], keep="last")
              .sort_values("Date Played", ascending=False)["Round Link"]
              .head(5)
              .tolist()
)
_prev5_df = _hist_excl[_hist_excl["Round Link"].isin(_prev5_round_ids)].copy()

# Last 100 holes (regardless of round boundaries)
_last100_df = _hist_excl.sort_values(["Date Played", "Hole"], ascending=[False, True]).head(100).copy()

# --- Compute historical metrics ---
prev5_gir = _gir_pct(_prev5_df)
last100_gir = _gir_pct(_last100_df)

prev5_fw = _fw_pct_p45(_prev5_df)
last100_fw = _fw_pct_p45(_last100_df)

prev5_pph = _putts_per_hole(_prev5_df)          # putts per hole
last100_pph = _putts_per_hole(_last100_df)      # putts per hole

prev5_ppr = _putts_per_round_mean(_prev5_df)    # average putts per round

# For "Putts / Round" against last 100 holes, use a round-equivalent (18 * putts/HOLE)
last100_ppr_equiv = (last100_pph * 18.0) if last100_pph is not None else None

# --- Current values (already computed earlier in your script) ---
curr_pph = putts_per_hole                              # float
curr_ppr = float(total_putts)                          # total putts this round
curr_gir = gir_pct                                     # %
curr_fw  = fw_pct                                      # %

# --- Build comparison table HTML ---
comp_html = f"""
<div style="margin-top:8px; padding:12px; background:#262626; border-radius:12px;">
  <div style="font-weight:700; margin-bottom:8px;">📊 Quick Comparisons</div>
  <table style="width:100%; border-collapse:collapse; font-size:13px;">
    <thead>
      <tr style="background:#333;">
        <th style="text-align:left; padding:6px;">Metric</th>
        <th style="text-align:center; padding:6px;">Current</th>
        <th style="text-align:center; padding:6px;">Prev 5 (value)</th>
        <th style="text-align:center; padding:6px;">Δ vs Prev 5</th>
        <th style="text-align:center; padding:6px;">Last 100 (value)</th>
        <th style="text-align:center; padding:6px;">Δ vs Last 100</th>
      </tr>
    </thead>
    <tbody>
      <tr style="background:#2d2d2d;">
        <td style="padding:6px; font-weight:600;">Putts / Hole</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(curr_pph)}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(prev5_pph)}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_pph, prev5_pph)}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(last100_pph)}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_pph, last100_pph)}</td>
      </tr>
      <tr style="background:#262626;">
        <td style="padding:6px; font-weight:600;">Putts / Round</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(curr_ppr)}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(prev5_ppr)}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_ppr, prev5_ppr)}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(last100_ppr_equiv)}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_ppr, last100_ppr_equiv)}</td>
      </tr>
      <tr style="background:#2d2d2d;">
        <td style="padding:6px; font-weight:600;">GIR %</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(curr_gir, '%')}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(prev5_gir, '%')}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_gir, prev5_gir, '%')}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(last100_gir, '%')}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_gir, last100_gir, '%')}</td>
      </tr>
      <tr style="background:#262626;">
        <td style="padding:6px; font-weight:600;">Fairway % (P4/P5)</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(curr_fw, '%')}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(prev5_fw, '%')}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_fw, prev5_fw, '%')}</td>
        <td style="text-align:center; padding:6px;">{_fmt_val(last100_fw, '%')}</td>
        <td style="text-align:center; padding:6px;">{_delta_str(curr_fw, last100_fw, '%')}</td>
      </tr>
    </tbody>
  </table>
  <div style="margin-top:6px; font-size:12px; color:#aaa;">
    Putts / Round for “Last 100” is a round-equivalent = 18 × (putts/HOLE over last 100 holes).
  </div>
</div>
"""

st.markdown(comp_html, unsafe_allow_html=True)
# =============== END QUICK COMPARISONS =======================================================

# ---- Comparison: current round vs prev 5 rounds & last 100 holes (player history) ----
try:
    # Work off the full dataset for the selected player, excluding the current round for comparisons
    _player_all = df[df["Player Name"] == player].copy()
    _player_hist = _player_all[_player_all["Round Link"] != selected_round].copy()

    # Ensure types
    _player_hist["Date Played"] = pd.to_datetime(_player_hist["Date Played"], errors="coerce")
    _player_hist["Par"] = pd.to_numeric(_player_hist["Par"], errors="coerce")
    _player_hist["Hole Score"] = pd.to_numeric(_player_hist["Hole Score"], errors="coerce")
    _player_hist["Putts"] = pd.to_numeric(_player_hist["Putts"], errors="coerce")
    _player_hist["GIR"] = pd.to_numeric(_player_hist["GIR"], errors="coerce")
    _player_hist["Fairway"] = pd.to_numeric(_player_hist["Fairway"], errors="coerce")

    # If no history, skip quietly
    if not _player_hist.empty:
        # ---- Last 5 rounds window
        _round_order = (
            _player_hist[["Round Link", "Date Played"]]
            .dropna()
            .drop_duplicates()
            .sort_values("Date Played")
        )
        _last5_links = _round_order["Round Link"].tail(5).tolist()
        _last5 = _player_hist[_player_hist["Round Link"].isin(_last5_links)].copy()

        # ---- Last 100 holes window
        _hist_sorted = _player_hist.sort_values(["Date Played", "Round Link", "Hole"])
        _last100 = _hist_sorted.tail(100).copy()

        def _fw_pct(df_block):
            # Fairway % only on Par 4 & 5 attempts
            _block = df_block[df_block["Par"].isin([4, 5])]
            if _block.empty:
                return 0.0
            return float((_block["Fairway"].fillna(0).astype(float).mean()) * 100.0)

        def _gir_pct(df_block):
            if df_block.empty:
                return 0.0
            return float((df_block["GIR"].fillna(0).astype(float).mean()) * 100.0)

        def _putts_per_hole(df_block):
            if df_block.empty:
                return 0.0
            return float(df_block["Putts"].fillna(0).astype(float).mean())

        # History metrics
        p5_putts = _putts_per_hole(_last5) if not _last5.empty else None
        p5_gir   = _gir_pct(_last5)        if not _last5.empty else None
        p5_fw    = _fw_pct(_last5)         if not _last5.empty else None

        h100_putts = _putts_per_hole(_last100) if not _last100.empty else None
        h100_gir   = _gir_pct(_last100)        if not _last100.empty else None
        h100_fw    = _fw_pct(_last100)         if not _last100.empty else None

        # Current round metrics (already computed above)
        cur_putts = float(putts_per_hole) if holes_played else 0.0
        cur_gir   = float(gir_pct)
        cur_fw    = float(fw_pct)

        def _delta_emoji(diff, higher_is_better=True):
            # For putts (lower is better), invert the sense
            if higher_is_better:
                return "🔺" if diff > 0 else ("🔻" if diff < 0 else "⟷")
            else:
                return "🔻" if diff > 0 else ("🔺" if diff < 0 else "⟷")

        def _fmt_cell(cur_val, ref_val, is_pct=False, higher_is_better=True):
            if ref_val is None:
                return f"{cur_val:.2f}{'%' if is_pct else ''} <span style='color:#aaa'>(n/a)</span>"
            diff = cur_val - ref_val
            arrow = _delta_emoji(diff, higher_is_better=higher_is_better)
            if is_pct:
                return f"{cur_val:.1f}% <span style='opacity:0.9'>({arrow} {diff:+.1f}%)</span>"
            else:
                return f"{cur_val:.2f} <span style='opacity:0.9'>({arrow} {diff:+.2f})</span>"

        # Build a compact comparison table
        comp_html = f"""
        <div style="margin-top:8px; padding:10px 12px; background:#2a2a2a; border-radius:10px;">
          <div style="font-weight:700; margin-bottom:6px;">📊 Quick Comparisons</div>
          <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <thead>
              <tr style="text-align:left; background:#333;">
                <th style="padding:6px;">Metric</th>
                <th style="padding:6px;">Current Round</th>
                <th style="padding:6px;">vs Prev 5 Rounds</th>
                <th style="padding:6px;">vs Last 100 Holes</th>
              </tr>
            </thead>
            <tbody>
              <tr style="background:#2f2f2f;">
                <td style="padding:6px; font-weight:600;">Putts / Hole</td>
                <td style="padding:6px;">{cur_putts:.2f}</td>
                <td style="padding:6px;">{_fmt_cell(cur_putts, p5_putts, is_pct=False, higher_is_better=False)}</td>
                <td style="padding:6px;">{_fmt_cell(cur_putts, h100_putts, is_pct=False, higher_is_better=False)}</td>
              </tr>
              <tr style="background:#282828;">
                <td style="padding:6px; font-weight:600;">GIR %</td>
                <td style="padding:6px;">{cur_gir:.1f}%</td>
                <td style="padding:6px;">{_fmt_cell(cur_gir, p5_gir, is_pct=True, higher_is_better=True)}</td>
                <td style="padding:6px;">{_fmt_cell(cur_gir, h100_gir, is_pct=True, higher_is_better=True)}</td>
              </tr>
              <tr style="background:#2f2f2f;">
                <td style="padding:6px; font-weight:600;">Fairway % (P4/P5)</td>
                <td style="padding:6px;">{cur_fw:.1f}%</td>
                <td style="padding:6px;">{_fmt_cell(cur_fw, p5_fw, is_pct=True, higher_is_better=True)}</td>
                <td style="padding:6px;">{_fmt_cell(cur_fw, h100_fw, is_pct=True, higher_is_better=True)}</td>
              </tr>
            </tbody>
          </table>
        </div>
        """.strip()

        st.markdown(comp_html, unsafe_allow_html=True)

except Exception as _comp_err:
    # Stay silent in UI if comparisons fail; log to Streamlit text for debugging (optional)
    st.debug(f"Comparison block skipped: {_comp_err}")





# Compact stat cards row
# --- Callouts: Best/Worst hole & longest Par-or-Better streak ---
_delta = pd.to_numeric(round_data["Hole Score"], errors="coerce") - pd.to_numeric(round_data["Par"], errors="coerce")
_holes = round_data["Hole"].astype(int)
_labels = round_data.get("Score Label", pd.Series([""] * len(round_data), index=round_data.index))

def _fmt_delta(n: float) -> str:
    n = int(n)
    return "E" if n == 0 else (f"+{n}" if n > 0 else f"{n}")

# Best (lowest to par) & worst (highest to par)
best_idx = _delta.idxmin()
worst_idx = _delta.idxmax()
best_hole_num  = int(round_data.loc[best_idx, "Hole"])
best_delta_str = _fmt_delta(_delta.loc[best_idx])
best_label     = str(_labels.loc[best_idx])

worst_hole_num  = int(round_data.loc[worst_idx, "Hole"])
worst_delta_str = _fmt_delta(_delta.loc[worst_idx])
worst_label     = str(_labels.loc[worst_idx])

# Longest Par-or-Better streak (score <= par), reported as Hstart–Hend
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
<div style="margin-top:8px; padding:10px 12px; background:#2a2a2a; border-radius:10px; line-height:1.6;">
  <b>⭐ Best Hole:</b> H{best_hole_num} ({best_delta_str}) — {best_label}<br>
  <b>⚠️ Worst Hole:</b> H{worst_hole_num} ({worst_delta_str}) — {worst_label}<br>
  <b>🔗 Longest Par-or-Better Streak:</b> {streak_text}
</div>
"""
st.markdown(callouts_html, unsafe_allow_html=True)

import random, json, os
from pathlib import Path

# ---- Visuals (one long progress bar with on-bar labels) ----
st.divider()
st.markdown("### 📊 Score Mix")

# Order & counts (use your existing score_type_counts)
order = ["Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]
counts = [int(score_type_counts.get(k, 0)) for k in order]
total = sum(counts) or 1

df_mix = pd.DataFrame({
    "Category": order,
    "Count": counts,
    "Percent": [c / total * 100 for c in counts],
    "Group": ["All Holes"] * len(order)  # single bar
})

# Hide zero-width segments in the chart layer
df_plot = df_mix[df_mix["Count"] > 0].copy()
if df_plot.empty:
    df_plot = df_mix.copy()

# Color palette (tweak if you like)
color_scale = alt.Scale(
    domain=order,
    range=["#71c7ec", "#64dfb5", "#bdbdbd", "#f2c14e", "#ee6c4d", "#b23a48"]
)

# Build a base with stacked positions so we can center labels per segment
base = (
    alt.Chart(df_plot)
    # round Percent to 1 decimal for labels
    .transform_calculate(pct='round(datum.Percent * 10) / 10')
    # compute start/end of each stacked segment
    .transform_stack(stack='Count', as_=['start', 'end'], groupby=['Group'])
    # mid-point for label placement
    .transform_calculate(mid='(datum.start + datum.end) / 2')
)

# Bar segments (full-width normalized stack)
bar = (
    base.mark_bar(height=38)
    .encode(
        y=alt.Y("Group:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
        x=alt.X("end:Q", stack=None, axis=None),
        x2="start:Q",
        color=alt.Color("Category:N", scale=color_scale, legend=alt.Legend(orient="bottom")),
        tooltip=[
            alt.Tooltip("Category:N"),
            alt.Tooltip("Count:Q", title="Holes"),
            alt.Tooltip("pct:Q", title="% of Round", format=".1f"),
        ],
    )
)

# Text labels centered in each segment: "Category 12.5%"
text = (
    base.mark_text(baseline="middle", dy=0, fontWeight="bold")
    .encode(
        y="Group:N",
        x="mid:Q",
        text=alt.Text("label:N"),
        # Only show label if the segment is at least ~8% wide (prevents overlap)
        opacity=alt.condition("datum.Percent < 8", alt.value(0), alt.value(1))
    )
    # Build the label string on the fly: Category + space + pct + '%'
    .transform_calculate(label='datum.Category + " " + (format(datum.pct, ".1f")) + "%"')
)

st.altair_chart(
    (bar + text).configure_view(stroke=None).configure_axis(grid=False, domain=False),
    use_container_width=True,
)

# A small line under the bar with exact numbers for reference/printing
counts_line = " • ".join(
    f"{row.Category}: {row.Count} ({row.Percent:.1f}%)" for _, row in df_mix.iterrows()
)
st.caption(counts_line)
st.markdown(summary_details_html, unsafe_allow_html=True)
# ---- Hole-by-hole (Score vs Par) sparkline (integer ticks only) ----
st.markdown("#### Hole-by-Hole (Score vs Par)")
df_line = pd.DataFrame({
    "Hole": round_data["Hole"].astype(int),
    "Delta": (pd.to_numeric(round_data["Hole Score"], errors="coerce")
              - pd.to_numeric(round_data["Par"], errors="coerce")).astype(int)
}).sort_values("Hole")

# Build integer tick marks
delta_min = int(df_line["Delta"].min())
delta_max = int(df_line["Delta"].max())
if delta_min == delta_max:   # avoid a single-tick axis
    delta_min -= 1
    delta_max += 1
tick_vals = list(range(delta_min, delta_max + 1))

# Zero reference line
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(opacity=0.4).encode(y="y:Q")

# Line + points, with integer-only y-axis
line = alt.Chart(df_line).mark_line().encode(
    x=alt.X("Hole:O", sort=None, axis=alt.Axis(title=None)),
    y=alt.Y(
        "Delta:Q",
        scale=alt.Scale(domain=[delta_min, delta_max], nice=False, clamp=True),
        axis=alt.Axis(title="To Par", values=tick_vals, format="d", tickCount=len(tick_vals))
    )
)

pts = alt.Chart(df_line).mark_point(size=70).encode(
    x="Hole:O",
    y="Delta:Q",
    color=alt.condition("datum.Delta <= 0", alt.value("#64dfb5"), alt.value("#ee6c4d")),
    tooltip=[alt.Tooltip("Hole:O"), alt.Tooltip("Delta:Q", title="To Par", format="d")]
)

st.altair_chart(zero + line + pts, use_container_width=True)
# --- Score-to-Par audit & comparison (non-destructive debug panel) ---
with st.expander("🔎 Debug: Score-to-Par audit (current vs last 5 rounds / last 100 holes)", expanded=False):
    # ----- CURRENT ROUND (per-hole & totals) -----
    audit = round_data.copy()
    audit["ParN"]   = pd.to_numeric(audit["Par"], errors="coerce").fillna(0)
    audit["ScoreN"] = pd.to_numeric(audit["Hole Score"], errors="coerce").fillna(0)
    audit["Delta"]  = audit["ScoreN"] - audit["ParN"]
    cur_delta_sum   = int(audit["Delta"].sum())
    cur_score_sum   = int(audit["ScoreN"].sum())
    cur_par_sum     = int(audit["ParN"].sum())

    st.markdown("**Current round — per hole**")
    st.dataframe(
        audit.sort_values("Hole")[["Hole", "ParN", "ScoreN", "Delta"]]
        .rename(columns={"ParN":"Par","ScoreN":"Score"}),
        use_container_width=True
    )
    st.write(
        f"**Current totals** — Score: {cur_score_sum} | Par: {cur_par_sum} | "
        f"Δ (Score−Par): {cur_delta_sum:+d}"
    )

    # ----- PLAYER-WIDE HISTORY PREP -----
    player_df = df[df["Player Name"] == player].copy()
    player_df["Date Played"] = pd.to_datetime(player_df["Date Played"], errors="coerce")
    player_df["ParN"]   = pd.to_numeric(player_df["Par"], errors="coerce")
    player_df["ScoreN"] = pd.to_numeric(player_df["Hole Score"], errors="coerce")
    player_df = player_df.dropna(subset=["Date Played", "ParN", "ScoreN"])
    player_df = player_df.sort_values(["Date Played", "Round Link", "Hole"])

    # ----- LAST 5 ROUNDS (round-by-round Δ, then average) -----
    per_round = (
        player_df
        .groupby(["Round Link", "Date Played"], as_index=False)
        .apply(lambda g: pd.Series({
            "holes": g["Hole"].count(),
            "delta_sum": (g["ScoreN"] - g["ParN"]).sum()
        }))
        .reset_index(drop=True)
        .sort_values("Date Played")
    )
    last5 = per_round.tail(5)
    st.markdown("**Last 5 rounds (by date)**")
    st.dataframe(last5[["Date Played", "Round Link", "holes", "delta_sum"]], use_container_width=True)

    if not last5.empty:
        stp_last5_avg = last5["delta_sum"].mean()
        st.write(f"**Average score-to-par over last {len(last5)} rounds:** {stp_last5_avg:+.1f}")
    else:
        st.write("**Average score-to-par over last 5 rounds:** —")

    # ----- LAST 100 HOLES (hole-by-hole Δ total and per-18 scaling) -----
    last100 = player_df.tail(100)
    if not last100.empty:
        last100_delta_total = (last100["ScoreN"] - last100["ParN"]).sum()
        last100_per18 = last100_delta_total / len(last100) * 18
        st.write(
            f"**Last {len(last100)} holes** — Δ total: {last100_delta_total:+.1f} | "
            f"per-18: {last100_per18:+.1f}"
        )
    else:
        st.write("**Last 100 holes:** —")

    # ----- QUICK SANITY FLAGS (help spot data issues) -----
    # Duplicate hole rows in current round
    dup_holes = int(round_data.duplicated(subset=["Hole"]).sum())
    # Non-18-hole rounds in history (just a heads-up)
    non18 = per_round.loc[per_round["holes"] != 18, "holes"].value_counts().to_dict()
    if dup_holes:
        st.warning(f"Found {dup_holes} duplicate hole number(s) in the current round.")
    if non18:
        st.info(f"Historical rounds with non-18 hole counts detected: {non18}")




# --- Random Fun Fact (auto only) ---
def load_fun_facts():
    json_path = Path(__file__).parent / "fun_facts.json"
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [fact for facts in data.values() for fact in facts]
    # fallback
    return [
        "Golf balls were once made of wood.",
        "The term 'birdie' originated at Atlantic City Country Club in 1903."
    ]

all_facts = load_fun_facts()
random_fact = random.choice(all_facts)

st.markdown(
    f"<br><b>💡 Fun Fact:</b> {random_fact}",
    unsafe_allow_html=True
)
# ---- Download this round (HTML) ----
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
<div style="margin-top:16px">
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
