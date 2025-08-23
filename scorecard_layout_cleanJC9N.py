import pandas as pd
import streamlit as st
import datetime

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

st.markdown(table_html, unsafe_allow_html=True)

# Summary shown below the table
st.markdown(f"""
<div style="margin-top:20px;">
  <h3 style="font-size:16px; font-weight:normal; text-align:center; color:#fff;">
    🏌️ {player} | {course} | {date}
  </h3>
  <div style="background:#222; padding:12px; border-radius:10px; color:#fff; font-size:13px; line-height:1.6; max-width:700px; margin:auto;">
    <b>📊 Round Totals</b><br>
    Score: {total_score}<br>
    Putts: {total_putts}<br>
    Fairways: {fw_hit}/{fw_att} ({fw_pct})<br>
    GIR: {gir_hit}/{gir_att} ({gir_pct})<br>
    Arnies: {arnie_total}<br><br>
    <b>📈 Scoring Averages</b><br>
    Par 3 Avg: {avg_par3}<br>
    Par 4 Avg: {avg_par4}<br>
    Par 5 Avg: {avg_par5}<br><br>
    <b>🔍 Advanced Insights</b><br>
    Total 1 Putts: {total_1_putts}<br>
    Total 3+ Putts: {total_3_plus_putts}<br>
    3-Putt Bogeys: {total_3_putt_bogeys}
  </div>
</div>
""", unsafe_allow_html=True)
