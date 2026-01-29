import pandas as pd
import streamlit as st
import altair as alt

# ===========================
# Page
# ===========================
st.set_page_config(page_title="Putting Stats", layout="wide")
st.title("⛳ Putting Proximity Table — Validation")

CSV_FILE = "Hole Data-Grid view (18).csv"

# ---------------------------
# Helpers
# ---------------------------
def _n(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

def _as_int(series, default=0):
    return _n(series, default=default).astype(int)

def _pct(numer, denom):
    return (numer / denom * 100.0) if denom else 0.0

def _pct_str(x):
    try:
        return f"{float(x):.1f}%"
    except:
        return "—"

PROX_COL = "Proximity to Hole - How far is your First Putt (FT)"
YARD_COL = "Approach Shot Distance (how far you had to the hole)"
CLUB_COL = "Approach Shot Club Used"
FEET_MADE_COL = "Feet of Putt Made (How far was the putt you made)"
SCORE_COL = "Hole Score"

# Bucket order (used for table + chart)
BUCKET_ORDER = ["0–3 ft", "3–6 ft", "6–10 ft", "10–16 ft", "16–22 ft", "23–30 ft", "30–40 ft", "> 40 ft"]

# ---------------------------
# Load
# ---------------------------
df = pd.read_csv(CSV_FILE, low_memory=False)
df["Date Played"] = pd.to_datetime(df.get("Date Played"), errors="coerce")
df["Year"] = df["Date Played"].dt.year

# Soft column safety
for col in [
    "Player Name", "Course Name", "Hole", "Par", "Putts",
    PROX_COL, YARD_COL, CLUB_COL, "3 Putt Bogey",
    "GIR", "Approach GIR",
    FEET_MADE_COL, SCORE_COL
]:
    if col not in df.columns:
        df[col] = pd.NA

# Normalize GIR fields (allow values like 1/0, True/False, Yes/No)
def _norm_yes_no(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().lower()
    if s in ["1", "true", "t", "yes", "y", "hit", "made"]:
        return "Yes"
    if s in ["0", "false", "f", "no", "n", "miss", "missed"]:
        return "No"
    if s == "yes":
        return "Yes"
    if s == "no":
        return "No"
    return pd.NA

df["GIR"] = df["GIR"].apply(_norm_yes_no)
df["Approach GIR"] = df["Approach GIR"].apply(_norm_yes_no)

# Numeric
df["Putts"] = _as_int(df["Putts"], 0)
df["Hole"] = _as_int(df["Hole"], 0)

par_num = pd.to_numeric(df.get("Par"), errors="coerce").round()
df["Par"] = par_num.where(par_num.isin([3, 4, 5]), pd.NA).astype("Int64")

df["HoleScoreN"] = pd.to_numeric(df[SCORE_COL], errors="coerce")
df["ProxN"] = pd.to_numeric(df[PROX_COL], errors="coerce")  # keep NaN if blank
df["YardN"] = pd.to_numeric(df[YARD_COL], errors="coerce")
df["FeetMadeN"] = pd.to_numeric(df[FEET_MADE_COL], errors="coerce")  # may be blank
df["3 Putt Bogey"] = _as_int(df["3 Putt Bogey"], 0)

# ---------------------------
# Year filter (All or single year)
# ---------------------------
years_available = sorted([int(y) for y in df["Year"].dropna().unique().tolist()])
year_options = ["All"] + years_available
sel_year = st.selectbox("Year", year_options, index=0)

if sel_year != "All":
    df = df[df["Year"] == int(sel_year)].copy()

if df.empty:
    st.warning("No rows found for the selected year.")
    st.stop()

yr_text = "All Years" if sel_year == "All" else f"Year={sel_year}"
st.caption(f"Showing: {yr_text}")

# ---------------------------
# Filters
# ---------------------------
c1, c2, c3, c4 = st.columns([1.6, 1.8, 1.2, 2.0], vertical_alignment="bottom")
with c1:
    players = sorted([x for x in df["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_player = st.selectbox("Player", ["All"] + players)
with c2:
    courses = sorted([x for x in df["Course Name"].dropna().unique().tolist() if str(x).strip() != ""])
    sel_course = st.selectbox("Course", ["All"] + courses)
with c3:
    pars = [3, 4, 5]
    sel_pars = st.multiselect("Par", pars, default=pars)
with c4:
    clubs = sorted([x for x in df[CLUB_COL].dropna().unique().tolist() if str(x).strip() != ""])
    sel_clubs = st.multiselect("Approach Club", clubs, default=[])

    sel_gir = st.selectbox("GIR", ["All", "Yes", "No"])
    sel_appr_gir = st.selectbox("Approach GIR", ["All", "Yes", "No"])

c5, c6 = st.columns([2.2, 1.8], vertical_alignment="bottom")
with c5:
    holes = sorted([int(x) for x in df["Hole"].dropna().unique().tolist() if int(x) > 0])
    sel_holes = st.multiselect("Hole Number", holes, default=[])
with c6:
    yard_series = df["YardN"].dropna()
    y_min = int(yard_series.min()) if not yard_series.empty else 0
    y_max = int(yard_series.max()) if not yard_series.empty else 300
    y_low, y_high = st.slider(
        "Approach Yardage Range",
        min_value=0,
        max_value=max(1, y_max),
        value=(max(0, y_min), y_max),
    )

f = df.copy()
if sel_player != "All":
    f = f[f["Player Name"] == sel_player]
if sel_course != "All":
    f = f[f["Course Name"] == sel_course]
if sel_pars:
    f = f[f["Par"].isin(sel_pars)]
if sel_clubs:
    f = f[f[CLUB_COL].isin(sel_clubs)]
if sel_holes:
    f = f[f["Hole"].isin(sel_holes)]
if sel_gir != "All":
    f = f[f["GIR"] == sel_gir]
if sel_appr_gir != "All":
    f = f[f["Approach GIR"] == sel_appr_gir]

# Yardage filter: keep blanks (do not exclude missing yardage)
f = f[(f["YardN"].isna()) | ((f["YardN"] >= y_low) & (f["YardN"] <= y_high))].copy()

# ---------------------------
# DATASET RULES
# ---------------------------
# Remove hole-outs entirely (Putts == 0)
# Exclude blanks in proximity, and exclude them from the dataset itself
f = f[(f["Putts"] > 0) & (f["ProxN"].notna())].copy()

if f.empty:
    st.info("No rows match filters after removing hole-outs and blank proximity.")
    st.stop()

# ---------------------------
# Bucket table
# ---------------------------
def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    tmp = frame.copy()
    tmp["PuttsN"] = _as_int(tmp["Putts"], 0)
    tmp["FeetMadeN"] = pd.to_numeric(tmp["FeetMadeN"], errors="coerce")
    tmp["ParN"] = pd.to_numeric(tmp["Par"], errors="coerce")
    tmp["HoleScoreN"] = pd.to_numeric(tmp["HoleScoreN"], errors="coerce")

    buckets = [
        ("0–3 ft",   (tmp["ProxN"] >= 0) & (tmp["ProxN"] <= 3)),
        ("3–6 ft",   (tmp["ProxN"] > 3) & (tmp["ProxN"] <= 6)),
        ("6–10 ft",  (tmp["ProxN"] > 6) & (tmp["ProxN"] <= 10)),
        ("10–16 ft", (tmp["ProxN"] > 10) & (tmp["ProxN"] <= 16)),
        ("16–22 ft", (tmp["ProxN"] > 16) & (tmp["ProxN"] <= 22)),
        ("23–30 ft", (tmp["ProxN"] >= 23) & (tmp["ProxN"] <= 30)),
        ("30–40 ft", (tmp["ProxN"] > 30) & (tmp["ProxN"] <= 40)),
        ("> 40 ft",  (tmp["ProxN"] > 40)),
    ]

    rows = []
    for label, mask in buckets:
        b = tmp[mask].copy()

        holes = int(len(b))
        total_putts = int(b["PuttsN"].sum())

        p1 = int((b["PuttsN"] == 1).sum())
        p2 = int((b["PuttsN"] == 2).sum())
        p3p = int((b["PuttsN"] >= 3).sum())
        bog3 = int(_as_int(b["3 Putt Bogey"], 0).sum())

        feet_made_total = float(_n(b["FeetMadeN"], 0).sum())
        feet_made_per_hole = (feet_made_total / holes) if holes else 0.0

        valid_ctx = b["HoleScoreN"].notna() & b["ParN"].notna()
        strokes_to_green = (b["HoleScoreN"] - b["PuttsN"])

        birdie_att_mask = valid_ctx & (strokes_to_green <= (b["ParN"] - 2))
        birdie_attempts = int(birdie_att_mask.sum())
        birdie_make_mask = valid_ctx & (b["HoleScoreN"] <= (b["ParN"] - 1))
        birdie_makes = int((birdie_att_mask & birdie_make_mask).sum())

        par_att_mask = valid_ctx & (strokes_to_green == (b["ParN"] - 1))
        par_attempts = int(par_att_mask.sum())
        par_make_mask = valid_ctx & (b["HoleScoreN"] == b["ParN"])
        par_makes = int((par_att_mask & par_make_mask).sum())

        rows.append({
            "Bucket": label,
            "Holes": holes,
            "Putts (Total)": total_putts,

            "1-putts": p1,
            "1-putt %": round(_pct(p1, holes), 1),

            "2-putts": p2,
            "2-putt %": round(_pct(p2, holes), 1),

            "3+ putts": p3p,
            "3+ putt %": round(_pct(p3p, holes), 1),

            "3-putt bogeys": bog3,
            "3-putt bogey %": round(_pct(bog3, holes), 1),

            "Birdie+ Attempts": birdie_attempts,
            "Birdie+ Makes": birdie_makes,
            "Birdie+ %": round(_pct(birdie_makes, birdie_attempts), 1),

            "Par Attempts": par_attempts,
            "Par Makes": par_makes,
            "Par %": round(_pct(par_makes, par_attempts), 1),

            "Feet Made (Total)": round(feet_made_total, 1),
            "Feet Made / Hole": round(feet_made_per_hole, 2),
        })

    out = pd.DataFrame(rows)
    out["Bucket"] = pd.Categorical(out["Bucket"], categories=BUCKET_ORDER, ordered=True)
    out = out.sort_values("Bucket").reset_index(drop=True)

    # Totals row
    if not out.empty:
        total_row = {"Bucket": "TOTAL"}
        for c in out.columns:
            if c != "Bucket":
                total_row[c] = float(out[c].sum())

        total_holes = int(total_row["Holes"]) if total_row["Holes"] else 0
        total_row["1-putt %"] = round(_pct(int(total_row["1-putts"]), total_holes), 1)
        total_row["2-putt %"] = round(_pct(int(total_row["2-putts"]), total_holes), 1)
        total_row["3+ putt %"] = round(_pct(int(total_row["3+ putts"]), total_holes), 1)
        total_row["3-putt bogey %"] = round(_pct(int(total_row["3-putt bogeys"]), total_holes), 1)

        total_ba = int(total_row["Birdie+ Attempts"])
        total_bm = int(total_row["Birdie+ Makes"])
        total_row["Birdie+ %"] = round(_pct(total_bm, total_ba), 1)

        total_pa = int(total_row["Par Attempts"])
        total_pm = int(total_row["Par Makes"])
        total_row["Par %"] = round(_pct(total_pm, total_pa), 1)

        feet_total = float(total_row["Feet Made (Total)"])
        total_row["Feet Made / Hole"] = round((feet_total / total_holes) if total_holes else 0.0, 2)

        out = pd.concat([out, pd.DataFrame([total_row])], ignore_index=True)

    return out

st.caption(
    f"Rules: {yr_text} • Hole-outs removed (Putts=0 excluded) • Blank proximity rows excluded • "
    "Buckets based on first-putt proximity • Totals shown in-table."
)

table = build_table(f)

# ---------------------------
# Table styling (no matplotlib)
# ---------------------------
percent_cols = ["1-putt %", "2-putt %", "3+ putt %", "3-putt bogey %", "Birdie+ %", "Par %"]

table_display = table.copy()
for col in percent_cols:
    table_display[col] = table_display[col].apply(_pct_str)

def _style_table(df_show: pd.DataFrame):
    def total_row_style(row):
        if str(row.get("Bucket", "")) == "TOTAL":
            return ["font-weight: 900; background-color: rgba(255,255,255,0.14); border-top: 2px solid rgba(255,255,255,0.35);"] * len(row)
        return [""] * len(row)

    def zebra_rows(row):
        idx = row.name
        base = "background-color: rgba(255,255,255,0.04);" if idx % 2 == 0 else "background-color: rgba(0,0,0,0.00);"
        return [base] * len(row)

    def pct_badge(val):
        try:
            v = float(str(val).replace("%", ""))
        except:
            return ""
        if v >= 60:
            return "background-color: rgba(46, 204, 113, 0.35); color: white; font-weight: 900;"
        if v >= 40:
            return "background-color: rgba(241, 196, 15, 0.28); color: white; font-weight: 900;"
        return "background-color: rgba(231, 76, 60, 0.26); color: white; font-weight: 900;"

    def feet_badge(val):
        try:
            v = float(val)
        except:
            return ""
        if v >= 3.0:
            return "background-color: rgba(52, 152, 219, 0.35); color: white; font-weight: 900;"
        if v >= 1.5:
            return "background-color: rgba(52, 152, 219, 0.22); color: white; font-weight: 900;"
        return "background-color: rgba(52, 152, 219, 0.10); color: white; font-weight: 800;"

    sty = (
        df_show.style
        .apply(zebra_rows, axis=1)
        .apply(total_row_style, axis=1)
        .set_properties(**{"text-align": "right", "padding": "6px 10px"})
        .set_properties(subset=["Bucket"], **{"text-align": "left", "font-weight": "900"})
        .applymap(pct_badge, subset=percent_cols)
        .applymap(feet_badge, subset=["Feet Made / Hole"])
        .format({
            "Holes": "{:,.0f}",
            "Putts (Total)": "{:,.0f}",
            "1-putts": "{:,.0f}",
            "2-putts": "{:,.0f}",
            "3+ putts": "{:,.0f}",
            "3-putt bogeys": "{:,.0f}",
            "Birdie+ Attempts": "{:,.0f}",
            "Birdie+ Makes": "{:,.0f}",
            "Par Attempts": "{:,.0f}",
            "Par Makes": "{:,.0f}",
            "Feet Made (Total)": "{:,.1f}",
            "Feet Made / Hole": "{:,.2f}",
        })
    )
    return sty

st.dataframe(_style_table(table_display), use_container_width=True, hide_index=True)

# ---------------------------
# Visual aid: Metric by distance (ALTair, labels show makes/attempts + %)
# ---------------------------
st.subheader("📊 Metric by Distance")

metric = st.radio(
    "Metric",
    options=[
        "Make % (1-putt %)",
        "Birdie+ %",
        "Par %",
        "3+ putt %",
        "3-putt bogey %",
    ],
    horizontal=True,
)

metric_map = {
    "Make % (1-putt %)": "1-putt %",
    "Birdie+ %": "Birdie+ %",
    "Par %": "Par %",
    "3+ putt %": "3+ putt %",
    "3-putt bogey %": "3-putt bogey %",
}
metric_col = metric_map[metric]

chart_df = table[table["Bucket"] != "TOTAL"][["Bucket", metric_col]].copy()
chart_df["Bucket"] = pd.Categorical(chart_df["Bucket"], categories=BUCKET_ORDER, ordered=True)
chart_df = chart_df.sort_values("Bucket")

chart_plot = chart_df.rename(columns={metric_col: "Value"}).copy()

metric_counts = {
    "Make % (1-putt %)": ("1-putts", "Holes"),
    "Birdie+ %": ("Birdie+ Makes", "Birdie+ Attempts"),
    "Par %": ("Par Makes", "Par Attempts"),
    "3+ putt %": ("3+ putts", "Holes"),
    "3-putt bogey %": ("3-putt bogeys", "Holes"),
}
num_col, den_col = metric_counts[metric]

counts_df = table[table["Bucket"] != "TOTAL"][["Bucket", num_col, den_col]].copy()
counts_df["Bucket"] = pd.Categorical(counts_df["Bucket"], categories=BUCKET_ORDER, ordered=True)
counts_df = counts_df.sort_values("Bucket")

chart_plot = chart_plot.merge(counts_df, on="Bucket", how="left")

def _label_row(r):
    try:
        num = int(r[num_col])
        den = int(r[den_col])
        val = float(r["Value"])
        return f"{num}/{den} {val:.1f}%"
    except:
        return "—"

chart_plot["Label"] = chart_plot.apply(_label_row, axis=1)

base = alt.Chart(chart_plot).encode(
    x=alt.X("Bucket:N", sort=BUCKET_ORDER, title=None),
    tooltip=[
        alt.Tooltip("Bucket:N", title="Bucket"),
        alt.Tooltip("Value:Q", title=metric, format=".1f"),
        alt.Tooltip(num_col + ":Q", title="Makes", format=",.0f"),
        alt.Tooltip(den_col + ":Q", title="Attempts", format=",.0f"),
    ],
)

# Keep your original colors
if metric in ["3+ putt %", "3-putt bogey %"]:
    bar_color = "#EF4444"   # red
    line_color = "#F97316"  # orange
else:
    bar_color = "#3B82F6"   # blue
    line_color = "#22C55E"  # green

bars = base.mark_bar(
    cornerRadiusTopLeft=7,
    cornerRadiusTopRight=7,
    opacity=0.95,
).encode(
    y=alt.Y("Value:Q", title=None),
    color=alt.value(bar_color),
)

line = base.mark_line(strokeWidth=3, opacity=0.9).encode(
    y="Value:Q",
    color=alt.value(line_color),
)

points = base.mark_point(size=120, filled=True, opacity=0.95).encode(
    y="Value:Q",
    color=alt.value(line_color),
)

labels = base.mark_text(
    dy=-10,
    fontSize=13,
    fontWeight="bold",
    color="white"
).encode(
    y="Value:Q",
    text="Label:N"
)

chart = (bars + line + points + labels).properties(height=380).configure_view(
    strokeOpacity=0
).configure_axis(
    labelColor="white",
    titleColor="white",
    gridColor="rgba(255,255,255,0.10)",
    tickColor="rgba(255,255,255,0.20)",
    domainColor="rgba(255,255,255,0.20)",
)

st.altair_chart(chart, use_container_width=True)

# ---------------------------
# Visual aid: Make % vs 3-Putt % by distance
# ---------------------------
st.subheader("⚔️ Make % vs 3-Putt % by Distance")

compare_df = table[table["Bucket"] != "TOTAL"][["Bucket", "1-putt %", "3+ putt %", "Holes", "1-putts", "3+ putts"]].copy()
compare_df["Bucket"] = pd.Categorical(compare_df["Bucket"], categories=BUCKET_ORDER, ordered=True)
compare_df = compare_df.sort_values("Bucket")

long_df = compare_df.melt(
    id_vars=["Bucket", "Holes", "1-putts", "3+ putts"],
    value_vars=["1-putt %", "3+ putt %"],
    var_name="Metric",
    value_name="Value"
)

metric_name = {
    "1-putt %": "Make % (1-putt %)",
    "3+ putt %": "3-Putt % (3+ putt %)"
}
long_df["Metric"] = long_df["Metric"].map(metric_name)

def _mk_label(r):
    try:
        if str(r["Metric"]).startswith("Make"):
            num = int(r["1-putts"])
            den = int(r["Holes"])
        else:
            num = int(r["3+ putts"])
            den = int(r["Holes"])
        return f"{num}/{den} {float(r['Value']):.1f}%"
    except:
        return "—"

long_df["Label"] = long_df.apply(_mk_label, axis=1)

base2 = alt.Chart(long_df).encode(
    x=alt.X("Bucket:N", sort=BUCKET_ORDER, title=None),
    y=alt.Y("Value:Q", title=None),
    color=alt.Color(
        "Metric:N",
        legend=alt.Legend(title=None, orient="top"),
        scale=alt.Scale(range=["#22C55E", "#EF4444"])
    ),
    tooltip=[
        alt.Tooltip("Bucket:N", title="Bucket"),
        alt.Tooltip("Metric:N", title="Metric"),
        alt.Tooltip("Value:Q", title="%", format=".1f"),
        alt.Tooltip("Holes:Q", title="Holes", format=",.0f"),
        alt.Tooltip("1-putts:Q", title="1-putts", format=",.0f"),
        alt.Tooltip("3+ putts:Q", title="3+ putts", format=",.0f"),
    ],
)

lines2 = base2.mark_line(strokeWidth=4, opacity=0.9)
points2 = base2.mark_point(size=140, filled=True, opacity=0.95)

labels2 = base2.mark_text(
    dy=-12,
    fontSize=12,
    fontWeight="bold",
    color="white"
).encode(
    text="Label:N"
)

chart2 = (lines2 + points2 + labels2).properties(height=360).configure_view(
    strokeOpacity=0
).configure_axis(
    labelColor="white",
    titleColor="white",
    gridColor="rgba(255,255,255,0.10)",
    tickColor="rgba(255,255,255,0.20)",
    domainColor="rgba(255,255,255,0.20)",
)

st.altair_chart(chart2, use_container_width=True)

# ---------------------------
# Visual aid: Attempts by Distance (bars) + Make % (line + dots + label)
# ---------------------------
st.subheader("📈 Attempts by Distance (Bars) + Make % (Line + Dots)")

att_df = table[table["Bucket"] != "TOTAL"][["Bucket", "Holes", "1-putts", "1-putt %"]].copy()
att_df["Bucket"] = pd.Categorical(att_df["Bucket"], categories=BUCKET_ORDER, ordered=True)
att_df = att_df.sort_values("Bucket")

att_df["Label"] = att_df.apply(
    lambda r: f"{int(r['1-putts'])}/{int(r['Holes'])} {float(r['1-putt %']):.1f}%" if int(r["Holes"]) else "—",
    axis=1
)

base3 = alt.Chart(att_df).encode(
    x=alt.X("Bucket:N", sort=BUCKET_ORDER, title=None),
    tooltip=[
        alt.Tooltip("Bucket:N", title="Bucket"),
        alt.Tooltip("Holes:Q", title="Attempts (Holes)", format=",.0f"),
        alt.Tooltip("1-putts:Q", title="Makes (1-putts)", format=",.0f"),
        alt.Tooltip("1-putt %:Q", title="Make %", format=".1f"),
    ],
)

# Bars = attempts (holes)
bars3 = base3.mark_bar(
    cornerRadiusTopLeft=7,
    cornerRadiusTopRight=7,
    opacity=0.90
).encode(
    y=alt.Y("Holes:Q", title="Attempts (Holes)"),
    color=alt.value("#60A5FA"),
)

# Line = make% (right axis)
line3 = base3.mark_line(
    strokeWidth=4,
    opacity=0.9
).encode(
    y=alt.Y("1-putt %:Q", title="Make % (1-putt %)", axis=alt.Axis(orient="right")),
    color=alt.value("#22C55E"),
)

# Dots = make% points
points3 = base3.mark_point(
    size=160,
    filled=True,
    opacity=0.95
).encode(
    y=alt.Y("1-putt %:Q", axis=alt.Axis(orient="right")),
    color=alt.value("#22C55E"),
)

# Labels next to dots
labels3 = base3.mark_text(
    dx=10,
    dy=-8,
    fontSize=12,
    fontWeight="bold",
    color="white"
).encode(
    y=alt.Y("1-putt %:Q", axis=alt.Axis(orient="right")),
    text="Label:N"
)

chart3 = alt.layer(bars3, line3, points3, labels3).resolve_scale(
    y="independent"
).properties(height=380).configure_view(
    strokeOpacity=0
).configure_axis(
    labelColor="white",
    titleColor="white",
    gridColor="rgba(255,255,255,0.10)",
    tickColor="rgba(255,255,255,0.20)",
    domainColor="rgba(255,255,255,0.20)",
)

st.altair_chart(chart3, use_container_width=True)
