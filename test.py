# slicer.py
# Streamlit “View Slicer” — works for ANY filtered slice of your hole data
# - Round scorecard
# - Slice summary (totals + score mix w/ counts + category mix)
# - Player comparison (2–4 players) with counts + % + best stat highlighting + badges + rank leaderboard + quick charts
# - Leaderboard Dashboard (all players in slice) with grouped mini leaderboards (Qty + %)
# - Visual Dashboard (charts-only) w/ same filters + simple/full density + focus section
# - NEW: Approach Analytics (Distance + Club + Heatmap "best of both worlds") for GIR (ignoring Approach GIR for now)
# - Robust numeric parsing
# - Windows-safe date formatting
# - Filter by "Golf Trip"
# - Putts: Total Putts + Putts/Hole + Putts/18
# - Par 3/4/5 scoring averages + Par 3/4/5 GIR% (Qty + %)
# - Date range filter (perfect for trips)
# - Min holes filter (prevents small-sample “wins”)
# - Trip Digest export (copy/paste + CSV download)

import pandas as pd
import streamlit as st
import datetime
import altair as alt

# =========================
# Config
# =========================
st.set_page_config(page_title="Golf Slicer", layout="wide")

CSV_FILE = "Hole Data-Grid view (18).csv"

PROX_COL = "Proximity to Hole - How far is your First Putt (FT)"
YARD_COL = "Approach Shot Distance (how far you had to the hole)"
CLUB_COL = "Approach Shot Club Used"
FEET_MADE_COL = "Feet of Putt Made (How far was the putt you made)"
GOLF_TRIP_COL = "Golf Trip"

NEEDED_COLS = [
    "Date Played", "Player Name", "Course Name", "Round Link", "Hole", "Par", "Yards",
    "Hole Score", "Putts", "Fairway", "GIR", "Approach GIR",
    "Arnie", "Seve", "Hole Out",
    "Scramble", "Scramble Opportunity",
    "Lost Ball Tee Shot Quantity", "Lost Ball Approach Shot Quantity",
    "Score Label",
    "3 Putt Bogey",
    "Pro Par", "Pro Birdie", "Pro Eagle+",
    "Approach Shot Direction Miss",
    GOLF_TRIP_COL,
    CLUB_COL, YARD_COL, PROX_COL, FEET_MADE_COL
]

# =========================
# Global style
# =========================
st.markdown(
    """
<style>
  .dash-wrap {margin-top: 0.25rem;}
  .dash-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 14px 14px 10px 14px;
    box-shadow: 0 10px 24px rgba(0,0,0,.18);
  }
  .dash-title {
    font-size: 1.05rem;
    font-weight: 800;
    letter-spacing: .2px;
    margin: 0 0 .35rem 0;
  }
  .dash-sub {
    font-size: .85rem;
    opacity: .8;
    margin: 0 0 .6rem 0;
  }
  .pill {
    display:inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.14);
    background: rgba(255,255,255,0.05);
    font-size: .78rem;
    margin-right: 6px;
    margin-bottom: 6px;
  }
  .section-h {
    margin: .4rem 0 .15rem 0;
    font-weight: 900;
    font-size: 1.05rem;
    letter-spacing: .2px;
  }
  .muted {opacity: .82;}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# Helpers (robust)
# =========================
def _ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _num(x, default=0):
    if isinstance(x, pd.DataFrame):
        out = x.apply(pd.to_numeric, errors="coerce")
        return out.fillna(default)
    if isinstance(x, pd.Series):
        return pd.to_numeric(x, errors="coerce").fillna(default)
    return pd.to_numeric(pd.Series(x), errors="coerce").fillna(default)

def _as_int(series, default=0):
    return _num(series, default=default).astype(int)

def _norm_yes_no(v):
    if pd.isna(v):
        return pd.NA
    s = str(v).strip().lower()
    if s in ["1", "true", "t", "yes", "y", "hit", "made"]:
        return "Yes"
    if s in ["0", "false", "f", "no", "n", "miss", "missed"]:
        return "No"
    return pd.NA

def _fmt_date(dt):
    if pd.isna(dt):
        return "—"
    try:
        return pd.to_datetime(dt).strftime("%m/%d/%y").lstrip("0").replace("/0", "/")
    except Exception:
        return "—"

def _pct(n, d):
    return (n / d * 100.0) if d else 0.0

def _fmt_pct1(p):
    # p is in 0-100 space
    if p is None or pd.isna(p):
        return "—"
    try:
        return f"{float(p):.1f}%"
    except Exception:
        return "—"

def _fmt_to_par(n: float):
    if pd.isna(n):
        return "—"
    try:
        n = float(n)
    except Exception:
        return "—"
    if abs(n) < 1e-9:
        return "E"
    if abs(n - round(n)) < 1e-9:
        n = int(round(n))
        return f"+{n}" if n > 0 else f"{n}"
    return f"+{n:.1f}" if n > 0 else f"{n:.1f}"

def _emoji(pct):
    if pct >= 50:
        return "🔥"
    if pct < 25:
        return "❄️"
    return ""

def _safe_str(x):
    return "" if pd.isna(x) else str(x)

def _fmt_avg(x):
    return "—" if (x is None or pd.isna(x)) else f"{float(x):.2f}"

def _cnt_pair(a, b):
    try:
        return f"{int(a)}/{int(b)}"
    except:
        return "—"

def _to_tsv(df_: pd.DataFrame) -> str:
    if df_ is None or df_.empty:
        return ""
    return df_.to_csv(sep="\t", index=False)

# =========================
# Load
# =========================
st.title("🧩 Golf Slicer — Any View / Any Slice")

df = pd.read_csv(CSV_FILE, low_memory=False)
df = _ensure_cols(df, NEEDED_COLS)

df["Date Played"] = pd.to_datetime(df["Date Played"], errors="coerce")
df = df.dropna(subset=["Date Played", "Player Name"]).copy()
df["Year"] = df["Date Played"].dt.year
df["Month"] = df["Date Played"].dt.strftime("%B")

# Normalize key numeric fields
df["Hole"] = _as_int(df["Hole"], 0)
df["Par"] = _as_int(df["Par"], 0)
df["Yards"] = _as_int(df["Yards"], 0)
df["Hole Score"] = _as_int(df["Hole Score"], 0)
df["Putts"] = _as_int(df["Putts"], 0)

# Normalize yes/no fields
df["GIR"] = df["GIR"].apply(_norm_yes_no)
df["Approach GIR"] = df["Approach GIR"].apply(_norm_yes_no)

# 1/0 fields that should be numeric for rollups
for col in [
    "Fairway", "Arnie", "Seve", "Hole Out",
    "Scramble", "Scramble Opportunity",
    "3 Putt Bogey",
    "Lost Ball Tee Shot Quantity", "Lost Ball Approach Shot Quantity",
    "Pro Par", "Pro Birdie", "Pro Eagle+"
]:
    df[col] = _as_int(df[col], 0)

# =========================
# Filters (Sidebar)
# =========================
st.sidebar.header("🔍 Filters (build your slice)")

players = sorted([x for x in df["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
courses = sorted([x for x in df["Course Name"].dropna().unique().tolist() if str(x).strip() != ""])
years = sorted([int(x) for x in df["Year"].dropna().unique().tolist()], reverse=True)
months = [datetime.date(2000, i, 1).strftime("%B") for i in range(1, 13)]

golf_trips = sorted([x for x in df[GOLF_TRIP_COL].dropna().unique().tolist() if str(x).strip() != ""])
sel_trips = st.sidebar.multiselect("Golf Trip", golf_trips, default=[])

# ✅ Date range (always available)
min_dt = df["Date Played"].min()
max_dt = df["Date Played"].max()
min_d = min_dt.date() if pd.notna(min_dt) else datetime.date.today()
max_d = max_dt.date() if pd.notna(max_dt) else datetime.date.today()

date_start, date_end = st.sidebar.date_input(
    "Date Range",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
)

# Normalize date_input edge-cases
if isinstance(date_start, (list, tuple)) and len(date_start) == 2:
    date_start, date_end = date_start[0], date_start[1]
if date_start > date_end:
    date_start, date_end = date_end, date_start

sel_players = st.sidebar.multiselect("Players", players, default=[])
sel_courses = st.sidebar.multiselect("Courses", courses, default=[])
sel_years = st.sidebar.multiselect("Years", years, default=[])
sel_months = st.sidebar.multiselect("Months", months, default=[])

sel_pars = st.sidebar.multiselect("Par", [3, 4, 5], default=[3, 4, 5])

clubs = sorted([x for x in df[CLUB_COL].dropna().unique().tolist() if str(x).strip() != ""])
sel_clubs = st.sidebar.multiselect("Approach Club", clubs, default=[])

sel_fw = st.sidebar.selectbox("Fairway (P4/P5 only)", ["All", "Yes", "No"], index=0)
sel_gir = st.sidebar.selectbox("GIR", ["All", "Yes", "No"], index=0)
sel_appr_gir = st.sidebar.selectbox("Approach GIR", ["All", "Yes", "No"], index=0)

holes = sorted([int(x) for x in df["Hole"].dropna().unique().tolist() if int(x) > 0])
sel_holes = st.sidebar.multiselect("Hole Number", holes, default=[])

yard_series = _num(df.get(YARD_COL), default=pd.NA).dropna()
y_min = int(yard_series.min()) if not yard_series.empty else 0
y_max = int(yard_series.max()) if not yard_series.empty else 300
y_low, y_high = st.sidebar.slider(
    "Approach Yardage Range",
    min_value=0,
    max_value=max(1, y_max),
    value=(max(0, y_min), y_max),
)

# =========================
# Apply filters -> base slice (NO player filter for compare/leaderboard/visual/analytics by default)
# =========================
base_f = df.copy()

# Date range filter
base_f = base_f[
    (base_f["Date Played"] >= pd.to_datetime(date_start)) &
    (base_f["Date Played"] <= pd.to_datetime(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
].copy()

if sel_trips:
    base_f = base_f[base_f[GOLF_TRIP_COL].isin(sel_trips)]

if sel_courses:
    base_f = base_f[base_f["Course Name"].isin(sel_courses)]
if sel_years:
    base_f = base_f[base_f["Year"].isin([int(x) for x in sel_years])]
if sel_months:
    base_f = base_f[base_f["Month"].isin(sel_months)]
if sel_pars:
    base_f = base_f[base_f["Par"].isin(sel_pars)]
if sel_clubs:
    base_f = base_f[base_f[CLUB_COL].isin(sel_clubs)]
if sel_holes:
    base_f = base_f[base_f["Hole"].isin(sel_holes)]

if sel_fw != "All":
    p45_mask = base_f["Par"].isin([4, 5])
    if sel_fw == "Yes":
        base_f = base_f[~p45_mask | (base_f["Fairway"] == 1)]
    else:
        base_f = base_f[~p45_mask | (base_f["Fairway"] == 0)]

if sel_gir != "All":
    base_f = base_f[base_f["GIR"] == sel_gir]
if sel_appr_gir != "All":
    base_f = base_f[base_f["Approach GIR"] == sel_appr_gir]

yard_n_base = _num(base_f.get(YARD_COL), default=pd.NA)
base_f = base_f[(yard_n_base.isna()) | ((yard_n_base >= y_low) & (yard_n_base <= y_high))].copy()

# =========================
# Single-slice frame (f) = base + optional sidebar Player filter
# =========================
f = base_f.copy()
if sel_players:
    f = f[f["Player Name"].isin(sel_players)]

if f.empty:
    st.warning("No rows match your filters.")
    st.stop()

slice_for_summary = f.copy()

# =========================
# Mode selector
# =========================
st.subheader("Mode")
mode = st.radio(
    "Choose what to render for the current slice",
    [
        "📦 Slice Summary (Any View)",
        "🧾 Round Scorecard (by Round Link)",
        "🎯 Hole Scorecard (last 18 for a specific hole)",
        "👥 Player Comparison (same slice)",
        "🏆 Leaderboard Dashboard (multi-stat)",
        "📊 Visual Dashboard (charts-only)",
        "📈 Approach Analytics (Distance + Club + Heatmap)"
    ],
    horizontal=True
)

# =========================
# Summary Builder (for ANY slice)
# =========================
def build_summary(b: pd.DataFrame) -> dict:
    out = {}

    holes_played = int(b.shape[0])
    out["holes_played"] = holes_played

    score_total = int(_num(b["Hole Score"], 0).sum())
    par_total = int(_num(b["Par"], 0).sum())
    out["score_total"] = score_total
    out["to_par"] = score_total - par_total

    putts_total = int(_num(b["Putts"], 0).sum())
    out["putts_total"] = putts_total
    out["putts_per_hole"] = (putts_total / holes_played) if holes_played else 0.0
    out["putts_per_18"] = out["putts_per_hole"] * 18.0

    p45 = b[b["Par"].isin([4, 5])]
    fw_made = int(_num(p45["Fairway"], 0).sum())
    fw_att = int(p45.shape[0])
    out["fw_made"] = fw_made
    out["fw_att"] = fw_att
    out["fw_pct"] = _pct(fw_made, fw_att)

    gir_made = int((b["GIR"] == "Yes").sum())
    out["gir_made"] = gir_made
    out["gir_att"] = holes_played
    out["gir_pct"] = _pct(gir_made, holes_played)

    p45_fw = p45[p45["Fairway"] == 1]
    fw_gir_att = int(p45_fw.shape[0])
    fw_gir_made = int((p45_fw["GIR"] == "Yes").sum())
    out["fw_gir_made"] = fw_gir_made
    out["fw_gir_att"] = fw_gir_att
    out["fw_gir_pct"] = _pct(fw_gir_made, fw_gir_att)

    scr_made = int(_num(b.get("Scramble", 0), 0).sum())
    scr_ops = int(_num(b.get("Scramble Opportunity", 0), 0).sum())
    out["scr_made"] = scr_made
    out["scr_ops"] = scr_ops
    out["scr_pct"] = _pct(scr_made, scr_ops)

    gir_yes = (b["GIR"] == "Yes")
    putt1 = (_as_int(b["Putts"], 0) == 1)
    up_made = int((~gir_yes & putt1).sum())
    out["ud_made"] = up_made
    out["ud_ops"] = scr_ops
    out["ud_pct"] = _pct(up_made, scr_ops)

    lb_tee = int(_num(b.get("Lost Ball Tee Shot Quantity", 0), 0).sum())
    lb_appr = int(_num(b.get("Lost Ball Approach Shot Quantity", 0), 0).sum())
    out["lb_tee"] = lb_tee
    out["lb_appr"] = lb_appr
    out["lb_total"] = lb_tee + lb_appr

    one_putts = int((_as_int(b["Putts"], 0) == 1).sum())
    out["one_putts"] = one_putts
    out["one_putt_pct"] = _pct(one_putts, holes_played)

    three_plus_putts = int((_as_int(b["Putts"], 0) >= 3).sum())
    out["three_plus_putts"] = three_plus_putts
    out["three_plus_putt_pct"] = _pct(three_plus_putts, holes_played)

    three_putt_bogeys = int(_num(b.get("3 Putt Bogey", 0), 0).sum())
    out["three_putt_bogeys"] = three_putt_bogeys
    out["three_putt_bogey_att"] = int(out["gir_made"])
    out["three_putt_bogey_pct"] = _pct(three_putt_bogeys, out["three_putt_bogey_att"])

    pro_cols = [c for c in ["Pro Par", "Pro Birdie", "Pro Eagle+"] if c in b.columns]
    out["pro_pars_plus"] = int(_num(b[pro_cols], 0).sum().sum()) if pro_cols else 0

    out["arnies"] = int(_num(b.get("Arnie", 0), 0).sum())
    out["seves"] = int(_num(b.get("Seve", 0), 0).sum())
    out["hole_outs"] = int(_num(b.get("Hole Out", 0), 0).sum())

    def _avg_for_par(p):
        block = b[b["Par"] == p]
        if block.empty:
            return None
        return float(_num(block["Hole Score"], 0).mean())

    out["avg_p3"] = _avg_for_par(3)
    out["avg_p4"] = _avg_for_par(4)
    out["avg_p5"] = _avg_for_par(5)

    if out["avg_p3"] is not None and out["avg_p4"] is not None and out["avg_p5"] is not None:
        par72_score = (out["avg_p3"] * 4) + (out["avg_p4"] * 10) + (out["avg_p5"] * 4)
        out["par72_score"] = float(par72_score)
        out["par72_to_par"] = float(par72_score - 72)
    else:
        out["par72_score"] = None
        out["par72_to_par"] = None

    def _gir_by_par(p):
        block = b[b["Par"] == p]
        t = int(block.shape[0])
        m = int((block["GIR"] == "Yes").sum())
        return m, t, _pct(m, t)

    out["gir3"] = _gir_by_par(3)
    out["gir4"] = _gir_by_par(4)
    out["gir5"] = _gir_by_par(5)

    def _fw_by_par(p):
        block = b[b["Par"] == p]
        t = int(block.shape[0])
        m = int(_num(block["Fairway"], 0).sum()) if t else 0
        return m, t, _pct(m, t)

    out["fw4"] = _fw_by_par(4)
    out["fw5"] = _fw_by_par(5)

    def _gir_from_fw_by_par(p):
        block = b[b["Par"] == p]
        if block.empty:
            return 0, 0, 0.0
        fw_block = block[block["Fairway"] == 1]
        att = int(fw_block.shape[0])
        made = int((fw_block["GIR"] == "Yes").sum())
        return made, att, _pct(made, att)

    out["fw_gir4"] = _gir_from_fw_by_par(4)
    out["fw_gir5"] = _gir_from_fw_by_par(5)

    return out

# =========================
# Summary cards
# =========================
def render_summary_cards(summary: dict, title="📦 Current Slice — Summary"):
    st.subheader(title)

    def _cnt(n):
        try:
            return f"{int(n):,}"
        except:
            return "—"

    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Holes", _cnt(summary["holes_played"]))
    c2.metric("Score", f'{_cnt(summary["score_total"])} ({_fmt_to_par(summary["to_par"])})')
    c3.metric("Putts", _cnt(summary["putts_total"]), delta=f'{summary["putts_per_hole"]:.2f}/hole • {summary["putts_per_18"]:.1f}/18')
    c4.metric("GIR", f'{summary["gir_pct"]:.1f}% {_emoji(summary["gir_pct"])}', delta=_cnt_pair(summary["gir_made"], summary["gir_att"]))
    c5.metric("FW (P4/P5)", f'{summary["fw_pct"]:.1f}% {_emoji(summary["fw_pct"])}', delta=_cnt_pair(summary["fw_made"], summary["fw_att"]))
    c6.metric("GIR from FW (P4/P5)", f'{summary["fw_gir_pct"]:.1f}% {_emoji(summary["fw_gir_pct"])}', delta=_cnt_pair(summary["fw_gir_made"], summary["fw_gir_att"]))
    c7.metric("Scrambles", f'{summary["scr_pct"]:.1f}%', delta=_cnt_pair(summary["scr_made"], summary["scr_ops"]))

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric("Up & Downs", f'{summary["ud_pct"]:.1f}%', delta=_cnt_pair(summary["ud_made"], summary["ud_ops"]))
    d2.metric("1 Putts", f'{summary["one_putt_pct"]:.1f}%', delta=f'{int(summary["one_putts"]):,}/{int(summary["holes_played"]):,}')
    d3.metric("3+ Putts", f'{summary["three_plus_putt_pct"]:.1f}%', delta=f'{int(summary["three_plus_putts"]):,}/{int(summary["holes_played"]):,}')
    d4.metric("3P Bogeys", f'{summary["three_putt_bogey_pct"]:.1f}%', delta=f'{int(summary["three_putt_bogeys"]):,}/{int(summary.get("three_putt_bogey_att", summary["gir_made"])):,}')
    d5.metric("Lost Balls", _cnt(summary["lb_total"]), delta=f"Tee {_cnt(summary['lb_tee'])} / Appr {_cnt(summary['lb_appr'])}")
    d6.metric("Pro Pars+", _cnt(summary["pro_pars_plus"]))

    st.markdown(
        f"""
**📈 Scoring Averages**
- Par 3: {_fmt_avg(summary["avg_p3"])}
- Par 4: {_fmt_avg(summary["avg_p4"])}
- Par 5: {_fmt_avg(summary["avg_p5"])}
- Par 72 Projection: {"—" if summary["par72_score"] is None else f'{summary["par72_score"]:.1f} ({_fmt_to_par(summary["par72_to_par"])})'}

**🎯 GIR by Hole Type**
- Par 3: {summary["gir3"][0]}/{summary["gir3"][1]} ({summary["gir3"][2]:.1f}%) {_emoji(summary["gir3"][2])}
- Par 4: {summary["gir4"][0]}/{summary["gir4"][1]} ({summary["gir4"][2]:.1f}%) {_emoji(summary["gir4"][2])}
- Par 5: {summary["gir5"][0]}/{summary["gir5"][1]} ({summary["gir5"][2]:.1f}%) {_emoji(summary["gir5"][2])}

**🎯 GIR from Fairway (by Hole Type)**
- Par 4: {summary["fw_gir4"][0]}/{summary["fw_gir4"][1]} ({summary["fw_gir4"][2]:.1f}%) {_emoji(summary["fw_gir4"][2])}
- Par 5: {summary["fw_gir5"][0]}/{summary["fw_gir5"][1]} ({summary["fw_gir5"][2]:.1f}%) {_emoji(summary["fw_gir5"][2])}

**🏹 Fairways by Hole Type**
- Par 4: {summary["fw4"][0]}/{summary["fw4"][1]} ({summary["fw4"][2]:.1f}%) {_emoji(summary["fw4"][2])}
- Par 5: {summary["fw5"][0]}/{summary["fw5"][1]} ({summary["fw5"][2]:.1f}%) {_emoji(summary["fw5"][2])}

**✨ Specials**
- Hole Outs: {summary["hole_outs"]} | Arnies: {summary["arnies"]} | Seves: {summary["seves"]}
        """.strip()
    )

# =========================
# Score Mix (Counts + %) + Category Mix (Counts + %)
# =========================
def render_score_mix(b: pd.DataFrame, title="📊 Score Mix — Current Slice"):
    st.subheader(title)

    def _fmt_count_pct(count, total):
        return f"{int(count)} ({_pct(count, total):.1f}%)" if total else f"{int(count)} (—)"

    # ---- Score Label Mix ----
    if "Score Label" in b.columns:
        order = ["Albatross", "Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]
        vc = b["Score Label"].value_counts()

        counts = [int(vc.get(k, 0)) for k in order]
        total = sum(counts) or 1

        df_mix = pd.DataFrame({"Category": order, "Count": counts})
        df_mix["Percent"] = df_mix["Count"].apply(lambda c: _pct(c, total))
        df_mix["Label"] = df_mix["Count"].apply(lambda c: _fmt_count_pct(c, total))
        df_mix["Group"] = "Score Labels"

        df_plot = df_mix[df_mix["Count"] > 0].copy()
        if df_plot.empty:
            df_plot = df_mix.copy()

        base = (
            alt.Chart(df_plot)
            .transform_calculate(pct='round(datum.Percent * 10) / 10')
            .transform_stack(stack='Count', as_=['start', 'end'], groupby=['Group'])
            .transform_calculate(mid='(datum.start + datum.end) / 2')
        )

        bar = (
            base.mark_bar(height=34)
            .encode(
                y=alt.Y("Group:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
                x=alt.X("end:Q", stack=None, axis=None),
                x2="start:Q",
                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                tooltip=[
                    alt.Tooltip("Category:N", title="Score Type"),
                    alt.Tooltip("Count:Q", title="Total", format=",.0f"),
                    alt.Tooltip("Percent:Q", title="%", format=".1f"),
                ],
            )
        )

        text = (
            base.mark_text(baseline="middle", fontWeight="bold")
            .encode(
                y="Group:N",
                x="mid:Q",
                text=alt.Text("Label:N"),
                opacity=alt.condition("datum.Percent < 6", alt.value(0), alt.value(1)),
            )
        )

        st.altair_chart(
            (bar + text).configure_view(stroke=None).configure_axis(grid=False, domain=False),
            use_container_width=True,
        )

        counts_line = " • ".join(
            f"{row.Category}: {int(row.Count)} ({row.Percent:.1f}%)"
            for _, row in df_mix.iterrows() if int(row.Count) > 0
        ) or "No score label counts found in this slice."
        st.caption(counts_line)
    else:
        st.caption("No Score Label column found for Score Mix chart.")

    # ---- Category Mix ----
    st.subheader("📊 Category Mix — Current Slice")

    if "Score Label" in b.columns:
        def _cat_from_label(lbl):
            s = str(lbl).strip()
            if s in ["Albatross", "Eagle", "Birdie", "Par"]:
                return "Par or Better"
            if s == "Bogey":
                return "Bogey"
            return "Double+"
        cats = b["Score Label"].apply(_cat_from_label)
    else:
        _s = pd.to_numeric(b.get("Hole Score"), errors="coerce")
        _p = pd.to_numeric(b.get("Par"), errors="coerce")

        def _cat_from_score_par(score, par):
            if pd.isna(score) or pd.isna(par):
                return None
            if score <= par:
                return "Par or Better"
            if score == par + 1:
                return "Bogey"
            return "Double+"

        cats = pd.Series([_cat_from_score_par(a, c) for a, c in zip(_s, _p)], index=b.index).dropna()

    cat_order = ["Par or Better", "Bogey", "Double+"]
    cat_vc = cats.value_counts()
    cat_counts = [int(cat_vc.get(k, 0)) for k in cat_order]
    cat_total = sum(cat_counts) or 1

    df_cat = pd.DataFrame({"Category": cat_order, "Count": cat_counts})
    df_cat["Percent"] = df_cat["Count"].apply(lambda c: _pct(c, cat_total))
    df_cat["Label"] = df_cat["Count"].apply(lambda c: _fmt_count_pct(c, cat_total))
    df_cat["Group"] = "Categories"

    df_cat_plot = df_cat[df_cat["Count"] > 0].copy()
    if df_cat_plot.empty:
        df_cat_plot = df_cat.copy()

    base2 = (
        alt.Chart(df_cat_plot)
        .transform_calculate(pct='round(datum.Percent * 10) / 10')
        .transform_stack(stack='Count', as_=['start', 'end'], groupby=['Group'])
        .transform_calculate(mid='(datum.start + datum.end) / 2')
    )

    bar2 = (
        base2.mark_bar(height=34)
        .encode(
            y=alt.Y("Group:N", title=None, axis=alt.Axis(labels=False, ticks=False)),
            x=alt.X("end:Q", stack=None, axis=None),
            x2="start:Q",
            color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Category:N"),
                alt.Tooltip("Count:Q", title="Total", format=",.0f"),
                alt.Tooltip("Percent:Q", title="%", format=".1f"),
            ],
        )
    )

    text2 = (
        base2.mark_text(baseline="middle", fontWeight="bold")
        .encode(
            y="Group:N",
            x="mid:Q",
            text=alt.Text("Label:N"),
            opacity=alt.condition("datum.Percent < 6", alt.value(0), alt.value(1)),
        )
    )

    st.altair_chart(
        (bar2 + text2).configure_view(stroke=None).configure_axis(grid=False, domain=False),
        use_container_width=True,
    )

    cat_line = " • ".join(
        f"{row.Category}: {int(row.Count)} ({row.Percent:.1f}%)"
        for _, row in df_cat.iterrows() if int(row.Count) > 0
    ) or "No category counts found in this slice."
    st.caption(cat_line)

# =========================
# Player Comparison table builder
# =========================
def _build_player_compare_table(frame: pd.DataFrame, selected_players: list) -> pd.DataFrame:
    rows = []
    for p in selected_players:
        b = frame[frame["Player Name"] == p].copy()
        if b.empty:
            continue

        s = build_summary(b)

        g3m, g3a, g3p = s["gir3"]
        g4m, g4a, g4p = s["gir4"]
        g5m, g5a, g5p = s["gir5"]

        rows.append({
            "Holes": int(s["holes_played"]),
            "Player": p,

            "Avg / 72": (None if s.get("par72_score") is None else float(s["par72_score"])),

            "Avg P3": (None if s["avg_p3"] is None else float(s["avg_p3"])),
            "Avg P4": (None if s["avg_p4"] is None else float(s["avg_p4"])),
            "Avg P5": (None if s["avg_p5"] is None else float(s["avg_p5"])),

            "Putts": int(s["putts_total"]),
            "Putts/Hole": float(s["putts_per_hole"]),
            "Putts/18": float(s["putts_per_18"]),

            "GIR": _cnt_pair(s["gir_made"], s["gir_att"]),
            "GIR%": float(s["gir_pct"]),

            "GIR P3": _cnt_pair(g3m, g3a),
            "GIR P3%": float(g3p),
            "GIR P4": _cnt_pair(g4m, g4a),
            "GIR P4%": float(g4p),
            "GIR P5": _cnt_pair(g5m, g5a),
            "GIR P5%": float(g5p),

            "FW": _cnt_pair(s["fw_made"], s["fw_att"]),
            "FW%": float(s["fw_pct"]),

            "GIR|FW": _cnt_pair(s["fw_gir_made"], s["fw_gir_att"]),
            "GIR|FW%": float(s["fw_gir_pct"]),

            "Scr": _cnt_pair(s["scr_made"], s["scr_ops"]),
            "Scr%": float(s["scr_pct"]),

            "U&D": _cnt_pair(s["ud_made"], s["ud_ops"]),
            "U&D%": float(s["ud_pct"]),

            "1P": f'{int(s["one_putts"])}/{int(s["holes_played"])}',
            "1P%": float(s["one_putt_pct"]),

            "3+P": f'{int(s["three_plus_putts"])}/{int(s["holes_played"])}',
            "3+P%": float(s["three_plus_putt_pct"]),

            "3P Bogey": f'{int(s["three_putt_bogeys"])}/{int(s.get("three_putt_bogey_att", s["gir_made"]))}',
            "3P Bogey%": float(s["three_putt_bogey_pct"]),

            "Lost Balls": int(s["lb_total"]),
            "Pro Pars+": int(s["pro_pars_plus"]),
            "Arnies": int(s["arnies"]),
            "Seves": int(s["seves"]),
            "Hole Outs": int(s["hole_outs"]),
        })

    dfc = pd.DataFrame(rows)
    if dfc.empty:
        return dfc

    cols = [
        "Holes","Player",
        "Avg / 72",
        "Avg P3","Avg P4","Avg P5",
        "Putts","Putts/Hole","Putts/18",
        "GIR","GIR%","GIR P3","GIR P3%","GIR P4","GIR P4%","GIR P5","GIR P5%",
        "FW","FW%","GIR|FW","GIR|FW%",
        "Scr","Scr%","U&D","U&D%","1P","1P%","3+P","3+P%","3P Bogey","3P Bogey%",
        "Lost Balls","Pro Pars+","Arnies","Seves","Hole Outs"
    ]
    dfc = dfc[cols].copy()

    dfc["Avg / 72"] = pd.to_numeric(dfc["Avg / 72"], errors="coerce").round(1)
    for c in ["Avg P3","Avg P4","Avg P5"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").round(2)

    dfc["Putts/Hole"] = pd.to_numeric(dfc["Putts/Hole"], errors="coerce").round(2)
    dfc["Putts/18"] = pd.to_numeric(dfc["Putts/18"], errors="coerce").round(1)

    for c in ["GIR%","GIR P3%","GIR P4%","GIR P5%","FW%","GIR|FW%","Scr%","U&D%","1P%","3+P%","3P Bogey%"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").round(1)

    return dfc

def _style_compare(dfc: pd.DataFrame):
    if dfc.empty:
        return dfc

    hi_good = {"GIR%","GIR P3%","GIR P4%","GIR P5%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"}
    lo_good = {"Avg / 72","Avg P3","Avg P4","Avg P5","Putts/Hole","Putts/18","Lost Balls","3+P%","3P Bogey%"}  # lower is better

    def _highlight(col):
        if col.name in hi_good:
            best = col.max()
            return ["background-color: rgba(0, 200, 0, 0.22); font-weight:700;" if v == best else "" for v in col]
        if col.name in lo_good:
            best = col.min()
            return ["background-color: rgba(0, 200, 0, 0.22); font-weight:700;" if v == best else "" for v in col]
        return [""] * len(col)

    ignore = {"Player","GIR","GIR P3","GIR P4","GIR P5","FW","GIR|FW","Scr","U&D","1P","3+P","3P Bogey"}
    numeric_cols = [c for c in dfc.columns if c not in ignore and pd.api.types.is_numeric_dtype(dfc[c])]
    sty = dfc.style.apply(_highlight, subset=numeric_cols)

    fmt = {
        "Avg / 72": "{:.1f}",
        "Avg P3": "{:.2f}",
        "Avg P4": "{:.2f}",
        "Avg P5": "{:.2f}",
        "Putts/Hole": "{:.2f}",
        "Putts/18": "{:.1f}",
        "GIR%": "{:.1f}",
        "GIR P3%": "{:.1f}",
        "GIR P4%": "{:.1f}",
        "GIR P5%": "{:.1f}",
        "FW%": "{:.1f}",
        "GIR|FW%": "{:.1f}",
        "Scr%": "{:.1f}",
        "U&D%": "{:.1f}",
        "1P%": "{:.1f}",
        "3+P%": "{:.1f}",
        "3P Bogey%": "{:.1f}",
    }
    return sty.format(fmt, na_rep="—")

def _rank_leaderboard(dfc: pd.DataFrame) -> pd.DataFrame:
    if dfc.empty:
        return dfc

    hi_good = ["GIR%","GIR P3%","GIR P4%","GIR P5%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"]
    lo_good = ["Avg / 72","Avg P3","Avg P4","Avg P5","Putts/Hole","Putts/18","Lost Balls","3+P%","3P Bogey%"]

    work = dfc.set_index("Player").copy()

    rank_cols = {}
    for c in hi_good:
        if c in work.columns:
            rank_cols[c] = work[c].rank(ascending=False, method="min")
    for c in lo_good:
        if c in work.columns:
            rank_cols[c] = work[c].rank(ascending=True, method="min")

    ranks = pd.DataFrame(rank_cols).astype(int)
    ranks["Total Rank Pts"] = ranks.sum(axis=1)
    ranks = ranks.sort_values("Total Rank Pts", ascending=True).reset_index()

    cols = ["Player", "Total Rank Pts"] + [c for c in ranks.columns if c not in ["Player", "Total Rank Pts"]]
    return ranks[cols]

def _award_badges(dfc: pd.DataFrame) -> pd.DataFrame:
    if dfc.empty:
        return dfc

    hi_good = {"GIR%","GIR P3%","GIR P4%","GIR P5%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"}
    lo_good = {"Avg / 72","Avg P3","Avg P4","Avg P5","Putts/Hole","Putts/18","Lost Balls","3+P%","3P Bogey%"}  # lower is better

    out = {"Player": "Badges"}
    out["Holes"] = ""

    for col in dfc.columns:
        if col in {"Player","Holes","GIR","GIR P3","GIR P4","GIR P5","FW","GIR|FW","Scr","U&D","1P","3+P","3P Bogey"}:
            out[col] = ""
            continue

        ser = dfc.set_index("Player")[col].copy()
        ser = pd.to_numeric(ser, errors="coerce").dropna()
        if ser.empty:
            out[col] = ""
            continue

        if col in hi_good:
            sorted_vals = ser.sort_values(ascending=False)
        elif col in lo_good:
            sorted_vals = ser.sort_values(ascending=True)
        else:
            out[col] = ""
            continue

        uniq = sorted_vals.unique().tolist()
        medals = []
        if len(uniq) >= 1: medals.append(("🥇", uniq[0]))
        if len(uniq) >= 2: medals.append(("🥈", uniq[1]))
        if len(uniq) >= 3: medals.append(("🥉", uniq[2]))

        parts = []
        for medal, v in medals:
            winners = sorted_vals[sorted_vals == v].index.tolist()
            if winners:
                parts.append(f"{medal} " + ", ".join(winners))

        out[col] = " | ".join(parts)

    return pd.DataFrame([out])[dfc.columns]

def _compare_chart(dfc: pd.DataFrame, metric: str):
    if dfc.empty or metric not in dfc.columns:
        return
    plot_df = dfc[["Player", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort="-y", title=None),
            y=alt.Y(f"{metric}:Q", title=metric),
            tooltip=[alt.Tooltip("Player:N"), alt.Tooltip(f"{metric}:Q", format=".2f")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Leaderboard Dashboard helpers (visual)
# =========================
def build_player_stats_for_slice(frame: pd.DataFrame) -> pd.DataFrame:
    players_in_slice = sorted([x for x in frame["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    if not players_in_slice:
        return pd.DataFrame()
    return _build_player_compare_table(frame, players_in_slice)

def _styled_top_table(t: pd.DataFrame, value_col: str):
    if t.empty:
        return t

    t2 = t.copy()
    medals = ["🥇", "🥈", "🥉"] + [""] * max(0, len(t2) - 3)
    t2.insert(0, " ", medals[:len(t2)])

    def hi_first(row):
        if row.name == 0:
            return ["background-color: rgba(245, 197, 24, 0.18); font-weight:800;"] * len(row)
        return [""] * len(row)

    sty = t2.style.apply(hi_first, axis=1)
    if value_col in t2.columns:
        sty = sty.set_properties(subset=[value_col], **{"font-weight": "800"})
    return sty

def render_mini_leaderboards(dfc: pd.DataFrame, top_n: int = 6, title: str = "Leaderboards"):
    if dfc.empty:
        st.info("No player stats for this slice.")
        return

    qty_map = {
        "GIR%": "GIR",
        "FW%": "FW",
        "GIR|FW%": "GIR|FW",
        "Scr%": "Scr",
        "U&D%": "U&D",
        "1P%": "1P",
        "3+P%": "3+P",
        "3P Bogey%": "3P Bogey",
        "GIR P3%": "GIR P3",
        "GIR P4%": "GIR P4",
        "GIR P5%": "GIR P5",
    }

    metrics = [
        ("Avg / 72", False, "{:.1f}"),
        ("Avg P3", False, "{:.2f}"),
        ("Avg P4", False, "{:.2f}"),
        ("Avg P5", False, "{:.2f}"),
        ("Putts/Hole", False, "{:.2f}"),
        ("Putts/18", False, "{:.1f}"),
        ("Putts", False, "{:.0f}"),
        ("GIR%", True, "{:.1f}%"),
        ("GIR P3%", True, "{:.1f}%"),
        ("GIR P4%", True, "{:.1f}%"),
        ("GIR P5%", True, "{:.1f}%"),
        ("FW%", True, "{:.1f}%"),
        ("GIR|FW%", True, "{:.1f}%"),
        ("Scr%", True, "{:.1f}%"),
        ("U&D%", True, "{:.1f}%"),
        ("1P%", True, "{:.1f}%"),
        ("3+P%", False, "{:.1f}%"),
        ("3P Bogey%", False, "{:.1f}%"),
        ("Lost Balls", False, "{:.0f}"),
        ("Pro Pars+", True, "{:.0f}"),
        ("Arnies", True, "{:.0f}"),
        ("Seves", True, "{:.0f}"),
        ("Hole Outs", True, "{:.0f}"),
    ]
    metrics = [(m, hib, fmt) for (m, hib, fmt) in metrics if m in dfc.columns]

    def _fmt(v, fmt):
        if pd.isna(v):
            return "—"
        try:
            return fmt.format(float(v))
        except:
            return str(v)

    st.markdown(f"<div class='section-h'>📌 {title}</div>", unsafe_allow_html=True)

    per_row = 3
    for i in range(0, len(metrics), per_row):
        cols = st.columns(per_row)
        for j, (metric, higher_better, fmt) in enumerate(metrics[i:i + per_row]):
            with cols[j]:
                st.markdown(
                    f"<div class='dash-card'><div class='dash-title'>{metric}</div>"
                    f"<div class='dash-sub muted'>Top {top_n}</div>",
                    unsafe_allow_html=True,
                )

                show_cols = ["Player", metric]
                qty_col = qty_map.get(metric)
                if qty_col and qty_col in dfc.columns:
                    show_cols.insert(1, qty_col)

                t = dfc[show_cols].copy()
                t[metric] = pd.to_numeric(t[metric], errors="coerce")
                t = t.dropna(subset=[metric]).sort_values(metric, ascending=not higher_better).head(top_n)

                if t.empty:
                    st.caption("No data.")
                    st.markdown("</div>", unsafe_allow_html=True)
                    continue

                t["Rank"] = range(1, len(t) + 1)
                t["Value"] = t[metric].apply(lambda v: _fmt(v, fmt))

                if qty_col and qty_col in t.columns:
                    t = t.rename(columns={qty_col: "Qty"})
                    view = t[["Rank", "Player", "Qty", "Value"]].reset_index(drop=True)
                else:
                    view = t[["Rank", "Player", "Value"]].reset_index(drop=True)

                st.dataframe(_styled_top_table(view, "Value"), hide_index=True, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

# =========================
# Visual Dashboard helpers
# =========================
def _bar_chart_players(dfc: pd.DataFrame, metric: str, title: str, higher_better: bool, fmt: str = ".2f"):
    if dfc.empty or metric not in dfc.columns:
        st.caption("No data.")
        return
    p = dfc[["Player", metric]].copy()
    p[metric] = pd.to_numeric(p[metric], errors="coerce")
    p = p.dropna(subset=[metric])
    if p.empty:
        st.caption("No data.")
        return

    p = p.sort_values(metric, ascending=not higher_better)

    ch = (
        alt.Chart(p)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort=p["Player"].tolist(), title=None),
            y=alt.Y(f"{metric}:Q", title=title),
            tooltip=[
                alt.Tooltip("Player:N"),
                alt.Tooltip(f"{metric}:Q", format=fmt, title=title),
            ],
        )
        .properties(height=240)
    )
    st.altair_chart(ch, use_container_width=True)

def _bar_chart_players_qty_pct(dfc: pd.DataFrame, pct_col: str, qty_col: str, title: str, higher_better: bool):
    if dfc.empty or pct_col not in dfc.columns or qty_col not in dfc.columns:
        st.caption("No data.")
        return
    p = dfc[["Player", pct_col, qty_col]].copy()
    p[pct_col] = pd.to_numeric(p[pct_col], errors="coerce")
    p = p.dropna(subset=[pct_col])
    if p.empty:
        st.caption("No data.")
        return

    p = p.sort_values(pct_col, ascending=not higher_better)
    p["Label"] = p[qty_col].astype(str) + " • " + p[pct_col].round(1).astype(str) + "%"

    ch = (
        alt.Chart(p)
        .mark_bar()
        .encode(
            x=alt.X("Player:N", sort=p["Player"].tolist(), title=None),
            y=alt.Y(f"{pct_col}:Q", title=title),
            tooltip=[
                alt.Tooltip("Player:N", title="Player"),
                alt.Tooltip(qty_col + ":N", title="Qty"),
                alt.Tooltip(f"{pct_col}:Q", format=".1f", title="%"),
            ],
        )
        .properties(height=240)
    )
    text = (
        alt.Chart(p)
        .mark_text(dy=-8)
        .encode(
            x=alt.X("Player:N", sort=p["Player"].tolist()),
            y=alt.Y(f"{pct_col}:Q"),
            text=alt.Text("Label:N"),
        )
    )
    st.altair_chart((ch + text), use_container_width=True)

def _stacked_score_mix_by_player(frame: pd.DataFrame, players_list: list):
    if frame.empty or "Score Label" not in frame.columns:
        st.caption("No Score Label data.")
        return
    b = frame[frame["Player Name"].isin(players_list)].copy()
    if b.empty:
        st.caption("No data.")
        return

    order = ["Albatross", "Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]
    b["Score Label"] = b["Score Label"].astype(str)

    g = (
        b.groupby(["Player Name", "Score Label"])
        .size()
        .reset_index(name="Count")
    )
    g["Score Label"] = pd.Categorical(g["Score Label"], categories=order, ordered=True)
    g = g.sort_values(["Player Name", "Score Label"])

    totals = g.groupby("Player Name")["Count"].sum().reset_index(name="Total")
    g = g.merge(totals, on="Player Name", how="left")
    g["Percent"] = g["Count"] / g["Total"] * 100.0

    ch = (
        alt.Chart(g)
        .mark_bar()
        .encode(
            x=alt.X("Player Name:N", title=None),
            y=alt.Y("Count:Q", title="Score Mix (Counts)"),
            color=alt.Color("Score Label:N", sort=order, legend=alt.Legend(orient="bottom")),
            tooltip=[
                alt.Tooltip("Player Name:N", title="Player"),
                alt.Tooltip("Score Label:N", title="Type"),
                alt.Tooltip("Count:Q", title="Count"),
                alt.Tooltip("Percent:Q", title="%", format=".1f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(ch, use_container_width=True)

# =========================
# NEW: Approach Analytics helpers (GIR only)
# =========================
def _distance_buckets_standard():
    return [
        ("0–49", 0, 49),
        ("50–74", 50, 74),
        ("75–99", 75, 99),
        ("100–124", 100, 124),
        ("125–149", 125, 149),
        ("150–174", 150, 174),
        ("175–199", 175, 199),
        ("200+", 200, None),
    ]

def _bucketize_distance(y: pd.Series, buckets):
    y = pd.to_numeric(y, errors="coerce")
    out = pd.Series([pd.NA] * len(y), index=y.index, dtype="object")
    for label, lo, hi in buckets:
        if hi is None:
            m = (y >= lo)
        else:
            m = (y >= lo) & (y <= hi)
        out.loc[m] = label
    return out

def _bucket_mid(label: str):
    if label.endswith("+"):
        try:
            return float(label.replace("+", ""))
        except:
            return 999.0
    if "–" in label:
        a, b = label.split("–", 1)
        try:
            return (float(a) + float(b)) / 2.0
        except:
            return None
    return None

def build_gir_by_distance(frame: pd.DataFrame, min_attempts: int = 1) -> pd.DataFrame:
    if frame.empty or YARD_COL not in frame.columns:
        return pd.DataFrame()

    b = frame.copy()
    y = pd.to_numeric(b[YARD_COL], errors="coerce")
    b = b[pd.notna(y)].copy()
    if b.empty:
        return pd.DataFrame()

    buckets = _distance_buckets_standard()
    b["Dist Bucket"] = _bucketize_distance(b[YARD_COL], buckets)

    b = b.dropna(subset=["Dist Bucket"]).copy()
    if b.empty:
        return pd.DataFrame()

    g = b.groupby("Dist Bucket", as_index=False).agg(
        Attempts=("Dist Bucket", "size"),
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum()))
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    g["Qty"] = g["GIR_Made"].astype(int).astype(str) + "/" + g["Attempts"].astype(int).astype(str)
    g["Label"] = g["Qty"] + " • " + g["GIR%"].round(1).astype(str) + "%"
    g["Mid"] = g["Dist Bucket"].astype(str).apply(_bucket_mid)

    order = [x[0] for x in buckets]
    g["Dist Bucket"] = pd.Categorical(g["Dist Bucket"], categories=order, ordered=True)
    g = g.sort_values("Dist Bucket").reset_index(drop=True)

    if min_attempts and min_attempts > 1:
        g = g[g["Attempts"] >= int(min_attempts)].copy()

    return g.reset_index(drop=True)

def build_gir_by_club(frame: pd.DataFrame, min_attempts: int = 1) -> pd.DataFrame:
    if frame.empty or CLUB_COL not in frame.columns:
        return pd.DataFrame()

    b = frame.copy()
    c = b[CLUB_COL].astype(str).str.strip()
    b = b[(c != "") & (c.str.lower() != "nan")].copy()
    if b.empty:
        return pd.DataFrame()

    b["Club"] = b[CLUB_COL].astype(str).str.strip()

    g = b.groupby("Club", as_index=False).agg(
        Attempts=("Club", "size"),
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum()))
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    g["Qty"] = g["GIR_Made"].astype(int).astype(str) + "/" + g["Attempts"].astype(int).astype(str)
    g["Label"] = g["Qty"] + " • " + g["GIR%"].round(1).astype(str) + "%"

    if min_attempts and min_attempts > 1:
        g = g[g["Attempts"] >= int(min_attempts)].copy()

    g = g.sort_values(["GIR%","Attempts"], ascending=[False, False]).reset_index(drop=True)
    return g

def build_gir_heatmap_distance_x_club(frame: pd.DataFrame, min_cell_attempts: int = 3) -> pd.DataFrame:
    if frame.empty or CLUB_COL not in frame.columns or YARD_COL not in frame.columns:
        return pd.DataFrame()

    b = frame.copy()
    y = pd.to_numeric(b[YARD_COL], errors="coerce")
    c = b[CLUB_COL].astype(str).str.strip()

    b = b[pd.notna(y)].copy()
    b = b[(c != "") & (c.str.lower() != "nan")].copy()
    if b.empty:
        return pd.DataFrame()

    buckets = _distance_buckets_standard()
    b["Dist Bucket"] = _bucketize_distance(b[YARD_COL], buckets)
    b["Club"] = b[CLUB_COL].astype(str).str.strip()

    b = b.dropna(subset=["Dist Bucket", "Club"]).copy()
    if b.empty:
        return pd.DataFrame()

    g = b.groupby(["Dist Bucket", "Club"], as_index=False).agg(
        Attempts=("GIR", "size"),
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum()))
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    g["Qty"] = g["GIR_Made"].astype(int).astype(str) + "/" + g["Attempts"].astype(int).astype(str)

    if min_cell_attempts and min_cell_attempts > 1:
        g = g[g["Attempts"] >= int(min_cell_attempts)].copy()

    order = [x[0] for x in buckets]
    g["Dist Bucket"] = pd.Categorical(g["Dist Bucket"], categories=order, ordered=True)
    g = g.sort_values(["Dist Bucket","Club"]).reset_index(drop=True)
    return g

def _line_chart_distance_gir(g: pd.DataFrame):
    if g.empty:
        st.caption("No distance data for GIR.")
        return

    bucket_order = [x[0] for x in _distance_buckets_standard()]

    base = alt.Chart(g).encode(
        x=alt.X("Dist Bucket:N", sort=bucket_order, title="Approach Distance Bucket"),
        y=alt.Y("GIR%:Q", title="GIR %"),
        tooltip=[
            alt.Tooltip("Dist Bucket:N", title="Bucket"),
            alt.Tooltip("Qty:N", title="Qty"),
            alt.Tooltip("GIR%:Q", title="GIR%", format=".1f"),
            alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
        ],
    )

    line = base.mark_line(point=True)
    text = base.mark_text(dy=-10).encode(text=alt.Text("Label:N"))

    st.altair_chart((line + text).properties(height=260), use_container_width=True)

def _bar_chart_club_gir(g: pd.DataFrame, top_n: int = 18):
    if g.empty:
        st.caption("No club data for GIR.")
        return

    gg = g.head(top_n).copy()
    ch = (
        alt.Chart(gg)
        .mark_bar()
        .encode(
            y=alt.Y("Club:N", sort="-x", title=None),
            x=alt.X("GIR%:Q", title="GIR %"),
            tooltip=[
                alt.Tooltip("Club:N"),
                alt.Tooltip("Qty:N", title="Qty"),
                alt.Tooltip("GIR%:Q", format=".1f"),
                alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
            ],
        )
        .properties(height=min(420, 24 * len(gg) + 40))
    )

    text = (
        alt.Chart(gg)
        .mark_text(align="left", dx=6)
        .encode(
            y=alt.Y("Club:N", sort="-x"),
            x=alt.X("GIR%:Q"),
            text=alt.Text("Label:N"),
        )
    )

    st.altair_chart(ch + text, use_container_width=True)

def _heatmap_distance_x_club(g: pd.DataFrame):
    if g.empty:
        st.caption("No rows where both Distance + Club exist (or cell min attempts is too high).")
        return

    club_order = (
        g.groupby("Club")["Attempts"].sum()
        .sort_values(ascending=False)
        .index.tolist()
    )

    ch = (
        alt.Chart(g)
        .mark_rect()
        .encode(
            x=alt.X("Dist Bucket:N", title="Distance Bucket", sort=[x[0] for x in _distance_buckets_standard()]),
            y=alt.Y("Club:N", sort=club_order, title="Club"),
            color=alt.Color("GIR%:Q", title="GIR %"),
            tooltip=[
                alt.Tooltip("Dist Bucket:N", title="Bucket"),
                alt.Tooltip("Club:N", title="Club"),
                alt.Tooltip("Qty:N", title="Qty"),
                alt.Tooltip("GIR%:Q", title="GIR%", format=".1f"),
                alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
            ],
        )
        .properties(height=520)
    )

    st.altair_chart(ch, use_container_width=True)

def build_approach_reference_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()

    b = frame.copy()

    def _row(label: str, block: pd.DataFrame):
        att = int(block.shape[0])
        made = int((block["GIR"] == "Yes").sum()) if att else 0
        pct = _pct(made, att)
        return {"Split": label, "GIR": _cnt_pair(made, att), "GIR%": pct, "Attempts": att}

    rows = []
    rows.append(_row("Overall", b))

    for p in [3, 4, 5]:
        bp = b[b["Par"] == p]
        rows.append(_row(f"Par {p}", bp))

    p45 = b[b["Par"].isin([4, 5])]
    rows.append(_row("Par 4/5 — FW Hit = Yes", p45[p45["Fairway"] == 1]))
    rows.append(_row("Par 4/5 — FW Hit = No", p45[p45["Fairway"] == 0]))

    out = pd.DataFrame(rows)
    return out

def _style_pct_table(df_: pd.DataFrame, pct_cols=("GIR%",)):
    if df_ is None or df_.empty:
        return df_
    sty = df_.style
    fmt = {}
    for c in pct_cols:
        if c in df_.columns:
            fmt[c] = "{:.1f}%"
    return sty.format(fmt, na_rep="—")

# =========================
# Round display helpers
# =========================
def _round_display_options(frame: pd.DataFrame):
    meta = (
        frame[["Round Link", "Date Played", "Player Name", "Course Name"]]
        .dropna(subset=["Round Link"])
        .drop_duplicates(subset=["Round Link"], keep="last")
        .sort_values("Date Played", ascending=False)
        .copy()
    )
    m = {}
    for _, r in meta.iterrows():
        rid = r["Round Link"]
        dt_str = _fmt_date(r["Date Played"])
        label = f'{_safe_str(r["Player Name"])} | {dt_str} | {_safe_str(r["Course Name"])}'
        if label in m:
            label = f"{label} • {rid}"
        m[label] = rid
    return m

def _scorecard_rows(round_data: pd.DataFrame):
    rd = round_data.sort_values("Hole").copy()
    rd["Hole"] = _as_int(rd["Hole"], 0)

    holes_v = rd["Hole"].tolist()
    pars = _as_int(rd["Par"], 0).tolist()
    yards = _as_int(rd["Yards"], 0).tolist()
    scores = _as_int(rd["Hole Score"], 0).tolist()
    putts = _as_int(rd["Putts"], 0).tolist()

    fairways = [("🟢" if int(v) == 1 else "") for v in _as_int(rd["Fairway"], 0).tolist()]
    girs = [("🟢" if (str(v).strip() == "Yes") else "") for v in rd["GIR"].tolist()]
    arnies = [("🅰️" if int(v) == 1 else "") for v in _as_int(rd["Arnie"], 0).tolist()]
    appr_gir = [("🟡" if (str(v).strip() == "Yes") else "") for v in rd["Approach GIR"].tolist()]
    appr_miss = rd["Approach Shot Direction Miss"].fillna("").astype(str).tolist()

    lost_balls = (_as_int(rd["Lost Ball Tee Shot Quantity"], 0) + _as_int(rd["Lost Ball Approach Shot Quantity"], 0)).tolist()
    appr_club = rd[CLUB_COL].fillna("").astype(str).tolist()
    appr_yard = _num(rd[YARD_COL], default=pd.NA).round(0).fillna("").tolist()
    prox = _num(rd[PROX_COL], default=pd.NA).round(0).fillna("").tolist()
    ft_made = _num(rd[FEET_MADE_COL], default=0).round(0).fillna(0).astype(int).tolist()

    def seg(vals, start, end):
        return sum([v for v in vals[start:end] if isinstance(v, (int, float))])

    def insert_seg(row, total=True):
        out = seg(row, 0, 9)
        inn = seg(row, 9, 18)
        return row[:9] + [out] + row[9:18] + [inn] + ([out + inn] if total else [""])

    def insert_icon(row, symbol):
        def cnt(a, b):
            return sum(symbol in str(x) for x in row[a:b])
        out = cnt(0, 9)
        inn = cnt(9, 18)
        return row[:9] + [out] + row[9:18] + [inn, out + inn]

    hole_nums = holes_v[:9] + ["Out"] + holes_v[9:18] + ["In", "Total"]

    rows = [
        ("Par", insert_seg(pars)),
        ("Yards", insert_seg(yards)),
        ("Score", insert_seg(scores)),
        ("Putts", insert_seg(putts)),
        ("Fairway", insert_icon(fairways, "🟢")),
        ("GIR", insert_icon(girs, "🟢")),
        ("Appr Miss Dir", appr_miss[:9] + [""] + appr_miss[9:18] + ["", ""]),
        ("Arnie", insert_icon(arnies, "🅰️")),
        ("Lost Balls", insert_seg(lost_balls)),
        ("Appr Club", appr_club[:9] + [""] + appr_club[9:18] + ["", ""]),
        ("Appr Yards", appr_yard[:9] + [""] + appr_yard[9:18] + ["", ""]),
        ("Appr GIR", insert_icon(appr_gir, "🟡")),
        ("Prox (FT)", prox[:9] + [""] + prox[9:18] + ["", ""]),
        ("FT Made", ft_made[:9] + [sum(ft_made[:9])] + ft_made[9:18] + [sum(ft_made[9:18]), sum(ft_made)]),
    ]
    return hole_nums, rows

def _render_scorecard_table(round_data: pd.DataFrame):
    hole_nums, rows = _scorecard_rows(round_data)
    par_row = next((r for lbl, r in rows if lbl == "Par"), None)

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

    for label, row in rows:
        table_html += f"<tr><td class='sc-label'>{label}</td>"
        for j, val in enumerate(row):
            if label == "Score":
                color = "#ffffff"
                if par_row and j not in [9, 19, 20] and isinstance(val, (int, float)):
                    try:
                        pv = int(par_row[j])
                        sv = int(val)
                        if sv <= pv - 1:
                            color = "#f5c518"
                        elif sv == pv + 1:
                            color = "#ff9999"
                        elif sv == pv + 2:
                            color = "#ff6666"
                        elif sv >= pv + 3:
                            color = "#cc0000"
                    except:
                        pass
                table_html += f"<td class='sc-score' style='color:{color};'>{val}</td>"
            elif label == "Lost Balls":
                is_total_col = j in [9, 19, 20]
                is_zero_num = isinstance(val, (int, float)) and float(val) == 0.0
                display_val = "" if (not is_total_col and is_zero_num) else val
                table_html += f"<td>{display_val}</td>"
            else:
                table_html += f"<td>{val}</td>"
        table_html += "</tr>"

    table_html += "</tbody></table></div>"
    st.markdown(table_html, unsafe_allow_html=True)

# =========================
# Trip Digest builder (copy/paste friendly)
# =========================
def build_trip_digest(dfc: pd.DataFrame) -> pd.DataFrame:
    if dfc.empty:
        return pd.DataFrame()

    ranks = _rank_leaderboard(dfc)
    if ranks.empty:
        return pd.DataFrame()

    digest = dfc.merge(ranks[["Player", "Total Rank Pts"]], on="Player", how="left")

    keep = [
        "Player", "Total Rank Pts", "Holes",
        "Avg / 72",
        "Putts/18",
        "GIR", "GIR%",
        "FW", "FW%",
        "Scr", "Scr%",
        "U&D", "U&D%",
        "Lost Balls",
    ]
    keep = [c for c in keep if c in digest.columns]
    digest = digest[keep].copy()

    for c in ["Avg / 72", "Putts/18"]:
        if c in digest.columns:
            digest[c] = pd.to_numeric(digest[c], errors="coerce").round(1)
    for c in ["GIR%", "FW%", "Scr%", "U&D%"]:
        if c in digest.columns:
            digest[c] = pd.to_numeric(digest[c], errors="coerce").round(1)

    digest = digest.sort_values(["Total Rank Pts", "Holes"], ascending=[True, False])
    return digest.reset_index(drop=True)

# =========================
# Render by mode
# =========================
if mode == "📦 Slice Summary (Any View)":
    st.caption(f"Rows in slice: {slice_for_summary.shape[0]:,}")

    summary = build_summary(slice_for_summary)
    render_summary_cards(summary, title="📦 Current Slice — Summary")
    render_score_mix(slice_for_summary, title="📊 Score Mix — Current Slice (Counts + %)")

    with st.expander("Show slice rows (preview)", expanded=False):
        show_cols = [c for c in [
            "Date Played", "Player Name", "Course Name", GOLF_TRIP_COL, "Round Link", "Hole", "Par", "Hole Score", "Putts",
            "Fairway", "GIR", "Approach GIR", "Score Label", CLUB_COL, YARD_COL
        ] if c in slice_for_summary.columns]
        st.dataframe(
            slice_for_summary.sort_values(["Date Played", "Round Link", "Hole"])[show_cols],
            use_container_width=True,
            hide_index=True
        )

elif mode == "🧾 Round Scorecard (by Round Link)":
    round_map = _round_display_options(slice_for_summary)
    if not round_map:
        st.warning("No rounds found in the current slice (missing Round Link).")
        st.stop()

    sel_label = st.selectbox("Select a Round", list(round_map.keys()))
    selected_round = round_map[sel_label]

    round_data = slice_for_summary[slice_for_summary["Round Link"] == selected_round].copy()
    round_data = round_data.sort_values("Hole")

    player = round_data["Player Name"].iloc[0]
    course = round_data["Course Name"].iloc[0]
    date_str = _fmt_date(round_data["Date Played"].iloc[0])

    st.markdown(
        f"🏌️ **{player}** | **{course}** | **{date_str}**  \n"
        f"*“Arnie steps to the tee with precision in mind. Seve follows, carving creativity from the rough.”*"
    )

    _render_scorecard_table(round_data)

    summary = build_summary(round_data)
    render_summary_cards(summary, title="📦 This Round — Summary")
    render_score_mix(round_data, title="📊 Score Mix — This Round (Counts + %)")

elif mode == "🎯 Hole Scorecard (last 18 for a specific hole)":
    st.caption("Shows the most recent 18 rows for the selected (Course, Hole) within your current date/trip/player filters.")

    c1, c2 = st.columns([2, 1])
    with c1:
        hole_course = st.selectbox("Course", ["All"] + courses, index=0)
    with c2:
        hole_num = st.selectbox("Hole", holes, index=0)

    h = df.copy()

    h = h[
        (h["Date Played"] >= pd.to_datetime(date_start)) &
        (h["Date Played"] <= pd.to_datetime(date_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    ].copy()

    if sel_trips:
        h = h[h[GOLF_TRIP_COL].isin(sel_trips)]
    if sel_players:
        h = h[h["Player Name"].isin(sel_players)]

    if hole_course != "All":
        h = h[h["Course Name"] == hole_course]
    h = h[h["Hole"] == int(hole_num)].copy()

    h = h.sort_values("Date Played", ascending=False).head(18).copy()
    if h.empty:
        st.warning("No rows found for that hole selection.")
        st.stop()

    st.subheader(f"🎯 Hole {hole_num} — Last {h.shape[0]} Results")
    show_cols = [c for c in [
        "Date Played", "Player Name", "Course Name", GOLF_TRIP_COL, "Round Link", "Hole", "Par", "Hole Score", "Putts",
        "Fairway", "GIR", "Approach GIR", "Score Label", CLUB_COL, YARD_COL, PROX_COL
    ] if c in h.columns]
    st.dataframe(h[show_cols], use_container_width=True, hide_index=True)

    summary = build_summary(h)
    render_summary_cards(summary, title="📦 Hole Slice — Summary")
    render_score_mix(h, title="📊 Score Mix — Hole Slice (Counts + %)")

elif mode == "👥 Player Comparison (same slice)":
    st.caption(f"Rows in slice (all players): {base_f.shape[0]:,}")

    all_players_in_slice = sorted([x for x in base_f["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    default_compare = sel_players[:4] if sel_players else (all_players_in_slice[:2] if len(all_players_in_slice) >= 2 else all_players_in_slice)

    sel_players_cmp = st.multiselect(
        "Select players to compare (up to 4)",
        options=all_players_in_slice,
        default=default_compare,
        max_selections=4
    )

    if len(sel_players_cmp) < 2:
        st.info("Pick at least 2 players to compare.")
        st.stop()

    dfc = _build_player_compare_table(base_f, sel_players_cmp)
    if dfc.empty:
        st.warning("No data for selected players in this slice.")
        st.stop()

    st.subheader("👥 Player Comparison — Counts + % + Best stats highlighted")
    st.dataframe(_style_compare(dfc), use_container_width=True, hide_index=True)

    st.subheader("🥇🥈🥉 Badge Row (who wins each stat)")
    badges = _award_badges(dfc)
    st.dataframe(badges, use_container_width=True, hide_index=True)
    st.caption("Badges show who’s 1st/2nd/3rd for each metric (ties supported).")

    st.subheader("🏆 Rank Leaderboard (sum of ranks across stats)")
    ranks = _rank_leaderboard(dfc)
    st.dataframe(ranks, use_container_width=True, hide_index=True)
    st.caption("Lower Total Rank Pts = better overall across the compared stats.")

    with st.expander("📊 Quick Comparison Charts", expanded=True):
        chart_metrics = [
            "Avg / 72","Avg P3","Avg P4","Avg P5",
            "Putts/Hole","Putts/18","Putts",
            "GIR%","GIR P3%","GIR P4%","GIR P5%",
            "FW%","GIR|FW%","Scr%","U&D%","1P%","3+P%","3P Bogey%","Lost Balls"
        ]
        metric = st.selectbox("Pick a metric to chart", [m for m in chart_metrics if m in dfc.columns], index=0)
        _compare_chart(dfc, metric)

elif mode == "🏆 Leaderboard Dashboard (multi-stat)":
    st.markdown("<div class='dash-wrap'></div>", unsafe_allow_html=True)

    dash_players = sorted([x for x in base_f["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    dash_default = sel_players if sel_players else dash_players

    bar1, bar2, bar3, bar4 = st.columns([2.2, 0.9, 1.0, 1.0])
    with bar1:
        sel_dash_players = st.multiselect("🎛️ Dashboard Players (optional)", options=dash_players, default=dash_default)
    with bar2:
        top_n = st.slider("Top N", 3, 15, 6)
    with bar3:
        min_holes = st.slider("Min Holes (leaderboards)", 0, 200, 18, step=9)
    with bar4:
        show_charts = st.toggle("Show quick charts", value=True)

    dash_frame = base_f.copy()
    if sel_dash_players:
        dash_frame = dash_frame[dash_frame["Player Name"].isin(sel_dash_players)].copy()

    if dash_frame.empty:
        st.warning("No rows match the dashboard selection.")
        st.stop()

    date_min = dash_frame["Date Played"].min()
    date_max = dash_frame["Date Played"].max()
    rounds = dash_frame["Round Link"].dropna().nunique() if "Round Link" in dash_frame.columns else 0
    courses_n = dash_frame["Course Name"].dropna().nunique()
    players_n = dash_frame["Player Name"].dropna().nunique()

    pills = []
    pills.append(f"📅 {_fmt_date(date_min)} → {_fmt_date(date_max)}")
    pills.append(f"🏌️ Players: {players_n}")
    pills.append(f"⛳ Courses: {courses_n}")
    if rounds:
        pills.append(f"🧾 Rounds: {rounds}")
    if sel_trips:
        pills.append(f"🧳 Trip: {', '.join([_safe_str(x) for x in sel_trips])}")

    st.markdown(
        "<div class='dash-card'>"
        "<div class='dash-title'>🏆 Leaderboard Dashboard</div>"
        "<div class='dash-sub'>Trip-friendly: who’s winning what (with Qty + %).</div>"
        + "".join([f"<span class='pill'>{p}</span>" for p in pills]) +
        "</div>",
        unsafe_allow_html=True,
    )

    summary_dash = build_summary(dash_frame)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Holes", f"{summary_dash['holes_played']:,}")
    k2.metric("Score", f"{summary_dash['score_total']:,} ({_fmt_to_par(summary_dash['to_par'])})")
    k3.metric("Putts", f"{summary_dash['putts_total']:,}", delta=f"{summary_dash['putts_per_18']:.1f}/18 • {summary_dash['putts_per_hole']:.2f}/hole")
    k4.metric("GIR", f"{summary_dash['gir_pct']:.1f}% {_emoji(summary_dash['gir_pct'])}", delta=_cnt_pair(summary_dash["gir_made"], summary_dash["gir_att"]))
    k5.metric("FW (P4/P5)", f"{summary_dash['fw_pct']:.1f}% {_emoji(summary_dash['fw_pct'])}", delta=_cnt_pair(summary_dash["fw_made"], summary_dash["fw_att"]))
    k6.metric("Scr%", f"{summary_dash['scr_pct']:.1f}%", delta=_cnt_pair(summary_dash["scr_made"], summary_dash["scr_ops"]))

    dfc_all = build_player_stats_for_slice(dash_frame)
    if dfc_all.empty:
        st.warning("No players found in this slice.")
        st.stop()

    if min_holes and min_holes > 0:
        dfc_all = dfc_all[dfc_all["Holes"] >= int(min_holes)].copy()

    if dfc_all.empty:
        st.warning("After Min Holes filter, no players remain.")
        st.stop()

    st.markdown("<div class='section-h'>🥇 Overall (sum of ranks across stats)</div>", unsafe_allow_html=True)
    ranks_all = _rank_leaderboard(dfc_all)
    st.dataframe(ranks_all, use_container_width=True, hide_index=True)
    st.caption("Lower Total Rank Pts = better overall across the included stats.")

    with st.expander("📌 Scoring (Avg / 72 + Par 3/4/5 Averages)", expanded=True):
        keep = [c for c in ["Player","Avg / 72","Avg P3","Avg P4","Avg P5","Putts","Putts/18","Putts/Hole"] if c in dfc_all.columns]
        render_mini_leaderboards(dfc_all[keep].copy(), top_n=top_n, title="Scoring + Putting")
        if show_charts:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Quick Chart — Avg / 72**")
                _compare_chart(dfc_all, "Avg / 72")
            with cB:
                st.markdown("**Quick Chart — Putts/18**")
                _compare_chart(dfc_all, "Putts/18")

    with st.expander("🎯 Ball Striking (GIR + FW + GIR|FW)", expanded=True):
        keep = [c for c in ["Player","GIR","GIR%","GIR P3","GIR P3%","GIR P4","GIR P4%","GIR P5","GIR P5%","FW","FW%","GIR|FW","GIR|FW%"] if c in dfc_all.columns]
        render_mini_leaderboards(dfc_all[keep].copy(), top_n=top_n, title="Ball Striking")
        if show_charts:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Quick Chart — GIR%**")
                _compare_chart(dfc_all, "GIR%")
            with cB:
                st.markdown("**Quick Chart — FW%**")
                _compare_chart(dfc_all, "FW%")

    with st.expander("🩹 Short Game (Scr, U&D, 1P, 3+P, 3P Bogey)", expanded=True):
        keep = [c for c in ["Player","Scr","Scr%","U&D","U&D%","1P","1P%","3+P","3+P%","3P Bogey","3P Bogey%"] if c in dfc_all.columns]
        render_mini_leaderboards(dfc_all[keep].copy(), top_n=top_n, title="Short Game")
        if show_charts:
            cA, cB = st.columns(2)
            with cA:
                st.markdown("**Quick Chart — Scr%**")
                _compare_chart(dfc_all, "Scr%")
            with cB:
                st.markdown("**Quick Chart — 3P Bogey%**")
                _compare_chart(dfc_all, "3P Bogey%")

    with st.expander("✨ Specials (Pro Pars+, Arnies, Seves, Hole Outs, Lost Balls)", expanded=False):
        keep = [c for c in ["Player","Pro Pars+","Arnies","Seves","Hole Outs","Lost Balls"] if c in dfc_all.columns]
        render_mini_leaderboards(dfc_all[keep].copy(), top_n=top_n, title="Specials")

    with st.expander("📤 Trip Digest (copy/paste + download)", expanded=True):
        digest = build_trip_digest(dfc_all)
        if digest.empty:
            st.caption("No digest available.")
        else:
            st.dataframe(digest, use_container_width=True, hide_index=True)

            tsv = _to_tsv(digest)
            st.text_area("Copy/Paste (TSV)", value=tsv, height=200)

            st.download_button(
                "Download Trip Digest (CSV)",
                data=digest.to_csv(index=False),
                file_name="trip_digest.csv",
                mime="text/csv",
            )

    with st.expander("Show full player table (all dashboard players)", expanded=False):
        st.dataframe(_style_compare(dfc_all), use_container_width=True, hide_index=True)

elif mode == "📊 Visual Dashboard (charts-only)":
    st.markdown("<div class='dash-wrap'></div>", unsafe_allow_html=True)

    vis_players = sorted([x for x in base_f["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    vis_default = sel_players if sel_players else vis_players

    v1, v2, v3, v4 = st.columns([2.2, 0.95, 1.05, 1.0])
    with v1:
        sel_vis_players = st.multiselect("🎛️ Visuals — Players (optional)", options=vis_players, default=vis_default)
    with v2:
        density = st.selectbox("Chart Density", ["Simple", "Full"], index=0)
    with v3:
        focus = st.selectbox("Focus", ["Overall", "Scoring", "Ball Striking", "Short Game", "Specials"], index=0)
    with v4:
        min_holes = st.slider("Min Holes (charts)", 0, 200, 18, step=9)

    vis_frame = base_f.copy()
    if sel_vis_players:
        vis_frame = vis_frame[vis_frame["Player Name"].isin(sel_vis_players)].copy()

    if vis_frame.empty:
        st.warning("No rows match the visual selection.")
        st.stop()

    dfc_all = build_player_stats_for_slice(vis_frame)
    if dfc_all.empty:
        st.warning("No players found in this slice.")
        st.stop()

    if min_holes and min_holes > 0:
        dfc_all = dfc_all[dfc_all["Holes"] >= int(min_holes)].copy()

    if dfc_all.empty:
        st.warning("After Min Holes filter, no players remain.")
        st.stop()

    date_min = vis_frame["Date Played"].min()
    date_max = vis_frame["Date Played"].max()
    rounds = vis_frame["Round Link"].dropna().nunique() if "Round Link" in vis_frame.columns else 0
    courses_n = vis_frame["Course Name"].dropna().nunique()
    players_n = vis_frame["Player Name"].dropna().nunique()

    pills = []
    pills.append(f"📅 {_fmt_date(date_min)} → {_fmt_date(date_max)}")
    pills.append(f"🏌️ Players: {players_n}")
    pills.append(f"⛳ Courses: {courses_n}")
    if rounds:
        pills.append(f"🧾 Rounds: {rounds}")
    if sel_trips:
        pills.append(f"🧳 Trip: {', '.join([_safe_str(x) for x in sel_trips])}")

    st.markdown(
        "<div class='dash-card'>"
        "<div class='dash-title'>📊 Visual Dashboard</div>"
        "<div class='dash-sub'>Charts only — same slice. Use Focus + Density + Min Holes to keep it clean.</div>"
        + "".join([f"<span class='pill'>{p}</span>" for p in pills]) +
        "</div>",
        unsafe_allow_html=True,
    )

    if focus in ["Overall", "Scoring"]:
        st.markdown("<div class='section-h'>📈 Scoring</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Avg / 72 (lower is better)**")
            _bar_chart_players(dfc_all, "Avg / 72", "Avg / 72", higher_better=False, fmt=".1f")
        with c2:
            st.markdown("**Putts/18 (lower is better)**")
            _bar_chart_players(dfc_all, "Putts/18", "Putts/18", higher_better=False, fmt=".1f")

        if density == "Full":
            c3, c4, c5 = st.columns(3)
            with c3:
                st.markdown("**Avg Par 3**")
                _bar_chart_players(dfc_all, "Avg P3", "Avg P3", higher_better=False, fmt=".2f")
            with c4:
                st.markdown("**Avg Par 4**")
                _bar_chart_players(dfc_all, "Avg P4", "Avg P4", higher_better=False, fmt=".2f")
            with c5:
                st.markdown("**Avg Par 5**")
                _bar_chart_players(dfc_all, "Avg P5", "Avg P5", higher_better=False, fmt=".2f")

    if focus in ["Overall", "Ball Striking"]:
        st.markdown("<div class='section-h'>🎯 Ball Striking</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**GIR% (higher is better)**")
            _bar_chart_players_qty_pct(dfc_all, "GIR%", "GIR", "GIR%", higher_better=True)
        with c2:
            st.markdown("**FW% (higher is better)**")
            _bar_chart_players_qty_pct(dfc_all, "FW%", "FW", "FW%", higher_better=True)

        if density == "Full":
            c3, c4, c5 = st.columns(3)
            with c3:
                st.markdown("**GIR Par 3%**")
                _bar_chart_players_qty_pct(dfc_all, "GIR P3%", "GIR P3", "GIR P3%", higher_better=True)
            with c4:
                st.markdown("**GIR Par 4%**")
                _bar_chart_players_qty_pct(dfc_all, "GIR P4%", "GIR P4", "GIR P4%", higher_better=True)
            with c5:
                st.markdown("**GIR Par 5%**")
                _bar_chart_players_qty_pct(dfc_all, "GIR P5%", "GIR P5", "GIR P5%", higher_better=True)

        st.markdown("**Score Mix by Player (stacked counts)**")
        _stacked_score_mix_by_player(vis_frame, dfc_all["Player"].tolist())

    if focus in ["Overall", "Short Game"]:
        st.markdown("<div class='section-h'>🩹 Short Game</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Scr% (higher is better)**")
            _bar_chart_players_qty_pct(dfc_all, "Scr%", "Scr", "Scr%", higher_better=True)
        with c2:
            st.markdown("**U&D% (higher is better)**")
            _bar_chart_players_qty_pct(dfc_all, "U&D%", "U&D", "U&D%", higher_better=True)
        with c3:
            st.markdown("**3P Bogey% (lower is better)**")
            _bar_chart_players_qty_pct(dfc_all, "3P Bogey%", "3P Bogey", "3P Bogey%", higher_better=False)

        if density == "Full":
            c4, c5 = st.columns(2)
            with c4:
                st.markdown("**1P% (higher is better)**")
                _bar_chart_players_qty_pct(dfc_all, "1P%", "1P", "1P%", higher_better=True)
            with c5:
                st.markdown("**3+P% (lower is better)**")
                _bar_chart_players_qty_pct(dfc_all, "3+P%", "3+P", "3+P%", higher_better=False)

    if focus in ["Overall", "Specials"]:
        st.markdown("<div class='section-h'>✨ Specials</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Pro Pars+ (higher is better)**")
            _bar_chart_players(dfc_all, "Pro Pars+", "Pro Pars+", higher_better=True, fmt=".0f")
        with c2:
            st.markdown("**Arnies (higher is better)**")
            _bar_chart_players(dfc_all, "Arnies", "Arnies", higher_better=True, fmt=".0f")
        with c3:
            st.markdown("**Lost Balls (lower is better)**")
            _bar_chart_players(dfc_all, "Lost Balls", "Lost Balls", higher_better=False, fmt=".0f")

        if density == "Full":
            c4, c5 = st.columns(2)
            with c4:
                st.markdown("**Seves (higher is better)**")
                _bar_chart_players(dfc_all, "Seves", "Seves", higher_better=True, fmt=".0f")
            with c5:
                st.markdown("**Hole Outs (higher is better)**")
                _bar_chart_players(dfc_all, "Hole Outs", "Hole Outs", higher_better=True, fmt=".0f")

elif mode == "📈 Approach Analytics (Distance + Club + Heatmap)":
    st.markdown("<div class='dash-wrap'></div>", unsafe_allow_html=True)

    a_players = sorted([x for x in base_f["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    a_default = sel_players if sel_players else a_players

    a1, a2, a3, a4 = st.columns([2.2, 1.0, 1.0, 1.1])
    with a1:
        sel_a_players = st.multiselect("🎛️ Analytics — Players (optional)", options=a_players, default=a_default)
    with a2:
        min_attempts = st.slider("Min Attempts (bucket/club)", 1, 30, 5)
    with a3:
        min_cell = st.slider("Min Attempts (heatmap cell)", 1, 20, 3)
    with a4:
        top_clubs = st.slider("Top Clubs (bar)", 6, 30, 18)

    a_frame = base_f.copy()
    if sel_a_players:
        a_frame = a_frame[a_frame["Player Name"].isin(sel_a_players)].copy()

    if a_frame.empty:
        st.warning("No rows match the analytics selection.")
        st.stop()

    # Quick coverage pills
    dist_present = pd.to_numeric(a_frame.get(YARD_COL), errors="coerce")
    club_present = a_frame.get(CLUB_COL).astype(str).str.strip()
    has_dist = int(pd.notna(dist_present).sum())
    has_club = int(((club_present != "") & (club_present.str.lower() != "nan")).sum())
    has_both = int(((pd.notna(dist_present)) & (club_present != "") & (club_present.str.lower() != "nan")).sum())

    pills = []
    pills.append(f"📅 {_fmt_date(a_frame['Date Played'].min())} → {_fmt_date(a_frame['Date Played'].max())}")
    pills.append(f"🏌️ Players: {a_frame['Player Name'].nunique()}")
    pills.append(f"⛳ Courses: {a_frame['Course Name'].nunique()}")
    pills.append(f"📏 Dist rows: {has_dist}")
    pills.append(f"🏷️ Club rows: {has_club}")
    pills.append(f"🧩 Both: {has_both}")
    if sel_trips:
        pills.append(f"🧳 Trip: {', '.join([_safe_str(x) for x in sel_trips])}")

    st.markdown(
        "<div class='dash-card'>"
        "<div class='dash-title'>📈 Approach Analytics — GIR</div>"
        "<div class='dash-sub'>A) Distance buckets • B) Club • Best of both worlds: Distance × Club heatmap (only rows that have BOTH).</div>"
        + "".join([f"<span class='pill'>{p}</span>" for p in pills]) +
        "</div>",
        unsafe_allow_html=True,
    )

    # ✅ Quick Reference / Grounding block
    st.markdown("<div class='section-h'>Quick Reference — GIR Grounding</div>", unsafe_allow_html=True)
    ref = build_approach_reference_table(a_frame)
    ref_view = ref[["Split", "GIR", "GIR%", "Attempts"]].copy()
    st.dataframe(_style_pct_table(ref_view, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)

    # A) Distance buckets
    st.markdown("<div class='section-h'>A) GIR by Distance Bucket</div>", unsafe_allow_html=True)
    dist_g = build_gir_by_distance(a_frame, min_attempts=min_attempts)
    _line_chart_distance_gir(dist_g)

    # B) Clubs
    st.markdown("<div class='section-h'>B) GIR by Club</div>", unsafe_allow_html=True)
    club_g = build_gir_by_club(a_frame, min_attempts=min_attempts)
    _bar_chart_club_gir(club_g, top_n=top_clubs)

    # Best of both worlds (heatmap)
    st.markdown("<div class='section-h'>Best of both worlds) Distance × Club Heatmap (GIR%)</div>", unsafe_allow_html=True)
    heat = build_gir_heatmap_distance_x_club(a_frame, min_cell_attempts=min_cell)
    _heatmap_distance_x_club(heat)

    with st.expander("Show tables (distance / club / heatmap)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distance buckets**")
            dtab = dist_g[["Dist Bucket","Qty","GIR%","Attempts"]].copy() if not dist_g.empty else pd.DataFrame()
            st.dataframe(_style_pct_table(dtab, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Clubs**")
            ctab = club_g[["Club","Qty","GIR%","Attempts"]].head(top_clubs).copy() if not club_g.empty else pd.DataFrame()
            st.dataframe(_style_pct_table(ctab, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)

        st.markdown("**Heatmap cells**")
        htab = heat[["Dist Bucket","Club","Qty","GIR%","Attempts"]].copy() if not heat.empty else pd.DataFrame()
        st.dataframe(_style_pct_table(htab, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)

    with st.expander("📤 Export Approach tables (CSV)", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            st.download_button(
                "Download Distance Buckets (CSV)",
                data=(dist_g.to_csv(index=False) if not dist_g.empty else "Dist Bucket,Attempts,GIR_Made,GIR%,Qty,Label,Mid\n"),
                file_name="approach_gir_by_distance.csv",
                mime="text/csv",
            )
        with colB:
            st.download_button(
                "Download Clubs (CSV)",
                data=(club_g.to_csv(index=False) if not club_g.empty else "Club,Attempts,GIR_Made,GIR%,Qty,Label\n"),
                file_name="approach_gir_by_club.csv",
                mime="text/csv",
            )
        with colC:
            st.download_button(
                "Download Heatmap Cells (CSV)",
                data=(heat.to_csv(index=False) if not heat.empty else "Dist Bucket,Club,Attempts,GIR_Made,GIR%,Qty\n"),
                file_name="approach_gir_heatmap_cells.csv",
                mime="text/csv",
            )

else:
    st.warning("Unknown mode selection.")