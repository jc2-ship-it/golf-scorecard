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

# ---------------------------
# Shared numeric helpers (used by putting + approach)
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

# =========================
# Config
# =========================
st.set_page_config(page_title="Golf Slicer", layout="wide")

CSV_FILE = "Hole Data-Grid view (18).csv"

PROX_COL = "Proximity to Hole - How far is your First Putt (FT)"
YARD_COL = "Approach Shot Distance (how far you had to the hole)"
CLUB_COL = "Approach Shot Club Used"
FEET_MADE_COL = "Feet of Putt Made (How far was the putt you made)"

SCORE_COL = "Hole Score"
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

# Default to a gentler landing page
if "mode" not in st.session_state:
    st.session_state["mode"] = "🏠 Home (Start Here)"

mode = st.radio(
    "Choose what to render for the current slice",
    [
        "🏠 Home (Start Here)",
        "📦 Slice Summary (Any View)",
        "🆚 Baseline Compare Dashboard",
        "🧾 Round Scorecard (by Round Link)",
        "🎯 Hole Scorecard (last 18 for a specific hole)",
        "👥 Player Comparison (same slice)",
        "🏆 Leaderboard Dashboard (multi-stat)",
        "📊 Visual Dashboard (charts-only)",
        "📈 Approach Analytics (Distance + Club + Heatmap)",
        "⛳ Putting Proximity (Validation)"
    ],
    horizontal=True,
    key="mode",
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
# Baseline Compare helpers
# =========================
def _apply_non_time_filters(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the SAME non-time filters used for the slice, but WITHOUT any date/year/month constraints.
    This makes 'June 2025 vs All of 2025' work correctly: slice picks June; baseline picks year.
    """
    out = frame.copy()

    # Trip
    if sel_trips:
        out = out[out[GOLF_TRIP_COL].isin(sel_trips)]

    # Players (match the slice's selected players, if any)
    if sel_players:
        out = out[out["Player Name"].isin(sel_players)]

    # Courses
    if sel_courses:
        out = out[out["Course Name"].isin(sel_courses)]

    # Par
    if sel_pars:
        out = out[out["Par"].isin(sel_pars)]

    # Club / holes
    if sel_clubs:
        out = out[out[CLUB_COL].isin(sel_clubs)]
    if sel_holes:
        out = out[out["Hole"].isin(sel_holes)]

    # Fairway (P4/P5 only)
    if sel_fw != "All":
        p45_mask = out["Par"].isin([4, 5])
        if sel_fw == "Yes":
            out = out[~p45_mask | (out["Fairway"] == 1)]
        else:
            out = out[~p45_mask | (out["Fairway"] == 0)]

    # GIR / Approach GIR
    if sel_gir != "All":
        out = out[out["GIR"] == sel_gir]
    if sel_appr_gir != "All":
        out = out[out["Approach GIR"] == sel_appr_gir]

    # Yardage range (only apply when yard col exists)
    yard_n = _num(out.get(YARD_COL), default=pd.NA)
    out = out[(yard_n.isna()) | ((yard_n >= y_low) & (yard_n <= y_high))].copy()

    return out


def _baseline_controls(df_all: pd.DataFrame, key_prefix: str = "base") -> pd.DataFrame:
    """
    Returns a baseline frame based on user selection.
    Baseline is independent of the current slice, but can optionally inherit the *non-time* filters.
    """
    st.markdown("<div class='dash-card'>"
                "<div class='dash-title'>🆚 Baseline Settings</div>"
                "<div class='dash-sub muted'>Pick what the slice should be compared against.</div>",
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.4, 1.2, 1.4])
    with c1:
        baseline_type = st.selectbox(
            "Baseline Type",
            ["All Time", "All of Year(s)", "Specific Months (Year + Month)", "Custom Date Range"],
            key=f"{key_prefix}_type",
        )
    with c2:
        inherit_filters = st.toggle(
            "Inherit non-time filters",
            value=True,
            help="If ON: baseline keeps Course/Par/Club/Hole/FW/GIR/etc filters, but uses its own time window.",
            key=f"{key_prefix}_inherit",
        )
    with c3:
        min_holes_base = st.number_input(
            "Min holes (baseline)",
            min_value=0,
            value=0,
            step=9,
            help="Optional: require at least this many holes in the baseline.",
            key=f"{key_prefix}_minholes",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Start from full data, then (optionally) inherit NON-time filters from the current slice.
    base_src = _apply_non_time_filters(df_all) if inherit_filters else df_all.copy()

    # Apply baseline time logic
    if baseline_type == "All Time":
        baseline_df = base_src

    elif baseline_type == "All of Year(s)":
        all_years = sorted([int(x) for x in df_all["Year"].dropna().unique().tolist()], reverse=True)
        yrs = st.multiselect("Select baseline year(s)", all_years, default=sel_years if sel_years else [], key=f"{key_prefix}_yrs")
        if yrs:
            baseline_df = base_src[base_src["Year"].isin([int(x) for x in yrs])].copy()
        else:
            baseline_df = base_src.copy()

    elif baseline_type == "Specific Months (Year + Month)":
        all_years = sorted([int(x) for x in df_all["Year"].dropna().unique().tolist()], reverse=True)
        yrs = st.multiselect("Year(s)", all_years, default=sel_years if sel_years else [], key=f"{key_prefix}_m_yrs")
        mos = st.multiselect("Month(s)", months, default=[], key=f"{key_prefix}_m_mos")
        baseline_df = base_src.copy()
        if yrs:
            baseline_df = baseline_df[baseline_df["Year"].isin([int(x) for x in yrs])].copy()
        if mos:
            baseline_df = baseline_df[baseline_df["Month"].isin(mos)].copy()

    else:  # Custom Date Range
        min_dt = df_all["Date Played"].min()
        max_dt = df_all["Date Played"].max()
        min_d = min_dt.date() if pd.notna(min_dt) else datetime.date.today()
        max_d = max_dt.date() if pd.notna(max_dt) else datetime.date.today()

        d1, d2 = st.columns(2)
        with d1:
            b_start = st.date_input("Baseline start", value=min_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_start")
        with d2:
            b_end = st.date_input("Baseline end", value=max_d, min_value=min_d, max_value=max_d, key=f"{key_prefix}_end")

        if b_start > b_end:
            b_start, b_end = b_end, b_start

        baseline_df = base_src[
            (base_src["Date Played"] >= pd.to_datetime(b_start)) &
            (base_src["Date Played"] <= pd.to_datetime(b_end) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
        ].copy()

    if min_holes_base and min_holes_base > 0 and not baseline_df.empty:
        if int(baseline_df.shape[0]) < int(min_holes_base):
            st.warning(f"Baseline only has {baseline_df.shape[0]:,} holes (below Min holes). Deltas may be noisy.")
        # We don't drop rows; we just warn.

    return baseline_df


def _delta_with_base(curr, base, invert=False, pct_points=False, fmt_curr=None, fmt_base=None):
    """
    Returns a delta string that ALSO shows the baseline value.
    Examples:
      - "base 38.0% • ▲ +4.1 pts"
      - "base 31.2 • ▼ -1.4 (↓ better)"
    """
    if curr is None or base is None or pd.isna(curr) or pd.isna(base):
        return "—"
    try:
        curr_f = float(curr)
        base_f = float(base)
    except Exception:
        return "—"

    d = curr_f - base_f
    arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")

    # Format baseline
    if fmt_base:
        try:
            base_txt = fmt_base.format(base_f)
        except Exception:
            base_txt = str(base)
    else:
        base_txt = f"{base_f:.1f}%" if pct_points else f"{base_f:.2f}"

    # Format delta
    if pct_points:
        d_txt = f"{arrow} {d:+.1f} pts"
    else:
        d_txt = f"{arrow} {d:+.2f}"

    if invert:
        d_txt = d_txt + " (↓ better)"

    return f"base {base_txt} • {d_txt}"

def _fmt_val(v, kind="num"):
    if v is None or pd.isna(v):
        return "—"
    try:
        f = float(v)
    except Exception:
        return str(v)
    if kind == "pct":
        return f"{f:.1f}%"
    if kind == "int":
        return f"{int(round(f)):,}"
    if kind == "to_par":
        return _fmt_to_par(f)
    return f"{f:.2f}"

def _compare_df(curr: dict, base: dict) -> pd.DataFrame:
    """
    Friendly compare table: Metric | Slice | Baseline | Δ
    For percentages, Δ is in percentage points.
    """
    def row(metric, slice_v, base_v, kind="num", invert=False, delta_kind=None):
        if delta_kind is None:
            delta_kind = kind
        # compute delta numeric
        d = None
        try:
            if slice_v is not None and base_v is not None and (not pd.isna(slice_v)) and (not pd.isna(base_v)):
                d = float(slice_v) - float(base_v)
        except Exception:
            d = None

        # format delta
        if d is None:
            d_txt = "—"
        else:
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
            if delta_kind == "pct":
                d_txt = f"{arrow} {d:+.1f} pts"
            elif delta_kind == "int":
                d_txt = f"{arrow} {d:+.0f}"
            else:
                d_txt = f"{arrow} {d:+.2f}"
            if invert:
                d_txt += " (↓ better)"

        return {
            "Metric": metric,
            "Slice": _fmt_val(slice_v, kind),
            "Baseline": _fmt_val(base_v, kind),
            "Δ": d_txt,
        }

    rows = []

    # Headliners
    rows += [
        row("Holes", curr["holes_played"], base["holes_played"], kind="int", delta_kind="int"),
        row("To Par", curr["to_par"], base["to_par"], kind="to_par", invert=True),
        row("Avg / 72", curr.get("par72_score"), base.get("par72_score"), kind="num", invert=True),
        row("Putts / 18", curr["putts_per_18"], base["putts_per_18"], kind="num", invert=True),
        row("GIR%", curr["gir_pct"], base["gir_pct"], kind="pct", delta_kind="pct"),
        row("FW% (P4/P5)", curr["fw_pct"], base["fw_pct"], kind="pct", delta_kind="pct"),
        row("Scr%", curr["scr_pct"], base["scr_pct"], kind="pct", delta_kind="pct"),
        row("U&D%", curr["ud_pct"], base["ud_pct"], kind="pct", delta_kind="pct"),
        row("1P%", curr["one_putt_pct"], base["one_putt_pct"], kind="pct", delta_kind="pct"),
        row("3+P%", curr["three_plus_putt_pct"], base["three_plus_putt_pct"], kind="pct", invert=True, delta_kind="pct"),
        row("3P Bogey%", curr["three_putt_bogey_pct"], base["three_putt_bogey_pct"], kind="pct", invert=True, delta_kind="pct"),
        row("Lost Balls", curr["lb_total"], base["lb_total"], kind="int", invert=True, delta_kind="int"),
        row("Pro Pars+", curr["pro_pars_plus"], base["pro_pars_plus"], kind="int", delta_kind="int"),
        row("Arnies", curr["arnies"], base["arnies"], kind="int", delta_kind="int"),
        row("Seves", curr["seves"], base["seves"], kind="int", delta_kind="int"),
        row("Hole Outs", curr["hole_outs"], base["hole_outs"], kind="int", delta_kind="int"),
    ]

    # Par splits
    rows += [
        row("Avg P3", curr["avg_p3"], base["avg_p3"], kind="num", invert=True),
        row("Avg P4", curr["avg_p4"], base["avg_p4"], kind="num", invert=True),
        row("Avg P5", curr["avg_p5"], base["avg_p5"], kind="num", invert=True),
        row("GIR P3%", curr["gir3"][2], base["gir3"][2], kind="pct", delta_kind="pct"),
        row("GIR P4%", curr["gir4"][2], base["gir4"][2], kind="pct", delta_kind="pct"),
        row("GIR P5%", curr["gir5"][2], base["gir5"][2], kind="pct", delta_kind="pct"),
        row("FW P4%", curr["fw4"][2], base["fw4"][2], kind="pct", delta_kind="pct"),
        row("FW P5%", curr["fw5"][2], base["fw5"][2], kind="pct", delta_kind="pct"),
        row("GIR|FW% P4", curr["fw_gir4"][2], base["fw_gir4"][2], kind="pct", delta_kind="pct"),
        row("GIR|FW% P5", curr["fw_gir5"][2], base["fw_gir5"][2], kind="pct", delta_kind="pct"),
    ]

    return pd.DataFrame(rows)

def _style_compare_table(t: pd.DataFrame):
    if t is None or t.empty:
        return t
    def shade_delta(val):
        s = str(val)
        if "—" in s:
            return ""
        # light green for ▲ on higher-better metrics, but we can't infer here reliably.
        # We'll just give direction-based subtle shading.
        if s.startswith("▲"):
            return "background-color: rgba(0, 200, 0, 0.14); font-weight:800;"
        if s.startswith("▼"):
            return "background-color: rgba(255, 80, 80, 0.14); font-weight:800;"
        return "opacity: .85;"
    return t.style.applymap(shade_delta, subset=["Δ"])


def render_baseline_compare_dashboard(curr_df: pd.DataFrame, baseline_df: pd.DataFrame, title: str):
    st.markdown("<div class='dash-wrap'></div>", unsafe_allow_html=True)
    st.subheader(title)

    # Summaries
    curr = build_summary(curr_df)
    base = build_summary(baseline_df) if (baseline_df is not None and not baseline_df.empty) else None

    # Friendly labels
    slice_label = f"{_fmt_date(curr_df['Date Played'].min())} → {_fmt_date(curr_df['Date Played'].max())}"
    base_label = "—"
    if baseline_df is not None and not baseline_df.empty:
        base_label = f"{_fmt_date(baseline_df['Date Played'].min())} → {_fmt_date(baseline_df['Date Played'].max())}"

    # Hero (clean summary)
    tags = []
    tags.append(f"📌 Slice: {curr['holes_played']:,} holes")
    if baseline_df is not None and not baseline_df.empty:
        tags.append(f"🧱 Baseline: {baseline_df.shape[0]:,} holes")
    tags.append(f"📅 Slice dates: {slice_label}")
    if baseline_df is not None and not baseline_df.empty:
        tags.append(f"📅 Baseline dates: {base_label}")

    st.markdown(
        "<div class='hero'>"
        "  <div class='hero-top'>"
        "    <div>"
        "      <div class='hero-title'>🆚 Slice vs Baseline</div>"
        "      <div class='hero-sub'>Deltas are <b>Slice − Baseline</b>. Use the Baseline Settings above to control what you’re comparing against.</div>"
        "    </div>"
        "  </div>"
        + "".join([f"<span class='tag'>{t}</span>" for t in tags]) +
        "</div>",
        unsafe_allow_html=True,
    )
    if base is None:
        st.warning("Baseline is empty with current settings.")
        return


    # Quick totals    # Overlay another player (same slice filters)
    with st.expander("🧑‍🤝‍🧑 Overlay another player (same slice filters)", expanded=False):
        overlay_player = st.selectbox("Overlay Player", options=["— None —"] + players, index=0, key="baseline_dash_overlay_player")
        if overlay_player and overlay_player != "— None —":
            o_df = base_f[base_f["Player Name"] == overlay_player].copy()
            if o_df.empty:
                st.info("No rows for that player in the current slice filters.")
            else:
                oth = build_summary(o_df)
                comp_o = _compare_df(curr, oth)
                st.dataframe(_style_compare_table(comp_o), use_container_width=True, hide_index=True)
                st.caption("Overlay compares the current slice vs the selected player (same filters).")


    # Quick totals (Slice vs Baseline)
    avg72_curr_txt = "—" if curr.get("par72_score") is None else f"{curr['par72_score']:.1f}"
    avg72_base_txt = "—" if base.get("par72_score") is None else f"{base['par72_score']:.1f}"
    st.markdown("<div class='compare-grid'>"
                "<div class='compare-box'>"
                "<div class='compare-h'>📌 Slice Totals</div>"
                f"<div class='compare-row'><div class='compare-k'>Holes</div><div class='compare-v'>{curr['holes_played']:,}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>To Par</div><div class='compare-v'>{_fmt_to_par(curr['to_par'])}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>Avg / 72</div><div class='compare-v'>{avg72_curr_txt}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>Putts / 18</div><div class='compare-v'>{curr['putts_per_18']:.1f}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>GIR%</div><div class='compare-v'>{curr['gir_pct']:.1f}% {_emoji(curr['gir_pct'])}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>FW% (P4/P5)</div><div class='compare-v'>{curr['fw_pct']:.1f}% {_emoji(curr['fw_pct'])}</div></div>"
                "</div>"
                "<div class='compare-box'>"
                "<div class='compare-h'>🧱 Baseline Totals</div>"
                f"<div class='compare-row'><div class='compare-k'>Holes</div><div class='compare-v'>{base['holes_played']:,}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>To Par</div><div class='compare-v'>{_fmt_to_par(base['to_par'])}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>Avg / 72</div><div class='compare-v'>{avg72_base_txt}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>Putts / 18</div><div class='compare-v'>{base['putts_per_18']:.1f}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>GIR%</div><div class='compare-v'>{base['gir_pct']:.1f}% {_emoji(base['gir_pct'])}</div></div>"
                f"<div class='compare-row'><div class='compare-k'>FW% (P4/P5)</div><div class='compare-v'>{base['fw_pct']:.1f}% {_emoji(base['fw_pct'])}</div></div>"
                "</div>"
                "</div>", unsafe_allow_html=True)

    # ===== Key cards (match dashboard feel) =====
    st.markdown("<div class='section-h'>📊 Compare Table</div>", unsafe_allow_html=True)
    comp = _compare_df(curr, base)
    # Make it easier to scan: show most important rows first
    top_metrics = [
        "Holes","To Par","Avg / 72","Putts / 18","GIR%","FW% (P4/P5)","Scr%","U&D%","1P%","3+P%","3P Bogey%","Lost Balls"
    ]
    comp["Order"] = comp["Metric"].apply(lambda x: top_metrics.index(x) if x in top_metrics else 999)
    comp = comp.sort_values(["Order","Metric"]).drop(columns=["Order"]).reset_index(drop=True)

    st.dataframe(_style_compare_table(comp), use_container_width=True, hide_index=True)
    st.caption("Tip: Δ is Slice − Baseline (percentages shown in points).")


    st.markdown("<div class='section-h'>🎯 Ball Striking</div>", unsafe_allow_html=True)
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("GIR", _cnt_pair(curr["gir_made"], curr["gir_att"]), _delta_with_base(curr["gir_pct"], base["gir_pct"], pct_points=True))
    b2.metric("FW (P4/P5)", _cnt_pair(curr["fw_made"], curr["fw_att"]), _delta_with_base(curr["fw_pct"], base["fw_pct"], pct_points=True))
    b3.metric("GIR|FW", _cnt_pair(curr["fw_gir_made"], curr["fw_gir_att"]), _delta_with_base(curr["fw_gir_pct"], base["fw_gir_pct"], pct_points=True))
    b4.metric("GIR|FW% (P4/P5)", f"{curr['fw_gir_pct']:.1f}% {_emoji(curr['fw_gir_pct'])}", _delta_with_base(curr["fw_gir_pct"], base["fw_gir_pct"], pct_points=True))

    st.markdown("<div class='section-h'>🩹 Short Game</div>", unsafe_allow_html=True)
    s1, s2, s3, s4, s5, s6 = st.columns(6)
    s1.metric("U&D%", f"{curr['ud_pct']:.1f}%", _delta_with_base(curr["ud_pct"], base["ud_pct"], pct_points=True))
    s2.metric("1P%", f"{curr['one_putt_pct']:.1f}%", _delta_with_base(curr["one_putt_pct"], base["one_putt_pct"], pct_points=True))
    s3.metric("3+P%", f"{curr['three_plus_putt_pct']:.1f}%", _delta_with_base(curr["three_plus_putt_pct"], base["three_plus_putt_pct"], invert=True, pct_points=True))
    s4.metric("3P Bogey%", f"{curr['three_putt_bogey_pct']:.1f}%", _delta_with_base(curr["three_putt_bogey_pct"], base["three_putt_bogey_pct"], invert=True, pct_points=True))
    s5.metric("Lost Balls", f"{curr['lb_total']:,}", _delta_with_base(curr["lb_total"], base["lb_total"], invert=True))
    s6.metric("Pro Pars+", f"{curr['pro_pars_plus']:,}", _delta_with_base(curr["pro_pars_plus"], base["pro_pars_plus"]))

    st.markdown("<div class='section-h'>📈 Par Type Splits</div>", unsafe_allow_html=True)
    p1, p2, p3 = st.columns(3)
    p1.metric("Avg P3", _fmt_avg(curr["avg_p3"]), _delta_with_base(curr["avg_p3"], base["avg_p3"], invert=True))
    p2.metric("Avg P4", _fmt_avg(curr["avg_p4"]), _delta_with_base(curr["avg_p4"], base["avg_p4"], invert=True))
    p3.metric("Avg P5", _fmt_avg(curr["avg_p5"]), _delta_with_base(curr["avg_p5"], base["avg_p5"], invert=True))

    g1, g2, g3 = st.columns(3)
    g1.metric("GIR P3%", f"{curr['gir3'][2]:.1f}% {_emoji(curr['gir3'][2])}", _delta_with_base(curr["gir3"][2], base["gir3"][2], pct_points=True))
    g2.metric("GIR P4%", f"{curr['gir4'][2]:.1f}% {_emoji(curr['gir4'][2])}", _delta_with_base(curr["gir4"][2], base["gir4"][2], pct_points=True))
    g3.metric("GIR P5%", f"{curr['gir5'][2]:.1f}% {_emoji(curr['gir5'][2])}", _delta_with_base(curr["gir5"][2], base["gir5"][2], pct_points=True))

    f1, f2 = st.columns(2)
    f1.metric("FW P4%", f"{curr['fw4'][2]:.1f}% {_emoji(curr['fw4'][2])}", _delta_with_base(curr["fw4"][2], base["fw4"][2], pct_points=True))
    f2.metric("FW P5%", f"{curr['fw5'][2]:.1f}% {_emoji(curr['fw5'][2])}", _delta_with_base(curr["fw5"][2], base["fw5"][2], pct_points=True))

    st.markdown("<div class='section-h'>✨ Specials</div>", unsafe_allow_html=True)
    x1, x2, x3 = st.columns(3)
    x1.metric("Arnies", f"{curr['arnies']:,}", _delta_with_base(curr["arnies"], base["arnies"]))
    x2.metric("Seves", f"{curr['seves']:,}", _delta_with_base(curr["seves"], base["seves"]))
    x3.metric("Hole Outs", f"{curr['hole_outs']:,}", _delta_with_base(curr["hole_outs"], base["hole_outs"]))

    with st.expander("Show baseline + slice summaries (raw dicts)", expanded=False):
        st.json({"slice": curr, "baseline": base})


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
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum())),
        Score_Total=("Hole Score", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        Par_Total=("Par", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    # Scoring context (relative to Par)
    g["ToPar_Total"] = g["Score_Total"] - g["Par_Total"]
    g["ToPar/Hole"] = g.apply(lambda r: (r["ToPar_Total"] / r["Attempts"]) if r["Attempts"] else 0.0, axis=1)
    g["Avg Score"] = g.apply(lambda r: (r["Score_Total"] / r["Attempts"]) if r["Attempts"] else 0.0, axis=1)
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
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum())),
        Score_Total=("Hole Score", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        Par_Total=("Par", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    # Scoring context (relative to Par)
    g["ToPar_Total"] = g["Score_Total"] - g["Par_Total"]
    g["ToPar/Hole"] = g.apply(lambda r: (r["ToPar_Total"] / r["Attempts"]) if r["Attempts"] else 0.0, axis=1)
    g["Avg Score"] = g.apply(lambda r: (r["Score_Total"] / r["Attempts"]) if r["Attempts"] else 0.0, axis=1)
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
        GIR_Made=("GIR", lambda s: int((s == "Yes").sum())),
        Score_Total=("Hole Score", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
        Par_Total=("Par", lambda s: int(pd.to_numeric(s, errors="coerce").fillna(0).sum())),
    )
    g["GIR%"] = g.apply(lambda r: _pct(r["GIR_Made"], r["Attempts"]), axis=1)
    # Scoring context (relative to Par)
    g["ToPar_Total"] = g["Score_Total"] - g["Par_Total"]
    g["ToPar/Hole"] = g.apply(lambda r: (r["ToPar_Total"] / r["Attempts"]) if r["Attempts"] else 0.0, axis=1)
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

    has_group = "Group" in g.columns

    base = alt.Chart(g).encode(
        x=alt.X("Dist Bucket:N", sort=bucket_order, title="Approach Distance Bucket"),
        y=alt.Y("GIR%:Q", title="GIR %"),
        tooltip=[
            alt.Tooltip("Dist Bucket:N", title="Bucket"),
            alt.Tooltip("Qty:N", title="Qty"),
            alt.Tooltip("GIR%:Q", title="GIR%", format=".1f"),
            alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
        ] + ([alt.Tooltip("Group:N", title="Group")] if has_group else []),
    )

    if has_group:
        line = base.mark_line(point=True).encode(color=alt.Color("Group:N", title=None))
        st.altair_chart(line.properties(height=260), use_container_width=True)
    else:
        line = base.mark_line(point=True)
        text = base.mark_text(dy=-10).encode(text=alt.Text("Label:N"))
        st.altair_chart((line + text).properties(height=260), use_container_width=True)


def _bar_chart_club_gir(g: pd.DataFrame, top_n: int = 18):
    if g.empty:
        st.caption("No club data for GIR.")
        return

    has_group = "Group" in g.columns

    if has_group:
        # Keep top clubs by primary group (or overall if not present)
        gg = g.copy()
        # Choose an ordering based on total attempts
        club_order = (
            gg.groupby("Club")["Attempts"].sum()
            .sort_values(ascending=False)
            .head(top_n)
            .index.tolist()
        )
        gg = gg[gg["Club"].isin(club_order)].copy()

        ch = (
            alt.Chart(gg)
            .mark_bar()
            .encode(
                y=alt.Y("Club:N", sort=club_order, title=None),
                x=alt.X("GIR%:Q", title="GIR %"),
                color=alt.Color("Group:N", title=None),
                xOffset=alt.XOffset("Group:N"),
                tooltip=[
                    alt.Tooltip("Group:N", title="Group"),
                    alt.Tooltip("Club:N", title="Club"),
                    alt.Tooltip("Qty:N", title="Qty"),
                    alt.Tooltip("GIR%:Q", format=".1f", title="GIR%"),
                    alt.Tooltip("Attempts:Q", title="Attempts", format=",.0f"),
                ],
            )
            .properties(height=min(520, 26 * len(club_order) + 60))
        )
        st.altair_chart(ch, use_container_width=True)
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
    """Small grounding table for the Approach Analytics tab.

    We intentionally show GIR as Qty (made/att) so 'Attempts' is redundant.
    Replace it with Score to Par (avg per hole) to make performance impact obvious.
    """
    if frame.empty:
        return pd.DataFrame()

    b = frame.copy()

    def _row(label: str, block: pd.DataFrame):
        att = int(block.shape[0])
        made = int((block["GIR"] == "Yes").sum()) if att else 0
        pct = _pct(made, att)

        # Avg score relative to par (per hole). Negative is better.
        score = pd.to_numeric(block.get("Hole Score"), errors="coerce")
        par = pd.to_numeric(block.get("Par"), errors="coerce")
        rel = (score - par)
        rel_avg = float(rel.mean()) if att and rel.notna().any() else pd.NA

        return {
            "Split": label,
            "GIR": _cnt_pair(made, att),
            "GIR%": pct,
            "Score to Par": rel_avg,
        }

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
    # Score context formatting
    if "ToPar/Hole" in df_.columns:
        fmt["ToPar/Hole"] = "{:+.2f}"
    if "ToPar_Total" in df_.columns:
        fmt["ToPar_Total"] = "{:+.0f}"
    if "Avg Score" in df_.columns:
        fmt["Avg Score"] = "{:.2f}"
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
if mode == "🆚 Baseline Compare Dashboard":
    st.caption(f"Rows in slice: {slice_for_summary.shape[0]:,}")

    # Baseline controls + baseline frame
    baseline_df = _baseline_controls(df, key_prefix="baseline_dash")

    # Render compare dashboard
    render_baseline_compare_dashboard(
        curr_df=slice_for_summary,
        baseline_df=baseline_df,
        title="🆚 Baseline Compare Dashboard — Full"
    )

# =========================
# Home (gentle landing)
# =========================
if mode == "🏠 Home (Start Here)":
    st.markdown("<div class='dash-wrap'></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='dash-card'>"
        "<div class='dash-title'>🏠 Golf Slicer — Start Here</div>"
        "<div class='dash-sub'>Pick a view below. Your filters on the left/top still define the slice.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Quick snapshot of the current slice (based on the already-filtered frame 'f')
    snap = build_summary(f)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Holes", f"{snap['holes_played']:,}")
    c2.metric("Score", f"{snap['score_total']:,} ({_fmt_to_par(snap['to_par'])})")
    c3.metric("Putts", f"{snap['putts_total']:,}", delta=f"{snap['putts_per_18']:.1f}/18")
    c4.metric("GIR%", f"{snap['gir_pct']:.1f}% {_emoji(snap['gir_pct'])}", delta=_cnt_pair(snap["gir_made"], snap["gir_att"]))
    c5.metric("FW% (P4/P5)", f"{snap['fw_pct']:.1f}% {_emoji(snap['fw_pct'])}", delta=_cnt_pair(snap["fw_made"], snap["fw_att"]))
    c6.metric("Scr%", f"{snap['scr_pct']:.1f}%", delta=_cnt_pair(snap["scr_made"], snap["scr_ops"]))

    st.markdown("### 🚀 Choose your next view")

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("📦 Slice Summary", use_container_width=True):
            st.session_state["mode"] = "📦 Slice Summary (Any View)"
            st.rerun()
        if st.button("👥 Player Comparison", use_container_width=True):
            st.session_state["mode"] = "👥 Player Comparison (same slice)"
            st.rerun()

    with b2:
        if st.button("🏆 Leaderboard Dashboard", use_container_width=True):
            st.session_state["mode"] = "🏆 Leaderboard Dashboard (multi-stat)"
            st.rerun()
        if st.button("📊 Visual Dashboard", use_container_width=True):
            st.session_state["mode"] = "📊 Visual Dashboard (charts-only)"
            st.rerun()

    with b3:
        if st.button("📈 Approach Analytics", use_container_width=True):
            st.session_state["mode"] = "📈 Approach Analytics (Distance + Club + Heatmap)"
            st.rerun()
        if st.button("⛳ Putting Proximity", use_container_width=True):
            st.session_state["mode"] = "⛳ Putting Proximity (Validation)"
            st.rerun()

    st.caption("Tip: Start with **Slice Summary** to validate your filters, then jump into leaderboards or approach insights.")


if mode == "📦 Slice Summary (Any View)":
    st.caption(f"Rows in slice: {slice_for_summary.shape[0]:,}")

    summary = build_summary(slice_for_summary)
    render_summary_cards(summary, title="📦 Current Slice — Summary")
    render_score_mix(slice_for_summary, title="📊 Score Mix — Current Slice (Counts + %)")


    # =========================
    # Overlay Player (same slice)
    # =========================
    with st.expander("🧑‍🤝‍🧑 Overlay another player (same slice filters)", expanded=False):
        overlay_player = st.selectbox("Overlay Player", options=["— None —"] + players, index=0, key="slice_overlay_player")
        if overlay_player and overlay_player != "— None —":
            o_df = base_f[base_f["Player Name"] == overlay_player].copy()
            if o_df.empty:
                st.info("No rows for that player in the current slice filters.")
            else:
                curr_s = build_summary(slice_for_summary)
                oth_s = build_summary(o_df)
                comp_o = _compare_df(curr_s, oth_s)
                st.dataframe(_style_compare_table(comp_o), use_container_width=True, hide_index=True)
                st.caption("Overlay compares the current slice vs the selected player (same filters).")


    # =========================
    # Baseline Comparison
    # =========================
    st.markdown("---")
    st.subheader("📊 Baseline Comparison")

    baseline_type = st.selectbox(
        "Choose Baseline",
        ["All Time", "Same Year(s) as Slice", "Custom Year(s)", "Custom Date Range"],
        key="baseline_selector"
    )

    baseline_df = df.copy()

    if baseline_type == "Same Year(s) as Slice" and sel_years:
        baseline_df = baseline_df[baseline_df["Year"].isin([int(x) for x in sel_years])]

    elif baseline_type == "Custom Year(s)":
        all_years = sorted(df["Year"].dropna().unique())
        custom_years = st.multiselect("Select Year(s)", all_years, key="baseline_years")
        if custom_years:
            baseline_df = baseline_df[baseline_df["Year"].isin(custom_years)]

    elif baseline_type == "Custom Date Range":
        b_start = st.date_input("Baseline Start Date", value=df["Date Played"].min(), key="baseline_start")
        b_end = st.date_input("Baseline End Date", value=df["Date Played"].max(), key="baseline_end")
        baseline_df = baseline_df[
            (baseline_df["Date Played"] >= pd.to_datetime(b_start)) &
            (baseline_df["Date Played"] <= pd.to_datetime(b_end))
        ]

    baseline_summary = build_summary(baseline_df)

    def _delta(a, b, invert=False):
        if b == 0:
            return "—"
        d = a - b
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
        if invert:
            arrow = "▲" if d < 0 else ("▼" if d > 0 else "→")
        return f"{arrow} {d:.2f}"

    st.markdown("#### Key Metric Comparison")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("To Par",
              _fmt_to_par(summary["to_par"]),
              _delta(summary["to_par"], baseline_summary["to_par"], invert=True))

    c2.metric("GIR %",
              _fmt_pct1(summary["gir_pct"]),
              _delta(summary["gir_pct"], baseline_summary["gir_pct"]))

    c3.metric("FW %",
              _fmt_pct1(summary["fw_pct"]),
              _delta(summary["fw_pct"], baseline_summary["fw_pct"]))

    c4.metric("Putts / 18",
              _fmt_avg(summary["putts_per_18"]),
              _delta(summary["putts_per_18"], baseline_summary["putts_per_18"], invert=True))

    # =========================
    # Baseline Comparison
    # =========================
    st.markdown("---")
    st.subheader("📊 Baseline Comparison")

    baseline_type = st.selectbox(
        "Choose Baseline",
        ["All Time", "Same Year(s) as Slice", "Custom Year(s)", "Custom Date Range"]
    )

    baseline_df = df.copy()

    if baseline_type == "Same Year(s) as Slice" and sel_years:
        baseline_df = baseline_df[baseline_df["Year"].isin([int(x) for x in sel_years])]

    elif baseline_type == "Custom Year(s)":
        all_years = sorted(df["Year"].dropna().unique())
        custom_years = st.multiselect("Select Year(s)", all_years)
        if custom_years:
            baseline_df = baseline_df[baseline_df["Year"].isin(custom_years)]

    elif baseline_type == "Custom Date Range":
        b_start = st.date_input("Baseline Start Date", value=df["Date Played"].min())
        b_end = st.date_input("Baseline End Date", value=df["Date Played"].max())
        baseline_df = baseline_df[
            (baseline_df["Date Played"] >= pd.to_datetime(b_start)) &
            (baseline_df["Date Played"] <= pd.to_datetime(b_end))
        ]

    baseline_summary = build_summary(baseline_df)

    def _delta(a, b, invert=False):
        if b == 0:
            return "—"
        d = a - b
        arrow = "▲" if d > 0 else ("▼" if d < 0 else "→")
        if invert:
            arrow = "▲" if d < 0 else ("▼" if d > 0 else "→")
        return f"{arrow} {d:.2f}"

    st.markdown("#### Key Metric Comparison")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("To Par",
              _fmt_to_par(summary["to_par"]),
              _delta(summary["to_par"], baseline_summary["to_par"], invert=True))

    c2.metric("GIR %",
              _fmt_pct1(summary["gir_pct"]),
              _delta(summary["gir_pct"], baseline_summary["gir_pct"]))

    c3.metric("FW %",
              _fmt_pct1(summary["fw_pct"]),
              _delta(summary["fw_pct"], baseline_summary["fw_pct"]))

    c4.metric("Putts / 18",
              _fmt_avg(summary["putts_per_18"]),
              _delta(summary["putts_per_18"], baseline_summary["putts_per_18"], invert=True))


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

    
    # ----------------------------------------------------------
    # NEW: Score Breakdown Leaderboards (Counts + % of Holes)
    # ----------------------------------------------------------
    with st.expander("🏌️ Score Breakdown (Counts + % of holes)", expanded=True):

        if "Score Label" not in dash_frame.columns:
            st.warning("No 'Score Label' column found in this slice.")
        else:
            score_order = ["Albatross", "Eagle", "Birdie", "Par", "Bogey", "Double Bogey", "Triple Bogey +"]

            rows_sb = []
            for p in dfc_all["Player"].tolist():
                b = dash_frame[dash_frame["Player Name"] == p].copy()
                holes = int(len(b))
                if holes <= 0:
                    continue

                vc = b["Score Label"].value_counts()

                def _cnt(lbl):
                    return int(vc.get(lbl, 0))

                al = _cnt("Albatross")
                ea = _cnt("Eagle")
                bi = _cnt("Birdie")
                pa = _cnt("Par")
                bo = _cnt("Bogey")
                db = _cnt("Double Bogey")
                tp = _cnt("Triple Bogey +")

                par_or_better = al + ea + bi + pa
                dbl_plus = db + tp

                r = {"Player": p, "Holes": holes}

                for lbl, c in [
                    ("Albatross", al),
                    ("Eagle", ea),
                    ("Birdie", bi),
                    ("Par", pa),
                    ("Bogey", bo),
                    ("Double Bogey", db),
                    ("Triple Bogey +", tp),
                ]:
                    r[lbl] = c
                    r[f"{lbl}%"] = _pct(c, holes)

                r["Par or Better"] = par_or_better
                r["Par or Better%"] = _pct(par_or_better, holes)
                r["Double Bogey+"] = dbl_plus
                r["Double Bogey+%"] = _pct(dbl_plus, holes)

                rows_sb.append(r)

            df_sb = pd.DataFrame(rows_sb)

            if df_sb.empty:
                st.info("No score breakdown rows available.")
            else:
                keep_full = ["Player"] + sum(
                    [[lbl, f"{lbl}%"] for lbl in score_order], []
                ) + ["Par or Better", "Par or Better%", "Double Bogey+", "Double Bogey+%"]

                keep_full = [c for c in keep_full if c in df_sb.columns]

                render_mini_leaderboards(df_sb[keep_full].copy(), top_n=top_n, title="Score Breakdown")

                # Table view (like other sections): Count + % in one cell
                def _fmt_cp(c, holes):
                    return f"{int(c)}/{int(holes)} ({_pct(c, holes):.1f}%)"

                table_cols = ["Player","Holes"] + score_order + ["Par or Better","Double Bogey+"]
                table_cols = [c for c in table_cols if c in df_sb.columns]
                df_tbl = df_sb[table_cols].copy()

                for lbl in score_order:
                    if lbl in df_tbl.columns:
                        df_tbl[lbl] = df_sb.apply(lambda r: _fmt_cp(r.get(lbl, 0), r["Holes"]), axis=1)

                if "Par or Better" in df_tbl.columns:
                    df_tbl["Par or Better"] = df_sb.apply(lambda r: _fmt_cp(r.get("Par or Better", 0), r["Holes"]), axis=1)

                if "Double Bogey+" in df_tbl.columns:
                    df_tbl["Double Bogey+"] = df_sb.apply(lambda r: _fmt_cp(r.get("Double Bogey+", 0), r["Holes"]), axis=1)

                st.markdown("**Table — Counts + % of holes**")
                st.dataframe(df_tbl, use_container_width=True, hide_index=True)

                if show_charts:
                    st.markdown("**Quick Chart — Categories (Par or Better / Bogey / Double+)**")
                    cat_rows = []
                    for _, r in df_sb.iterrows():
                        holes = float(r["Holes"]) if r.get("Holes") else 0.0
                        pob = float(r.get("Par or Better", 0))
                        bog = float(r.get("Bogey", 0))
                        dplus = float(r.get("Double Bogey+", 0))
                        for cat, c in [("Par or Better", pob), ("Bogey", bog), ("Double Bogey+", dplus)]:
                            cat_rows.append({
                                "Player": r["Player"],
                                "Category": cat,
                                "Count": c,
                                "Percent": _pct(c, holes) if holes else 0.0
                            })
                    df_cat = pd.DataFrame(cat_rows)

                    if not df_cat.empty and df_cat["Count"].sum() > 0:
                        cat_chart = (
                            alt.Chart(df_cat)
                            .mark_bar()
                            .encode(
                                y=alt.Y("Player:N", title=None),
                                x=alt.X("Percent:Q", title="% of holes", stack="normalize"),
                                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                                tooltip=[
                                    "Player:N",
                                    "Category:N",
                                    alt.Tooltip("Count:Q", title="Count", format=",.0f"),
                                    alt.Tooltip("Percent:Q", title="%", format=".1f"),
                                ],
                            )
                            .properties(height=min(450, 40 * df_sb.shape[0] + 60))
                        )
                        st.altair_chart(cat_chart, use_container_width=True)
                    else:
                        st.caption("No category counts found for this slice.")

                if show_charts:
                    st.markdown("**Quick Chart — Score Mix by Player (% of holes)**")

                    long_rows = []
                    for _, r in df_sb.iterrows():
                        for lbl in score_order:
                            long_rows.append({
                                "Player": r["Player"],
                                "Category": lbl,
                                "Count": r.get(lbl, 0),
                                "Percent": r.get(f"{lbl}%", 0.0),
                            })

                    df_long = pd.DataFrame(long_rows)

                    if df_long["Count"].sum() == 0:
                        st.caption("No score label counts found.")
                    else:
                        chart = (
                            alt.Chart(df_long)
                            .mark_bar()
                            .encode(
                                y=alt.Y("Player:N", title=None),
                                x=alt.X("Percent:Q", title="% of holes", stack="normalize"),
                                color=alt.Color("Category:N", legend=alt.Legend(orient="bottom")),
                                tooltip=[
                                    "Player:N",
                                    "Category:N",
                                    alt.Tooltip("Count:Q", title="Count"),
                                    alt.Tooltip("Percent:Q", title="%", format=".1f"),
                                ],
                            )
                        )
                        st.altair_chart(chart, use_container_width=True)

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

    a1, a2, a3, a4, a5 = st.columns([2.2, 1.0, 1.0, 1.1, 1.3])
    with a1:
        sel_a_players = st.multiselect("🎛️ Analytics — Players (optional)", options=a_players, default=a_default)
    with a2:
        min_attempts = st.slider("Min Attempts (bucket/club)", 1, 30, 5)
    with a3:
        min_cell = st.slider("Min Attempts (heatmap cell)", 1, 20, 3)
    with a4:
        top_clubs = st.slider("Top Clubs (bar)", 6, 30, 18)

    with a5:
        # Overlay up to 4 additional players on the charts (kept separate from the primary selection)
        try:
            overlay_players = st.multiselect(
                "Overlay Players (optional — up to 4)",
                options=a_players,
                default=[],
                max_selections=4,
                help="Overlay up to 4 additional players on the distance + club charts (same slice filters).",
            )
        except TypeError:
            # Older Streamlit: no max_selections
            overlay_players = st.multiselect(
                "Overlay Players (optional — up to 4)",
                options=a_players,
                default=[],
                help="Overlay up to 4 additional players on the distance + club charts (same slice filters).",
            )
            overlay_players = overlay_players[:4]

    a_frame = base_f.copy()
    if sel_a_players:
        a_frame = a_frame[a_frame["Player Name"].isin(sel_a_players)].copy()

    cmp_frames = {}
    if 'overlay_players' in locals() and overlay_players:
        for _p in overlay_players:
            try:
                cmp_frames[_p] = base_f[base_f["Player Name"] == _p].copy()
            except Exception:
                continue

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
    ref_view = ref[["Split", "GIR", "GIR%", "Score to Par"]].copy()
    # Friendly formatting (avg score relative to par per hole)
    if "Score to Par" in ref_view.columns:
        ref_view["Score to Par"] = ref_view["Score to Par"].apply(_fmt_to_par)
    st.dataframe(_style_pct_table(ref_view, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)

    # A) Distance buckets
    st.markdown("<div class='section-h'>A) GIR by Distance Bucket</div>", unsafe_allow_html=True)
    dist_g = build_gir_by_distance(a_frame, min_attempts=min_attempts)

    parts = []
    if dist_g is not None and not dist_g.empty:
        d0 = dist_g.copy()
        d0["Group"] = "Primary"
        parts.append(d0)

    if isinstance(locals().get("cmp_frames", None), dict) and cmp_frames:
        for _p, _dfp in cmp_frames.items():
            if _dfp is None or _dfp.empty:
                continue
            _d = build_gir_by_distance(_dfp, min_attempts=min_attempts)
            if _d is None or _d.empty:
                continue
            _d = _d.copy()
            _d["Group"] = str(_p)
            parts.append(_d)

    dist_plot = pd.concat(parts, ignore_index=True) if parts else dist_g
    _line_chart_distance_gir(dist_plot)

    # B) Clubs
    st.markdown("<div class='section-h'>B) GIR by Club</div>", unsafe_allow_html=True)
    club_g = build_gir_by_club(a_frame, min_attempts=min_attempts)

    parts = []
    if club_g is not None and not club_g.empty:
        c0 = club_g.copy()
        c0["Group"] = "Primary"
        parts.append(c0)

    if isinstance(locals().get("cmp_frames", None), dict) and cmp_frames:
        for _p, _dfp in cmp_frames.items():
            if _dfp is None or _dfp.empty:
                continue
            _c = build_gir_by_club(_dfp, min_attempts=min_attempts)
            if _c is None or _c.empty:
                continue
            _c = _c.copy()
            _c["Group"] = str(_p)
            parts.append(_c)

    club_plot = pd.concat(parts, ignore_index=True) if parts else club_g
    _bar_chart_club_gir(club_plot, top_n=top_clubs)

    # Best of both worlds (heatmap)
    st.markdown("<div class='section-h'>Best of both worlds) Distance × Club Heatmap (GIR%)</div>", unsafe_allow_html=True)
    heat = build_gir_heatmap_distance_x_club(a_frame, min_cell_attempts=min_cell)

    # Heatmap overlay: show primary + the *first* overlay player (keeps layout readable)
    _overlay_for_heat = None
    if isinstance(locals().get("cmp_frames", None), dict) and cmp_frames:
        _overlay_for_heat = list(cmp_frames.keys())[0] if len(cmp_frames.keys()) > 0 else None

    if _overlay_for_heat:
        heat2 = build_gir_heatmap_distance_x_club(cmp_frames.get(_overlay_for_heat, pd.DataFrame()), min_cell_attempts=min_cell)
        hL, hR = st.columns(2)
        with hL:
            st.markdown("**Primary**")
            _heatmap_distance_x_club(heat)
        with hR:
            st.markdown(f"**Overlay: {_overlay_for_heat}**")
            _heatmap_distance_x_club(heat2)
        # Keep primary heat in variable for exports/expanders below
    else:
        _heatmap_distance_x_club(heat)

    with st.expander("Show tables (distance / club / heatmap)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Distance buckets**")
            dtab = dist_g[["Dist Bucket","Qty","GIR%","Attempts","ToPar/Hole"]].copy() if not dist_g.empty else pd.DataFrame()
            if not dtab.empty and "ToPar/Hole" in dtab.columns:
                dtab["Score to Par"] = dtab["ToPar/Hole"].apply(_fmt_to_par)
                dtab = dtab.drop(columns=["ToPar/Hole"])
            st.dataframe(_style_pct_table(dtab, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**Clubs**")
            ctab = club_g[["Club","Qty","GIR%","Attempts","ToPar/Hole"]].head(top_clubs).copy() if not club_g.empty else pd.DataFrame()
            if not ctab.empty and "ToPar/Hole" in ctab.columns:
                ctab["Score to Par"] = ctab["ToPar/Hole"].apply(_fmt_to_par)
                ctab = ctab.drop(columns=["ToPar/Hole"])
            st.dataframe(_style_pct_table(ctab, pct_cols=("GIR%",)), hide_index=True, use_container_width=True)

        st.markdown("**Heatmap cells**")
        htab = heat[["Dist Bucket","Club","Qty","GIR%","Attempts","ToPar/Hole"]].copy() if not heat.empty else pd.DataFrame()
        if not htab.empty and "ToPar/Hole" in htab.columns:
            htab["Score to Par"] = htab["ToPar/Hole"].apply(_fmt_to_par)
            htab = htab.drop(columns=["ToPar/Hole"])
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

elif mode == "⛳ Putting Proximity (Validation)":
    st.subheader("⛳ Putting Proximity Table — Validation")

    # Start from the current filtered slice (f), then apply putting-specific rules/filters below
    dfp = f.copy()

    # Ensure date/year
    dfp["Date Played"] = pd.to_datetime(dfp.get("Date Played"), errors="coerce")
    dfp["Year"] = dfp["Date Played"].dt.year

    # Soft safety for required columns
    for col in [
        "Player Name", "Course Name", "Hole", "Par", "Putts",
        PROX_COL, YARD_COL, CLUB_COL, "3 Putt Bogey",
        "GIR", "Approach GIR",
        FEET_MADE_COL, SCORE_COL,
    ]:
        if col not in dfp.columns:
            dfp[col] = pd.NA

    # Normalize GIR fields (allow values like 1/0, True/False, Yes/No)
    dfp["GIR"] = dfp["GIR"].apply(_norm_yes_no)
    dfp["Approach GIR"] = dfp["Approach GIR"].apply(_norm_yes_no)

    # Numeric parsing
    dfp["Putts"] = _as_int(dfp["Putts"], 0)
    dfp["Hole"] = _as_int(dfp["Hole"], 0)

    par_num = pd.to_numeric(dfp.get("Par"), errors="coerce").round()
    dfp["Par"] = par_num.where(par_num.isin([3, 4, 5]), pd.NA).astype("Int64")

    dfp["HoleScoreN"] = pd.to_numeric(dfp[SCORE_COL], errors="coerce")
    dfp["ProxN"] = pd.to_numeric(dfp[PROX_COL], errors="coerce")  # keep NaN if blank
    dfp["YardN"] = pd.to_numeric(dfp[YARD_COL], errors="coerce")
    dfp["FeetMadeN"] = pd.to_numeric(dfp[FEET_MADE_COL], errors="coerce")
    dfp["3 Putt Bogey"] = _as_int(dfp["3 Putt Bogey"], 0)

    # ---------------------------
    # Filters (local to this view)
    # ---------------------------
    years_available = sorted([int(y) for y in dfp["Year"].dropna().unique().tolist()])
    year_options = ["All"] + years_available
    sel_year = st.selectbox("Year", year_options, index=0)

    if sel_year != "All":
        dfp = dfp[dfp["Year"] == int(sel_year)].copy()

    if dfp.empty:
        st.warning("No rows found for the selected year.")
        st.stop()

    yr_text = "All Years" if sel_year == "All" else f"Year={sel_year}"
    st.caption(f"Showing: {yr_text} (based on your current top-level filters)")

    c1, c2, c3, c4 = st.columns([1.6, 1.8, 1.2, 2.0], vertical_alignment="bottom")
    with c1:
        players = sorted([x for x in dfp["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
        sel_player = st.selectbox("Player", ["All"] + players, key="pp_player")
    with c2:
        courses = sorted([x for x in dfp["Course Name"].dropna().unique().tolist() if str(x).strip() != ""])
        sel_course = st.selectbox("Course", ["All"] + courses, key="pp_course")
    with c3:
        pars = [3, 4, 5]
        sel_pars = st.multiselect("Par", pars, default=pars, key="pp_pars")
    with c4:
        clubs = sorted([x for x in dfp[CLUB_COL].dropna().unique().tolist() if str(x).strip() != ""])
        sel_clubs = st.multiselect("Approach Club", clubs, default=[], key="pp_clubs")
        sel_gir = st.selectbox("GIR", ["All", "Yes", "No"], key="pp_gir")
        sel_appr_gir = st.selectbox("Approach GIR", ["All", "Yes", "No"], key="pp_agir")

    c5, c6 = st.columns([2.2, 1.8], vertical_alignment="bottom")
    with c5:
        holes = sorted([int(x) for x in dfp["Hole"].dropna().unique().tolist() if int(x) > 0])
        sel_holes = st.multiselect("Hole Number", holes, default=[], key="pp_holes")
    with c6:
        yard_series = dfp["YardN"].dropna()
        y_min = int(yard_series.min()) if not yard_series.empty else 0
        y_max = int(yard_series.max()) if not yard_series.empty else 300
        y_low, y_high = st.slider(
            "Approach Yardage Range",
            min_value=0,
            max_value=max(1, y_max),
            value=(max(0, y_min), y_max),
            key="pp_yards",
        )

    g = dfp.copy()
    if sel_player != "All":
        g = g[g["Player Name"] == sel_player]
    if sel_course != "All":
        g = g[g["Course Name"] == sel_course]
    if sel_pars:
        g = g[g["Par"].isin(sel_pars)]
    if sel_clubs:
        g = g[g[CLUB_COL].isin(sel_clubs)]
    if sel_holes:
        g = g[g["Hole"].isin(sel_holes)]
    if sel_gir != "All":
        g = g[g["GIR"] == sel_gir]
    if sel_appr_gir != "All":
        g = g[g["Approach GIR"] == sel_appr_gir]

    # Yardage filter: keep blanks (do not exclude missing yardage)
    g = g[(g["YardN"].isna()) | ((g["YardN"] >= y_low) & (g["YardN"] <= y_high))].copy()

    # ---------------------------
    # DATASET RULES
    # ---------------------------
    # Remove hole-outs entirely (Putts == 0)
    # Exclude blanks in proximity, and exclude them from the dataset itself
    g = g[(g["Putts"] > 0) & (g["ProxN"].notna())].copy()

    if g.empty:
        st.info("No rows match filters after removing hole-outs and blank proximity.")
        st.stop()

    # ---------------------------
    # Bucket table
    # ---------------------------
    BUCKET_ORDER = ["0–3 ft", "3–6 ft", "6–10 ft", "10–16 ft", "16–22 ft", "23–30 ft", "30–40 ft", "> 40 ft"]

    def build_putting_prox_table(frame: pd.DataFrame) -> pd.DataFrame:
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
            ("23–30 ft", (tmp["ProxN"] > 22) & (tmp["ProxN"] <= 30)),
            ("30–40 ft", (tmp["ProxN"] > 30) & (tmp["ProxN"] <= 40)),
            ("> 40 ft",  (tmp["ProxN"] > 40)),
        ]

        # DEBUG: show any rows that don't land in a bucket
        all_bucket_mask = False
        for _, m in buckets:
            all_bucket_mask = all_bucket_mask | m

        not_bucketed = tmp[~all_bucket_mask].copy()
        if len(not_bucketed) > 0:
            st.warning(f"⚠️ {len(not_bucketed)} row(s) have ProxN that don't match any bucket.")
            st.dataframe(
                not_bucketed[["Date Played", "Player Name", "Course Name", "Hole", "PuttsN", "ProxN", "FeetMadeN", "HoleScoreN"]]
                .sort_values(["Date Played", "Hole"]),
                use_container_width=True,
                hide_index=True
            )

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

    table = build_putting_prox_table(g)

    # ---------------------------
    # Table styling
    # ---------------------------
    def _pct_str(x):
        try:
            return f"{float(x):.1f}%"
        except:
            return "—"

    percent_cols = ["1-putt %", "2-putt %", "3+ putt %", "3-putt bogey %", "Birdie+ %", "Par %"]

    table_display = table.copy()
    for col in percent_cols:
        table_display[col] = table_display[col].apply(_pct_str)

    def _style_putting_table(df_show: pd.DataFrame):
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

    st.dataframe(_style_putting_table(table_display), use_container_width=True, hide_index=True)

    # ---------------------------
    # Visual aid: Metric by distance
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
        key="pp_metric",
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

    if metric in ["3+ putt %", "3-putt bogey %"]:
        bar_color = "#EF4444"
        line_color = "#F97316"
    else:
        bar_color = "#3B82F6"
        line_color = "#22C55E"

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

    bars3 = base3.mark_bar(
        cornerRadiusTopLeft=7,
        cornerRadiusTopRight=7,
        opacity=0.90
    ).encode(
        y=alt.Y("Holes:Q", title="Attempts (Holes)"),
        color=alt.value("#60A5FA"),
    )

    line3 = base3.mark_line(
        strokeWidth=4,
        opacity=0.9
    ).encode(
        y=alt.Y("1-putt %:Q", title="Make % (1-putt %)", axis=alt.Axis(orient="right")),
        color=alt.value("#22C55E"),
    )

    points3 = base3.mark_point(
        size=160,
        filled=True,
        opacity=0.95
    ).encode(
        y=alt.Y("1-putt %:Q", axis=alt.Axis(orient="right")),
        color=alt.value("#22C55E"),
    )

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

    # ---------------------------
    # Overlay: Make % by Distance (multi-player comparison)
    # ---------------------------
    st.subheader("📉 Overlay — Make % by Distance (Multi-Player)")

    # Build an overlay dataset using ALL players (same filters, except player)
    ov = dfp.copy()

    # Apply same non-player filters
    if sel_course != "All":
        ov = ov[ov["Course Name"] == sel_course]
    if sel_pars:
        ov = ov[ov["Par"].isin(sel_pars)]
    if sel_clubs:
        ov = ov[ov[CLUB_COL].isin(sel_clubs)]
    if sel_holes:
        ov = ov[ov["Hole"].isin(sel_holes)]
    if sel_gir != "All":
        ov = ov[ov["GIR"] == sel_gir]
    if sel_appr_gir != "All":
        ov = ov[ov["Approach GIR"] == sel_appr_gir]

    # Yardage filter: keep blanks (do not exclude missing yardage)
    ov = ov[(ov["YardN"].isna()) | ((ov["YardN"] >= y_low) & (ov["YardN"] <= y_high))].copy()

    # Same dataset rules (remove hole-outs + require proximity)
    ov = ov[(ov["Putts"] > 0) & (ov["ProxN"].notna())].copy()

    ov_players = sorted([x for x in ov["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
    default_overlay = []
    if sel_player != "All" and sel_player in ov_players:
        default_overlay = [sel_player]
    elif len(ov_players) >= 2:
        default_overlay = ov_players[:2]
    elif len(ov_players) == 1:
        default_overlay = ov_players

    overlay_players = st.multiselect(
        "Overlay Players (line per player)",
        options=ov_players,
        default=default_overlay,
        help="Uses the same filters above (Course/Par/Club/Yardage/GIR), but lets you compare multiple players at once."
    )

    def _make_pct_overlay(frame: pd.DataFrame) -> pd.DataFrame:
        tmp = frame.copy()
        tmp["PuttsN"] = _as_int(tmp["Putts"], 0)

        buckets = [
            ("0–3 ft",   (tmp["ProxN"] >= 0) & (tmp["ProxN"] <= 3)),
            ("3–6 ft",   (tmp["ProxN"] > 3) & (tmp["ProxN"] <= 6)),
            ("6–10 ft",  (tmp["ProxN"] > 6) & (tmp["ProxN"] <= 10)),
            ("10–16 ft", (tmp["ProxN"] > 10) & (tmp["ProxN"] <= 16)),
            ("16–22 ft", (tmp["ProxN"] > 16) & (tmp["ProxN"] <= 22)),
            ("23–30 ft", (tmp["ProxN"] > 22) & (tmp["ProxN"] <= 30)),
            ("30–40 ft", (tmp["ProxN"] > 30) & (tmp["ProxN"] <= 40)),
            ("> 40 ft",  (tmp["ProxN"] > 40)),
        ]

        out_rows = []
        for label, mask in buckets:
            b = tmp[mask].copy()
            holes = int(len(b))
            makes = int((b["PuttsN"] == 1).sum())
            pct = (makes / holes * 100.0) if holes else 0.0
            out_rows.append({"Bucket": label, "Holes": holes, "1-putts": makes, "Make %": round(pct, 1)})

        out = pd.DataFrame(out_rows)
        out["Bucket"] = pd.Categorical(out["Bucket"], categories=BUCKET_ORDER, ordered=True)
        out = out.sort_values("Bucket").reset_index(drop=True)
        return out

    if len(overlay_players) < 1:
        st.info("Select at least one player to overlay.")
    else:
        ov_plot_parts = []
        for p in overlay_players:
            pv = ov[ov["Player Name"] == p].copy()
            if pv.empty:
                continue
            ptab = _make_pct_overlay(pv)
            ptab["Player"] = p
            ov_plot_parts.append(ptab)

        if not ov_plot_parts:
            st.info("No overlay data available for the selected filters.")
        else:
            ov_plot = pd.concat(ov_plot_parts, ignore_index=True)

            # Pretty labels: "makes/holes pct"
            ov_plot["Label"] = ov_plot.apply(
                lambda r: f"{int(r['1-putts'])}/{int(r['Holes'])} {float(r['Make %']):.1f}%" if int(r["Holes"]) else "—",
                axis=1
            )

            base_ov = alt.Chart(ov_plot).encode(
                x=alt.X("Bucket:N", sort=BUCKET_ORDER, title=None),
                y=alt.Y("Make %:Q", title="Make % (1-putt %)"),
                color=alt.Color("Player:N", legend=alt.Legend(title=None, orient="top")),
                tooltip=[
                    alt.Tooltip("Player:N", title="Player"),
                    alt.Tooltip("Bucket:N", title="Bucket"),
                    alt.Tooltip("Make %:Q", title="Make %", format=".1f"),
                    alt.Tooltip("1-putts:Q", title="1-putts", format=",.0f"),
                    alt.Tooltip("Holes:Q", title="Attempts (holes)", format=",.0f"),
                ],
            )

            lines_ov = base_ov.mark_line(strokeWidth=4, opacity=0.9)
            points_ov = base_ov.mark_point(size=150, filled=True, opacity=0.95)

            labels_ov = base_ov.mark_text(
                dy=-12,
                fontSize=12,
                fontWeight="bold",
                color="white"
            ).encode(text="Label:N")

            chart_ov = (lines_ov + points_ov + labels_ov).properties(height=380).configure_view(
                strokeOpacity=0
            ).configure_axis(
                labelColor="white",
                titleColor="white",
                gridColor="rgba(255,255,255,0.10)",
                tickColor="rgba(255,255,255,0.20)",
                domainColor="rgba(255,255,255,0.20)",
            )

            st.altair_chart(chart_ov, use_container_width=True)


else:
    st.warning("Unknown mode selection.")


# ==========================================================
# NEW: Club vs GIR Overlay (Inside Approach Analytics)
# ==========================================================

st.markdown("## 🎯 Club vs GIR — Overlay View")

if CLUB_COL in df.columns and "GIR" in df.columns:

    players_available = sorted(df["Player Name"].dropna().unique())

    col1, col2 = st.columns(2)
    with col1:
        primary_player = st.selectbox("Primary Player", players_available, key="club_primary")
    with col2:
        compare_player = st.selectbox("Compare Player (Optional)", [""] + players_available, key="club_compare")

    min_attempts = st.slider("Minimum Attempts per Club", 1, 20, 3, key="club_min_attempts")

    
    def _gir_flag(series: pd.Series) -> pd.Series:
        """Return 1/0 for GIR across common encodings: Yes/No, Y/N, True/False, 1/0."""
        s = series.copy()
        # Numeric already?
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().any():
            # Treat any positive as 1, zero/negative as 0 (common imports are 1/0)
            return (num.fillna(0) > 0).astype(int)
        # Strings / bool-like
        st_s = s.astype(str).str.strip().str.lower()
        yes = {"yes","y","true","t","1","gir","hit","made"}
        no  = {"no","n","false","f","0","miss","na","nan",""}
        out = st_s.map(lambda v: 1 if v in yes else (0 if v in no else 0))
        return out.fillna(0).astype(int)

    def compute_club_gir(data, player):
        d = data[data["Player Name"] == player].copy()
        d = d[d[CLUB_COL].notna() & (d[CLUB_COL].astype(str).str.strip() != "")]
        # GIR flag + attempts
        d["GIR_flag"] = _gir_flag(d["GIR"])
        grp = (
            d.groupby(CLUB_COL, dropna=False)
            .agg(
                attempts=("GIR_flag", "size"),
                gir_made=("GIR_flag", "sum"),
            )
            .reset_index()
        )
        grp = grp[grp["attempts"] >= min_attempts]
        grp["GIR_pct"] = grp.apply(lambda r: _pct(r["gir_made"], r["attempts"]), axis=1)
        grp["Player"] = player
        return grp

    base_df = df.copy()

    # Debug (helps when chart shows no points)
    with st.expander("🔎 Debug: Club vs GIR data", expanded=False):
        st.write("Total rows in current dataset:", len(base_df))
        st.write("Rows for primary player:", int((base_df["Player Name"] == primary_player).sum()))
        st.write("Unique GIR values (sample):", base_df["GIR"].dropna().astype(str).str.strip().str.lower().value_counts().head(12))
        st.write("Unique clubs (sample):", base_df[CLUB_COL].dropna().astype(str).str.strip().value_counts().head(20))


    p1_df = compute_club_gir(base_df, primary_player)

    if compare_player:
        p2_df = compute_club_gir(base_df, compare_player)
        chart_df = pd.concat([p1_df, p2_df])
    else:
        chart_df = p1_df

    if not chart_df.empty:
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True)
            .encode(
                x=alt.X(f"{CLUB_COL}:N", sort=None, title="Club"),
                y=alt.Y("GIR_pct:Q", title="GIR %"),
                color="Player:N",
                tooltip=[
                    "Player",
                    alt.Tooltip(CLUB_COL, title="Club"),
                    alt.Tooltip("attempts", title="Attempts"),
                    alt.Tooltip("gir_made", title="GIR Made"),
                    alt.Tooltip("GIR_pct", title="GIR %", format=".1f")
                ]
            )
            .properties(height=400)
        )

        st.altair_chart(chart, use_container_width=True)

    else:
        st.info("Not enough data for selected player(s) with current minimum attempts filter.")

else:
    st.warning("Required columns missing for Club vs GIR view.")