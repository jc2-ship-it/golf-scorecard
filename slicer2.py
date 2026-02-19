# slicer.py
# Streamlit “View Slicer” — works for ANY filtered slice of your hole data
# - Round scorecard
# - Slice summary (totals + score mix w/ counts + category mix)
# - Player comparison (2–4 players) with:
#     - Avg / 72 (par72 projection)
#     - clean rounding (no “all the digits”)
#     - counts + % side-by-side (GIR, FW, GIR-from-FW, Scr, U&D, 1-putts, 3+ putts, 3P bogeys)
#     - best-stat highlighting
#     - badge row (🥇/🥈/🥉 per stat)
#     - rank leaderboard
#     - quick charts
# - Adds: GIR from Fairway by Hole Type (Par 4/5)
# - Robust numeric parsing (prevents pd.to_numeric DataFrame TypeError)
# - Windows-safe date formatting (no %-m / %-d)

import pandas as pd
import streamlit as st
import datetime
import altair as alt

# =========================
# Config
# =========================
st.set_page_config(page_title="Golf Slicer", layout="wide")

CSV_FILE = "Hole Data-Grid view (18).csv"  # keep name if you run from same folder

PROX_COL = "Proximity to Hole - How far is your First Putt (FT)"
YARD_COL = "Approach Shot Distance (how far you had to the hole)"
CLUB_COL = "Approach Shot Club Used"
FEET_MADE_COL = "Feet of Putt Made (How far was the putt you made)"

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
    CLUB_COL, YARD_COL, PROX_COL, FEET_MADE_COL
]

# =========================
# Helpers (robust)
# =========================
def _ensure_cols(df: pd.DataFrame, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df

def _num(x, default=0):
    """
    Robust numeric coercion:
    - Series -> Series numeric
    - DataFrame -> DataFrame numeric
    - scalar/list -> Series numeric
    """
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
# Filters (Sidebar) — MULTISELECT for Player/Course/Year/Month
# =========================
st.sidebar.header("🔍 Filters (build your slice)")

players = sorted([x for x in df["Player Name"].dropna().unique().tolist() if str(x).strip() != ""])
courses = sorted([x for x in df["Course Name"].dropna().unique().tolist() if str(x).strip() != ""])
years = sorted([int(x) for x in df["Year"].dropna().unique().tolist()], reverse=True)
months = [datetime.date(2000, i, 1).strftime("%B") for i in range(1, 13)]

# ✅ Multi-selects (empty = All)
sel_players = st.sidebar.multiselect("Players", players, default=[])
sel_courses = st.sidebar.multiselect("Courses", courses, default=[])
sel_years = st.sidebar.multiselect("Years", years, default=[])
sel_months = st.sidebar.multiselect("Months", months, default=[])

sel_pars = st.sidebar.multiselect("Par", [3, 4, 5], default=[3, 4, 5])

clubs = sorted([x for x in df[CLUB_COL].dropna().unique().tolist() if str(x).strip() != ""])
sel_clubs = st.sidebar.multiselect("Approach Club", clubs, default=[])

# ✅ Fairway filter (before GIR)
sel_fw = st.sidebar.selectbox("Fairway (P4/P5 only)", ["All", "Yes", "No"], index=0)

sel_gir = st.sidebar.selectbox("GIR", ["All", "Yes", "No"], index=0)
sel_appr_gir = st.sidebar.selectbox("Approach GIR", ["All", "Yes", "No"], index=0)

holes = sorted([int(x) for x in df["Hole"].dropna().unique().tolist() if int(x) > 0])
sel_holes = st.sidebar.multiselect("Hole Number", holes, default=[])

# Yard slider (keeps blank yardage rows)
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
# Apply filters -> base slice (NO player filter for compare)
# =========================
base_f = df.copy()

# ✅ Apply multi-filters (empty list means "All")
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

# ✅ Fairway filter (only meaningful on Par 4/5; keep Par 3s in slice)
if sel_fw != "All":
    p45_mask = base_f["Par"].isin([4, 5])
    if sel_fw == "Yes":
        base_f = base_f[~p45_mask | (base_f["Fairway"] == 1)]
    else:  # "No"
        base_f = base_f[~p45_mask | (base_f["Fairway"] == 0)]

if sel_gir != "All":
    base_f = base_f[base_f["GIR"] == sel_gir]
if sel_appr_gir != "All":
    base_f = base_f[base_f["Approach GIR"] == sel_appr_gir]

# Yard filter: keep blanks
yard_n_base = _num(base_f.get(YARD_COL), default=pd.NA)
base_f = base_f[(yard_n_base.isna()) | ((yard_n_base >= y_low) & (yard_n_base <= y_high))].copy()

# =========================
# Single-slice frame (f)
# - For single-player modes, we still need a single "selected player"
#   but now it should be derived from sel_players:
#     - if exactly 1 selected: use it
#     - if 0 selected: treat as All (no filter)
#     - if 2+ selected: keep them in slice (no forced single)
# =========================
f = base_f.copy()

# If user selected players, filter the slice to those players
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
        "👥 Player Comparison (same slice)"
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

    # Fairways: only par 4/5 attempts
    p45 = b[b["Par"].isin([4, 5])]
    fw_made = int(_num(p45["Fairway"], 0).sum())
    fw_att = int(p45.shape[0])
    out["fw_made"] = fw_made
    out["fw_att"] = fw_att
    out["fw_pct"] = _pct(fw_made, fw_att)

    # GIR overall (Yes count)
    gir_made = int((b["GIR"] == "Yes").sum())
    out["gir_made"] = gir_made
    out["gir_att"] = holes_played
    out["gir_pct"] = _pct(gir_made, holes_played)

    # GIR from Fairway (Par 4/5 only): attempts = FW hit, makes = GIR Yes
    p45_fw = p45[p45["Fairway"] == 1]
    fw_gir_att = int(p45_fw.shape[0])
    fw_gir_made = int((p45_fw["GIR"] == "Yes").sum())
    out["fw_gir_made"] = fw_gir_made
    out["fw_gir_att"] = fw_gir_att
    out["fw_gir_pct"] = _pct(fw_gir_made, fw_gir_att)

    # Scrambles / Up&Downs
    scr_made = int(_num(b.get("Scramble", 0), 0).sum())
    scr_ops = int(_num(b.get("Scramble Opportunity", 0), 0).sum())
    out["scr_made"] = scr_made
    out["scr_ops"] = scr_ops
    out["scr_pct"] = _pct(scr_made, scr_ops)

    # Up & downs = no GIR + 1 putt; ops = scramble opps
    gir_yes = (b["GIR"] == "Yes")
    putt1 = (_as_int(b["Putts"], 0) == 1)
    up_made = int((~gir_yes & putt1).sum())
    out["ud_made"] = up_made
    out["ud_ops"] = scr_ops
    out["ud_pct"] = _pct(up_made, scr_ops)

    # Lost balls
    lb_tee = int(_num(b.get("Lost Ball Tee Shot Quantity", 0), 0).sum())
    lb_appr = int(_num(b.get("Lost Ball Approach Shot Quantity", 0), 0).sum())
    out["lb_tee"] = lb_tee
    out["lb_appr"] = lb_appr
    out["lb_total"] = lb_tee + lb_appr

    # One-putts / 3+
    one_putts = int((_as_int(b["Putts"], 0) == 1).sum())
    out["one_putts"] = one_putts
    out["one_putt_pct"] = _pct(one_putts, holes_played)

    three_plus_putts = int((_as_int(b["Putts"], 0) >= 3).sum())
    out["three_plus_putts"] = three_plus_putts
    out["three_plus_putt_pct"] = _pct(three_plus_putts, holes_played)

    # ✅ 3 Putt Bogey %: attempts should be GIR hits (not holes played)
    three_putt_bogeys = int(_num(b.get("3 Putt Bogey", 0), 0).sum())
    out["three_putt_bogeys"] = three_putt_bogeys
    out["three_putt_bogey_att"] = int(out["gir_made"])
    out["three_putt_bogey_pct"] = _pct(three_putt_bogeys, out["three_putt_bogey_att"])

    # Pro Pars+
    pro_cols = [c for c in ["Pro Par", "Pro Birdie", "Pro Eagle+"] if c in b.columns]
    out["pro_pars_plus"] = int(_num(b[pro_cols], 0).sum().sum()) if pro_cols else 0

    # Arnies / Seves / Hole outs
    out["arnies"] = int(_num(b.get("Arnie", 0), 0).sum())
    out["seves"] = int(_num(b.get("Seve", 0), 0).sum())
    out["hole_outs"] = int(_num(b.get("Hole Out", 0), 0).sum())

    # Scoring averages by par
    def _avg_for_par(p):
        block = b[b["Par"] == p]
        if block.empty:
            return None
        return float(_num(block["Hole Score"], 0).mean())

    out["avg_p3"] = _avg_for_par(3)
    out["avg_p4"] = _avg_for_par(4)
    out["avg_p5"] = _avg_for_par(5)

    # Par 72 projection (typical: 4x Par 3, 10x Par 4, 4x Par 5)
    if out["avg_p3"] is not None and out["avg_p4"] is not None and out["avg_p5"] is not None:
        par72_score = (out["avg_p3"] * 4) + (out["avg_p4"] * 10) + (out["avg_p5"] * 4)
        out["par72_score"] = float(par72_score)
        out["par72_to_par"] = float(par72_score - 72)
    else:
        out["par72_score"] = None
        out["par72_to_par"] = None

    # GIR by hole type
    def _gir_by_par(p):
        block = b[b["Par"] == p]
        t = int(block.shape[0])
        m = int((block["GIR"] == "Yes").sum())
        return m, t, _pct(m, t)

    out["gir3"] = _gir_by_par(3)
    out["gir4"] = _gir_by_par(4)
    out["gir5"] = _gir_by_par(5)

    # FW by par 4/5
    def _fw_by_par(p):
        block = b[b["Par"] == p]
        t = int(block.shape[0])
        m = int(_num(block["Fairway"], 0).sum()) if t else 0
        return m, t, _pct(m, t)

    out["fw4"] = _fw_by_par(4)
    out["fw5"] = _fw_by_par(5)

    # GIR from Fairway by hole type (Par 4/5 only)
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
# Summary cards (single slice / round / hole)
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
    c3.metric("Putts/Hole", f'{summary["putts_per_hole"]:.2f}')

    c4.metric(
        "GIR",
        f'{summary["gir_pct"]:.1f}% {_emoji(summary["gir_pct"])}',
        delta=_cnt_pair(summary["gir_made"], summary["gir_att"]),
    )
    c5.metric(
        "FW (P4/P5)",
        f'{summary["fw_pct"]:.1f}% {_emoji(summary["fw_pct"])}',
        delta=_cnt_pair(summary["fw_made"], summary["fw_att"]),
    )
    c6.metric(
        "GIR from FW (P4/P5)",
        f'{summary["fw_gir_pct"]:.1f}% {_emoji(summary["fw_gir_pct"])}',
        delta=_cnt_pair(summary["fw_gir_made"], summary["fw_gir_att"]),
    )
    c7.metric(
        "Scrambles",
        f'{summary["scr_pct"]:.1f}%',
        delta=_cnt_pair(summary["scr_made"], summary["scr_ops"]),
    )

    d1, d2, d3, d4, d5, d6 = st.columns(6)
    d1.metric(
        "Up & Downs",
        f'{summary["ud_pct"]:.1f}%',
        delta=_cnt_pair(summary["ud_made"], summary["ud_ops"]),
    )
    d2.metric(
        "1 Putts",
        f'{summary["one_putt_pct"]:.1f}%',
        delta=f'{int(summary["one_putts"]):,}/{int(summary["holes_played"]):,}',
    )
    d3.metric(
        "3+ Putts",
        f'{summary["three_plus_putt_pct"]:.1f}%',
        delta=f'{int(summary["three_plus_putts"]):,}/{int(summary["holes_played"]):,}',
    )
    d4.metric(
        "3P Bogeys",
        f'{summary["three_putt_bogey_pct"]:.1f}%',
        delta=f'{int(summary["three_putt_bogeys"]):,}/{int(summary.get("three_putt_bogey_att", summary["gir_made"])):,}',
    )
    d5.metric(
        "Lost Balls",
        _cnt(summary["lb_total"]),
        delta=f"Tee {_cnt(summary['lb_tee'])} / Appr {_cnt(summary['lb_appr'])}",
    )
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
# Player Comparison (2–4 players) — with counts + % columns + clean formatting
# =========================
def _build_player_compare_table(frame: pd.DataFrame, selected_players: list) -> pd.DataFrame:
    rows = []
    for p in selected_players:
        b = frame[frame["Player Name"] == p].copy()
        if b.empty:
            continue

        s = build_summary(b)

        rows.append({
            "Holes": int(s["holes_played"]),
            "Player": p,

            "Avg / 72": (None if s.get("par72_score") is None else float(s["par72_score"])),
            "Putts/Hole": float(s["putts_per_hole"]),

            "GIR": _cnt_pair(s["gir_made"], s["gir_att"]),
            "GIR%": float(s["gir_pct"]),

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

            # ✅ denominator = GIR hits
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
        "Avg / 72","Putts/Hole",
        "GIR","GIR%","FW","FW%","GIR|FW","GIR|FW%",
        "Scr","Scr%","U&D","U&D%","1P","1P%","3+P","3+P%","3P Bogey","3P Bogey%",
        "Lost Balls","Pro Pars+","Arnies","Seves","Hole Outs"
    ]
    dfc = dfc[cols].copy()

    for c in ["Avg / 72"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").round(1)
    for c in ["Putts/Hole"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").round(2)
    for c in ["GIR%","FW%","GIR|FW%","Scr%","U&D%","1P%","3+P%","3P Bogey%"]:
        dfc[c] = pd.to_numeric(dfc[c], errors="coerce").round(1)

    return dfc

def _style_compare(dfc: pd.DataFrame):
    if dfc.empty:
        return dfc

    hi_good = {"GIR%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"}
    lo_good = {"Avg / 72","Putts/Hole","Lost Balls","3+P%","3P Bogey%"}  # lower is better

    def _highlight(col):
        if col.name in hi_good:
            best = col.max()
            return ["background-color: rgba(0, 200, 0, 0.22); font-weight:700;" if v == best else "" for v in col]
        if col.name in lo_good:
            best = col.min()
            return ["background-color: rgba(0, 200, 0, 0.22); font-weight:700;" if v == best else "" for v in col]
        return [""] * len(col)

    ignore = {"Player","GIR","FW","GIR|FW","Scr","U&D","1P","3+P","3P Bogey"}
    numeric_cols = [c for c in dfc.columns if c not in ignore and pd.api.types.is_numeric_dtype(dfc[c])]
    sty = dfc.style.apply(_highlight, subset=numeric_cols)

    fmt = {
        "Avg / 72": "{:.1f}",
        "Putts/Hole": "{:.2f}",
        "GIR%": "{:.1f}",
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

    hi_good = ["GIR%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"]
    lo_good = ["Avg / 72","Putts/Hole","Lost Balls","3+P%","3P Bogey%"]

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

    hi_good = {"GIR%","FW%","GIR|FW%","Scr%","U&D%","1P%","Pro Pars+","Arnies","Seves","Hole Outs"}
    lo_good = {"Avg / 72","Putts/Hole","Lost Balls","3+P%","3P Bogey%"}  # lower is better

    out = {"Player": "Badges"}
    out["Holes"] = ""

    for col in dfc.columns:
        if col in {"Player","Holes","GIR","FW","GIR|FW","Scr","U&D","1P","3+P","3P Bogey"}:
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
# Round display helpers (clean names)
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
# Render by mode
# =========================
if mode == "📦 Slice Summary (Any View)":
    st.caption(f"Rows in slice: {slice_for_summary.shape[0]:,}")

    summary = build_summary(slice_for_summary)
    render_summary_cards(summary, title="📦 Current Slice — Summary")
    render_score_mix(slice_for_summary, title="📊 Score Mix — Current Slice (Counts + %)")

    with st.expander("Show slice rows (preview)", expanded=False):
        show_cols = [c for c in [
            "Date Played", "Player Name", "Course Name", "Round Link", "Hole", "Par", "Hole Score", "Putts",
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
    st.caption("Shows the most recent 18 rows for the selected (Course, Hole).")

    c1, c2 = st.columns([2, 1])
    with c1:
        hole_course = st.selectbox("Course", ["All"] + courses, index=0)
    with c2:
        hole_num = st.selectbox("Hole", holes, index=0)

    h = df.copy()
    if hole_course != "All":
        h = h[h["Course Name"] == hole_course]
    h = h[h["Hole"] == int(hole_num)].copy()

    # If user selected players, filter hole view to them
    if sel_players:
        h = h[h["Player Name"].isin(sel_players)]

    h = h.sort_values("Date Played", ascending=False).head(18).copy()
    if h.empty:
        st.warning("No rows found for that hole selection.")
        st.stop()

    st.subheader(f"🎯 Hole {hole_num} — Last {h.shape[0]} Results")
    show_cols = [c for c in [
        "Date Played", "Player Name", "Course Name", "Round Link", "Hole", "Par", "Hole Score", "Putts",
        "Fairway", "GIR", "Approach GIR", "Score Label", CLUB_COL, YARD_COL, PROX_COL
    ] if c in h.columns]
    st.dataframe(h[show_cols], use_container_width=True, hide_index=True)

    summary = build_summary(h)
    render_summary_cards(summary, title="📦 Hole Slice — Summary")
    render_score_mix(h, title="📊 Score Mix — Hole Slice (Counts + %)")

elif mode == "👥 Player Comparison (same slice)":
    st.caption(f"Rows in slice (all players): {base_f.shape[0]:,}")

    # If sidebar Players multi-select is set, use that as the default compare list.
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
            "Avg / 72",
            "Putts/Hole",
            "GIR%",
            "FW%",
            "GIR|FW%",
            "Scr%",
            "U&D%",
            "1P%",
            "3+P%",
            "3P Bogey%",
            "Lost Balls",
        ]
        metric = st.selectbox("Pick a metric to chart", [m for m in chart_metrics if m in dfc.columns], index=0)
        _compare_chart(dfc, metric)
