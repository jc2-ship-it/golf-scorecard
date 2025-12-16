import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Monthly Trends", layout="wide")
st.title("📈 Monthly Trends — 2025")

# ------------------ Load CSV ------------------
df = pd.read_csv("Hole Data-Grid view (18).csv")
df["Date Played"] = pd.to_datetime(df.get("Date Played"), errors="coerce")

required = ["Date Played", "Player Name", "Hole Score"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df = df.dropna(subset=["Date Played", "Player Name", "Hole Score"]).copy()
df["Year"] = df["Date Played"].dt.year
df = df[df["Year"] == 2025].copy()

if df.empty:
    st.warning("No rows found for year 2025 in the CSV.")
    st.stop()

df["YearMonth"] = df["Date Played"].dt.to_period("M").astype(str)

# ------------------ Safe numeric helpers ------------------
def _n(series, default=0):
    return pd.to_numeric(series, errors="coerce").fillna(default)

df["Hole Score"] = _n(df["Hole Score"])
df["Putts"] = _n(df.get("Putts", 0))
df["GIR"] = _n(df.get("GIR", 0))
df["Fairway"] = _n(df.get("Fairway", 0))
df["Fairway Attempt"] = _n(df.get("Fairway Attempt", 0))
df["Par"] = pd.to_numeric(df.get("Par"), errors="coerce")

df["Lost Ball Tee Shot Quantity"] = _n(df.get("Lost Ball Tee Shot Quantity", 0))
df["Lost Ball Approach Shot Quantity"] = _n(df.get("Lost Ball Approach Shot Quantity", 0))
df["Lost Balls"] = df["Lost Ball Tee Shot Quantity"] + df["Lost Ball Approach Shot Quantity"]

if "Score Label" not in df.columns:
    df["Score Label"] = ""

# ------------------ Controls ------------------
METRICS = [
    "Average Score Per Round",
    "GIR %",
    "Fairway %",
    "Putts Per Round",
    "Lost Balls Per Round",
    "Par or Better %",
    "Bogey %",
    "Double Bogey or Worse %",
]

players = sorted(df["Player Name"].dropna().unique().tolist())

c1, c2 = st.columns([2.4, 3.6], vertical_alignment="bottom")

with c1:
    metric = st.radio(
        "Metric",
        METRICS,
        index=0,
        horizontal=True,
        label_visibility="visible",
    )

with c2:
    selected_players = st.multiselect("Players", players, default=players)

if not selected_players:
    st.info("Select at least one player.")
    st.stop()

df = df[df["Player Name"].isin(selected_players)].copy()
if df.empty:
    st.info("No data for the current selection.")
    st.stop()

has_round = "Round Link" in df.columns and df["Round Link"].notna().any()

# ------------------ Aggregation helpers ------------------
def month_order_from(frame: pd.DataFrame):
    return sorted(frame["YearMonth"].dropna().unique().tolist())

def per_hole_pct_table(frame: pd.DataFrame, mask_fn):
    tmp = frame.copy()
    tmp["Hit"] = mask_fn(tmp).astype(int)
    out = (
        tmp.groupby(["YearMonth", "Player Name"], as_index=False)
           .agg(Hit=("Hit", "sum"), Den=("Hit", "count"))
    )
    out["Value"] = out.apply(lambda r: (r["Hit"] / r["Den"] * 100.0) if r["Den"] else 0.0, axis=1)
    return out

def fairway_pct_table(frame: pd.DataFrame):
    tmp = frame.copy()
    if "Fairway Attempt" in tmp.columns and tmp["Fairway Attempt"].sum() > 0:
        g = (
            tmp.groupby(["YearMonth", "Player Name"], as_index=False)
               .agg(FW_Hit=("Fairway", "sum"), FW_Att=("Fairway Attempt", "sum"))
        )
        g["Value"] = g.apply(lambda r: (r["FW_Hit"] / r["FW_Att"] * 100.0) if r["FW_Att"] else 0.0, axis=1)
        return g

    # fallback: Par 4/5 inferred attempts
    if "Par" in tmp.columns:
        p45 = tmp[tmp["Par"].isin([4, 5])].copy()
    else:
        p45 = tmp.copy()

    g = (
        p45.groupby(["YearMonth", "Player Name"], as_index=False)
           .agg(FW_Hit=("Fairway", "sum"), FW_Att=("Fairway", "count"))
    )
    g["Value"] = g.apply(lambda r: (r["FW_Hit"] / r["FW_Att"] * 100.0) if r["FW_Att"] else 0.0, axis=1)
    return g

def per_round_avg_table(frame: pd.DataFrame, value_col: str):
    # Uses Round Link if possible; else normalizes per-18 from holes
    if has_round:
        per_round = (
            frame.groupby(["YearMonth", "Player Name", "Round Link"], as_index=False)
                 .agg(RoundValue=(value_col, "sum"), Holes=("Hole Score", "count"))
        )
        per_round["RoundValue18"] = per_round.apply(
            lambda r: (r["RoundValue"] / r["Holes"] * 18.0) if r["Holes"] else 0.0,
            axis=1
        )
        out = (
            per_round.groupby(["YearMonth", "Player Name"], as_index=False)
                    .agg(Value=("RoundValue18", "mean"), Rounds=("Round Link", "nunique"))
        )
        return out

    g = (
        frame.groupby(["YearMonth", "Player Name"], as_index=False)
             .agg(Total=(value_col, "sum"), Holes=("Hole Score", "count"))
    )
    g["Value"] = g.apply(lambda r: (r["Total"] / r["Holes"] * 18.0) if r["Holes"] else 0.0, axis=1)
    return g

def metric_table(frame: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    if metric_name == "Average Score Per Round":
        return per_round_avg_table(frame, "Hole Score")

    if metric_name == "Putts Per Round":
        return per_round_avg_table(frame, "Putts")

    if metric_name == "Lost Balls Per Round":
        return per_round_avg_table(frame, "Lost Balls")

    if metric_name == "GIR %":
        out = (
            frame.groupby(["YearMonth", "Player Name"], as_index=False)
                 .agg(GIR_Made=("GIR", "sum"), Holes=("GIR", "count"))
        )
        out["Value"] = out.apply(lambda r: (r["GIR_Made"] / r["Holes"] * 100.0) if r["Holes"] else 0.0, axis=1)
        return out

    if metric_name == "Fairway %":
        return fairway_pct_table(frame)

    if metric_name == "Par or Better %":
        return per_hole_pct_table(
            frame,
            lambda t: t["Score Label"].isin(["Eagle", "Birdie", "Par"]) if "Score Label" in t.columns
            else (_n(t["Hole Score"]) <= _n(t["Par"]))
        )

    if metric_name == "Bogey %":
        return per_hole_pct_table(
            frame,
            lambda t: (t["Score Label"] == "Bogey") if "Score Label" in t.columns
            else (_n(t["Hole Score"]) == (_n(t["Par"]) + 1))
        )

    if metric_name == "Double Bogey or Worse %":
        return per_hole_pct_table(
            frame,
            lambda t: t["Score Label"].isin(["Double Bogey", "Triple Bogey +"]) if "Score Label" in t.columns
            else (_n(t["Hole Score"]) >= (_n(t["Par"]) + 2))
        )

    return pd.DataFrame()

# ------------------ Chart ------------------
def build_chart(metric_df: pd.DataFrame, metric_name: str):
    if metric_df.empty:
        st.info(f"No data after aggregation for: {metric_name}")
        return None

    month_order = month_order_from(metric_df)
    baseline = float(metric_df["Value"].mean())

    band = 2.0 if "%" in metric_name else 1.0

    baseline_df = pd.DataFrame({"baseline": [baseline]})
    band_df = pd.DataFrame({"low": [baseline - band], "high": [baseline + band]})

    avg_band = (
        alt.Chart(band_df)
        .mark_rect(opacity=0.08)
        .encode(y="low:Q", y2="high:Q")
    )

    # Leaner baseline (less thick, slightly less opaque)
    avg_line = (
        alt.Chart(baseline_df)
        .mark_rule(strokeWidth=2.0, color="#dc2626", opacity=0.85)
        .encode(y="baseline:Q")
    )

    avg_label = (
        alt.Chart(baseline_df)
        .mark_text(align="left", dx=6, dy=-10, fontWeight="bold", fontSize=13, color="#dc2626", opacity=0.9)
        .encode(y="baseline:Q", text=alt.value(f"Avg: {baseline:.1f}"))
    )

    lines = (
        alt.Chart(metric_df)
        .mark_line(point=alt.OverlayMarkDef(size=60), strokeWidth=2.6)
        .encode(
            x=alt.X("YearMonth:N", sort=month_order, title="Month"),
            y=alt.Y("Value:Q", title=metric_name),
            color=alt.Color("Player Name:N", title="Player"),
            tooltip=[
                alt.Tooltip("YearMonth:N", title="Month"),
                alt.Tooltip("Player Name:N", title="Player"),
                alt.Tooltip("Value:Q", title=metric_name, format=".1f"),
            ],
        )
    )

    return (avg_band + avg_line + avg_label + lines).properties(height=340)

# ------------------ Render ------------------
metric_df = metric_table(df, metric)
chart = build_chart(metric_df, metric)
if chart is not None:
    st.altair_chart(chart, use_container_width=True)
