
import io
import math
import re
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Launch Monitor Analyzer", layout="wide")
alt.data_transformers.disable_max_rows()

DIRECTIONAL_COLS = {
    "Offline",
    "Curve",
    "Launch Direction",
    "Side Spin",
    "Spin Axis Tilt",
}

NUMERIC_COLS = {
    "Carry",
    "Total",
    "Peak Height",
    "Descent Angle",
    "Hang Time",
    "Ball Speed",
    "Launch Angle",
    "Back Spin",
    "Total Spin",
    "Club Speed",
    "Smash Factor",
}

METRIC_PRIORITY = [
    "Carry",
    "Total",
    "Ball Speed",
    "Club Speed",
    "Smash Factor",
    "Launch Angle",
    "Back Spin",
    "Total Spin",
    "Peak Height",
    "Descent Angle",
    "Hang Time",
    "Offline Num",
    "Curve Num",
    "Launch Direction Num",
    "Side Spin Num",
    "Spin Axis Tilt Num",
]

AUTO_FILE_CANDIDATES = [
    "Launch.csv",
    "Launch .csv",
    "launch.csv",
    "launch_data.csv",
]

# Illustrative benchmark baselines for comparison mode.
# These are meant as useful reference targets, not absolute truths.
BENCHMARKS = {
    "Tour / Pro Reference": {
        "DR": {"Carry": 275, "Ball Speed": 167, "Club Speed": 113, "Smash Factor": 1.48},
        "3W": {"Carry": 242, "Ball Speed": 156, "Club Speed": 107, "Smash Factor": 1.46},
        "5W": {"Carry": 230, "Ball Speed": 151, "Club Speed": 103, "Smash Factor": 1.46},
        "4I": {"Carry": 215, "Ball Speed": 141, "Club Speed": 98, "Smash Factor": 1.44},
        "5I": {"Carry": 205, "Ball Speed": 136, "Club Speed": 95, "Smash Factor": 1.43},
        "6I": {"Carry": 193, "Ball Speed": 131, "Club Speed": 92, "Smash Factor": 1.42},
        "7I": {"Carry": 181, "Ball Speed": 126, "Club Speed": 89, "Smash Factor": 1.41},
        "8I": {"Carry": 169, "Ball Speed": 120, "Club Speed": 85, "Smash Factor": 1.40},
        "9I": {"Carry": 157, "Ball Speed": 114, "Club Speed": 80, "Smash Factor": 1.39},
        "PW": {"Carry": 145, "Ball Speed": 108, "Club Speed": 76, "Smash Factor": 1.38},
        "GW": {"Carry": 130, "Ball Speed": 100, "Club Speed": 72, "Smash Factor": 1.37},
        "SW": {"Carry": 115, "Ball Speed": 92, "Club Speed": 68, "Smash Factor": 1.35},
        "LW": {"Carry": 100, "Ball Speed": 86, "Club Speed": 64, "Smash Factor": 1.34},
    },
    "Scratch Reference": {
        "DR": {"Carry": 255, "Ball Speed": 154, "Club Speed": 105, "Smash Factor": 1.47},
        "3W": {"Carry": 228, "Ball Speed": 145, "Club Speed": 100, "Smash Factor": 1.45},
        "5W": {"Carry": 216, "Ball Speed": 139, "Club Speed": 96, "Smash Factor": 1.45},
        "4I": {"Carry": 198, "Ball Speed": 129, "Club Speed": 91, "Smash Factor": 1.42},
        "5I": {"Carry": 188, "Ball Speed": 124, "Club Speed": 88, "Smash Factor": 1.41},
        "6I": {"Carry": 177, "Ball Speed": 119, "Club Speed": 85, "Smash Factor": 1.40},
        "7I": {"Carry": 166, "Ball Speed": 114, "Club Speed": 82, "Smash Factor": 1.39},
        "8I": {"Carry": 155, "Ball Speed": 108, "Club Speed": 78, "Smash Factor": 1.38},
        "9I": {"Carry": 144, "Ball Speed": 102, "Club Speed": 74, "Smash Factor": 1.38},
        "PW": {"Carry": 132, "Ball Speed": 96, "Club Speed": 70, "Smash Factor": 1.37},
        "GW": {"Carry": 118, "Ball Speed": 90, "Club Speed": 66, "Smash Factor": 1.36},
        "SW": {"Carry": 103, "Ball Speed": 84, "Club Speed": 62, "Smash Factor": 1.35},
        "LW": {"Carry": 90, "Ball Speed": 79, "Club Speed": 59, "Smash Factor": 1.34},
    },
    "10 Handicap Reference": {
        "DR": {"Carry": 225, "Ball Speed": 140, "Club Speed": 96, "Smash Factor": 1.46},
        "3W": {"Carry": 205, "Ball Speed": 132, "Club Speed": 92, "Smash Factor": 1.43},
        "5W": {"Carry": 194, "Ball Speed": 126, "Club Speed": 88, "Smash Factor": 1.43},
        "4I": {"Carry": 180, "Ball Speed": 118, "Club Speed": 84, "Smash Factor": 1.40},
        "5I": {"Carry": 171, "Ball Speed": 113, "Club Speed": 81, "Smash Factor": 1.39},
        "6I": {"Carry": 162, "Ball Speed": 108, "Club Speed": 78, "Smash Factor": 1.38},
        "7I": {"Carry": 153, "Ball Speed": 103, "Club Speed": 75, "Smash Factor": 1.37},
        "8I": {"Carry": 143, "Ball Speed": 98, "Club Speed": 72, "Smash Factor": 1.36},
        "9I": {"Carry": 133, "Ball Speed": 92, "Club Speed": 68, "Smash Factor": 1.35},
        "PW": {"Carry": 122, "Ball Speed": 87, "Club Speed": 64, "Smash Factor": 1.35},
        "GW": {"Carry": 108, "Ball Speed": 81, "Club Speed": 60, "Smash Factor": 1.34},
        "SW": {"Carry": 95, "Ball Speed": 76, "Club Speed": 57, "Smash Factor": 1.33},
        "LW": {"Carry": 82, "Ball Speed": 71, "Club Speed": 54, "Smash Factor": 1.31},
    },
}


def _clean_text(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


def parse_directional(value):
    s = _clean_text(value)
    if s == "":
        return np.nan
    if s in {"0", "0.0"}:
        return 0.0

    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*([A-Za-z\-]+)?\s*$", s)
    if not m:
        try:
            return float(s)
        except Exception:
            return np.nan

    num = float(m.group(1))
    suffix = (m.group(2) or "").upper()

    negative_tags = {"L", "DN", "O-I", "CLOSED", "IN"}
    positive_tags = {"R", "UP", "I-O", "OPEN", "OUT"}

    if suffix in negative_tags:
        return -abs(num)
    if suffix in positive_tags:
        return abs(num)
    return num


def to_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace(",", "", regex=False), errors="coerce")


def detect_table_type(df):
    cols = [str(c).strip() for c in df.columns]
    normalized_markers = {"Player", "Club", "Carry"}
    if normalized_markers.issubset(set(cols)):
        return "normalized"

    rows = df.fillna("")
    for _, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        if "Date" in vals and "Carry" in vals and "Offline" in vals:
            return "monitor_export"
    return "unknown"


def normalize_shot_table(df):
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]

    if "Player" not in out.columns:
        out["Player"] = "Unknown Player"
    if "Club" not in out.columns:
        out["Club"] = "Unknown Club"
    if "Source File" not in out.columns:
        out["Source File"] = "uploaded file"

    for col in DIRECTIONAL_COLS:
        if col in out.columns:
            out[col + " Num"] = out[col].apply(parse_directional)

    for col in NUMERIC_COLS:
        if col in out.columns:
            out[col] = to_numeric(out[col])

    if "Date" in out.columns:
        out["Date Parsed"] = pd.to_datetime(out["Date"], errors="coerce")
    else:
        out["Date Parsed"] = pd.NaT

    out["Session Label"] = np.where(
        out["Date Parsed"].notna(),
        out["Date Parsed"].dt.strftime("%Y-%m-%d"),
        out["Source File"].astype(str),
    )

    club_order = {
        "DR": 1, "DRIVER": 1, "3W": 2, "5W": 3, "7W": 4,
        "3H": 5, "4H": 6, "5H": 7,
        "3I": 8, "4I": 9, "5I": 10, "6I": 11, "7I": 12,
        "8I": 13, "9I": 14, "PW": 15, "GW": 16, "AW": 17,
        "SW": 18, "LW": 19,
    }

    def club_key(x):
        s = str(x).strip().upper().replace(" ", "")
        return club_order.get(s, 999)

    out["Club Clean"] = out["Club"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    out["Club Sort"] = out["Club"].map(club_key)
    return out


def parse_monitor_export(df, source_name="Launch.csv"):
    rows = df.fillna("")
    header_idxs = []
    for i, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        if "Date" in vals and "Carry" in vals and "Offline" in vals:
            header_idxs.append(i)

    player_name = str(df.columns[0]).strip()
    if player_name.startswith("Unnamed"):
        player_name = ""

    records = []
    current_club = None
    current_header = None

    for idx, row in rows.iterrows():
        vals = [_clean_text(v) for v in row.tolist()]
        nonempty = [v for v in vals if v != ""]
        if not nonempty:
            continue

        if idx in header_idxs:
            current_header = vals
            continue

        if player_name and vals[0] == player_name and len(nonempty) <= 2:
            continue

        if vals[0] and len(nonempty) == 1 and vals[0] not in {"Average", "Std Dev", "Longest", "Shortest"}:
            current_club = vals[0]
            continue

        if vals[0] in {"Average", "Std Dev", "Longest", "Shortest"}:
            continue

        if re.fullmatch(r"\d+", vals[0] or "") and current_header is not None:
            rec = {}
            for c, v in zip(current_header, vals):
                if c:
                    rec[c] = v
            rec["Club"] = current_club
            rec["Player"] = player_name or "Unknown Player"
            rec["Source File"] = source_name
            records.append(rec)

    out = pd.DataFrame(records)
    if out.empty:
        return out
    return normalize_shot_table(out)


def load_single_csv(file_like, source_name="Launch.csv"):
    raw = pd.read_csv(file_like)
    t = detect_table_type(raw)
    if t == "normalized":
        return normalize_shot_table(raw)
    if t == "monitor_export":
        return parse_monitor_export(raw, source_name=source_name)
    return normalize_shot_table(raw)




def ensure_shot_schema(df):
    out = df.copy()
    if "Player" not in out.columns:
        out["Player"] = "Unknown Player"
    if "Club" not in out.columns:
        out["Club"] = "Unknown Club"

    club_order = {
        "DR": 1, "DRIVER": 1, "3W": 2, "5W": 3, "7W": 4,
        "3H": 5, "4H": 6, "5H": 7,
        "3I": 8, "4I": 9, "5I": 10, "6I": 11, "7I": 12,
        "8I": 13, "9I": 14, "PW": 15, "GW": 16, "AW": 17,
        "SW": 18, "LW": 19,
    }

    def club_key(x):
        s = str(x).strip().upper().replace(" ", "")
        return club_order.get(s, 999)

    if "Club Clean" not in out.columns:
        out["Club Clean"] = out["Club"].astype(str).str.strip().str.upper().str.replace(" ", "", regex=False)
    if "Club Sort" not in out.columns:
        out["Club Sort"] = out["Club"].map(club_key)
    else:
        out["Club Sort"] = pd.to_numeric(out["Club Sort"], errors="coerce")
        missing_sort = out["Club Sort"].isna()
        if missing_sort.any():
            out.loc[missing_sort, "Club Sort"] = out.loc[missing_sort, "Club"].map(club_key)

    if "Session Label" not in out.columns:
        if "Date" in out.columns:
            dt = pd.to_datetime(out["Date"], errors="coerce")
            out["Session Label"] = np.where(dt.notna(), dt.dt.strftime("%Y-%m-%d"), "Session")
        else:
            out["Session Label"] = "Session"

    if "Is Clean Shot" not in out.columns:
        out["Is Clean Shot"] = True
    if "Exclusion Reason" not in out.columns:
        out["Exclusion Reason"] = ""

    return out

def add_quality_flags(df):
    out = df.copy()
    needed = {"Carry", "Ball Speed", "Smash Factor"}
    if not needed.issubset(out.columns):
        out["Is Clean Shot"] = True
        out["Exclusion Reason"] = ""
        return out

    out["Exclusion Reason"] = ""
    missing_core = out["Carry"].isna() | out["Ball Speed"].isna() | out["Smash Factor"].isna()
    out.loc[missing_core, "Exclusion Reason"] = "Missing core data"

    hard_floor = (
        (out["Carry"] <= 0)
        | (out["Ball Speed"] < 40)
        | (out["Smash Factor"] < 0.80)
    )
    out.loc[hard_floor & (out["Exclusion Reason"] == ""), "Exclusion Reason"] = "Invalid / obvious bad read"

    for _, idx in out.groupby("Club").groups.items():
        g = out.loc[idx]
        if len(g) < 6:
            continue
        carry_cut = g["Carry"].quantile(0.10)
        speed_cut = g["Ball Speed"].quantile(0.10)
        smash_cut = g["Smash Factor"].quantile(0.10)
        low_tail = (
            out.index.isin(idx)
            & (out["Exclusion Reason"] == "")
            & (
                (out["Carry"] < carry_cut)
                | (out["Ball Speed"] < speed_cut)
                | (out["Smash Factor"] < smash_cut)
            )
        )
        out.loc[low_tail, "Exclusion Reason"] = "Low-end outlier / likely mishit"

    out["Is Clean Shot"] = out["Exclusion Reason"].eq("")
    return out


def pct(n, d):
    return 0 if d == 0 else 100.0 * n / d


def consistency_score(group):
    if len(group) < 3 or group["Carry"].dropna().empty:
        return np.nan
    med = group["Carry"].median()
    sd = group["Carry"].std(ddof=0)
    if pd.isna(sd) or med <= 0:
        return np.nan
    score = 10 - (sd / med) * 35
    return max(1, min(10, score))


def pattern_label(group):
    if group.empty or "Offline Num" not in group.columns or group["Offline Num"].dropna().empty:
        return "Need more direction data"

    off_mean = group["Offline Num"].mean()
    off_sd = group["Offline Num"].std(ddof=0)
    carry_sd = group["Carry"].std(ddof=0) if "Carry" in group.columns else np.nan

    labels = []
    if abs(off_mean) <= 3:
        labels.append("🎯 Neutral")
    elif off_mean > 3:
        labels.append("➡️ Right bias")
    else:
        labels.append("⬅️ Left bias")

    if pd.notna(off_sd):
        if off_sd <= 6:
            labels.append("🧵 Tight")
        elif off_sd >= 12:
            labels.append("🌪 Wide")

    if pd.notna(carry_sd):
        if carry_sd <= 4:
            labels.append("📏 Stable distance")
        elif carry_sd >= 9:
            labels.append("📉 Loose distance")

    return " • ".join(labels) if labels else "Pattern still developing"


@st.cache_data(show_spinner=False)
def prepare_data_from_bytes(file_bytes, file_name):
    buffer = io.BytesIO(file_bytes)
    return ensure_shot_schema(add_quality_flags(load_single_csv(buffer, source_name=file_name)))


@st.cache_data(show_spinner=False)
def prepare_data_from_path(path_str):
    path = Path(path_str)
    with open(path, "rb") as f:
        file_bytes = f.read()
    return prepare_data_from_bytes(file_bytes, path.name)


def make_download(df):
    return df.to_csv(index=False).encode("utf-8")


def find_auto_file():
    search_dirs = [Path.cwd(), Path(__file__).resolve().parent, Path("/mnt/data")]
    for directory in search_dirs:
        for name in AUTO_FILE_CANDIDATES:
            candidate = directory / name
            if candidate.exists():
                return candidate
    return None


def available_metrics(df):
    metrics = []
    for metric in METRIC_PRIORITY:
        if metric in df.columns and pd.to_numeric(df[metric], errors="coerce").notna().any():
            metrics.append(metric)
    return metrics


def metric_axis_label(metric):
    mapping = {
        "Offline Num": "Offline (Left negative / Right positive)",
        "Curve Num": "Curve",
        "Launch Direction Num": "Launch Direction",
        "Side Spin Num": "Side Spin",
        "Spin Axis Tilt Num": "Spin Axis Tilt",
    }
    return mapping.get(metric, metric)


def nice_format(metric):
    if metric in {"Smash Factor"}:
        return ".2f"
    return ".1f"


def delta_text(player_val, baseline_val, unit=""):
    if pd.isna(player_val) or pd.isna(baseline_val):
        return "—"
    diff = player_val - baseline_val
    sign = "+" if diff > 0 else ""
    suffix = f" {unit}" if unit else ""
    return f"{sign}{diff:.1f}{suffix}"


def baseline_table_for_set(name):
    rows = []
    for club, metrics in BENCHMARKS[name].items():
        row = {"Club Clean": club}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def metric_unit(metric):
    if metric in {"Carry", "Total", "Peak Height", "Offline Num", "Curve Num", "Launch Direction Num"}:
        return "yds"
    if metric in {"Ball Speed", "Club Speed"}:
        return "mph"
    if metric in {"Launch Angle", "Descent Angle", "Spin Axis Tilt Num"}:
        return "°"
    if metric in {"Back Spin", "Total Spin", "Side Spin Num"}:
        return "rpm"
    return ""


def player_baseline_summary(filtered, benchmark_name):
    club_player = (
        filtered.groupby(["Club", "Club Clean", "Club Sort"], dropna=False)
        .agg(
            Shots=("Club", "size"),
            Carry=("Carry", "median"),
            BallSpeed=("Ball Speed", "median"),
            ClubSpeed=("Club Speed", "median"),
            Smash=("Smash Factor", "median"),
        )
        .reset_index()
    )
    club_player = club_player.rename(columns={"BallSpeed": "Ball Speed", "ClubSpeed": "Club Speed", "Smash": "Smash Factor"})

    base = baseline_table_for_set(benchmark_name)
    merged = club_player.merge(base, on="Club Clean", how="left", suffixes=(" Player", " Baseline"))
    rename_map = {
        "Carry Player": "Carry Player", "Carry Baseline": "Carry Baseline",
        "Ball Speed Player": "Ball Speed Player", "Ball Speed Baseline": "Ball Speed Baseline",
        "Club Speed Player": "Club Speed Player", "Club Speed Baseline": "Club Speed Baseline",
        "Smash Factor Player": "Smash Factor Player", "Smash Factor Baseline": "Smash Factor Baseline",
    }
    # normalize columns after merge
    for raw in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        if raw in merged.columns and f"{raw} Baseline" not in merged.columns:
            merged[f"{raw} Baseline"] = np.nan
        if raw in merged.columns and f"{raw} Player" not in merged.columns:
            merged[f"{raw} Player"] = merged[raw]
    for raw in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        if raw in merged.columns:
            merged.drop(columns=[raw], inplace=True)
    merged = merged.sort_values(["Club Sort", "Club"]).copy()
    return merged


def metric_eval_label(delta, metric):
    if pd.isna(delta):
        return "No baseline"
    abs_delta = abs(delta)
    if metric == "Smash Factor":
        if abs_delta <= 0.02:
            return "Right there"
        if delta > 0:
            return "Above baseline"
        if delta >= -0.05:
            return "Close"
        return "Needs strike quality"
    if abs_delta <= 3:
        return "Right there"
    if delta > 0:
        return "Above baseline"
    if delta >= -8:
        return "Close"
    return "Gap to close"


def metric_card_html(club, stock, p25, p75, shots, consistency, pattern, baseline=None):
    baseline_row = ""
    if baseline is not None and not pd.isna(baseline):
        diff = stock - baseline
        baseline_row = f"<div class='club-sub'>vs baseline: <b>{diff:+.0f}</b></div>"
    consistency_txt = "—" if pd.isna(consistency) else f"{consistency:.1f}/10"
    width = "—" if pd.isna(p25) or pd.isna(p75) else f"{p25:.0f}–{p75:.0f}"
    stock_txt = "—" if pd.isna(stock) else f"{stock:.0f}"
    return f"""
    <div class='club-card'>
        <div class='club-top'>
            <div class='club-name'>{club}</div>
            <div class='club-shots'>{int(shots)} shots</div>
        </div>
        <div class='club-stock'>{stock_txt}<span class='club-unit'> yds</span></div>
        <div class='club-sub'>stock window: <b>{width}</b></div>
        {baseline_row}
        <div class='club-sub'>consistency: <b>{consistency_txt}</b></div>
        <div class='club-pattern'>{pattern}</div>
    </div>
    """



def _club_strength_reason(row):
    reasons = []
    cons = row.get("Consistency", np.nan)
    if pd.notna(cons):
        if cons >= 8:
            reasons.append("tight carry window")
        elif cons <= 5.5:
            reasons.append("loose carry spread")
    avg_off = row.get("AvgOffline", np.nan)
    if pd.notna(avg_off):
        if abs(avg_off) <= 3:
            reasons.append("starts near center")
        elif abs(avg_off) >= 8:
            reasons.append("direction bias shows up")
    smash = row.get("Smash", np.nan)
    baseline_smash = row.get("Baseline Smash", np.nan)
    if pd.notna(smash) and pd.notna(baseline_smash):
        diff = smash - baseline_smash
        if diff >= 0.01:
            reasons.append("solid strike efficiency")
        elif diff <= -0.03:
            reasons.append("strike efficiency trails benchmark")
    elif pd.notna(smash):
        if smash >= 1.35:
            reasons.append("solid strike efficiency")
        elif smash <= 1.25:
            reasons.append("strike efficiency is inconsistent")
    carry_delta = row.get("Carry Delta vs Benchmark", np.nan)
    if pd.notna(carry_delta):
        if carry_delta >= 3:
            reasons.append("carry is above baseline")
        elif carry_delta <= -8:
            reasons.append("carry trails baseline")
    if not reasons:
        reasons.append("small sample / still forming")
    return ", ".join(reasons[:3])


def build_best_worst_lists(club_summary):
    if club_summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = club_summary.copy()
    work["CarryStdFill"] = work["CarryStd"].fillna(work["CarryStd"].median())
    work["ConsistencyFill"] = work["Consistency"].fillna(work["Consistency"].median())
    work["AbsOffline"] = work["AvgOffline"].abs().fillna(work["AvgOffline"].abs().median())
    work["SmashDelta"] = (work["Smash"] - work["Baseline Smash"]).fillna(0)
    # Higher is better
    work["Club Score"] = (
        work["ConsistencyFill"] * 1.7
        - work["CarryStdFill"] * 0.22
        - work["AbsOffline"] * 0.16
        + work["SmashDelta"] * 40
        + work["Carry Delta vs Benchmark"].fillna(0) * 0.04
    )
    work["Why"] = work.apply(_club_strength_reason, axis=1)
    best = work.sort_values(["Club Score", "Shots"], ascending=[False, False]).head(3).copy()
    worst = work.sort_values(["Club Score", "Shots"], ascending=[True, False]).head(3).copy()
    return best, worst


def benchmark_reason(row, benchmark_name):
    bits = []
    for metric in ["Carry", "Ball Speed", "Club Speed", "Smash Factor"]:
        pcol = f"{metric} Player"
        bcol = f"{metric} Baseline"
        if pcol not in row or bcol not in row:
            continue
        pv, bv = row[pcol], row[bcol]
        if pd.isna(pv) or pd.isna(bv):
            continue
        delta = pv - bv
        if metric == "Smash Factor":
            if delta >= 0.01:
                bits.append("strike efficiency is at or above the benchmark")
            elif delta >= -0.02:
                bits.append("strike efficiency is basically even")
            else:
                bits.append("strike efficiency is behind the benchmark")
        elif metric == "Carry":
            if abs(delta) <= 3:
                bits.append("carry is basically even")
            elif delta > 0:
                bits.append("carry is beating the benchmark")
            else:
                bits.append("carry is trailing the benchmark")
        elif metric == "Ball Speed":
            if abs(delta) <= 2:
                bits.append("ball speed is right there")
            elif delta > 0:
                bits.append("ball speed is strong")
            else:
                bits.append("ball speed is the main gap")
        elif metric == "Club Speed":
            if abs(delta) <= 2:
                bits.append("speed is in range")
            elif delta > 0:
                bits.append("speed is plenty")
            else:
                bits.append("speed is a little light")
    if not bits:
        return f"Not enough data to compare to {benchmark_name.lower()}."
    return "; ".join(bits[:3]).capitalize() + "."

st.markdown(
    """
    <style>
    .club-card {
        background: linear-gradient(180deg, #10131a 0%, #171b23 100%);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 18px;
        padding: 14px 16px;
        min-height: 158px;
        box-shadow: 0 8px 22px rgba(0,0,0,.16);
        margin-bottom: 10px;
    }
    .club-top {
        display:flex;
        justify-content:space-between;
        align-items:center;
        margin-bottom: 10px;
    }
    .club-name {
        font-size: 1.1rem;
        font-weight: 700;
    }
    .club-shots {
        font-size: 0.82rem;
        opacity: 0.75;
    }
    .club-stock {
        font-size: 2rem;
        font-weight: 800;
        line-height: 1.0;
        margin-bottom: 8px;
    }
    .club-unit {
        font-size: .95rem;
        font-weight: 600;
        opacity: .8;
    }
    .club-sub {
        font-size: .9rem;
        margin-bottom: 4px;
        opacity: .92;
    }
    .club-pattern {
        margin-top: 10px;
        font-size: .88rem;
        opacity: .92;
    }
    .note-box {
        padding: 10px 12px;
        border-radius: 12px;
        background: rgba(255,255,255,.04);
        border: 1px solid rgba(255,255,255,.08);
        margin-bottom: 10px;
    }
    .hero-card {
        background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 18px;
        padding: 14px 16px;
        min-height: 146px;
        box-shadow: 0 10px 24px rgba(0,0,0,.14);
        margin-bottom: 12px;
    }
    .hero-title {font-size: .85rem; opacity: .72; margin-bottom: 8px;}
    .hero-value {font-size: 1.9rem; font-weight: 800; line-height:1.05; margin-bottom:6px;}
    .hero-sub {font-size: .92rem; opacity:.88;}
    .rank-card {
        background: rgba(255,255,255,.035);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 16px;
        padding: 12px 14px;
        margin-bottom: 10px;
    }
    .rank-label {font-size: .8rem; opacity: .72; text-transform: uppercase; letter-spacing: .04em;}
    .rank-club {font-size: 1.12rem; font-weight: 700; margin: 4px 0 6px 0;}
    .rank-why {font-size: .9rem; opacity: .92; line-height: 1.35;}
    .bench-chip {
        display:inline-block; padding: 4px 8px; border-radius: 999px;
        border: 1px solid rgba(255,255,255,.08); background: rgba(255,255,255,.05);
        margin-right: 6px; margin-bottom: 6px; font-size: .82rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("⛳ Launch Monitor Analyzer")
st.caption("Auto-loads Launch.csv, keeps the dispersion views, upgrades the carry section, and adds a benchmark comparison tab.")

with st.sidebar:
    st.header("Data Source")
    auto_file = find_auto_file()
    source_mode = st.radio(
        "How should the app load data?",
        ["Auto-load Launch.csv", "Upload a different CSV"],
        index=0,
    )

    shots = None
    source_label = None

    if source_mode == "Auto-load Launch.csv":
        if auto_file is None:
            st.warning("No Launch.csv file was found next to the app or in the working folder.")
        else:
            shots = prepare_data_from_path(str(auto_file))
            source_label = str(auto_file)
            st.success(f"Loaded: {auto_file.name}")
    else:
        uploaded = st.file_uploader("Upload one master CSV file", type=["csv"])
        if uploaded is not None:
            shots = prepare_data_from_bytes(uploaded.getvalue(), uploaded.name)
            source_label = uploaded.name

if shots is None:
    st.info("Put Launch.csv in the same folder as this app, or choose a file from the sidebar.")
    st.stop()

shots = ensure_shot_schema(shots)

if shots.empty:
    st.error("Could not parse any shot rows from that file.")
    st.stop()

st.caption(f"Current source: {source_label}")

metrics = available_metrics(shots)
if not metrics:
    st.error("No usable numeric metrics were found in this file.")
    st.stop()

st.subheader("Filters")
f1, f2, f3, f4, f5, f6 = st.columns(6)
players = ["All Players"] + sorted(shots["Player"].dropna().astype(str).unique().tolist())
sessions = ["All Sessions"] + sorted(shots["Session Label"].dropna().astype(str).unique().tolist())
clubs = ["All Clubs"] + sorted(
    shots["Club"].dropna().astype(str).unique().tolist(),
    key=lambda x: (shots.loc[shots["Club"] == x, "Club Sort"].min(), str(x)),
)

with f1:
    player_sel = st.selectbox("Player", players)
with f2:
    session_sel = st.selectbox("Session / Date", sessions)
with f3:
    club_sel = st.selectbox("Club", clubs)
with f4:
    quality_sel = st.selectbox("Shot Set", ["Clean Shots Only", "All Shots"])
with f5:
    metric_sel = st.selectbox("Analysis Metric", metrics, index=0)
with f6:
    benchmark_sel = st.selectbox("Benchmark", list(BENCHMARKS.keys()), index=1)

filtered = shots.copy()
if player_sel != "All Players":
    filtered = filtered[filtered["Player"] == player_sel]
if session_sel != "All Sessions":
    filtered = filtered[filtered["Session Label"] == session_sel]
if club_sel != "All Clubs":
    filtered = filtered[filtered["Club"] == club_sel]
if quality_sel == "Clean Shots Only":
    filtered = filtered[filtered["Is Clean Shot"]]

if filtered.empty:
    st.warning("No shots match the current filters.")
    st.stop()

all_count = len(shots)
clean_count = int(shots["Is Clean Shot"].sum())
filtered_count = len(filtered)
metric_series = pd.to_numeric(filtered[metric_sel], errors="coerce") if metric_sel in filtered.columns else pd.Series(dtype=float)
metric_fmt = nice_format(metric_sel)
metric_median = metric_series.median() if not metric_series.dropna().empty else np.nan

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Shots in file", f"{all_count}")
m2.metric("Clean shots", f"{clean_count}")
m3.metric("Shots in current view", f"{filtered_count}")
m4.metric("Exclusion rate", f"{pct(all_count-clean_count, all_count):.1f}%")
m5.metric(f"Median {metric_sel}", "—" if pd.isna(metric_median) else format(metric_median, metric_fmt))

club_summary = (
    filtered.groupby(["Club", "Club Clean", "Club Sort"], dropna=False)
    .agg(
        Shots=("Club", "size"),
        CarryP10=("Carry", lambda s: s.quantile(0.10)),
        CarryP25=("Carry", lambda s: s.quantile(0.25)),
        MedianCarry=("Carry", "median"),
        CarryP75=("Carry", lambda s: s.quantile(0.75)),
        CarryP90=("Carry", lambda s: s.quantile(0.90)),
        AvgCarry=("Carry", "mean"),
        CarryStd=("Carry", "std"),
        AvgOffline=("Offline Num", "mean"),
        BallSpeed=("Ball Speed", "mean"),
        Smash=("Smash Factor", "mean"),
        MetricMean=(metric_sel, "mean"),
        MetricMedian=(metric_sel, "median"),
        MetricStd=(metric_sel, "std"),
        MetricP25=(metric_sel, lambda s: s.quantile(0.25)),
        MetricP75=(metric_sel, lambda s: s.quantile(0.75)),
    )
    .reset_index()
    .sort_values(["Club Sort", "Club"])
)

if club_summary.empty:
    st.warning("No summary rows available for the current filters.")
    st.stop()

cons_map = filtered.groupby("Club").apply(consistency_score).to_dict()
pat_map = filtered.groupby("Club").apply(pattern_label).to_dict()
club_summary["Consistency"] = club_summary["Club"].map(cons_map)
club_summary["Pattern"] = club_summary["Club"].map(pat_map)

base_df = baseline_table_for_set(benchmark_sel)
base_carry_map = base_df.set_index("Club Clean")["Carry"].to_dict()
club_summary["Baseline Carry"] = club_summary["Club Clean"].map(base_carry_map)
base_smash_map = base_df.set_index("Club Clean")["Smash Factor"].to_dict() if "Smash Factor" in base_df.columns else {}
club_summary["Baseline Smash"] = club_summary["Club Clean"].map(base_smash_map)
club_summary["Carry Delta vs Benchmark"] = club_summary["MedianCarry"] - club_summary["Baseline Carry"]

summary_show = club_summary.copy()
for c in [
    "CarryP10", "CarryP25", "MedianCarry", "CarryP75", "CarryP90", "AvgCarry", "CarryStd",
    "AvgOffline", "BallSpeed", "Smash", "Consistency", "MetricMean", "MetricMedian", "MetricStd",
    "MetricP25", "MetricP75", "Baseline Carry", "Baseline Smash", "Carry Delta vs Benchmark",
]:
    if c in summary_show.columns:
        summary_show[c] = summary_show[c].round(2 if c in {"Smash", "Baseline Smash"} else 1)

best_three, worst_three = build_best_worst_lists(club_summary)

overview_tab, benchmark_tab, data_tab = st.tabs(["Overview", "Benchmark Analysis", "Shot Tables"])

with overview_tab:
    st.subheader("At-a-Glance Read")
    h1, h2, h3 = st.columns(3)
    best_name = best_three.iloc[0]["Club"] if not best_three.empty else "—"
    worst_name = worst_three.iloc[0]["Club"] if not worst_three.empty else "—"
    stock_gap = club_summary["MedianCarry"].max() - club_summary["MedianCarry"].min() if len(club_summary) > 1 else np.nan
    with h1:
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Best current gamer</div><div class='hero-value'>{best_name}</div><div class='hero-sub'>{best_three.iloc[0]['Why'] if not best_three.empty else 'Need more data.'}</div></div>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Needs the most love</div><div class='hero-value'>{worst_name}</div><div class='hero-sub'>{worst_three.iloc[0]['Why'] if not worst_three.empty else 'Need more data.'}</div></div>",
            unsafe_allow_html=True,
        )
    with h3:
        sub = "Need at least two clubs." if pd.isna(stock_gap) else f"{stock_gap:.0f} yds from shortest median to longest median in this filtered view."
        st.markdown(
            f"<div class='hero-card'><div class='hero-title'>Current gapping window</div><div class='hero-value'>{'—' if pd.isna(stock_gap) else f'{stock_gap:.0f} yds'}</div><div class='hero-sub'>{sub}</div></div>",
            unsafe_allow_html=True,
        )

    st.subheader("Stock Carry Cards")
    st.markdown(
        "<div class='note-box'>This version makes stock carry the star and pushes the range into a cleaner horizontal view. Median carry is the main number. The range bar is your middle 50% window. Average carry is there for context only.</div>",
        unsafe_allow_html=True,
    )

    rows = [club_summary.iloc[i:i+4] for i in range(0, len(club_summary), 4)]
    for chunk in rows:
        cols = st.columns(len(chunk))
        for col, (_, r) in zip(cols, chunk.iterrows()):
            with col:
                st.markdown(
                    metric_card_html(
                        r["Club"],
                        r["MedianCarry"],
                        r["CarryP25"],
                        r["CarryP75"],
                        r["Shots"],
                        r["Consistency"],
                        r["Pattern"],
                        r.get("Baseline Carry", np.nan),
                    ),
                    unsafe_allow_html=True,
                )

    st.subheader("Carry Window by Club")
    carry_plot_df = club_summary.sort_values(["Club Sort", "Club"], ascending=[False, True]).copy()
    carry_base = alt.Chart(carry_plot_df)
    range_bar = carry_base.mark_bar(size=16, cornerRadius=8, opacity=0.42).encode(
        y=alt.Y("Club:N", sort=None, title="Club"),
        x=alt.X("CarryP25:Q", title="Carry Distance (yds)"),
        x2="CarryP75:Q",
        tooltip=[
            "Club",
            alt.Tooltip("MedianCarry:Q", title="Stock Carry", format=".0f"),
            alt.Tooltip("CarryP25:Q", title="25th %", format=".0f"),
            alt.Tooltip("CarryP75:Q", title="75th %", format=".0f"),
            alt.Tooltip("AvgCarry:Q", title="Average", format=".1f"),
            alt.Tooltip("Carry Delta vs Benchmark:Q", title="vs Benchmark", format="+.0f"),
        ],
    )
    median_dot = carry_base.mark_circle(size=220, filled=True).encode(
        y=alt.Y("Club:N", sort=None),
        x=alt.X("MedianCarry:Q"),
        color=alt.Color("Club:N", legend=None),
        tooltip=["Club", alt.Tooltip("MedianCarry:Q", title="Stock Carry", format=".0f")],
    )
    avg_tick = carry_base.mark_tick(size=24, thickness=2, color="white").encode(
        y=alt.Y("Club:N", sort=None),
        x="AvgCarry:Q",
        tooltip=["Club", alt.Tooltip("AvgCarry:Q", title="Average Carry", format=".1f")],
    )
    carry_label = carry_base.mark_text(dx=22, fontSize=12, align="left", fontWeight="bold").encode(
        y=alt.Y("Club:N", sort=None),
        x="MedianCarry:Q",
        text=alt.Text("MedianCarry:Q", format=".0f"),
    )
    st.altair_chart((range_bar + median_dot + avg_tick + carry_label).properties(height=max(260, 42 * len(carry_plot_df))), use_container_width=True)
    st.caption("Dot = stock carry. Thick bar = middle 50% carry window. White tick = average carry. This should read closer to a real club-gapping board.")

    st.subheader("What you hit best right now")
    if best_three.empty:
        st.info("Need more club data to rank your best clubs.")
    else:
        cols = st.columns(len(best_three))
        medals = ["🥇", "🥈", "🥉"]
        for col, (_, r), medal in zip(cols, best_three.iterrows(), medals):
            with col:
                st.markdown(
                    f"<div class='rank-card'><div class='rank-label'>{medal} Best club</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>Stock carry: {r['MedianCarry']:.0f} yds</div><div class='bench-chip'>Consistency: {r['Consistency']:.1f}/10</div><div class='bench-chip'>Smash: {r['Smash']:.2f}</div></div>",
                    unsafe_allow_html=True,
                )

    st.subheader("What needs work")
    if worst_three.empty:
        st.info("Need more club data to rank the weaker clubs.")
    else:
        cols = st.columns(len(worst_three))
        badges = ["🛠️ 1", "🛠️ 2", "🛠️ 3"]
        for col, (_, r), badge in zip(cols, worst_three.iterrows(), badges):
            with col:
                st.markdown(
                    f"<div class='rank-card'><div class='rank-label'>{badge} Focus club</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>Carry SD: {0 if pd.isna(r['CarryStd']) else r['CarryStd']:.1f}</div><div class='bench-chip'>Offline avg: {0 if pd.isna(r['AvgOffline']) else r['AvgOffline']:+.1f}</div><div class='bench-chip'>vs benchmark: {0 if pd.isna(r['Carry Delta vs Benchmark']) else r['Carry Delta vs Benchmark']:+.0f}</div></div>",
                    unsafe_allow_html=True,
                )

    st.subheader(f"{metric_sel} Analysis")
    metric_df = filtered.dropna(subset=[metric_sel]).copy() if metric_sel in filtered.columns else pd.DataFrame()
    if metric_df.empty:
        st.info(f"No usable {metric_sel} values are available for the current selection.")
    else:
        left_metric, right_metric = st.columns(2)

        metric_by_club = alt.Chart(club_summary).mark_bar(size=24, cornerRadiusTopLeft=6, cornerRadiusTopRight=6, opacity=0.65).encode(
            x=alt.X("Club:N", sort=None, title="Club"),
            y=alt.Y("MetricMedian:Q", title=metric_axis_label(metric_sel)),
            color=alt.Color("Club:N", legend=None),
            tooltip=[
                "Club",
                alt.Tooltip("MetricMedian:Q", title="Median", format=metric_fmt),
                alt.Tooltip("MetricMean:Q", title="Mean", format=metric_fmt),
                alt.Tooltip("MetricP25:Q", title="25th %", format=metric_fmt),
                alt.Tooltip("MetricP75:Q", title="75th %", format=metric_fmt),
            ],
        )
        metric_text = alt.Chart(club_summary).mark_text(dy=-10, fontSize=12, fontWeight="bold").encode(
            x=alt.X("Club:N", sort=None),
            y=alt.Y("MetricMedian:Q"),
            text=alt.Text("MetricMedian:Q", format=metric_fmt),
        )

        with left_metric:
            st.markdown("**Metric by Club**")
            st.altair_chart((metric_by_club + metric_text).properties(height=340), use_container_width=True)

        compare_field = "Session Label" if session_sel == "All Sessions" else "Club"
        with right_metric:
            st.markdown(f"**{metric_sel} Distribution by {compare_field}**")
            metric_box = alt.Chart(metric_df).mark_boxplot(size=40).encode(
                x=alt.X(f"{compare_field}:N", title=compare_field, sort=None),
                y=alt.Y(f"{metric_sel}:Q", title=metric_axis_label(metric_sel)),
                color=alt.Color(f"{compare_field}:N", legend=None),
                tooltip=[compare_field],
            ).properties(height=340)
            st.altair_chart(metric_box, use_container_width=True)

    st.subheader("Dispersion")
    left_col, right_col = st.columns(2)

    has_offline = "Offline Num" in filtered.columns and filtered["Offline Num"].notna().any()
    has_carry = "Carry" in filtered.columns and filtered["Carry"].notna().any()

    with left_col:
        st.markdown("**Left / Right Pattern**")
        if has_offline:
            offline_df = filtered.dropna(subset=["Offline Num"]).copy()
            scatter_lr = alt.Chart(offline_df).mark_circle(size=82, opacity=0.68).encode(
                x=alt.X("Offline Num:Q", title="Offline (Left negative / Right positive)"),
                y=alt.Y("jitter:Q", axis=None),
                color=alt.Color("Club:N", legend=None),
                tooltip=[
                    "Player", "Club", "Session Label",
                    alt.Tooltip("Offline Num:Q", format=".1f"),
                    alt.Tooltip("Carry:Q", format=".1f"),
                    alt.Tooltip(metric_sel + ":Q", title=metric_sel, format=metric_fmt),
                ],
            ).transform_calculate(jitter="sqrt(-2*log(random()))*cos(2*PI*random())")
            zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], size=2).encode(x="x:Q")
            st.altair_chart((scatter_lr + zero_rule).properties(height=320), use_container_width=True)
        else:
            st.info("No usable offline column was found for this selection.")

    with right_col:
        st.markdown("**Carry + Direction Map**")
        if has_offline and has_carry:
            disp_df = filtered.dropna(subset=["Offline Num", "Carry"]).copy()
            color_field = "Club" if club_sel == "All Clubs" else "Player"
            scatter = alt.Chart(disp_df).mark_circle(size=95, opacity=0.72).encode(
                x=alt.X("Offline Num:Q", title="Offline"),
                y=alt.Y("Carry:Q", title="Carry"),
                color=alt.Color(f"{color_field}:N", legend=alt.Legend(title=color_field)),
                tooltip=[
                    "Player", "Club", "Session Label",
                    alt.Tooltip("Carry:Q", format=".1f"),
                    alt.Tooltip("Offline Num:Q", format=".1f"),
                    alt.Tooltip(metric_sel + ":Q", title=metric_sel, format=metric_fmt),
                ],
            )
            vline = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[5, 5], size=2).encode(x="x:Q")
            st.altair_chart((scatter + vline).properties(height=320), use_container_width=True)
        else:
            st.info("Need both carry and offline data for the combo dispersion view.")

    st.subheader("Distance Consistency")
    box_df = filtered.dropna(subset=["Carry"]).copy()
    if not box_df.empty:
        box = alt.Chart(box_df).mark_boxplot(size=38).encode(
            x=alt.X("Club:N", sort=None, title="Club"),
            y=alt.Y("Carry:Q", title="Carry Distance"),
            color=alt.Color("Club:N", legend=None),
            tooltip=["Club"],
        ).properties(height=340)
        st.altair_chart(box, use_container_width=True)
    else:
        st.info("No carry data available for the current selection.")

    st.subheader("Club Summary Table")
    st.dataframe(
        summary_show[[
            "Club", "Shots", "CarryP25", "MedianCarry", "CarryP75", "Baseline Carry",
            "Carry Delta vs Benchmark", "MetricMedian", "MetricMean", "MetricStd",
            "BallSpeed", "Smash", "Consistency", "Pattern"
        ]],
        use_container_width=True,
        hide_index=True,
    )

with benchmark_tab:
    st.subheader("Benchmark View")
    st.markdown(
        f"<div class='note-box'>This tab compares your filtered shot set against the <b>{benchmark_sel}</b> reference profile. Treat these as directional baselines you can tweak over time, not commandments carved in stone.</div>",
        unsafe_allow_html=True,
    )

    bench_summary = player_baseline_summary(filtered, benchmark_sel)
    if bench_summary.empty:
        st.info("No benchmark comparison is available for the current filters.")
    else:
        focus_metric = st.selectbox(
            "Benchmark metric",
            ["Carry", "Ball Speed", "Club Speed", "Smash Factor"],
            index=0,
            key="benchmark_metric",
        )
        player_col = f"{focus_metric} Player"
        base_col = f"{focus_metric} Baseline"
        if player_col not in bench_summary.columns:
            bench_summary[player_col] = np.nan
        if base_col not in bench_summary.columns:
            bench_summary[base_col] = np.nan
        bench_summary["Delta"] = bench_summary[player_col] - bench_summary[base_col]
        bench_summary["Read"] = bench_summary["Delta"].apply(lambda x: metric_eval_label(x, focus_metric))
        bench_summary["Why"] = bench_summary.apply(lambda r: benchmark_reason(r, benchmark_sel), axis=1)

        b1, b2, b3 = st.columns(3)
        overall_player = bench_summary[player_col].mean()
        overall_base = bench_summary[base_col].mean()
        delta = overall_player - overall_base if pd.notna(overall_player) and pd.notna(overall_base) else np.nan
        b1.metric(f"Avg {focus_metric}", "—" if pd.isna(overall_player) else f"{overall_player:.1f}")
        b2.metric(f"{benchmark_sel}", "—" if pd.isna(overall_base) else f"{overall_base:.1f}")
        b3.metric("Delta", "—" if pd.isna(delta) else delta_text(overall_player, overall_base, metric_unit(focus_metric)))

        same_or_better = bench_summary[bench_summary["Read"].isin(["Right there", "Above baseline"])]
        needs_work = bench_summary[bench_summary["Read"].isin(["Close", "Gap to close", "Needs strike quality"])]
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Where you are even or better**")
            if same_or_better.empty:
                st.info("Nothing is clearly at or above the selected benchmark on this filter yet.")
            else:
                for _, r in same_or_better.head(5).iterrows():
                    st.markdown(
                        f"<div class='rank-card'><div class='rank-label'>{r['Read']}</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>{focus_metric}: {r[player_col]:.1f} vs {r[base_col]:.1f}</div><div class='bench-chip'>Delta: {r['Delta']:+.1f}</div></div>",
                        unsafe_allow_html=True,
                    )
        with c2:
            st.markdown("**Where the gap still lives**")
            if needs_work.empty:
                st.info("Nothing major to flag here with the current benchmark and filter.")
            else:
                for _, r in needs_work.head(5).iterrows():
                    st.markdown(
                        f"<div class='rank-card'><div class='rank-label'>{r['Read']}</div><div class='rank-club'>{r['Club']}</div><div class='rank-why'>{r['Why']}</div><div class='bench-chip'>{focus_metric}: {r[player_col]:.1f} vs {r[base_col]:.1f}</div><div class='bench-chip'>Delta: {r['Delta']:+.1f}</div></div>",
                        unsafe_allow_html=True,
                    )

        compare_plot_df = bench_summary.dropna(subset=[player_col, base_col]).copy()
        if not compare_plot_df.empty:
            melted = pd.concat([
                compare_plot_df[["Club", player_col]].rename(columns={player_col: "Value"}).assign(Group="You"),
                compare_plot_df[["Club", base_col]].rename(columns={base_col: "Value"}).assign(Group=benchmark_sel),
            ], ignore_index=True)

            rules = alt.Chart(compare_plot_df).mark_rule(strokeWidth=4, opacity=0.30).encode(
                x=alt.X("Club:N", sort=None, title="Club"),
                y=alt.Y(base_col + ":Q", title=f"{focus_metric} ({metric_unit(focus_metric)})" if metric_unit(focus_metric) else focus_metric),
                y2=player_col + ":Q",
                tooltip=[
                    "Club",
                    alt.Tooltip(player_col + ":Q", title="You", format=nice_format(focus_metric)),
                    alt.Tooltip(base_col + ":Q", title=benchmark_sel, format=nice_format(focus_metric)),
                ],
            )
            points = alt.Chart(melted).mark_circle(size=155, opacity=0.88).encode(
                x=alt.X("Club:N", sort=None),
                y=alt.Y("Value:Q"),
                color=alt.Color("Group:N", title="Series"),
                tooltip=["Club", "Group", alt.Tooltip("Value:Q", format=nice_format(focus_metric))],
            )
            labels = alt.Chart(compare_plot_df).mark_text(dy=-12, fontSize=11, fontWeight="bold").encode(
                x=alt.X("Club:N", sort=None),
                y=alt.Y(player_col + ":Q"),
                text=alt.Text("Delta:Q", format="+.1f"),
            )
            st.altair_chart((rules + points + labels).properties(height=390), use_container_width=True)
            st.caption("Connector line shows the gap between your median and the chosen baseline. Positive label = you are above it. Negative = the remaining gap.")

        show_cols = [
            "Club", "Shots", player_col, base_col, "Delta", "Read", "Why",
            "Carry Player", "Carry Baseline", "Ball Speed Player", "Ball Speed Baseline",
            "Club Speed Player", "Club Speed Baseline", "Smash Factor Player", "Smash Factor Baseline",
        ]
        show_cols = [c for c in show_cols if c in bench_summary.columns]
        bench_show = bench_summary.loc[:, show_cols].copy()
        bench_show = bench_show.loc[:, ~bench_show.columns.duplicated()].copy()
        for c in list(bench_show.columns):
            if c in {"Club", "Read", "Why"}:
                continue
            col_obj = bench_show[c]
            if isinstance(col_obj, pd.DataFrame):
                col_obj = col_obj.iloc[:, 0]
            bench_show[c] = pd.to_numeric(col_obj, errors="coerce")
            if pd.api.types.is_numeric_dtype(bench_show[c]):
                bench_show[c] = bench_show[c].round(2 if "Smash" in c else 1)
        st.dataframe(bench_show, use_container_width=True, hide_index=True)

with data_tab:
    st.subheader("Shot Tables")
    t1, t2 = st.tabs(["Clean / Current View", "Excluded Shots"])
    with t1:
        st.dataframe(filtered, use_container_width=True, hide_index=True)
        st.download_button(
            "Download current filtered shots",
            data=make_download(filtered),
            file_name="launch_filtered_shots.csv",
            mime="text/csv",
        )
    with t2:
        excluded = shots[~shots["Is Clean Shot"]].copy()
        st.dataframe(excluded, use_container_width=True, hide_index=True)
        st.download_button(
            "Download excluded shots",
            data=make_download(excluded),
            file_name="launch_excluded_shots.csv",
            mime="text/csv",
        )
