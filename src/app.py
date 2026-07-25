from pathlib import Path

import streamlit as st

from src.dashboard import build_accuracy_table, build_coverage_heatmap, get_latest_run, get_runs

DB_PATH = Path("data/metrics.db")

st.set_page_config(page_title="PerceptionGuard", layout="wide")
st.title("PerceptionGuard Dashboard")

if not DB_PATH.exists():
    st.warning("No metrics database found. Run the evaluation pipeline first.")
    st.stop()

runs_df = get_runs(DB_PATH)

if runs_df.empty:
    st.info("No runs recorded yet.")
    st.stop()

st.subheader("Latest Run Verdict")
latest = get_latest_run(DB_PATH)
if not latest.empty:
    overall = latest["verdict"].iloc[0] if "verdict" in latest.columns else "unknown"
    if overall == "pass":
        st.success(f"Overall: {overall.upper()}")
    elif overall == "warning":
        st.warning(f"Overall: {overall.upper()}")
    else:
        st.error(f"Overall: {overall.upper()}")

    st.dataframe(
        latest[["weather", "scene", "time_of_day", "mean_ap", "verdict"]],
        use_container_width=True,
    )

st.subheader("Accuracy by Scenario Over Time")
accuracy_table = build_accuracy_table(runs_df)
if not accuracy_table.empty:
    st.line_chart(accuracy_table)
else:
    st.info("Not enough data for trend chart.")

st.subheader("Coverage Heatmap (Latest Run)")
heatmap = build_coverage_heatmap(runs_df)
if not heatmap.empty:
    st.dataframe(heatmap.style.background_gradient(cmap="RdYlGn", vmin=0, vmax=1))
else:
    st.info("Not enough data for heatmap.")
