import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.express as px
import streamlit as st

from dashboard import (
    build_coverage_heatmap,
    get_bucket_summary,
    get_coverage_data,
    get_latest_run,
    get_runs,
    get_stats,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = DATA_DIR / "metrics.db"

st.set_page_config(page_title="PerceptionGuard", layout="wide", page_icon="🛡️")

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 16px; border: 1px solid #0f3460; }
    .stMetric label { color: #a0aec0 !important; font-size: 0.85rem !important; }
    .stMetric [data-testid="stMetricValue"] {
        color: #e2e8f0 !important; font-size: 1.8rem !important;
    }
    .stMetric [data-testid="stMetricDelta"] { font-size: 0.85rem !important; }
    .verdict-pass { background: linear-gradient(90deg, #065f46, #047857); padding: 1rem 1.5rem;
                    border-radius: 12px; color: white; text-align: center; font-size: 1.4rem;
                    font-weight: 700; letter-spacing: 0.05em; margin-bottom: 1rem;
                    border: 1px solid #10b981; }
    .verdict-warning { background: linear-gradient(90deg, #92400e, #b45309); padding: 1rem 1.5rem;
                       border-radius: 12px; color: white; text-align: center; font-size: 1.4rem;
                       font-weight: 700; letter-spacing: 0.05em; margin-bottom: 1rem;
                       border: 1px solid #f59e0b; }
    .verdict-critical { background: linear-gradient(90deg, #991b1b, #dc2626); padding: 1rem 1.5rem;
                        border-radius: 12px; color: white; text-align: center; font-size: 1.4rem;
                        font-weight: 700; letter-spacing: 0.05em; margin-bottom: 1rem;
                        border: 1px solid #ef4444; }
    .section-header { color: #e2e8f0; font-size: 1.1rem; font-weight: 600;
                      margin: 0.5rem 0 0.25rem 0; padding-bottom: 0.25rem;
                      border-bottom: 2px solid #0f3460; }
    div[data-testid="stSidebar"] { background: linear-gradient(180deg, #0a0e1a 0%, #111827 100%); }
    div[data-testid="stSidebar"] .stMarkdown h1 { color: #60a5fa !important; }
    div[data-testid="stSidebar"] .stMarkdown h3 { color: #94a3b8 !important; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


if not DB_PATH.exists():
    st.markdown("## 🛡️ PerceptionGuard")
    st.warning("No metrics database found. Run the evaluation pipeline first.")
    st.stop()

runs_df = get_runs(DB_PATH)
coverage_df = get_coverage_data(DATA_DIR)

if runs_df.empty:
    st.markdown("## 🛡️ PerceptionGuard")
    st.info("No runs recorded yet. Run the evaluation pipeline first.")
    st.stop()

latest_run = get_latest_run(DB_PATH)
stats = get_stats(runs_df, coverage_df)

with st.sidebar:
    st.markdown("# 🛡️ PerceptionGuard")
    st.markdown("### Navigation")
    tab = st.radio("Go to", ["Overview", "Coverage", "Details"], label_visibility="collapsed")
    st.markdown("---")
    st.markdown("### Filters")
    weather_opts = sorted(runs_df["weather"].unique())
    scene_opts = sorted(runs_df["scene"].unique())
    time_opts = sorted(runs_df["time_of_day"].unique())
    sel_weather = st.multiselect("Weather", weather_opts, default=weather_opts)
    sel_scene = st.multiselect("Scene", scene_opts, default=scene_opts)
    sel_time = st.multiselect("Time of Day", time_opts, default=time_opts)

if tab == "Overview":
    st.markdown("## 🛡️ PerceptionGuard Dashboard")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Images", f"{stats['total_images']:,}")
    c2.metric("Scenario Buckets", stats["total_buckets"])
    c3.metric("Avg mAP", f"{stats['avg_map']:.3f}")
    c4.metric("Coverage Gaps", stats["coverage_gaps"])
    verdict_color = {"pass": "🟢", "warning": "🟡", "critical": "🔴"}.get(stats["verdict"], "⚪")
    c5.metric("Overall Verdict", f"{verdict_color} {stats['verdict'].upper()}")

    verdict = stats["verdict"]
    css_class = f"verdict-{verdict}"
    st.markdown(
        f'<div class="{css_class}">Overall Verdict: {verdict.upper()}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-header">Coverage Heatmap — mAP by Weather × Scene</p>',
        unsafe_allow_html=True,
    )
    heatmap = build_coverage_heatmap(runs_df)
    if not heatmap.empty:
        fig_heat = px.imshow(
            heatmap.values,
            x=heatmap.columns.tolist(),
            y=heatmap.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto",
            text_auto=".2f",
            labels={"color": "mAP"},
        )
        fig_heat.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            xaxis_title="Scene",
            yaxis_title="Weather",
            height=350,
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="mAP", tickformat=".1f"),
        )
        fig_heat.update_xaxes(side="bottom")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.info("Not enough data for heatmap.")

    st.markdown('<p class="section-header">mAP Distribution by Weather</p>', unsafe_allow_html=True)
    filtered = latest_run[
        latest_run["weather"].isin(sel_weather)
        & latest_run["scene"].isin(sel_scene)
        & latest_run["time_of_day"].isin(sel_time)
    ]
    if not filtered.empty:
        fig_box = px.box(
            filtered,
            x="weather",
            y="mean_ap",
            color="weather",
            color_discrete_sequence=px.colors.qualitative.Set2,
            points="all",
        )
        fig_box.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            showlegend=False,
            height=350,
            xaxis_title="",
            yaxis_title="mAP",
            yaxis=dict(range=[0, 1]),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<p class="section-header">mAP by Scenario (Top 20)</p>', unsafe_allow_html=True)
    if not filtered.empty:
        filtered_sorted = filtered.sort_values("mean_ap", ascending=False).head(20)
        filtered_sorted["label"] = (
            filtered_sorted["weather"]
            + " · "
            + filtered_sorted["scene"]
            + " · "
            + filtered_sorted["time_of_day"]
        )
        fig_bar = px.bar(
            filtered_sorted,
            x="mean_ap",
            y="label",
            orientation="h",
            color="mean_ap",
            color_continuous_scale="Viridis",
            text="mean_ap",
        )
        fig_bar.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=500,
            xaxis_title="mAP",
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
            coloraxis_colorbar=dict(title="mAP"),
        )
        fig_bar.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

elif tab == "Coverage":
    st.markdown("## 📊 Coverage Analysis")

    if not coverage_df.empty:
        total_buckets = len(coverage_df)
        low_count = len(coverage_df[coverage_df["status"] == "low"])
        critical_count = len(coverage_df[coverage_df["status"] == "critical"])
        adequate_count = len(coverage_df[coverage_df["status"] == "adequate"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Buckets", total_buckets)
        c2.metric(
            "Adequate",
            adequate_count,
            delta=f"{adequate_count / total_buckets * 100:.0f}%",
        )
        c3.metric(
            "Low",
            low_count,
            delta=f"{low_count / total_buckets * 100:.0f}%",
            delta_color="inverse",
        )
        c4.metric(
            "Critical",
            critical_count,
            delta=f"{critical_count / total_buckets * 100:.0f}%",
            delta_color="inverse",
        )

        st.markdown('<p class="section-header">Coverage by Scenario</p>', unsafe_allow_html=True)
        fig_cov = px.treemap(
            coverage_df,
            path=["weather", "scene", "time_of_day"],
            values="count",
            color="status",
            color_discrete_map={"adequate": "#10b981", "low": "#f59e0b", "critical": "#ef4444"},
        )
        fig_cov.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=500,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_cov, use_container_width=True)

        st.markdown('<p class="section-header">Coverage Table</p>', unsafe_allow_html=True)
        display_df = coverage_df.copy()
        display_df = display_df.sort_values(["status", "count"], ascending=[True, True])
        st.dataframe(
            display_df,
            use_container_width=True,
            column_config={
                "count": st.column_config.ProgressColumn(
                    "Sample Count",
                    min_value=0,
                    max_value=int(display_df["count"].max()),
                ),
                "status": st.column_config.TextColumn("Status"),
            },
        )
    else:
        st.info("No coverage data available.")

elif tab == "Details":
    st.markdown("## 🔍 Bucket Details")

    filtered = latest_run[
        latest_run["weather"].isin(sel_weather)
        & latest_run["scene"].isin(sel_scene)
        & latest_run["time_of_day"].isin(sel_time)
    ]

    summary = get_bucket_summary(filtered)
    if not summary.empty:
        summary = summary.sort_values("mean_ap", ascending=False)

        st.markdown('<p class="section-header">All Scenario Buckets</p>', unsafe_allow_html=True)

        def style_map(val):
            if val >= 0.5:
                return "background-color: rgba(16, 185, 129, 0.2); color: #34d399"
            elif val >= 0.25:
                return "background-color: rgba(245, 158, 11, 0.15); color: #fbbf24"
            elif val > 0:
                return "background-color: rgba(239, 68, 68, 0.15); color: #f87171"
            return "color: #64748b"

        styled = summary.style.map(style_map, subset=["mean_ap"])
        st.dataframe(styled, use_container_width=True, height=500)

        st.markdown('<p class="section-header">mAP Distribution</p>', unsafe_allow_html=True)
        fig_hist = px.histogram(
            summary,
            x="mean_ap",
            nbins=20,
            color="verdict",
            color_discrete_map={"pass": "#10b981", "warning": "#f59e0b", "critical": "#ef4444"},
        )
        fig_hist.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e2e8f0"),
            height=300,
            xaxis_title="mAP",
            yaxis_title="Count",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No bucket details available.")
