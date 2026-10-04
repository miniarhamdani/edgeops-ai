"""
EdgeOps AI — Analytics Page
Performance benchmarks, anomaly distributions, latency stats.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from api import api_get, SEVERITY_COLORS

st.set_page_config(page_title="Analytics — EdgeOps AI", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #161B2A 100%); }
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📈 Analytics & Performance")

# ── Filters ─────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    hours = st.selectbox("Time window", [1, 6, 12, 24, 48, 168], index=3,
                         format_func=lambda h: f"Last {h}h")
with col2:
    device_filter = st.text_input("Filter by device ID", "")


# ── Load data ────────────────────────────────────────────────────────────
params = {"hours": hours, "limit": 1000}
if device_filter:
    params["device_id"] = device_filter

events_raw = api_get("/api/events", params=params) or []
timeline_raw = api_get("/api/analytics/timeline", params={"hours": hours, "bucket_minutes": max(hours, 1)}) or []

if not events_raw:
    st.info("No anomaly events in the selected time window.")
    st.stop()

df = pd.DataFrame(events_raw)
df["received_at"] = pd.to_datetime(df["received_at"])

# ── KPIs ─────────────────────────────────────────────────────────────────
st.markdown("### 📊 Summary Statistics")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Events", len(df))
c2.metric("Unique Devices", df["device_id"].nunique())
c3.metric("Avg Anomaly Score", f"{df['normalized_score'].mean():.3f}")
c4.metric("Avg Detection Latency", f"{df['detection_latency_ms'].mean():.2f}ms")
c5.metric("Avg Confidence", f"{df['confidence'].mean():.1%}")

st.markdown("---")

# ── Score Distribution ────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🎯 Score Distribution")
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=df["normalized_score"],
        nbinsx=30,
        marker=dict(
            color=df["normalized_score"],
            colorscale=[[0, "#10B981"], [0.5, "#F59E0B"], [1, "#FF2D55"]],
        ),
        name="Score",
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#F59E0B", annotation_text="Medium threshold")
    fig.add_vline(x=0.7, line_dash="dash", line_color="#FF6B35", annotation_text="High threshold")
    fig.add_vline(x=0.85, line_dash="dash", line_color="#FF2D55", annotation_text="Critical threshold")
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,41,0.8)",
        font=dict(color="#94A3B8", family="Inter"),
        margin=dict(l=10, r=10, t=20, b=10), height=320,
        xaxis=dict(title="Normalized Anomaly Score", gridcolor="rgba(99,102,241,0.1)"),
        yaxis=dict(title="Count", gridcolor="rgba(99,102,241,0.1)"),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_b:
    st.markdown("### 🔬 Anomaly Type Breakdown")
    type_counts = df["anomaly_type"].value_counts().reset_index()
    type_counts.columns = ["type", "count"]
    colors = ["#6366F1", "#8B5CF6", "#EC4899", "#10B981", "#F59E0B"]
    fig2 = go.Figure(go.Pie(
        labels=type_counts["type"],
        values=type_counts["count"],
        hole=0.45,
        marker=dict(colors=colors[:len(type_counts)], line=dict(color="#0A0E1A", width=2)),
        textfont=dict(color="white"),
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter"),
        margin=dict(l=10, r=10, t=20, b=10), height=320,
        legend=dict(font=dict(color="#94A3B8")),
        annotations=[dict(text="Types", x=0.5, y=0.5, font_size=16, showarrow=False, font_color="#94A3B8")],
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Detection Latency ─────────────────────────────────────────────────────
st.markdown("### ⚡ Detection Latency Distribution")
fig3 = go.Figure()
for device in df["device_id"].unique()[:6]:
    sub = df[df["device_id"] == device]
    fig3.add_trace(go.Box(
        y=sub["detection_latency_ms"],
        name=device,
        boxpoints="outliers",
        marker_color="#6366F1",
        line_color="#8B5CF6",
    ))
fig3.update_layout(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(20,24,41,0.8)",
    font=dict(color="#94A3B8", family="Inter"),
    margin=dict(l=10, r=10, t=20, b=10), height=300,
    yaxis=dict(title="Latency (ms)", gridcolor="rgba(99,102,241,0.1)"),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)"),
    showlegend=False,
)
st.plotly_chart(fig3, use_container_width=True)

# ── Sensor Correlations ──────────────────────────────────────────────────
st.markdown("### 🔗 Sensor Metric Correlations")
numeric_cols = ["cpu_usage", "memory_usage", "temperature", "network_in",
                "network_out", "disk_io", "error_rate", "latency_ms", "normalized_score"]
corr = df[numeric_cols].corr()

fig4 = go.Figure(go.Heatmap(
    z=corr.values,
    x=corr.columns.tolist(),
    y=corr.columns.tolist(),
    colorscale=[[0, "#0A0E1A"], [0.5, "#6366F1"], [1, "#FF2D55"]],
    zmid=0,
    text=corr.round(2).values,
    texttemplate="%{text}",
    textfont=dict(size=10, color="white"),
    hovertemplate="%{x} vs %{y}: %{z:.3f}<extra></extra>",
))
fig4.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94A3B8", family="Inter"),
    margin=dict(l=10, r=10, t=20, b=10), height=420,
)
st.plotly_chart(fig4, use_container_width=True)

# ── Per-device Anomaly Rate ─────────────────────────────────────────────
st.markdown("### 📡 Per-Device Anomaly Profile")
device_stats = (
    df.groupby("device_id")
    .agg(
        event_count=("id", "count"),
        avg_score=("normalized_score", "mean"),
        max_score=("normalized_score", "max"),
        avg_cpu=("cpu_usage", "mean"),
        avg_mem=("memory_usage", "mean"),
        avg_temp=("temperature", "mean"),
    )
    .reset_index()
    .sort_values("event_count", ascending=False)
)
device_stats = device_stats.round(3)
st.dataframe(device_stats, use_container_width=True, hide_index=True)
