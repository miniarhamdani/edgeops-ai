"""
EdgeOps AI — Main Dashboard Entry Point (Overview)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "utils"))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from streamlit_autorefresh import st_autorefresh
from api import api_get, fmt_score, SEVERITY_COLORS, ANOMALY_TYPE_ICONS

st.set_page_config(
    page_title="EdgeOps AI — Dashboard",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0A0E1A; }

  .metric-card {
    background: linear-gradient(135deg, #1a1f35 0%, #141829 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); }
  .metric-value {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #6366F1, #8B5CF6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
  }
  .metric-label {
    color: #94A3B8;
    font-size: 0.85rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
  }
  .alert-critical { color: #FF2D55 !important; font-weight: 700; }
  .alert-high     { color: #FF6B35 !important; font-weight: 600; }
  .alert-medium   { color: #F59E0B !important; }
  .alert-low      { color: #10B981 !important; }

  .stDataFrame { border-radius: 12px; overflow: hidden; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B2A 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.15);
  }

  /* Hide default streamlit elements */
  #MainMenu, footer { visibility: hidden; }

  .header-logo {
    text-align: center;
    padding: 10px 0;
    border-bottom: 1px solid rgba(99, 102, 241, 0.2);
    margin-bottom: 20px;
  }
</style>
""", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="header-logo">
          <h2 style="color:#6366F1;margin:0">🛰️ EdgeOps AI</h2>
          <p style="color:#475569;font-size:0.8rem;margin:4px 0">Distributed Anomaly Detection</p>
        </div>
        """, unsafe_allow_html=True)

        refresh_interval = st.selectbox(
            "🔄 Auto-refresh",
            [5, 10, 30, 60, 0],
            format_func=lambda x: f"{x}s" if x > 0 else "Off",
        )
        if refresh_interval > 0:
            st_autorefresh(interval=refresh_interval * 1000, key="main_refresh")

        st.markdown("---")
        st.markdown("**🔗 API Status**")
        health = api_get("/health")
        if health:
            st.success("Cloud Backend ✓")
        else:
            st.error("Cloud Backend ✗")

        llm_status = api_get("/api/llm/status")
        if llm_status:
            if llm_status.get("available"):
                st.success(f"Ollama ({llm_status.get('model')}) ✓")
            else:
                st.warning("Ollama ✗ (rule-based)")

        st.markdown("---")
        st.markdown("**📖 Navigation**")
        st.page_link("app.py", label="📊 Overview", icon="🏠")
        st.page_link("pages/1_📊_Incidents.py", label="🚨 Incidents")
        st.page_link("pages/2_📈_Analytics.py", label="📈 Analytics")
        st.page_link("pages/3_🤖_LLM_Insights.py", label="🤖 LLM Insights")


def render_kpi_cards(stats: dict):
    cols = st.columns(5)
    kpis = [
        ("Total Devices", stats.get("total_devices", 0), "🖥️"),
        ("Active Devices", stats.get("active_devices", 0), "💚"),
        ("Total Anomalies", stats.get("total_events", 0), "⚡"),
        ("Open Incidents", stats.get("open_incidents", 0), "🚨"),
        ("Events (1h)", stats.get("events_last_hour", 0), "📡"),
    ]
    for col, (label, value, icon) in zip(cols, kpis):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)


def render_timeline_chart():
    st.markdown("### 📈 Anomaly Timeline (Last 24h)")
    timeline = api_get("/api/analytics/timeline", params={"hours": 24, "bucket_minutes": 5})
    if not timeline:
        st.info("No timeline data available yet.")
        return

    df = pd.DataFrame(timeline)
    if df.empty:
        st.info("No anomaly events in the last 24h.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df["timestamp"],
        y=df["count"],
        name="Anomaly Count",
        marker=dict(
            color=df["avg_score"],
            colorscale=[[0, "#10B981"], [0.35, "#F59E0B"], [0.75, "#FF2D55"]],
            colorbar=dict(title="Avg Score", thickness=12),
        ),
        hovertemplate="<b>%{x|%H:%M}</b><br>Count: %{y}<br>Avg Score: %{marker.color:.3f}<extra></extra>",
    ))

    xaxis_layout = dict(
        gridcolor="rgba(99,102,241,0.1)",
        showgrid=True,
        tickformat="%H:%M",
        type="date",
    )
    if len(df) == 1:
        # Prevent microsecond zoom on a single data bucket
        t0 = df["timestamp"].iloc[0]
        xaxis_layout["range"] = [t0 - pd.Timedelta(minutes=30), t0 + pd.Timedelta(minutes=30)]

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,41,0.8)",
        font=dict(color="#94A3B8", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=xaxis_layout,
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)", showgrid=True, title="Anomaly Count"),
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_top_devices(stats: dict):
    st.markdown("### 🏆 Most Anomalous Devices (24h)")
    top = stats.get("top_anomalous_devices", [])
    if not top:
        st.info("No device data yet.")
        return

    df = pd.DataFrame(top)
    fig = px.bar(
        df, x="device_id", y="event_count",
        color="avg_score",
        color_continuous_scale=["#10B981", "#F59E0B", "#FF2D55"],
        labels={"event_count": "Anomaly Count", "device_id": "Device", "avg_score": "Avg Score"},
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(20,24,41,0.8)",
        font=dict(color="#94A3B8", family="Inter"),
        margin=dict(l=10, r=10, t=10, b=10),
        coloraxis_colorbar=dict(thickness=12),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_recent_events():
    st.markdown("### 🔴 Recent Anomaly Events")
    events = api_get("/api/events", params={"limit": 20, "hours": 24})
    if not events:
        st.info("No recent events. Waiting for edge nodes...")
        return

    rows = []
    for ev in events:
        rows.append({
            "Device": ev["device_id"],
            "Type": ANOMALY_TYPE_ICONS.get(ev.get("anomaly_type", "unknown"), "❓") + " " + ev.get("anomaly_type", "unknown"),
            "Score": fmt_score(ev.get("normalized_score", 0)),
            "CPU%": f"{ev.get('cpu_usage', 0):.1f}",
            "Mem%": f"{ev.get('memory_usage', 0):.1f}",
            "Temp°C": f"{ev.get('temperature', 0):.1f}",
            "LLM": "✓" if ev.get("llm_explanation") else "⏳",
            "Incident": f"#{ev.get('incident_id', '—')}",
        })

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_comm_reduction(stats: dict):
    reduction = stats.get("communication_reduction_pct", 95.0)
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=reduction,
        title={"text": "Communication Reduction %", "font": {"color": "#94A3B8", "family": "Inter"}},
        number={"suffix": "%", "font": {"color": "#6366F1", "size": 40}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 20, 40, 60, 80, 100],
                "ticktext": ["0%", "20%", "40%", "60%", "80%", "100%"],
                "tickcolor": "#475569",
                "tickfont": {"size": 11, "color": "#94A3B8"},
            },
            "bar": {"color": "#6366F1"},
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 80], "color": "rgba(245,158,11,0.15)"},
                {"range": [80, 100], "color": "rgba(16,185,129,0.15)"},
            ],
            "threshold": {
                "line": {"color": "#10B981", "width": 3},
                "thickness": 0.75,
                "value": reduction,
            },
            "bgcolor": "rgba(20,24,41,0.8)",
        },
    ))
    gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94A3B8", family="Inter"),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(gauge, use_container_width=True)


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    render_sidebar()

    st.markdown("""
    <h1 style="color:#F1F5F9;font-weight:700;font-size:2rem;margin-bottom:4px">
      🛰️ EdgeOps AI
    </h1>
    <p style="color:#64748B;font-size:1rem;margin-bottom:24px">
      Distributed Edge/Cloud Anomaly Detection & Intelligent Monitoring
    </p>
    """, unsafe_allow_html=True)

    stats = api_get("/api/stats")
    if not stats:
        st.warning("⚠️ Could not fetch stats. Is the cloud backend running?")
        st.code("cd cloud_backend && uvicorn main:app --reload --port 8000", language="bash")
        return

    render_kpi_cards(stats)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        render_timeline_chart()
    with col2:
        render_comm_reduction(stats)

    col3, col4 = st.columns([2, 3])
    with col3:
        render_top_devices(stats)
    with col4:
        render_recent_events()


if __name__ == "__main__" or True:
    main()
