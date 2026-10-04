"""
EdgeOps AI — Incidents Page
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import streamlit as st
import pandas as pd
from api import api_get, api_patch, api_post, SEVERITY_COLORS, STATUS_COLORS

st.set_page_config(page_title="Incidents — EdgeOps AI", page_icon="🚨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #161B2A 100%); }
#MainMenu, footer { visibility: hidden; }
.incident-card {
    background: linear-gradient(135deg, #1a1f35 0%, #141829 100%);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid;
    box-shadow: 0 2px 12px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🚨 Incident Management")

# Filters
col1, col2, col3 = st.columns(3)
with col1:
    status_filter = st.selectbox("Status", ["all", "open", "acknowledged", "resolved"])
with col2:
    device_filter = st.text_input("Filter by device", "")
with col3:
    limit = st.slider("Max results", 10, 200, 50)

# Auto-resolve button
if st.button("🔄 Auto-resolve stale incidents"):
    result = api_post("/api/incidents/auto-resolve", {})
    if result:
        st.success(f"Auto-resolved {result.get('resolved_count', 0)} incidents")

params = {"limit": limit}
if status_filter != "all":
    params["status"] = status_filter
if device_filter:
    params["device_id"] = device_filter

incidents = api_get("/api/incidents", params=params)
if not incidents:
    st.info("No incidents found. Edge nodes may still be in warmup phase.")
    st.stop()

st.markdown(f"**{len(incidents)} incidents found**")

for inc in incidents:
    sev_color = SEVERITY_COLORS.get(inc["severity"], "#888")
    status_color = STATUS_COLORS.get(inc["status"], "#888")

    with st.container():
        st.markdown(f"""
        <div class="incident-card" style="border-left-color:{sev_color}">
          <div style="display:flex;justify-content:space-between;align-items:center">
            <span style="color:#F1F5F9;font-weight:600;font-size:1rem">
              #{inc['id']} — {inc.get('title', 'Anomaly incident')}
            </span>
            <span>
              <span style="background:{sev_color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700;margin-right:8px">
                {inc['severity'].upper()}
              </span>
              <span style="background:{status_color};color:white;padding:2px 8px;border-radius:4px;font-size:0.75rem;font-weight:700">
                {inc['status'].upper()}
              </span>
            </span>
          </div>
          <div style="color:#64748B;font-size:0.85rem;margin-top:6px">
            📍 {inc['device_id']} &nbsp;|&nbsp;
            ⚡ {inc['anomaly_count']} anomalies &nbsp;|&nbsp;
            📊 Peak: {inc.get('peak_score', 0):.3f} &nbsp;|&nbsp;
            🕐 {inc['created_at'][:19]}
          </div>
          {f'<div style="color:#94A3B8;font-size:0.85rem;margin-top:6px">{inc["summary"]}</div>' if inc.get("summary") else ""}
        </div>
        """, unsafe_allow_html=True)

        # Expandable actions
        with st.expander(f"Manage Incident #{inc['id']}"):
            tab1, tab2 = st.tabs(["📋 Details", "⚙️ Actions"])

            with tab1:
                cols = st.columns(3)
                cols[0].metric("Anomaly Count", inc["anomaly_count"])
                cols[1].metric("Avg Score", f"{inc.get('avg_score', 0):.3f}")
                cols[2].metric("Peak Score", f"{inc.get('peak_score', 0):.3f}")

                if inc.get("llm_root_cause"):
                    st.markdown("**🤖 LLM Root Cause Analysis:**")
                    st.info(inc["llm_root_cause"])

                if inc.get("llm_action_plan"):
                    st.markdown("**📋 Recommended Action Plan:**")
                    st.success(inc["llm_action_plan"])

                if inc.get("resolution_notes"):
                    st.markdown("**📝 Resolution Notes:**")
                    st.text(inc["resolution_notes"])

            with tab2:
                new_status = st.selectbox(
                    "Update Status",
                    ["open", "acknowledged", "resolved"],
                    index=["open", "acknowledged", "resolved"].index(inc["status"]),
                    key=f"status_{inc['id']}",
                )
                new_severity = st.selectbox(
                    "Update Severity",
                    ["low", "medium", "high", "critical"],
                    index=["low", "medium", "high", "critical"].index(inc["severity"]),
                    key=f"sev_{inc['id']}",
                )
                notes = st.text_area("Resolution Notes", value=inc.get("resolution_notes") or "", key=f"notes_{inc['id']}")

                if st.button("💾 Save Changes", key=f"save_{inc['id']}"):
                    result = api_patch(f"/api/incidents/{inc['id']}", {
                        "status": new_status,
                        "severity": new_severity,
                        "resolution_notes": notes if notes else None,
                    })
                    if result:
                        st.success(f"Incident #{inc['id']} updated!")
                        st.rerun()
