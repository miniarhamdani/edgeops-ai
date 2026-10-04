"""
EdgeOps AI — LLM Insights Page
Browse LLM-generated anomaly explanations and trigger analysis on demand.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))

import streamlit as st
import pandas as pd
from api import api_get, api_post, SEVERITY_COLORS, ANOMALY_TYPE_ICONS, fmt_score, fmt_ts

st.set_page_config(page_title="LLM Insights — EdgeOps AI", page_icon="🤖", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0D1117 0%, #161B2A 100%); }
#MainMenu, footer { visibility: hidden; }

.llm-card {
    background: linear-gradient(135deg, #1a1f35 0%, #141829 100%);
    border-radius: 16px;
    padding: 20px 24px;
    margin-bottom: 16px;
    border: 1px solid rgba(99,102,241,0.2);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
.llm-explanation {
    background: rgba(99,102,241,0.08);
    border-left: 3px solid #6366F1;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #CBD5E1;
    font-size: 0.92rem;
    line-height: 1.6;
}
.llm-action {
    background: rgba(16,185,129,0.08);
    border-left: 3px solid #10B981;
    border-radius: 0 8px 8px 0;
    padding: 12px 16px;
    margin: 10px 0;
    color: #CBD5E1;
    font-size: 0.92rem;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🤖 LLM Anomaly Insights")
st.markdown("*Powered by Ollama — local LLM explanations with zero data leaving your infrastructure*")

# LLM status
llm_status = api_get("/api/llm/status")
if llm_status:
    if llm_status.get("available"):
        st.success(f"✅ Ollama running — model: `{llm_status.get('model')}`")
    else:
        st.warning("⚠️ Ollama not available. Using rule-based explanations. Run `ollama serve` to enable LLM.")

st.markdown("---")

# ── Filters ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    hours = st.selectbox("Time window", [1, 6, 24, 48], index=2, format_func=lambda h: f"Last {h}h")
with col2:
    min_score = st.slider("Min anomaly score", 0.0, 1.0, 0.3, 0.05)
with col3:
    show_pending = st.checkbox("Show pending LLM analysis", value=True)

events = api_get("/api/events", params={"hours": hours, "limit": 50, "min_score": min_score}) or []

if not events:
    st.info("No events found in the selected time window.")
    st.stop()

# Separate processed vs pending
processed = [e for e in events if e.get("llm_explanation")]
pending = [e for e in events if not e.get("llm_explanation")]

st.markdown(f"**{len(processed)} explained** | **{len(pending)} pending LLM analysis**")

# ── Trigger batch LLM for pending ────────────────────────────────────────
if pending and st.button(f"🤖 Analyze {len(pending)} pending events with LLM"):
    for ev in pending[:10]:  # limit to 10 at a time
        api_post(f"/api/events/{ev['id']}/explain", {"event_id": ev["id"], "force_refresh": False})
    st.success("LLM analysis triggered! Refresh in a few seconds.")
    st.rerun()

# ── Processed Events ─────────────────────────────────────────────────────
st.markdown("### ✅ LLM-Explained Anomalies")

if not processed:
    st.info("No LLM explanations yet. Events are processed in the background.")
else:
    for ev in processed[:20]:
        sev_color = SEVERITY_COLORS.get(ev.get("llm_severity", "medium"), "#888")
        atype_icon = ANOMALY_TYPE_ICONS.get(ev.get("anomaly_type", "unknown"), "❓")

        st.markdown(f"""
        <div class="llm-card">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px">
            <div>
              <span style="color:#F1F5F9;font-weight:600;font-size:1rem">
                {atype_icon} {ev['device_id']} — Event #{ev['id']}
              </span>
              <div style="color:#64748B;font-size:0.8rem;margin-top:3px">
                🕐 {fmt_ts(ev['timestamp'])} &nbsp;|&nbsp; Score: {fmt_score(ev.get('normalized_score', 0))}
              </div>
            </div>
            <span style="background:{sev_color};color:white;padding:3px 10px;border-radius:6px;font-size:0.75rem;font-weight:700">
              {(ev.get('llm_severity') or 'N/A').upper()}
            </span>
          </div>

          <div style="color:#94A3B8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:4px">
            📖 Explanation
          </div>
          <div class="llm-explanation">{ev.get('llm_explanation', '—')}</div>

          <div style="color:#94A3B8;font-size:0.8rem;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;margin-top:10px;margin-bottom:4px">
            🔧 Recommended Action
          </div>
          <div class="llm-action">{ev.get('llm_recommended_action', '—')}</div>

          <div style="display:flex;gap:16px;margin-top:10px;font-size:0.78rem;color:#475569">
            <span>CPU: <b style="color:#CBD5E1">{ev.get('cpu_usage', 0):.1f}%</b></span>
            <span>Mem: <b style="color:#CBD5E1">{ev.get('memory_usage', 0):.1f}%</b></span>
            <span>Temp: <b style="color:#CBD5E1">{ev.get('temperature', 0):.1f}°C</b></span>
            <span>Err: <b style="color:#CBD5E1">{ev.get('error_rate', 0):.2%}</b></span>
            <span>Latency: <b style="color:#CBD5E1">{ev.get('latency_ms', 0):.1f}ms</b></span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ── Pending Events ────────────────────────────────────────────────────────
if show_pending and pending:
    st.markdown("### ⏳ Pending LLM Analysis")
    pending_df = pd.DataFrame([{
        "Event ID": ev["id"],
        "Device": ev["device_id"],
        "Type": ANOMALY_TYPE_ICONS.get(ev.get("anomaly_type", ""), "") + " " + ev.get("anomaly_type", ""),
        "Score": f"{ev.get('normalized_score', 0):.3f}",
        "Time": fmt_ts(ev.get("timestamp", 0)),
    } for ev in pending])
    st.dataframe(pending_df, use_container_width=True, hide_index=True)
