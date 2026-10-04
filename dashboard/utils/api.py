"""
EdgeOps AI — Streamlit Dashboard
Shared API client and utilities.
"""
import os
import requests
from datetime import datetime
from typing import Optional
import streamlit as st

CLOUD_API_URL = os.getenv("CLOUD_API_URL", "http://localhost:8000")
API_KEY = os.getenv("CLOUD_API_KEY", "dev-secret-key-change-in-prod")

HEADERS = {"X-API-Key": API_KEY}

SEVERITY_COLORS = {
    "critical": "#FF2D55",
    "high": "#FF6B35",
    "medium": "#F59E0B",
    "low": "#10B981",
}

STATUS_COLORS = {
    "open": "#FF4757",
    "acknowledged": "#F59E0B",
    "resolved": "#10B981",
}

ANOMALY_TYPE_ICONS = {
    "spike": "⚡",
    "drop": "📉",
    "drift": "📈",
    "freeze": "🧊",
    "unknown": "❓",
}


def api_get(path: str, params: Optional[dict] = None) -> Optional[dict | list]:
    try:
        resp = requests.get(f"{CLOUD_API_URL}{path}", headers=HEADERS, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to Cloud API at `{CLOUD_API_URL}`")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(path: str, json: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{CLOUD_API_URL}{path}", headers=HEADERS, json=json, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_patch(path: str, json: dict) -> Optional[dict]:
    try:
        resp = requests.patch(f"{CLOUD_API_URL}{path}", headers=HEADERS, json=json, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def fmt_score(score: float) -> str:
    if score >= 0.85:
        return f"🔴 {score:.3f}"
    elif score >= 0.70:
        return f"🟠 {score:.3f}"
    elif score >= 0.50:
        return f"🟡 {score:.3f}"
    else:
        return f"🟢 {score:.3f}"


def fmt_ts(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def severity_badge(severity: str) -> str:
    color = SEVERITY_COLORS.get(severity, "#888")
    return f'<span style="background:{color};color:white;padding:2px 8px;border-radius:4px;font-size:0.8em;font-weight:bold">{severity.upper()}</span>'
