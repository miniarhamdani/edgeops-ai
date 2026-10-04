"""
EdgeOps AI — Pydantic Schemas (Request/Response)
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ────────────────────────────────────────────────
# Device
# ────────────────────────────────────────────────

class DeviceCreate(BaseModel):
    device_id: str
    device_type: str = "simulated_iot"
    location: Optional[str] = None
    metadata: dict = Field(default_factory=dict)


class DeviceResponse(BaseModel):
    id: int
    device_id: str
    device_type: str
    location: Optional[str]
    registered_at: datetime
    last_seen_at: datetime
    is_active: bool

    class Config:
        from_attributes = True


# ────────────────────────────────────────────────
# Anomaly Event
# ────────────────────────────────────────────────

class AnomalyEventCreate(BaseModel):
    device_id: str
    timestamp: float
    cpu_usage: float
    memory_usage: float
    temperature: float
    network_in: float
    network_out: float
    disk_io: float
    error_rate: float
    latency_ms: float
    anomaly_score: float
    normalized_score: float
    confidence: float
    detection_latency_ms: float = 0.0
    anomaly_type: str = "unknown"
    is_ground_truth_anomaly: bool = False
    source: str = "edge"


class AnomalyEventResponse(BaseModel):
    id: int
    device_id: str
    timestamp: float
    received_at: datetime
    cpu_usage: float
    memory_usage: float
    temperature: float
    network_in: float
    network_out: float
    disk_io: float
    error_rate: float
    latency_ms: float
    anomaly_score: float
    normalized_score: float
    confidence: float
    detection_latency_ms: float
    anomaly_type: str
    is_ground_truth_anomaly: bool
    llm_explanation: Optional[str] = None
    llm_recommended_action: Optional[str] = None
    llm_severity: Optional[str] = None
    incident_id: Optional[int] = None

    class Config:
        from_attributes = True


# ────────────────────────────────────────────────
# Incident
# ────────────────────────────────────────────────

class IncidentCreate(BaseModel):
    device_id: str
    severity: str = "medium"
    title: Optional[str] = None
    summary: Optional[str] = None


class IncidentUpdate(BaseModel):
    status: Optional[str] = None
    severity: Optional[str] = None
    resolution_notes: Optional[str] = None


class IncidentResponse(BaseModel):
    id: int
    device_id: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    status: str
    severity: str
    title: Optional[str]
    summary: Optional[str]
    resolution_notes: Optional[str]
    anomaly_count: int
    avg_score: Optional[float]
    peak_score: Optional[float]
    llm_root_cause: Optional[str] = None
    llm_action_plan: Optional[str] = None

    class Config:
        from_attributes = True


# ────────────────────────────────────────────────
# Stats / Analytics
# ────────────────────────────────────────────────

class SystemStats(BaseModel):
    total_devices: int
    active_devices: int
    total_events: int
    open_incidents: int
    events_last_hour: int
    avg_anomaly_score: Optional[float]
    top_anomalous_devices: list[dict]
    communication_reduction_pct: float


class LLMExplainRequest(BaseModel):
    event_id: int
    force_refresh: bool = False
