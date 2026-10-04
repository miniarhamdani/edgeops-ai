"""
EdgeOps AI — SQLAlchemy Database Models
"""
from datetime import datetime
from sqlalchemy import (
    Column, String, Float, Boolean, DateTime,
    Integer, Text, JSON, ForeignKey, Index,
)
from sqlalchemy.orm import relationship
from database import Base


class Device(Base):
    """Registered edge device."""
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(64), unique=True, nullable=False, index=True)
    device_type = Column(String(64), default="simulated_iot")
    location = Column(String(128), nullable=True)
    registered_at = Column(DateTime, default=datetime.utcnow)
    last_seen_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    metadata_ = Column("metadata", JSON, default=dict)

    events = relationship("AnomalyEvent", back_populates="device", lazy="dynamic")
    incidents = relationship("Incident", back_populates="device", lazy="dynamic")

    def __repr__(self):
        return f"<Device {self.device_id}>"


class AnomalyEvent(Base):
    """A single anomaly event forwarded from an edge node."""
    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(64), ForeignKey("devices.device_id"), nullable=False, index=True)
    timestamp = Column(Float, nullable=False)       # Unix timestamp from edge
    received_at = Column(DateTime, default=datetime.utcnow)

    # Sensor readings
    cpu_usage = Column(Float)
    memory_usage = Column(Float)
    temperature = Column(Float)
    network_in = Column(Float)
    network_out = Column(Float)
    disk_io = Column(Float)
    error_rate = Column(Float)
    latency_ms = Column(Float)

    # Detection metadata
    anomaly_score = Column(Float)
    normalized_score = Column(Float)
    confidence = Column(Float)
    detection_latency_ms = Column(Float)
    anomaly_type = Column(String(32), default="unknown")
    is_ground_truth_anomaly = Column(Boolean, default=False)  # from simulator label
    source = Column(String(32), default="edge")

    # LLM explanation (filled asynchronously)
    llm_explanation = Column(Text, nullable=True)
    llm_recommended_action = Column(Text, nullable=True)
    llm_severity = Column(String(16), nullable=True)   # low/medium/high/critical
    llm_processed_at = Column(DateTime, nullable=True)

    # Incident link
    incident_id = Column(Integer, ForeignKey("incidents.id"), nullable=True)

    device = relationship("Device", back_populates="events")
    incident = relationship("Incident", back_populates="events")

    __table_args__ = (
        Index("ix_anomaly_events_device_ts", "device_id", "timestamp"),
        Index("ix_anomaly_events_received", "received_at"),
    )

    def __repr__(self):
        return f"<AnomalyEvent {self.id} [{self.device_id}] score={self.normalized_score:.3f}>"


class Incident(Base):
    """
    An incident groups related anomaly events for a device into a manageable alert.
    Auto-created when a device exceeds the anomaly threshold.
    """
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(String(64), ForeignKey("devices.device_id"), nullable=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

    status = Column(String(32), default="open")   # open / acknowledged / resolved
    severity = Column(String(16), default="medium")  # low/medium/high/critical
    title = Column(String(256), nullable=True)
    summary = Column(Text, nullable=True)
    resolution_notes = Column(Text, nullable=True)

    anomaly_count = Column(Integer, default=1)
    avg_score = Column(Float, nullable=True)
    peak_score = Column(Float, nullable=True)

    # LLM-generated root cause
    llm_root_cause = Column(Text, nullable=True)
    llm_action_plan = Column(Text, nullable=True)

    device = relationship("Device", back_populates="incidents")
    events = relationship("AnomalyEvent", back_populates="incident")

    def __repr__(self):
        return f"<Incident {self.id} [{self.device_id}] {self.status}>"
