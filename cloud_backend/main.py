"""
EdgeOps AI — FastAPI Cloud Backend
Central hub for anomaly events, incident management, LLM explanations, and analytics.
"""
import os
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

import models
import schemas
from database import engine, get_db
from llm_service import get_llm_service
from incident_handler import upsert_incident, auto_resolve_stale_incidents

CLOUD_API_KEY = os.getenv("CLOUD_API_KEY", "dev-secret-key-change-in-prod")


# ────────────────────────────────────────────────
# App lifecycle
# ────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 EdgeOps AI Cloud Backend starting...")
    models.Base.metadata.create_all(bind=engine)
    logger.info("✓ Database tables created/verified")

    # Pre-init LLM service
    llm = get_llm_service()
    logger.info(f"✓ LLM service: {'Ollama' if llm.available else 'Rule-based fallback'}")
    yield
    logger.info("EdgeOps AI Cloud Backend stopped.")


app = FastAPI(
    title="EdgeOps AI — Cloud Backend",
    description="Centralized anomaly event store, incident management, and LLM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────────────────────────────────
# Auth dependency (simple API key)
# ────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="")):
    if CLOUD_API_KEY and x_api_key != CLOUD_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ────────────────────────────────────────────────
# Health
# ────────────────────────────────────────────────

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/api/llm/status", tags=["llm"])
def llm_status():
    llm = get_llm_service()
    return {
        "available": llm.available,
        "model": llm.model,
        "backend": "ollama" if llm.available else "rule_based",
    }


# ────────────────────────────────────────────────
# Devices
# ────────────────────────────────────────────────

@app.post("/api/devices", response_model=schemas.DeviceResponse, tags=["devices"])
def register_device(
    payload: schemas.DeviceCreate,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
):
    device = db.query(models.Device).filter_by(device_id=payload.device_id).first()
    if device:
        device.last_seen_at = datetime.utcnow()
        device.is_active = True
    else:
        device = models.Device(
            device_id=payload.device_id,
            device_type=payload.device_type,
            location=payload.location,
            metadata_=payload.metadata,
        )
        db.add(device)
    db.commit()
    db.refresh(device)
    return device


@app.get("/api/devices", response_model=list[schemas.DeviceResponse], tags=["devices"])
def list_devices(db: Session = Depends(get_db)):
    return db.query(models.Device).order_by(models.Device.last_seen_at.desc()).all()


@app.get("/api/devices/{device_id}", response_model=schemas.DeviceResponse, tags=["devices"])
def get_device(device_id: str, db: Session = Depends(get_db)):
    device = db.query(models.Device).filter_by(device_id=device_id).first()
    if not device:
        raise HTTPException(404, f"Device '{device_id}' not found")
    return device


# ────────────────────────────────────────────────
# Anomaly Events
# ────────────────────────────────────────────────

def _enrich_with_llm(event_id: int):
    """Background task: generate LLM explanation and attach to event."""
    from database import SessionLocal
    db = SessionLocal()
    try:
        event = db.query(models.AnomalyEvent).filter_by(id=event_id).first()
        if not event or event.llm_explanation:
            return

        llm = get_llm_service()
        event_dict = {
            "device_id": event.device_id,
            "timestamp": event.timestamp,
            "cpu_usage": event.cpu_usage,
            "memory_usage": event.memory_usage,
            "temperature": event.temperature,
            "network_in": event.network_in,
            "network_out": event.network_out,
            "disk_io": event.disk_io,
            "error_rate": event.error_rate,
            "latency_ms": event.latency_ms,
            "normalized_score": event.normalized_score,
            "confidence": event.confidence,
            "anomaly_type": event.anomaly_type,
        }

        result = llm.explain(event_dict)
        event.llm_explanation = result.get("explanation")
        event.llm_recommended_action = result.get("recommended_action")
        event.llm_severity = result.get("severity")
        event.llm_processed_at = datetime.utcnow()

        # Also enrich the incident if linked
        if event.incident_id:
            incident = db.query(models.Incident).filter_by(id=event.incident_id).first()
            if incident and not incident.llm_root_cause:
                incident.llm_root_cause = result.get("root_cause_hypothesis")
                incident.llm_action_plan = result.get("recommended_action")

        db.commit()
        logger.success(f"LLM explanation generated for event #{event_id}")
    except Exception as e:
        logger.error(f"LLM enrichment failed for event #{event_id}: {e}")
    finally:
        db.close()


@app.post("/api/events", response_model=schemas.AnomalyEventResponse, status_code=201, tags=["events"])
def ingest_event(
    payload: schemas.AnomalyEventCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    _: None = Depends(verify_api_key),
):
    # Ensure device exists
    device = db.query(models.Device).filter_by(device_id=payload.device_id).first()
    if not device:
        device = models.Device(device_id=payload.device_id)
        db.add(device)
        db.flush()

    device.last_seen_at = datetime.utcnow()

    # Create event
    event = models.AnomalyEvent(**payload.model_dump())
    db.add(event)
    db.flush()

    # Auto-create/update incident
    incident = upsert_incident(db, event)
    event.incident_id = incident.id

    db.commit()
    db.refresh(event)

    # Async LLM enrichment
    background_tasks.add_task(_enrich_with_llm, event.id)

    logger.info(f"Event #{event.id} ingested [{payload.device_id}] score={payload.normalized_score:.3f}")
    return event


@app.get("/api/events", response_model=list[schemas.AnomalyEventResponse], tags=["events"])
def list_events(
    device_id: Optional[str] = Query(None),
    hours: int = Query(24, ge=1, le=168),
    limit: int = Query(100, ge=1, le=1000),
    min_score: float = Query(0.0, ge=0.0, le=1.0),
    db: Session = Depends(get_db),
):
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    q = (
        db.query(models.AnomalyEvent)
        .filter(
            models.AnomalyEvent.received_at >= cutoff,
            models.AnomalyEvent.normalized_score >= min_score,
        )
        .order_by(desc(models.AnomalyEvent.received_at))
    )
    if device_id:
        q = q.filter(models.AnomalyEvent.device_id == device_id)
    return q.limit(limit).all()


@app.get("/api/events/{event_id}", response_model=schemas.AnomalyEventResponse, tags=["events"])
def get_event(event_id: int, db: Session = Depends(get_db)):
    event = db.query(models.AnomalyEvent).filter_by(id=event_id).first()
    if not event:
        raise HTTPException(404, f"Event #{event_id} not found")
    return event


@app.post("/api/events/{event_id}/explain", tags=["llm"])
def explain_event(
    event_id: int,
    req: schemas.LLMExplainRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    event = db.query(models.AnomalyEvent).filter_by(id=event_id).first()
    if not event:
        raise HTTPException(404, f"Event #{event_id} not found")

    if req.force_refresh or not event.llm_explanation:
        event.llm_explanation = None
        db.commit()
        background_tasks.add_task(_enrich_with_llm, event_id)
        return {"message": "LLM explanation scheduled", "event_id": event_id}

    return {
        "event_id": event_id,
        "explanation": event.llm_explanation,
        "recommended_action": event.llm_recommended_action,
        "severity": event.llm_severity,
    }


# ────────────────────────────────────────────────
# Incidents
# ────────────────────────────────────────────────

@app.get("/api/incidents", response_model=list[schemas.IncidentResponse], tags=["incidents"])
def list_incidents(
    status: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db),
):
    q = db.query(models.Incident).order_by(desc(models.Incident.created_at))
    if status:
        q = q.filter(models.Incident.status == status)
    if device_id:
        q = q.filter(models.Incident.device_id == device_id)
    return q.limit(limit).all()


@app.get("/api/incidents/{incident_id}", response_model=schemas.IncidentResponse, tags=["incidents"])
def get_incident(incident_id: int, db: Session = Depends(get_db)):
    inc = db.query(models.Incident).filter_by(id=incident_id).first()
    if not inc:
        raise HTTPException(404, f"Incident #{incident_id} not found")
    return inc


@app.patch("/api/incidents/{incident_id}", response_model=schemas.IncidentResponse, tags=["incidents"])
def update_incident(
    incident_id: int,
    update: schemas.IncidentUpdate,
    db: Session = Depends(get_db),
):
    inc = db.query(models.Incident).filter_by(id=incident_id).first()
    if not inc:
        raise HTTPException(404, f"Incident #{incident_id} not found")

    if update.status:
        inc.status = update.status
        if update.status == "resolved":
            inc.resolved_at = datetime.utcnow()
    if update.severity:
        inc.severity = update.severity
    if update.resolution_notes:
        inc.resolution_notes = update.resolution_notes

    inc.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(inc)
    return inc


@app.post("/api/incidents/auto-resolve", tags=["incidents"])
def trigger_auto_resolve(db: Session = Depends(get_db)):
    count = auto_resolve_stale_incidents(db)
    return {"resolved_count": count}


# ────────────────────────────────────────────────
# Analytics / Stats
# ────────────────────────────────────────────────

@app.get("/api/stats", response_model=schemas.SystemStats, tags=["analytics"])
def system_stats(db: Session = Depends(get_db)):
    total_devices = db.query(func.count(models.Device.id)).scalar() or 0
    active_devices = (
        db.query(func.count(models.Device.id))
        .filter(models.Device.is_active == True)
        .scalar() or 0
    )
    total_events = db.query(func.count(models.AnomalyEvent.id)).scalar() or 0
    open_incidents = (
        db.query(func.count(models.Incident.id))
        .filter(models.Incident.status.in_(["open", "acknowledged"]))
        .scalar() or 0
    )

    cutoff_1h = datetime.utcnow() - timedelta(hours=1)
    events_last_hour = (
        db.query(func.count(models.AnomalyEvent.id))
        .filter(models.AnomalyEvent.received_at >= cutoff_1h)
        .scalar() or 0
    )

    avg_score = (
        db.query(func.avg(models.AnomalyEvent.normalized_score)).scalar()
    )

    # Top anomalous devices (last 24h)
    cutoff_24h = datetime.utcnow() - timedelta(hours=24)
    top_devices = (
        db.query(
            models.AnomalyEvent.device_id,
            func.count(models.AnomalyEvent.id).label("event_count"),
            func.avg(models.AnomalyEvent.normalized_score).label("avg_score"),
        )
        .filter(models.AnomalyEvent.received_at >= cutoff_24h)
        .group_by(models.AnomalyEvent.device_id)
        .order_by(desc("event_count"))
        .limit(5)
        .all()
    )

    top_anomalous = [
        {"device_id": r.device_id, "event_count": r.event_count, "avg_score": round(float(r.avg_score), 3)}
        for r in top_devices
    ]

    # Communication reduction: edge only sends anomalies
    # Estimate: if anomaly rate is ~5%, we saved ~95% of traffic
    comm_reduction = 95.0  # Would be computed from edge stats in production

    return schemas.SystemStats(
        total_devices=total_devices,
        active_devices=active_devices,
        total_events=total_events,
        open_incidents=open_incidents,
        events_last_hour=events_last_hour,
        avg_anomaly_score=round(float(avg_score), 4) if avg_score else None,
        top_anomalous_devices=top_anomalous,
        communication_reduction_pct=comm_reduction,
    )


@app.get("/api/analytics/timeline", tags=["analytics"])
def anomaly_timeline(
    hours: int = Query(24, ge=1, le=168),
    bucket_minutes: int = Query(60, ge=5, le=1440),
    device_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
):
    """Return anomaly counts bucketed by time for timeline charts."""
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    q = db.query(
        models.AnomalyEvent.received_at,
        models.AnomalyEvent.normalized_score,
        models.AnomalyEvent.device_id,
        models.AnomalyEvent.anomaly_type,
    ).filter(models.AnomalyEvent.received_at >= cutoff)

    if device_id:
        q = q.filter(models.AnomalyEvent.device_id == device_id)

    events = q.all()

    # Bucket into time slots
    buckets: dict = {}
    for ev in events:
        slot = ev.received_at.replace(
            minute=(ev.received_at.minute // bucket_minutes) * bucket_minutes,
            second=0, microsecond=0,
        )
        key = slot.isoformat()
        if key not in buckets:
            buckets[key] = {"timestamp": key, "count": 0, "avg_score": 0.0, "scores": []}
        buckets[key]["count"] += 1
        buckets[key]["scores"].append(ev.normalized_score)

    result = []
    for key in sorted(buckets):
        b = buckets[key]
        b["avg_score"] = round(sum(b["scores"]) / len(b["scores"]), 4)
        del b["scores"]
        result.append(b)

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
