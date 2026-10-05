"""
EdgeOps AI — Automated Incident Handler
Groups anomaly events into incidents and manages lifecycle.
"""
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from loguru import logger
import models


# Configuration
INCIDENT_GROUP_WINDOW_MINUTES = 15    # Events within this window belong to same incident
INCIDENT_OPEN_THRESHOLD = 2           # Min anomalies before creating incident
SEVERITY_MAP = {
    "low": 1, "medium": 2, "high": 3, "critical": 4
}


def upsert_incident(db: Session, event: models.AnomalyEvent) -> models.Incident:
    """
    Find or create an incident for this anomaly event.
    Groups events from the same device within the time window.
    """
    cutoff = datetime.utcnow() - timedelta(minutes=INCIDENT_GROUP_WINDOW_MINUTES)

    # Look for an open incident for this device within the window
    incident = (
        db.query(models.Incident)
        .filter(
            models.Incident.device_id == event.device_id,
            models.Incident.status.in_(["open", "acknowledged"]),
            models.Incident.updated_at >= cutoff,
        )
        .order_by(models.Incident.created_at.desc())
        .first()
    )

    severity = _compute_severity(event.normalized_score)

    # Determine human-friendly anomaly pattern name
    raw_type = (event.anomaly_type or "unclassified").lower()
    type_display = "statistical_deviation" if raw_type in ("none", "unknown", "") else raw_type

    if incident is None:
        # Create new incident
        incident = models.Incident(
            device_id=event.device_id,
            severity=severity,
            title=f"Anomaly detected on {event.device_id} [{type_display}]",
            summary=(
                f"First anomaly: {type_display} pattern detected at "
                f"{datetime.fromtimestamp(event.timestamp).strftime('%H:%M:%S')}. "
                f"Score: {event.normalized_score:.3f}."
            ),
            anomaly_count=1,
            avg_score=event.normalized_score,
            peak_score=event.normalized_score,
        )
        db.add(incident)
        db.flush()  # get ID
        logger.info(f"New incident #{incident.id} created for {event.device_id} [{severity}]")
    else:
        # Update existing incident
        incident.anomaly_count += 1
        incident.updated_at = datetime.utcnow()

        # Update title if current title was unclassified/none and we now have a specific type
        if type_display != "statistical_deviation" and any(
            marker in incident.title for marker in ["[none]", "[statistical_deviation]", "[unclassified]"]
        ):
            incident.title = f"Anomaly detected on {event.device_id} [{type_display}]"
            logger.info(f"Incident #{incident.id} title updated to dominant type [{type_display}]")

        # Update running average
        old_total = (incident.avg_score or 0) * (incident.anomaly_count - 1)
        incident.avg_score = (old_total + event.normalized_score) / incident.anomaly_count

        # Update peak
        if event.normalized_score > (incident.peak_score or 0):
            incident.peak_score = event.normalized_score

        # Escalate severity if needed
        new_sev_rank = SEVERITY_MAP.get(severity, 1)
        cur_sev_rank = SEVERITY_MAP.get(incident.severity, 1)
        if new_sev_rank > cur_sev_rank:
            incident.severity = severity
            logger.info(f"Incident #{incident.id} escalated to {severity}")

    return incident


def auto_resolve_stale_incidents(db: Session, stale_minutes: int = 60) -> int:
    """Auto-resolve incidents that haven't received new events in `stale_minutes`."""
    cutoff = datetime.utcnow() - timedelta(minutes=stale_minutes)
    stale = (
        db.query(models.Incident)
        .filter(
            models.Incident.status == "open",
            models.Incident.updated_at < cutoff,
        )
        .all()
    )

    count = 0
    for incident in stale:
        incident.status = "resolved"
        incident.resolved_at = datetime.utcnow()
        incident.resolution_notes = f"Auto-resolved: no new events for {stale_minutes} minutes."
        count += 1

    if count > 0:
        db.commit()
        logger.info(f"Auto-resolved {count} stale incidents.")
    return count


def _compute_severity(score: float) -> str:
    if score >= 0.75:
        return "critical"
    elif score >= 0.55:
        return "high"
    elif score >= 0.35:
        return "medium"
    else:
        return "low"
