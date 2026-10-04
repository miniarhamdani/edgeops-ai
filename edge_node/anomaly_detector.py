"""
EdgeOps AI — Edge Anomaly Detection Pipeline
Implements a two-phase Isolation Forest pipeline with incremental retraining.
"""
import time
import numpy as np
import joblib
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sensor import SensorReading


@dataclass
class AnomalyResult:
    """Result of anomaly detection for a single reading."""
    device_id: str
    timestamp: float
    reading: SensorReading
    score: float            # Raw anomaly score (negative = anomalous)
    normalized_score: float # 0–1 (higher = more anomalous)
    is_anomaly: bool
    confidence: float       # 0–1
    latency_ms: float       # Detection latency

    def to_dict(self) -> dict:
        d = self.reading.to_dict()
        d.update({
            "anomaly_score": round(self.score, 4),
            "normalized_score": round(self.normalized_score, 4),
            "is_anomaly_detected": self.is_anomaly,
            "confidence": round(self.confidence, 4),
            "detection_latency_ms": round(self.latency_ms, 3),
        })
        return d


class EdgeAnomalyDetector:
    """
    Local Isolation Forest–based anomaly detector for an edge node.

    Lifecycle:
      1. WARMUP  — collect `training_samples` readings, no detection
      2. TRAINED — model fitted, detect on each new reading
      3. RETRAIN — periodically refit on a rolling window

    The detector is designed to run entirely offline on the edge device
    without sending data to the cloud until an anomaly is found.
    """

    def __init__(
        self,
        device_id: str,
        contamination: float = 0.05,
        training_samples: int = 200,
        retrain_interval: int = 500,
        model_path: Optional[Path] = None,
    ):
        self.device_id = device_id
        self.contamination = contamination
        self.training_samples = training_samples
        self.retrain_interval = retrain_interval
        self.model_path = model_path or Path(f"models/{device_id}_iforest.pkl")

        self._pipeline: Optional[Pipeline] = None
        self._buffer: list[list[float]] = []
        self._total_seen = 0
        self._total_anomalies = 0
        self._is_trained = False

        # Try to load a persisted model
        if self.model_path.exists():
            self._load_model()

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("iforest", IsolationForest(
                n_estimators=100,
                contamination=self.contamination,
                max_samples="auto",
                random_state=42,
                n_jobs=-1,
            )),
        ])

    def _train(self, X: np.ndarray) -> None:
        logger.info(f"[{self.device_id}] Training Isolation Forest on {len(X)} samples...")
        self._pipeline = self._build_pipeline()
        self._pipeline.fit(X)
        self._is_trained = True
        self._save_model()
        logger.success(f"[{self.device_id}] Model trained. Contamination={self.contamination}")

    def _save_model(self) -> None:
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._pipeline, self.model_path)
        logger.debug(f"[{self.device_id}] Model saved → {self.model_path}")

    def _load_model(self) -> None:
        try:
            self._pipeline = joblib.load(self.model_path)
            self._is_trained = True
            logger.info(f"[{self.device_id}] Loaded persisted model from {self.model_path}")
        except Exception as e:
            logger.warning(f"[{self.device_id}] Failed to load model: {e}")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------
    def _score_to_confidence(self, score: float) -> float:
        """Convert raw IF score to a 0–1 confidence value."""
        # IsolationForest scores are typically in [-0.5, 0.5]
        # Negative scores are anomalies; more negative = more anomalous
        normalized = np.clip(-score, 0, 1)
        return float(normalized)

    def process(self, reading: SensorReading) -> Optional[AnomalyResult]:
        """
        Process a single sensor reading.

        Returns:
            AnomalyResult if model is trained, None during warmup.
        """
        t_start = time.perf_counter()
        features = reading.to_feature_vector()
        self._buffer.append(features)
        self._total_seen += 1

        # -- Warmup phase: collect training data
        if not self._is_trained:
            if len(self._buffer) >= self.training_samples:
                X = np.array(self._buffer)
                self._train(X)
            else:
                remaining = self.training_samples - len(self._buffer)
                logger.debug(f"[{self.device_id}] Warmup: {remaining} samples remaining")
                return None

        # -- Periodic retraining on rolling window
        if self._total_seen % self.retrain_interval == 0 and self._total_seen > 0:
            window = self._buffer[-self.training_samples:]
            X = np.array(window)
            self._train(X)

        # -- Detection
        X_sample = np.array([features])
        score = float(self._pipeline.decision_function(X_sample)[0])
        prediction = int(self._pipeline.predict(X_sample)[0])
        is_anomaly = (prediction == -1)
        normalized = self._score_to_confidence(score)
        confidence = normalized if is_anomaly else 1 - normalized

        if is_anomaly:
            self._total_anomalies += 1

        # Keep buffer bounded
        if len(self._buffer) > self.training_samples * 5:
            self._buffer = self._buffer[-self.training_samples * 2:]

        latency_ms = (time.perf_counter() - t_start) * 1000

        return AnomalyResult(
            device_id=self.device_id,
            timestamp=reading.timestamp,
            reading=reading,
            score=score,
            normalized_score=normalized,
            is_anomaly=is_anomaly,
            confidence=confidence,
            latency_ms=latency_ms,
        )

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @property
    def anomaly_rate(self) -> float:
        if self._total_seen == 0:
            return 0.0
        return self._total_anomalies / self._total_seen

    def stats(self) -> dict:
        return {
            "device_id": self.device_id,
            "is_trained": self._is_trained,
            "total_readings": self._total_seen,
            "total_anomalies": self._total_anomalies,
            "anomaly_rate": round(self.anomaly_rate, 4),
            "buffer_size": len(self._buffer),
            "contamination": self.contamination,
        }
