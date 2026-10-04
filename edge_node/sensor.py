"""
EdgeOps AI — Synthetic Sensor Data Generator
Simulates realistic multi-metric IoT device telemetry with controllable anomaly injection.
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class AnomalyType(str, Enum):
    SPIKE = "spike"
    DROP = "drop"
    DRIFT = "drift"
    FREEZE = "freeze"
    NONE = "none"


@dataclass
class SensorReading:
    """A single multi-metric reading from a simulated edge device."""
    device_id: str
    timestamp: float
    cpu_usage: float          # 0–100 %
    memory_usage: float       # 0–100 %
    temperature: float        # Celsius
    network_in: float         # KB/s
    network_out: float        # KB/s
    disk_io: float            # MB/s
    error_rate: float         # 0–1
    latency_ms: float         # milliseconds
    anomaly_type: AnomalyType = AnomalyType.NONE
    is_anomaly: bool = False

    def to_feature_vector(self) -> list[float]:
        """Return the numeric features used for ML."""
        return [
            self.cpu_usage,
            self.memory_usage,
            self.temperature,
            self.network_in,
            self.network_out,
            self.disk_io,
            self.error_rate,
            self.latency_ms,
        ]

    def to_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "cpu_usage": round(self.cpu_usage, 2),
            "memory_usage": round(self.memory_usage, 2),
            "temperature": round(self.temperature, 2),
            "network_in": round(self.network_in, 2),
            "network_out": round(self.network_out, 2),
            "disk_io": round(self.disk_io, 2),
            "error_rate": round(self.error_rate, 4),
            "latency_ms": round(self.latency_ms, 2),
            "anomaly_type": self.anomaly_type.value,
            "is_anomaly": self.is_anomaly,
        }


class SensorSimulator:
    """
    Generates realistic telemetry for a single edge device.

    Normal behaviour follows correlated Gaussian distributions with
    time-of-day patterns. Anomalies are injected probabilistically.
    """

    FEATURE_NAMES = [
        "cpu_usage", "memory_usage", "temperature",
        "network_in", "network_out", "disk_io",
        "error_rate", "latency_ms",
    ]

    def __init__(
        self,
        device_id: str,
        anomaly_probability: float = 0.05,
        seed: Optional[int] = None,
    ):
        self.device_id = device_id
        self.anomaly_probability = anomaly_probability
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._drift_active = False
        self._drift_duration = 0
        self._freeze_value: Optional[np.ndarray] = None
        self._freeze_duration = 0

    # ------------------------------------------------------------------
    # Normal baseline (correlated, time-aware)
    # ------------------------------------------------------------------
    def _normal_reading(self) -> np.ndarray:
        """Simulate a correlated normal reading."""
        hour = (self._step / 12) % 24  # 5-second intervals → hourly pattern
        # Daytime load factor (higher during 8–18h)
        load = 0.5 + 0.3 * np.sin(np.pi * max(0, hour - 6) / 12)

        cpu = np.clip(self._rng.normal(30 * load + 20, 8), 5, 95)
        mem = np.clip(self._rng.normal(45 + 10 * load, 7), 20, 90)
        temp = np.clip(self._rng.normal(40 + 0.25 * cpu, 3), 25, 85)
        net_in = np.clip(self._rng.exponential(50 * (1 + load)), 0.5, 800)
        net_out = np.clip(self._rng.exponential(30 * (1 + load)), 0.5, 600)
        disk_io = np.clip(self._rng.exponential(5 + 2 * load), 0.1, 120)
        error_rate = np.clip(self._rng.beta(0.5, 50), 0, 0.15)
        latency = np.clip(self._rng.lognormal(3.5, 0.4), 10, 500)

        return np.array([cpu, mem, temp, net_in, net_out, disk_io, error_rate, latency])

    # ------------------------------------------------------------------
    # Anomaly injection
    # ------------------------------------------------------------------
    def _inject_anomaly(self, base: np.ndarray) -> tuple[np.ndarray, AnomalyType]:
        """Choose and apply a random anomaly pattern."""
        atype = self._rng.choice(list(AnomalyType)[:-1])  # exclude NONE

        if atype == AnomalyType.SPIKE:
            # Sudden spike on CPU, temp and error_rate
            base[0] = np.clip(base[0] + self._rng.uniform(40, 65), 0, 100)
            base[2] = np.clip(base[2] + self._rng.uniform(20, 40), 0, 105)
            base[6] = np.clip(base[6] + self._rng.uniform(0.3, 0.8), 0, 1)
            base[7] = np.clip(base[7] * self._rng.uniform(3, 10), 0, 5000)

        elif atype == AnomalyType.DROP:
            # Sudden drop in network traffic (possible link failure)
            base[3] = self._rng.uniform(0.01, 1.0)
            base[4] = self._rng.uniform(0.01, 0.5)
            base[5] = self._rng.uniform(0.01, 0.3)

        elif atype == AnomalyType.DRIFT:
            # Slow drift — memory leak / gradual temp increase
            if not self._drift_active:
                self._drift_active = True
                self._drift_duration = self._rng.integers(5, 20)
            drift_factor = 1 + (self._drift_duration / 20) * 0.6
            base[1] = np.clip(base[1] * drift_factor, 0, 100)
            base[2] = np.clip(base[2] * (1 + drift_factor * 0.1), 0, 105)
            self._drift_duration -= 1
            if self._drift_duration <= 0:
                self._drift_active = False

        elif atype == AnomalyType.FREEZE:
            # Sensor freeze: same values for several cycles
            if self._freeze_value is None:
                self._freeze_value = base.copy()
                self._freeze_duration = self._rng.integers(3, 8)
            base = self._freeze_value.copy()
            self._freeze_duration -= 1
            if self._freeze_duration <= 0:
                self._freeze_value = None

        return base, atype

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def next_reading(self) -> SensorReading:
        """Generate the next sensor reading."""
        self._step += 1
        values = self._normal_reading()
        is_anomaly = False
        atype = AnomalyType.NONE

        if self._rng.random() < self.anomaly_probability:
            values, atype = self._inject_anomaly(values)
            is_anomaly = True

        return SensorReading(
            device_id=self.device_id,
            timestamp=time.time(),
            cpu_usage=float(values[0]),
            memory_usage=float(values[1]),
            temperature=float(values[2]),
            network_in=float(values[3]),
            network_out=float(values[4]),
            disk_io=float(values[5]),
            error_rate=float(values[6]),
            latency_ms=float(values[7]),
            anomaly_type=atype,
            is_anomaly=is_anomaly,
        )

    def generate_batch(self, n: int) -> list[SensorReading]:
        """Generate a batch of n readings (useful for training)."""
        return [self.next_reading() for _ in range(n)]
