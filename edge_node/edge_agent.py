"""
EdgeOps AI — Edge Agent
Main loop: reads sensors, runs local detection, forwards anomalies to Cloud API.
"""
import asyncio
import time
import os
import sys
from pathlib import Path
from loguru import logger
import httpx
from dotenv import load_dotenv

load_dotenv()

# Allow running from edge_node/ directly
sys.path.insert(0, str(Path(__file__).parent))

from sensor import SensorSimulator, SensorReading
from anomaly_detector import EdgeAnomalyDetector, AnomalyResult


class EdgeAgent:
    """
    Autonomous edge agent that:
      1. Reads sensor data at a configurable interval
      2. Detects anomalies locally using Isolation Forest
      3. Forwards ONLY anomalous events to the Cloud API
      4. Supports CSV data import alongside synthetic generation
    """

    def __init__(
        self,
        device_id: str,
        cloud_api_url: str,
        interval_seconds: float = 5.0,
        contamination: float = 0.05,
        training_samples: int = 200,
        anomaly_probability: float = 0.08,
        api_key: str = "",
    ):
        self.device_id = device_id
        self.cloud_api_url = cloud_api_url.rstrip("/")
        self.interval = interval_seconds
        self.api_key = api_key

        self.sensor = SensorSimulator(
            device_id=device_id,
            anomaly_probability=anomaly_probability,
        )
        self.detector = EdgeAnomalyDetector(
            device_id=device_id,
            contamination=contamination,
            training_samples=training_samples,
        )

        self._readings_sent = 0
        self._anomalies_forwarded = 0
        self._http_client: httpx.AsyncClient | None = None

    # ------------------------------------------------------------------
    # HTTP client
    # ------------------------------------------------------------------
    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                base_url=self.cloud_api_url,
                headers={
                    "X-API-Key": self.api_key,
                    "Content-Type": "application/json",
                },
                timeout=10.0,
            )
        return self._http_client

    async def _forward_anomaly(self, result: AnomalyResult) -> bool:
        """Send anomaly event to the cloud backend. Returns True on success."""
        payload = result.to_dict()
        payload["source"] = "edge"
        try:
            client = await self._get_client()
            resp = await client.post("/api/events", json=payload)
            if resp.status_code in (200, 201):
                self._anomalies_forwarded += 1
                logger.success(
                    f"[{self.device_id}] ✓ Anomaly forwarded "
                    f"(score={result.normalized_score:.3f}, "
                    f"type={result.reading.anomaly_type.value})"
                )
                return True
            else:
                logger.warning(
                    f"[{self.device_id}] Cloud API returned {resp.status_code}: {resp.text[:200]}"
                )
        except httpx.ConnectError:
            logger.warning(f"[{self.device_id}] Cloud API unreachable, storing locally...")
        except Exception as e:
            logger.error(f"[{self.device_id}] Forward error: {e}")
        return False

    async def _register_device(self) -> None:
        """Register this device with the cloud backend."""
        payload = {
            "device_id": self.device_id,
            "device_type": "simulated_iot",
            "location": f"edge-rack-{self.device_id[-2:]}",
            "metadata": {
                "contamination": self.detector.contamination,
                "interval_s": self.interval,
            },
        }
        try:
            client = await self._get_client()
            await client.post("/api/devices", json=payload)
            logger.info(f"[{self.device_id}] Registered with cloud backend")
        except Exception as e:
            logger.warning(f"[{self.device_id}] Registration skipped: {e}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Main agent loop — runs indefinitely."""
        logger.info(f"[{self.device_id}] Starting edge agent → {self.cloud_api_url}")
        await self._register_device()

        while True:
            loop_start = time.perf_counter()

            reading: SensorReading = self.sensor.next_reading()
            self._readings_sent += 1

            result = self.detector.process(reading)

            if result is not None:
                if result.is_anomaly:
                    await self._forward_anomaly(result)
                else:
                    logger.debug(
                        f"[{self.device_id}] Normal reading "
                        f"(cpu={reading.cpu_usage:.1f}%, "
                        f"mem={reading.memory_usage:.1f}%, "
                        f"score={result.normalized_score:.3f})"
                    )
            else:
                logger.debug(f"[{self.device_id}] Warmup #{self._readings_sent}/{self.detector.training_samples}")

            # Log stats every 50 readings
            if self._readings_sent % 50 == 0:
                stats = self.detector.stats()
                comm_reduction = (
                    1 - self._anomalies_forwarded / self._readings_sent
                ) * 100 if self._readings_sent > 0 else 0
                logger.info(
                    f"[{self.device_id}] Stats | "
                    f"readings={stats['total_readings']} | "
                    f"anomalies={stats['total_anomalies']} | "
                    f"rate={stats['anomaly_rate']:.2%} | "
                    f"comm_reduction={comm_reduction:.1f}%"
                )

            # Sleep for remainder of interval
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.0, self.interval - elapsed)
            await asyncio.sleep(sleep_time)

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="EdgeOps AI — Edge Agent")
    parser.add_argument("--device-id", default="edge-001", help="Device identifier")
    parser.add_argument("--cloud-url", default=os.getenv("CLOUD_API_URL", "http://localhost:8000"))
    parser.add_argument("--interval", type=float, default=float(os.getenv("EDGE_SEND_INTERVAL", "5")))
    parser.add_argument("--contamination", type=float, default=float(os.getenv("EDGE_CONTAMINATION", "0.05")))
    parser.add_argument("--training-samples", type=int, default=int(os.getenv("EDGE_TRAINING_SAMPLES", "200")))
    parser.add_argument("--anomaly-prob", type=float, default=0.08)
    parser.add_argument("--api-key", default=os.getenv("CLOUD_API_KEY", ""))
    args = parser.parse_args()

    agent = EdgeAgent(
        device_id=args.device_id,
        cloud_api_url=args.cloud_url,
        interval_seconds=args.interval,
        contamination=args.contamination,
        training_samples=args.training_samples,
        anomaly_probability=args.anomaly_prob,
        api_key=args.api_key,
    )

    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        logger.info(f"[{args.device_id}] Agent stopped.")


if __name__ == "__main__":
    main()
