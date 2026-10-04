"""
EdgeOps AI — Multi-Node Edge Simulator
Launches multiple edge agents concurrently via asyncio.
"""
import asyncio
import argparse
import os
import sys
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent))

from edge_agent import EdgeAgent


async def run_all_agents(
    n_nodes: int,
    cloud_url: str,
    interval: float,
    contamination: float,
    training_samples: int,
    base_anomaly_prob: float,
) -> None:
    """Spin up n_nodes edge agents concurrently."""
    agents = [
        EdgeAgent(
            device_id=f"edge-{i+1:03d}",
            cloud_api_url=cloud_url,
            interval_seconds=interval,
            contamination=contamination,
            training_samples=training_samples,
            # Slightly vary anomaly probability per node for realism
            anomaly_probability=base_anomaly_prob * (0.5 + 0.1 * i),
            api_key=os.getenv("CLOUD_API_KEY", ""),
        )
        for i in range(n_nodes)
    ]

    logger.info(f"🚀 Starting {n_nodes} edge nodes → {cloud_url}")

    tasks = [asyncio.create_task(agent.run(), name=agent.device_id) for agent in agents]

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        logger.info("Shutting down edge nodes...")
    finally:
        for agent in agents:
            await agent.close()


def main():
    parser = argparse.ArgumentParser(description="EdgeOps AI — Multi-Node Simulator")
    parser.add_argument("--nodes", type=int, default=int(os.getenv("NUM_EDGE_NODES", "5")),
                        help="Number of simulated edge nodes")
    parser.add_argument("--cloud-url", default=os.getenv("CLOUD_API_URL", "http://localhost:8000"),
                        help="Cloud backend URL")
    parser.add_argument("--interval", type=float, default=float(os.getenv("EDGE_SEND_INTERVAL", "5")),
                        help="Seconds between readings")
    parser.add_argument("--contamination", type=float, default=float(os.getenv("EDGE_CONTAMINATION", "0.05")),
                        help="IsolationForest contamination rate")
    parser.add_argument("--training-samples", type=int,
                        default=int(os.getenv("EDGE_TRAINING_SAMPLES", "200")),
                        help="Samples before model is trained")
    parser.add_argument("--anomaly-prob", type=float, default=0.08,
                        help="Base anomaly injection probability per node")
    args = parser.parse_args()

    logger.configure(
        handlers=[{"sink": sys.stdout, "colorize": True, "level": "INFO"}]
    )

    try:
        asyncio.run(
            run_all_agents(
                n_nodes=args.nodes,
                cloud_url=args.cloud_url,
                interval=args.interval,
                contamination=args.contamination,
                training_samples=args.training_samples,
                base_anomaly_prob=args.anomaly_prob,
            )
        )
    except KeyboardInterrupt:
        logger.info("Simulator stopped.")


if __name__ == "__main__":
    main()
