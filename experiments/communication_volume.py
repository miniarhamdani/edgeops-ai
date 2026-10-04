"""
EdgeOps AI — Communication Volume Experiment
Measures bandwidth reduction from edge-side anomaly filtering.
"""
import sys
import json
import statistics
from pathlib import Path
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "edge_node"))

from sensor import SensorSimulator
from anomaly_detector import EdgeAnomalyDetector

console = Console()

# Estimated payload sizes in bytes
FULL_READING_BYTES = 512      # All sensor metrics + metadata (JSON)
ANOMALY_EVENT_BYTES = 768     # Anomaly event with detection scores (JSON)
OVERHEAD_BYTES = 200          # HTTP headers + overhead


def simulate_communication_volume(
    n_readings: int = 2000,
    n_nodes: int = 5,
    contamination_rates: list[float] = None,
    anomaly_probs: list[float] = None,
) -> list[dict]:
    """
    Compare baseline (send every reading) vs edge-filtered (send only anomalies).
    """
    if contamination_rates is None:
        contamination_rates = [0.02, 0.05, 0.10]
    if anomaly_probs is None:
        anomaly_probs = [0.05, 0.08, 0.15]

    results = []

    for contamination, anomaly_prob in zip(contamination_rates, anomaly_probs):
        total_readings = 0
        total_anomalies_detected = 0
        total_gt_anomalies = 0

        for node_idx in range(n_nodes):
            simulator = SensorSimulator(
                device_id=f"sim-{node_idx:03d}",
                anomaly_probability=anomaly_prob,
                seed=node_idx * 42,
            )
            detector = EdgeAnomalyDetector(
                device_id=f"sim-{node_idx:03d}",
                contamination=contamination,
                training_samples=100,
            )

            for _ in range(n_readings):
                reading = simulator.next_reading()
                result = detector.process(reading)

                total_readings += 1
                if reading.is_anomaly:
                    total_gt_anomalies += 1
                if result and result.is_anomaly:
                    total_anomalies_detected += 1

        # Compute volumes
        baseline_bytes = total_readings * (FULL_READING_BYTES + OVERHEAD_BYTES)
        edge_filtered_bytes = total_anomalies_detected * (ANOMALY_EVENT_BYTES + OVERHEAD_BYTES)
        reduction_pct = (1 - edge_filtered_bytes / baseline_bytes) * 100 if baseline_bytes > 0 else 0

        true_positives = min(total_anomalies_detected, total_gt_anomalies)
        false_negatives = max(0, total_gt_anomalies - total_anomalies_detected)
        false_positives = max(0, total_anomalies_detected - total_gt_anomalies)
        precision = true_positives / total_anomalies_detected if total_anomalies_detected > 0 else 0
        recall = true_positives / total_gt_anomalies if total_gt_anomalies > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            "contamination": contamination,
            "anomaly_prob": anomaly_prob,
            "n_nodes": n_nodes,
            "total_readings": total_readings,
            "gt_anomalies": total_gt_anomalies,
            "detected_anomalies": total_anomalies_detected,
            "baseline_kb": round(baseline_bytes / 1024, 1),
            "edge_filtered_kb": round(edge_filtered_bytes / 1024, 1),
            "reduction_pct": round(reduction_pct, 2),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1_score": round(f1, 3),
        })

    return results


def print_results(results: list[dict]) -> None:
    table = Table(show_header=True, header_style="bold magenta", title="Communication Volume Results")
    headers = [
        "Contamin.", "Anom.Prob", "Readings", "GT Anom.", "Detected",
        "Baseline KB", "Filtered KB", "Reduction %", "Precision", "Recall", "F1"
    ]
    for h in headers:
        table.add_column(h, justify="right")

    for r in results:
        table.add_row(
            str(r["contamination"]),
            str(r["anomaly_prob"]),
            str(r["total_readings"]),
            str(r["gt_anomalies"]),
            str(r["detected_anomalies"]),
            str(r["baseline_kb"]),
            str(r["edge_filtered_kb"]),
            f"[green]{r['reduction_pct']}%[/green]",
            str(r["precision"]),
            str(r["recall"]),
            str(r["f1_score"]),
        )

    console.print(table)


def main():
    console.print("[bold green]═══ EdgeOps AI — Communication Volume Experiment ═══[/bold green]\n")

    results = simulate_communication_volume(
        n_readings=1000,
        n_nodes=5,
    )
    print_results(results)

    avg_reduction = statistics.mean(r["reduction_pct"] for r in results)
    avg_f1 = statistics.mean(r["f1_score"] for r in results)

    console.print(f"\n[bold cyan]Average communication reduction: {avg_reduction:.1f}%[/bold cyan]")
    console.print(f"[bold cyan]Average F1 score: {avg_f1:.3f}[/bold cyan]")

    output_path = Path(__file__).parent / "results" / "communication_volume.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
