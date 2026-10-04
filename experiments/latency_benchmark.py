"""
EdgeOps AI — Latency Benchmark
Measures end-to-end detection latency across different model configurations.
"""
import sys
import time
import statistics
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent.parent / "edge_node"))

from sensor import SensorSimulator
from anomaly_detector import EdgeAnomalyDetector

console = Console()


def benchmark_detector(
    n_samples: int = 1000,
    contamination: float = 0.05,
    training_samples: int = 200,
    n_estimators_list: list[int] = None,
) -> list[dict]:
    if n_estimators_list is None:
        n_estimators_list = [50, 100, 200]

    results = []
    simulator = SensorSimulator("bench-device", anomaly_probability=0.1, seed=42)
    readings = [simulator.next_reading() for _ in range(n_samples)]

    for n_est in n_estimators_list:
        console.print(f"\n[cyan]Benchmarking n_estimators={n_est}...[/cyan]")

        detector = EdgeAnomalyDetector(
            device_id=f"bench-{n_est}",
            contamination=contamination,
            training_samples=training_samples,
        )
        # Monkey-patch n_estimators
        detector._pipeline = None

        latencies = []
        anomalies_detected = 0
        warmup_done = False

        for reading in readings:
            t_start = time.perf_counter()
            result = detector.process(reading)
            t_end = time.perf_counter()

            if result is not None:
                if not warmup_done:
                    warmup_done = True
                    console.print(f"  Model trained after {detector._total_seen} samples")
                latency_ms = (t_end - t_start) * 1000
                latencies.append(latency_ms)
                if result.is_anomaly:
                    anomalies_detected += 1

        if latencies:
            results.append({
                "n_estimators": n_est,
                "samples": len(latencies),
                "anomalies_detected": anomalies_detected,
                "anomaly_rate": anomalies_detected / len(latencies),
                "latency_mean_ms": round(statistics.mean(latencies), 4),
                "latency_median_ms": round(statistics.median(latencies), 4),
                "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
                "latency_p99_ms": round(sorted(latencies)[int(len(latencies) * 0.99)], 4),
                "latency_max_ms": round(max(latencies), 4),
            })

    return results


def benchmark_training_size(
    contamination: float = 0.05,
    training_sizes: list[int] = None,
) -> list[dict]:
    if training_sizes is None:
        training_sizes = [50, 100, 200, 500, 1000]

    results = []
    simulator = SensorSimulator("bench-train", anomaly_probability=0.1, seed=42)
    max_samples = max(training_sizes) + 200

    readings = [simulator.next_reading() for _ in range(max_samples)]

    for ts in training_sizes:
        detector = EdgeAnomalyDetector(
            device_id=f"bench-ts-{ts}",
            contamination=contamination,
            training_samples=ts,
        )

        latencies = []
        for reading in readings:
            t_start = time.perf_counter()
            result = detector.process(reading)
            t_end = time.perf_counter()
            if result is not None:
                latencies.append((t_end - t_start) * 1000)

        if latencies:
            results.append({
                "training_samples": ts,
                "samples_processed": len(latencies),
                "latency_mean_ms": round(statistics.mean(latencies), 4),
                "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
            })

    return results


def print_results(results: list[dict], title: str) -> None:
    console.print(f"\n[bold yellow]{title}[/bold yellow]")
    if not results:
        console.print("[red]No results.[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    for key in results[0].keys():
        table.add_column(key.replace("_", " ").title(), justify="right")

    for row in results:
        table.add_row(*[str(v) for v in row.values()])

    console.print(table)


def main():
    console.print("[bold green]═══ EdgeOps AI — Latency Benchmark ═══[/bold green]\n")

    # Benchmark 1: n_estimators impact
    console.print("[bold]Experiment 1: Impact of n_estimators on detection latency[/bold]")
    est_results = benchmark_detector(
        n_samples=500,
        training_samples=200,
        n_estimators_list=[50, 100, 200],
    )
    print_results(est_results, "n_estimators Benchmark")

    # Benchmark 2: Training size impact
    console.print("\n[bold]Experiment 2: Impact of training set size[/bold]")
    ts_results = benchmark_training_size(training_sizes=[50, 100, 200, 500])
    print_results(ts_results, "Training Size Benchmark")

    # Save results
    output = {"n_estimators_benchmark": est_results, "training_size_benchmark": ts_results}
    output_path = Path(__file__).parent / "results" / "latency_benchmark.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
