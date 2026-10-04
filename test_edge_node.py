"""
EdgeOps AI — Quick validation test (no external dependencies beyond scikit-learn/numpy/pandas)
Verifies: sensor generation, anomaly detection, scoring
"""
import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent / "edge_node"))

from sensor import SensorSimulator, AnomalyType
from anomaly_detector import EdgeAnomalyDetector


def test_sensor_generation():
    print("Testing sensor generation...")
    sim = SensorSimulator("test-001", anomaly_probability=0.2, seed=42)
    readings = sim.generate_batch(50)

    assert len(readings) == 50
    for r in readings:
        assert 0 <= r.cpu_usage <= 100
        assert 0 <= r.memory_usage <= 100
        assert r.temperature >= 0
        assert r.error_rate >= 0

    anomalies = [r for r in readings if r.is_anomaly]
    print(f"  ✓ Generated 50 readings, {len(anomalies)} anomalies injected")
    return True


def test_anomaly_detection():
    print("Testing Isolation Forest detection...")
    sim = SensorSimulator("test-002", anomaly_probability=0.10, seed=99)
    detector = EdgeAnomalyDetector(
        device_id="test-002",
        contamination=0.05,
        training_samples=100,
        model_path=Path("edge_node/models/test-002.pkl"),
    )

    results = []
    warmup_done = False
    for i in range(300):
        reading = sim.next_reading()
        result = detector.process(reading)
        if result is not None:
            if not warmup_done:
                print(f"  ✓ Model trained after {i+1} samples")
                warmup_done = True
            results.append(result)

    assert len(results) > 0, "No results produced!"
    anomalies = [r for r in results if r.is_anomaly]
    normals = [r for r in results if not r.is_anomaly]

    print(f"  ✓ Processed {len(results)} readings post-training")
    print(f"  ✓ Detected {len(anomalies)} anomalies ({len(anomalies)/len(results):.1%})")
    print(f"  ✓ Normal readings: {len(normals)}")

    # Check latencies are reasonable
    latencies = [r.latency_ms for r in results]
    avg_lat = sum(latencies) / len(latencies)
    max_lat = max(latencies)
    print(f"  ✓ Avg detection latency: {avg_lat:.4f}ms | Max: {max_lat:.4f}ms")
    assert avg_lat < 100, f"Latency too high: {avg_lat}ms"

    # Check scores
    for r in results:
        assert 0.0 <= r.normalized_score <= 1.0
        assert 0.0 <= r.confidence <= 1.0

    return True


def test_serialization():
    print("Testing serialization...")
    from anomaly_detector import AnomalyResult
    sim = SensorSimulator("test-003", seed=42)
    reading = sim.next_reading()

    fv = reading.to_feature_vector()
    d = reading.to_dict()

    assert len(fv) == 8
    assert "device_id" in d
    assert "cpu_usage" in d
    print(f"  ✓ Feature vector: {len(fv)} dims")
    print(f"  ✓ Dict keys: {list(d.keys())}")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  EdgeOps AI — Validation Test")
    print("=" * 50)

    tests = [
        ("Sensor Generation", test_sensor_generation),
        ("Anomaly Detection", test_anomaly_detection),
        ("Serialization", test_serialization),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n[{name}]")
        try:
            fn()
            print(f"  PASSED ✓")
            passed += 1
        except Exception as e:
            print(f"  FAILED ✗ — {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"  Results: {passed}/{len(tests)} tests passed")
    print("=" * 50)
