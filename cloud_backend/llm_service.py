"""
EdgeOps AI — Ollama LLM Service
Generates human-readable anomaly explanations and recommended actions.
"""
import os
import json
from datetime import datetime
from typing import Optional
from loguru import logger

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

# Severity thresholds
SEVERITY_THRESHOLDS = {
    "low": 0.20,
    "medium": 0.35,
    "high": 0.55,
    "critical": 0.75,
}


def score_to_severity(score: float) -> str:
    """Map normalized anomaly score to severity label."""
    if score >= SEVERITY_THRESHOLDS["critical"]:
        return "critical"
    elif score >= SEVERITY_THRESHOLDS["high"]:
        return "high"
    elif score >= SEVERITY_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "low"


def build_prompt(event: dict) -> str:
    """Build a structured prompt for anomaly explanation with strict context constraints."""
    severity = score_to_severity(event.get("normalized_score", 0))
    raw_type = (event.get("anomaly_type") or "unknown").lower()
    anomaly_type = "statistical deviation" if raw_type in ("none", "unknown", "") else raw_type

    # Identify primary outliers to ground the LLM
    deviations = []
    if event.get("cpu_usage", 0) > 75:
        deviations.append(f"CPU at {event['cpu_usage']:.1f}%")
    if event.get("memory_usage", 0) > 75:
        deviations.append(f"Memory at {event['memory_usage']:.1f}%")
    if event.get("temperature", 0) > 70:
        deviations.append(f"Temperature at {event['temperature']:.1f}°C")
    if event.get("error_rate", 0) > 0.05:
        deviations.append(f"Error rate at {event['error_rate']:.2%}")
    if event.get("latency_ms", 0) > 150:
        deviations.append(f"Latency at {event['latency_ms']:.1f}ms")
    if event.get("network_in", 0) < 1.0 and event.get("network_out", 0) < 1.0:
        deviations.append("Network traffic dropped to near zero")

    deviations_str = ", ".join(deviations) if deviations else "multi-metric correlation shift"

    return f"""You are a senior Edge SRE engineer analyzing IoT gateway telemetry.
An anomaly was detected on node '{event.get("device_id", "unknown")}' at {datetime.fromtimestamp(event.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S')}.

ANOMALY PROFILE:
- Classified Pattern: {anomaly_type.upper()}
- Assessed Severity: {severity.upper()}
- Anomaly Score: {event.get("normalized_score", 0):.3f} (0=baseline, 1=extreme outlier)
- Isolation Forest Confidence: {event.get("confidence", 0):.1%}
- Key Outliers Identified: {deviations_str}

LIVE SENSOR READINGS:
- CPU Usage: {event.get("cpu_usage", 0):.1f}%
- Memory Usage: {event.get("memory_usage", 0):.1f}%
- Temperature: {event.get("temperature", 0):.1f}°C
- Network IN / OUT: {event.get("network_in", 0):.2f} / {event.get("network_out", 0):.2f} KB/s
- Disk I/O: {event.get("disk_io", 0):.2f} MB/s
- Error Rate: {event.get("error_rate", 0):.2%}
- Response Latency: {event.get("latency_ms", 0):.1f} ms

CRITICAL GUIDELINES:
1. Base your explanation strictly on the specific sensor numbers above ({deviations_str}).
2. DO NOT use vague generic phrases like "temporary spike", "routine maintenance", or "software update".
3. Provide a concrete technical hypothesis (e.g., thread starvation, memory leak accumulation, thermal throttling, packet drop on cellular backhaul, sensor freeze).
4. Provide 2-3 precise, actionable remediation commands or steps.

Respond with a JSON object only (no markdown wrapping, no comments):
{{
  "explanation": "<concise 2-3 sentence technical analysis citing the exact anomalous numbers>",
  "recommended_action": "<concrete diagnostic and mitigation steps for an edge node>",
  "severity": "{severity}",
  "root_cause_hypothesis": "<precise root cause tied to the metrics>",
  "urgency": "<immediate|within_1h|within_24h|monitor>"
}}"""


class LLMService:
    """
    Wrapper around Ollama for generating anomaly explanations.
    Falls back to rule-based explanations if Ollama is unavailable.
    """

    def __init__(self):
        self.model = OLLAMA_MODEL
        self.available = self._check_availability()

    def _check_availability(self) -> bool:
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama Python package not installed.")
            return False
        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            client.list()
            logger.info(f"✓ Ollama available at {OLLAMA_BASE_URL}, model={self.model}")
            return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}. Using rule-based fallback.")
            return False

    def explain(self, event: dict) -> dict:
        """
        Generate explanation for an anomaly event.

        Returns:
            dict with keys: explanation, recommended_action, severity,
                            root_cause_hypothesis, urgency
        """
        if self.available:
            return self._ollama_explain(event)
        else:
            return self._rule_based_explain(event)

    def _ollama_explain(self, event: dict) -> dict:
        """Call Ollama to generate explanation."""
        prompt = build_prompt(event)
        try:
            client = ollama.Client(host=OLLAMA_BASE_URL)
            response = client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.3, "num_predict": 512},
            )
            content = response["message"]["content"].strip()

            # Extract JSON (handle possible markdown wrapping)
            if "```" in content:
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            result = json.loads(content)
            result["source"] = "ollama"
            result["model"] = self.model
            return result

        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}. Using fallback.")
            return self._rule_based_explain(event)
        except Exception as e:
            logger.error(f"Ollama call failed: {e}")
            return self._rule_based_explain(event)

    def _rule_based_explain(self, event: dict) -> dict:
        """
        Rule-based fallback when Ollama is not available.
        Produces deterministic, human-readable explanations based on sensor values.
        """
        anomaly_type = event.get("anomaly_type", "unknown")
        score = event.get("normalized_score", 0)
        severity = score_to_severity(score)

        # Identify dominant metrics
        issues = []
        if event.get("cpu_usage", 0) > 85:
            issues.append(f"high CPU usage ({event['cpu_usage']:.1f}%)")
        if event.get("memory_usage", 0) > 85:
            issues.append(f"high memory usage ({event['memory_usage']:.1f}%)")
        if event.get("temperature", 0) > 75:
            issues.append(f"elevated temperature ({event['temperature']:.1f}°C)")
        if event.get("error_rate", 0) > 0.1:
            issues.append(f"high error rate ({event['error_rate']:.1%})")
        if event.get("latency_ms", 0) > 300:
            issues.append(f"high latency ({event['latency_ms']:.0f}ms)")
        if event.get("network_in", 0) < 1 and event.get("network_out", 0) < 1:
            issues.append("near-zero network traffic (possible link failure)")

        issue_str = ", ".join(issues) if issues else "multiple metrics out of range"

        explanations = {
            "spike": f"A sudden spike anomaly was detected showing {issue_str}. "
                     f"This suggests a sudden workload burst, process runaway, or hardware thermal event.",
            "drop": f"A traffic drop anomaly was detected with {issue_str}. "
                    f"This may indicate a network failure, service crash, or power issue.",
            "drift": f"A gradual drift anomaly was detected with {issue_str}. "
                     f"This pattern often indicates a memory leak, disk filling up, or thermal throttling.",
            "freeze": f"A sensor freeze anomaly was detected — readings are static, suggesting "
                      f"sensor malfunction, process hang, or data pipeline issue.",
            "statistical_deviation": f"A multi-metric statistical deviation was detected with {issue_str}. "
                                     f"Telemetry points lie significantly outside the normal feature distribution.",
            "none": f"A subtle statistical outlier was detected by the Isolation Forest ({issue_str}).",
        }

        actions = {
            "spike": "1. Check running processes (top/htop). 2. Review recent deployments. "
                     "3. Inspect system logs for errors. 4. Consider scaling if load is legitimate.",
            "drop": "1. Verify network connectivity (ping, traceroute). 2. Check service health. "
                    "3. Review firewall rules. 4. Inspect hardware connections.",
            "drift": "1. Check for memory leaks (heap profiler). 2. Review disk usage. "
                     "3. Monitor for thermal throttling. 4. Schedule maintenance window.",
            "freeze": "1. Verify sensor/agent is responsive. 2. Restart monitoring agent. "
                      "3. Check data pipeline health. 4. Run hardware diagnostics.",
            "statistical_deviation": "1. Inspect joint metric distribution. 2. Monitor for recurrence. "
                                     "3. Verify sensor calibration and edge node clock sync.",
            "none": "1. Check baseline telemetry. 2. Monitor next readings to confirm transient fluctuation.",
        }

        return {
            "explanation": explanations.get(anomaly_type, f"Anomaly detected: {issue_str}."),
            "recommended_action": actions.get(anomaly_type, "Investigate device logs and metrics."),
            "severity": severity,
            "root_cause_hypothesis": f"{anomaly_type.title()} pattern with {issue_str}",
            "urgency": "immediate" if severity == "critical" else "within_1h" if severity == "high" else "within_24h",
            "source": "rule_based",
            "model": "rule_based_fallback",
        }

    def batch_explain(self, events: list[dict]) -> list[dict]:
        """Explain multiple events (serial, not parallel to avoid Ollama overload)."""
        return [self.explain(event) for event in events]


# Singleton
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service
