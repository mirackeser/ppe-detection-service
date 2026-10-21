"""Small per-process Prometheus counters/histogram; worker updates stay on event loop."""
from dataclasses import dataclass, field
import math

BUCKETS = (.005, .01, .025, .05, .1, .25, .5, 1, 2, 5, 10, math.inf)


@dataclass
class Metrics:
    completed: int = 0
    rejected: int = 0
    invalid: int = 0
    failed: int = 0
    dropped: int = 0
    events: int = 0
    duration_sum: float = 0
    buckets: list[int] = field(default_factory=lambda: [0] * len(BUCKETS))

    def observe(self, seconds: float):
        self.completed += 1
        self.duration_sum += seconds
        for index, boundary in enumerate(BUCKETS):
            if seconds <= boundary:
                self.buckets[index] += 1

    def render(self, busy: bool):
        lines = ["# HELP ppe_inference_completed_total Successfully processed frames.",
                 "# TYPE ppe_inference_completed_total counter", f"ppe_inference_completed_total {self.completed}"]
        for name, value in (("rejected", self.rejected), ("invalid", self.invalid),
                            ("failed", self.failed), ("dropped_frames", self.dropped), ("events", self.events)):
            lines += [f"# TYPE ppe_{name}_total counter", f"ppe_{name}_total {value}"]
        lines += ["# TYPE ppe_worker_busy gauge", f"ppe_worker_busy {int(busy)}",
                  "# HELP ppe_pipeline_seconds Decode, prediction and assessment latency, excluding transport.",
                  "# TYPE ppe_pipeline_seconds histogram"]
        for boundary, count in zip(BUCKETS, self.buckets):
            label = "+Inf" if math.isinf(boundary) else str(boundary)
            lines.append(f'ppe_pipeline_seconds_bucket{{le="{label}"}} {count}')
        lines += [f"ppe_pipeline_seconds_sum {self.duration_sum}",
                  f"ppe_pipeline_seconds_count {self.completed}"]
        return "\n".join(lines) + "\n"
