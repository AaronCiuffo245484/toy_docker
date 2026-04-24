# src/ml_toy_repo/state.py
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class JobStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class TrainingMetrics:
    device: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    final_accuracy: Optional[float] = None
    epochs: Optional[int] = None


@dataclass
class JobState:
    status: JobStatus = JobStatus.IDLE
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    error: Optional[str] = None

    def reset(self) -> None:
        self.status = JobStatus.IDLE
        self.metrics = TrainingMetrics()
        self.error = None


job_state = JobState()