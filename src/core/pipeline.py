"""
Pipeline orchestration for the Semantic Codebase Graph Engine.

Implements a state-machine based pipeline that coordinates all
processing stages with clear input/output contracts, serialization
support, and graceful failure handling.
"""

import logging
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.config import PipelineConfig, Config
from src.core.exceptions import PipelineError

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result from a pipeline stage execution."""

    stage_name: str
    status: StageStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    output: Any = None
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage_name": self.stage_name,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "metrics": self.metrics,
        }


@dataclass
class PipelineState:
    """
    Maintains the complete state of a pipeline execution.
    
    Supports serialization for checkpointing and resumption.
    """

    pipeline_id: str
    source_repo: str
    target_repo: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    stage_results: Dict[str, StageResult] = field(default_factory=dict)
    current_stage: Optional[str] = None
    data: Dict[str, Any] = field(default_factory=dict)

    def get_stage_status(self, stage_name: str) -> StageStatus:
        """Get the status of a specific stage."""
        if stage_name in self.stage_results:
            return self.stage_results[stage_name].status
        return StageStatus.PENDING

    def is_stage_completed(self, stage_name: str) -> bool:
        """Check if a stage has completed successfully."""
        return self.get_stage_status(stage_name) == StageStatus.COMPLETED

    def record_stage_start(self, stage_name: str) -> None:
        """Record that a stage has started."""
        self.current_stage = stage_name
        self.stage_results[stage_name] = StageResult(
            stage_name=stage_name,
            status=StageStatus.RUNNING,
            started_at=datetime.now(),
        )

    def record_stage_completion(
        self, stage_name: str, output: Any, metrics: Dict[str, Any] = None
    ) -> None:
        """Record that a stage has completed successfully."""
        if stage_name in self.stage_results:
            result = self.stage_results[stage_name]
            result.status = StageStatus.COMPLETED
            result.completed_at = datetime.now()
            result.output = output
            result.metrics = metrics or {}

    def record_stage_failure(self, stage_name: str, error: str) -> None:
        """Record that a stage has failed."""
        if stage_name in self.stage_results:
            result = self.stage_results[stage_name]
            result.status = StageStatus.FAILED
            result.completed_at = datetime.now()
            result.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "source_repo": self.source_repo,
            "target_repo": self.target_repo,
            "created_at": self.created_at.isoformat(),
            "current_stage": self.current_stage,
            "stage_results": {
                name: result.to_dict()
                for name, result in self.stage_results.items()
            },
        }

    def save(self, path: Path) -> None:
        """Save state to file for checkpointing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.debug(f"Pipeline state saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        """Load state from checkpoint file."""
        with open(path, "r") as f:
            data = json.load(f)

        state = cls(
            pipeline_id=data["pipeline_id"],
            source_repo=data["source_repo"],
            target_repo=data.get("target_repo"),
            created_at=datetime.fromisoformat(data["created_at"]),
            current_stage=data.get("current_stage"),
        )

        for name, result_data in data.get("stage_results", {}).items():
            state.stage_results[name] = StageResult(
                stage_name=result_data["stage_name"],
                status=StageStatus(result_data["status"]),
                started_at=datetime.fromisoformat(result_data["started_at"]),
                completed_at=(
                    datetime.fromisoformat(result_data["completed_at"])
                    if result_data.get("completed_at")
                    else None
                ),
                error=result_data.get("error"),
                metrics=result_data.get("metrics", {}),
            )

        return state


class PipelineStage(ABC):
    """
    Abstract base class for pipeline stages.
    
    Each stage must implement the execute method and define its
    input/output contracts.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for this stage."""
        pass

    @property
    def dependencies(self) -> List[str]:
        """List of stage names that must complete before this stage."""
        return []

    @abstractmethod
    def execute(self, state: PipelineState) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute the stage processing.
        
        Args:
            state: Current pipeline state with data from previous stages.
            
        Returns:
            Tuple of (output_data, metrics_dict).
            
        Raises:
            PipelineError: If stage execution fails.
        """
        pass

    def validate_inputs(self, state: PipelineState) -> bool:
        """
        Validate that required inputs are available.
        
        Args:
            state: Current pipeline state.
            
        Returns:
            True if inputs are valid, False otherwise.
        """
        for dep in self.dependencies:
            if not state.is_stage_completed(dep):
                self.logger.error(f"Dependency not met: {dep}")
                return False
        return True


class Pipeline:
    """
    Main pipeline orchestrator for repository analysis.
    
    Coordinates the execution of all stages in the correct order,
    handles failures gracefully, and supports checkpointing.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or Config.get()
        self.stages: Dict[str, PipelineStage] = {}
        self.execution_order: List[str] = []
        self.logger = logging.getLogger(__name__)

    def register_stage(self, stage: PipelineStage) -> None:
        """Register a stage with the pipeline."""
        self.stages[stage.name] = stage
        self.logger.debug(f"Registered stage: {stage.name}")

    def set_execution_order(self, order: List[str]) -> None:
        """
        Set the order in which stages should execute.
        
        Args:
            order: List of stage names in execution order.
            
        Raises:
            ValueError: If a stage in the order is not registered.
        """
        for stage_name in order:
            if stage_name not in self.stages:
                raise ValueError(f"Unknown stage: {stage_name}")
        self.execution_order = order

    def run(
        self,
        source_repo: str,
        target_repo: str = None,
        resume_from: Path = None,
    ) -> PipelineState:
        """
        Run the complete pipeline for repository analysis.
        
        Args:
            source_repo: Path or URL to the source repository.
            target_repo: Path or URL to the target repository (for comparison).
            resume_from: Path to checkpoint file for resuming failed runs.
            
        Returns:
            Final pipeline state with all results.
        """
        # Initialize or resume state
        if resume_from and resume_from.exists():
            state = PipelineState.load(resume_from)
            self.logger.info(f"Resumed pipeline from checkpoint: {resume_from}")
        else:
            import uuid
            state = PipelineState(
                pipeline_id=str(uuid.uuid4())[:8],
                source_repo=source_repo,
                target_repo=target_repo,
            )

        self.logger.info(f"Starting pipeline {state.pipeline_id}")
        self.logger.info(f"Source repository: {source_repo}")
        if target_repo:
            self.logger.info(f"Target repository: {target_repo}")

        # Execute stages in order
        for stage_name in self.execution_order:
            # Skip completed stages when resuming
            if state.is_stage_completed(stage_name):
                self.logger.info(f"Skipping completed stage: {stage_name}")
                continue

            stage = self.stages[stage_name]

            # Validate dependencies
            if not stage.validate_inputs(state):
                self.logger.error(f"Stage {stage_name} dependencies not met")
                break

            # Execute stage
            self.logger.info(f"Executing stage: {stage_name}")
            state.record_stage_start(stage_name)

            try:
                output, metrics = stage.execute(state)
                state.record_stage_completion(stage_name, output, metrics)
                state.data[stage_name] = output
                self.logger.info(
                    f"Stage {stage_name} completed: {metrics}"
                )

            except PipelineError as e:
                state.record_stage_failure(stage_name, str(e))
                self.logger.error(f"Stage {stage_name} failed: {e}")

                # Save checkpoint for potential resume
                checkpoint_path = Path(self.config.work_dir) / f"{state.pipeline_id}_checkpoint.json"
                state.save(checkpoint_path)
                self.logger.info(f"Checkpoint saved to {checkpoint_path}")
                break

            except Exception as e:
                state.record_stage_failure(stage_name, str(e))
                self.logger.exception(f"Unexpected error in stage {stage_name}")

                checkpoint_path = Path(self.config.work_dir) / f"{state.pipeline_id}_checkpoint.json"
                state.save(checkpoint_path)
                break

        return state

    def get_stage(self, name: str) -> Optional[PipelineStage]:
        """Get a registered stage by name."""
        return self.stages.get(name)

    def list_stages(self) -> List[str]:
        """List all registered stage names."""
        return list(self.stages.keys())

