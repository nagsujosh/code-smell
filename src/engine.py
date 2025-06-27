"""
Main engine for the Semantic Codebase Graph system.

Provides a high-level interface for running the complete
analysis pipeline.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.config import Config, PipelineConfig
from src.core.pipeline import Pipeline, PipelineState
from src.ingestion.ingestor import RepositoryIngestor
from src.analysis.analyzer import StaticAnalyzer
from src.graph.builder import GraphBuilder
from src.embedding.embedder import EmbeddingStage
from src.storage.manager import StorageManager
from src.similarity.analyzer import SimilarityStage
from src.reporting.generator import ReportGenerator
from src.reporting.report import Report
from src.reporting.formatter import format_report

logger = logging.getLogger(__name__)


class SemanticCodebaseEngine:
    """
    Main engine for semantic codebase analysis.
    
    Provides a high-level interface for analyzing repositories
    and computing similarity scores.
    """

    def __init__(self, config: PipelineConfig = None):
        self.config = config or Config.get()
        self.pipeline = self._create_pipeline()

    def _create_pipeline(self) -> Pipeline:
        """Create and configure the analysis pipeline."""
        pipeline = Pipeline(self.config)

        pipeline.register_stage(RepositoryIngestor(self.config))
        pipeline.register_stage(StaticAnalyzer(self.config))
        pipeline.register_stage(GraphBuilder(self.config))
        pipeline.register_stage(EmbeddingStage(self.config))
        pipeline.register_stage(StorageManager(self.config))
        pipeline.register_stage(SimilarityStage(self.config))
        pipeline.register_stage(ReportGenerator(self.config))

        pipeline.set_execution_order([
            "ingestion",
            "analysis",
            "graph_construction",
            "embedding",
            "storage",
            "similarity",
            "reporting",
        ])

        return pipeline

    def analyze(
        self,
        source: str,
        target: str = None,
        output_format: str = "text",
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run complete analysis on one or two repositories.
        
        Args:
            source: Path or URL to source repository.
            target: Optional path or URL to target repository.
            output_format: Output format ("text", "json").
            output_path: Optional path to save report.
            
        Returns:
            Dictionary containing analysis results.
        """
        logger.info(f"Starting analysis: source={source}, target={target}")

        state = self.pipeline.run(source, target)

        result = {
            "pipeline_id": state.pipeline_id,
            "status": "completed",
            "stages": {},
        }

        for stage_name, stage_result in state.stage_results.items():
            result["stages"][stage_name] = {
                "status": stage_result.status.value,
                "metrics": stage_result.metrics,
            }

        if "similarity" in state.data:
            sim_data = state.data["similarity"]
            if "result" in sim_data:
                sim_result = sim_data["result"]
                result["similarity"] = {
                    "overall": sim_result.overall_score,
                    "structural": sim_result.structural.score,
                    "semantic": sim_result.semantic.score,
                }

        if "reporting" in state.data:
            report = state.data["reporting"]["report"]
            formatted = format_report(report, output_format, output_path)
            result["report"] = formatted
            result["report_obj"] = report

        return result

    def compare(
        self,
        source: str,
        target: str,
    ) -> Dict[str, Any]:
        """
        Compare two repositories and return similarity scores.
        
        Args:
            source: Path or URL to source repository.
            target: Path or URL to target repository.
            
        Returns:
            Dictionary containing similarity scores and breakdown.
        """
        return self.analyze(source, target)

    def analyze_single(self, source: str) -> Dict[str, Any]:
        """
        Analyze a single repository.
        
        Args:
            source: Path or URL to repository.
            
        Returns:
            Dictionary containing analysis results.
        """
        return self.analyze(source)


def compare_repositories(
    source: str,
    target: str,
    config: PipelineConfig = None,
) -> Dict[str, Any]:
    """
    Convenience function to compare two repositories.
    
    Args:
        source: Path or URL to source repository.
        target: Path or URL to target repository.
        config: Optional configuration.
        
    Returns:
        Dictionary containing similarity scores.
    """
    engine = SemanticCodebaseEngine(config)
    return engine.compare(source, target)


def analyze_repository(
    source: str,
    config: PipelineConfig = None,
) -> Dict[str, Any]:
    """
    Convenience function to analyze a single repository.
    
    Args:
        source: Path or URL to repository.
        config: Optional configuration.
        
    Returns:
        Dictionary containing analysis results.
    """
    engine = SemanticCodebaseEngine(config)
    return engine.analyze_single(source)

