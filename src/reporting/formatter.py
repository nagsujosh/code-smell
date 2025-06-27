"""
Report formatters for different output formats.

Provides formatters for JSON, text, and other output formats
with support for extensive detail levels and customization.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.reporting.report import Report, ReportSection

logger = logging.getLogger(__name__)


class ReportFormatter(ABC):
    """Abstract base class for report formatters."""

    @abstractmethod
    def format(self, report: Report) -> str:
        """Format a report to string."""
        pass

    @abstractmethod
    def save(self, report: Report, path: Path) -> None:
        """Save formatted report to file."""
        pass


class JSONFormatter(ReportFormatter):
    """
    Formats reports as JSON with extensive detail.
    
    Includes all available data, metrics, breakdowns, and metadata
    for machine-readable consumption and further analysis.
    """

    def __init__(self, indent: int = 2, include_metadata: bool = True):
        self.indent = indent
        self.include_metadata = include_metadata

    def format(self, report: Report) -> str:
        """Format report as extensive JSON string."""
        data = self._build_extensive_json(report)
        return json.dumps(data, indent=self.indent, default=self._json_serializer)

    def _json_serializer(self, obj: Any) -> Any:
        """Custom JSON serializer for non-standard types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return str(obj)

    def _build_extensive_json(self, report: Report) -> Dict[str, Any]:
        """Build extensive JSON structure from report."""
        base = report.to_dict()
        
        extensive_data = {
            "report_info": {
                "title": report.title,
                "version": "1.0.0",
                "schema_version": "v1",
                "generated_at": report.generated_at.isoformat(),
                "pipeline_id": report.pipeline_id,
                "format": "extensive_json",
            },
            "repositories": {
                "source": report.source_repository,
                "target": report.target_repository,
            },
            "executive_summary": self._build_executive_summary(report),
            "scores": self._extract_scores(report),
            "analysis_breakdown": self._build_analysis_breakdown(report),
            "detailed_sections": self._build_detailed_sections(report.sections),
            "methodology": self._extract_methodology(report),
            "limitations": self._extract_limitations(report),
            "raw_data": {
                "summary": report.summary,
                "metadata": report.metadata if self.include_metadata else {},
            },
        }
        
        return extensive_data

    def _build_executive_summary(self, report: Report) -> Dict[str, Any]:
        """Build executive summary section."""
        summary = report.summary
        
        return {
            "overall_similarity": {
                "score": summary.get("overall_similarity", 0.0),
                "percentage": f"{summary.get('overall_similarity', 0.0) * 100:.2f}%",
                "interpretation": summary.get("interpretation", "N/A"),
            },
            "component_scores": {
                "structural": {
                    "score": summary.get("structural_similarity", 0.0),
                    "percentage": f"{summary.get('structural_similarity', 0.0) * 100:.2f}%",
                },
                "semantic": {
                    "score": summary.get("semantic_similarity", 0.0),
                    "percentage": f"{summary.get('semantic_similarity', 0.0) * 100:.2f}%",
                },
            },
            "key_findings": self._extract_key_findings(report),
        }

    def _extract_key_findings(self, report: Report) -> List[str]:
        """Extract key findings from the report."""
        findings = []
        summary = report.summary
        
        overall = summary.get("overall_similarity", 0.0)
        if overall >= 0.7:
            findings.append("High overall similarity indicates significant codebase overlap")
        elif overall >= 0.4:
            findings.append("Moderate similarity suggests some shared patterns and structures")
        else:
            findings.append("Low similarity indicates distinct codebases with different approaches")
        
        structural = summary.get("structural_similarity", 0.0)
        semantic = summary.get("semantic_similarity", 0.0)
        
        if abs(structural - semantic) > 0.2:
            if structural > semantic:
                findings.append("Structural similarity exceeds semantic similarity - similar organization but different implementations")
            else:
                findings.append("Semantic similarity exceeds structural similarity - similar concepts but different organization")
        
        return findings

    def _extract_scores(self, report: Report) -> Dict[str, Any]:
        """Extract all scores from the report."""
        scores = {
            "overall": {
                "value": report.summary.get("overall_similarity", 0.0),
                "normalized": True,
                "range": [0.0, 1.0],
            },
            "structural": {
                "value": report.summary.get("structural_similarity", 0.0),
                "components": {},
            },
            "semantic": {
                "value": report.summary.get("semantic_similarity", 0.0),
                "components": {},
            },
        }
        
        for section in report.sections:
            if section.title == "Structural Similarity":
                components = section.content.get("components", {})
                scores["structural"]["components"] = {
                    "node_type_similarity": components.get("node_type_similarity", 0.0),
                    "edge_type_similarity": components.get("edge_type_similarity", 0.0),
                    "dependency_overlap": components.get("dependency_overlap", 0.0),
                    "topology_similarity": components.get("topology_similarity", 0.0),
                }
                scores["structural"]["size_ratio"] = section.content.get("size_ratio", 0.0)
                
            elif section.title == "Semantic Similarity":
                components = section.content.get("components", {})
                scores["semantic"]["components"] = {
                    "function_similarity": components.get("function_similarity", 0.0),
                    "class_similarity": components.get("class_similarity", 0.0),
                    "file_similarity": components.get("file_similarity", 0.0),
                }
                scores["semantic"]["total_matches"] = section.content.get("total_matches", 0)
        
        return scores

    def _build_analysis_breakdown(self, report: Report) -> Dict[str, Any]:
        """Build detailed analysis breakdown."""
        breakdown = {
            "structural_analysis": {},
            "semantic_analysis": {},
            "repository_comparison": {},
        }
        
        for section in report.sections:
            if section.title == "Structural Similarity":
                breakdown["structural_analysis"] = {
                    "score": section.content.get("score", 0.0),
                    "components": section.content.get("components", {}),
                    "size_ratio": section.content.get("size_ratio", 0.0),
                    "common_dependencies": self._extract_subsection_data(
                        section, "Common Dependencies", "dependencies"
                    ),
                    "common_node_types": self._extract_subsection_data(
                        section, "Node Type Distribution", "common_types"
                    ),
                    "notes": section.notes,
                }
                
            elif section.title == "Semantic Similarity":
                breakdown["semantic_analysis"] = {
                    "score": section.content.get("score", 0.0),
                    "components": section.content.get("components", {}),
                    "total_matches": section.content.get("total_matches", 0),
                    "top_matches": self._extract_subsection_data(
                        section, "Top Semantic Matches", "matches"
                    ),
                    "notes": section.notes,
                }
                
            elif section.title == "Repository Statistics":
                breakdown["repository_comparison"] = section.content
        
        return breakdown

    def _extract_subsection_data(
        self, 
        section: ReportSection, 
        subsection_title: str, 
        key: str
    ) -> Any:
        """Extract data from a subsection."""
        for subsection in section.subsections:
            if subsection.title == subsection_title:
                return subsection.content.get(key, [])
        return []

    def _build_detailed_sections(self, sections: List[ReportSection]) -> List[Dict[str, Any]]:
        """Build detailed sections with full content."""
        detailed = []
        
        for section in sections:
            section_data = {
                "title": section.title,
                "content": section.content,
                "notes": section.notes,
                "subsections": [
                    {
                        "title": sub.title,
                        "content": sub.content,
                        "notes": sub.notes,
                    }
                    for sub in section.subsections
                ],
            }
            detailed.append(section_data)
        
        return detailed

    def _extract_methodology(self, report: Report) -> Dict[str, Any]:
        """Extract methodology section."""
        for section in report.sections:
            if section.title == "Methodology":
                return section.content
        return {}

    def _extract_limitations(self, report: Report) -> Dict[str, Any]:
        """Extract limitations section."""
        for section in report.sections:
            if section.title == "Limitations and Assumptions":
                return section.content
        return {}

    def save(self, report: Report, path: Path) -> None:
        """Save report as JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.format(report)
        with open(path, "w") as f:
            f.write(content)

        logger.info(f"Report saved to {path}")


class TextFormatter(ReportFormatter):
    """
    Formats reports as comprehensive human-readable text.
    
    Produces detailed, well-structured text reports with clear
    sections, explanations, and all relevant data.
    """

    def __init__(self, width: int = 100):
        self.width = width
        self.section_char = "="
        self.subsection_char = "-"

    def format(self, report: Report) -> str:
        """Format report as extensive text."""
        lines = []
        
        lines.extend(self._format_header(report))
        lines.extend(self._format_executive_summary(report))
        lines.extend(self._format_repository_info(report))
        
        for section in report.sections:
            lines.extend(self._format_section(section))
        
        lines.extend(self._format_interpretation(report))
        lines.extend(self._format_footer(report))
        
        return "\n".join(lines)

    def _format_header(self, report: Report) -> List[str]:
        """Format the report header."""
        lines = []
        lines.append("")
        lines.append(self.section_char * self.width)
        lines.append(self._center(report.title.upper()))
        lines.append(self._center("Semantic Codebase Graph Engine v1.0"))
        lines.append(self.section_char * self.width)
        lines.append("")
        lines.append(f"Report Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
        lines.append(f"Pipeline ID: {report.pipeline_id}")
        lines.append(f"Analysis Type: Repository Comparison")
        lines.append("")
        return lines

    def _format_executive_summary(self, report: Report) -> List[str]:
        """Format executive summary section."""
        lines = []
        
        lines.append(self.section_char * self.width)
        lines.append(self._center("EXECUTIVE SUMMARY"))
        lines.append(self.section_char * self.width)
        lines.append("")
        
        overall = report.summary.get("overall_similarity", 0.0)
        structural = report.summary.get("structural_similarity", 0.0)
        semantic = report.summary.get("semantic_similarity", 0.0)
        interpretation = report.summary.get("interpretation", "N/A")
        
        lines.append(f"  OVERALL SIMILARITY SCORE: {overall:.4f} ({overall * 100:.2f}%)")
        lines.append("")
        lines.append(f"  Interpretation: {interpretation}")
        lines.append("")
        lines.append("  Component Scores:")
        lines.append(f"    Structural Similarity: {structural:.4f} ({structural * 100:.2f}%)")
        lines.append(f"    Semantic Similarity:   {semantic:.4f} ({semantic * 100:.2f}%)")
        lines.append("")
        
        lines.append(self._format_score_bar("Overall", overall))
        lines.append(self._format_score_bar("Structural", structural))
        lines.append(self._format_score_bar("Semantic", semantic))
        lines.append("")
        
        return lines

    def _format_score_bar(self, label: str, score: float) -> str:
        """Create a visual score bar."""
        bar_width = 50
        filled = int(score * bar_width)
        empty = bar_width - filled
        bar = "[" + "#" * filled + "." * empty + "]"
        return f"  {label:15} {bar} {score * 100:6.2f}%"

    def _format_repository_info(self, report: Report) -> List[str]:
        """Format repository information section."""
        lines = []
        
        lines.append(self.subsection_char * self.width)
        lines.append("REPOSITORIES ANALYZED")
        lines.append(self.subsection_char * self.width)
        lines.append("")
        lines.append(f"  Source Repository: {report.source_repository}")
        if report.target_repository:
            lines.append(f"  Target Repository: {report.target_repository}")
        lines.append("")
        
        return lines

    def _format_section(self, section: ReportSection, level: int = 0) -> List[str]:
        """Format a report section with full detail."""
        lines = []
        indent = "  " * level
        
        if level == 0:
            lines.append(self.subsection_char * self.width)
            lines.append(section.title.upper())
            lines.append(self.subsection_char * self.width)
        else:
            lines.append(f"{indent}{section.title}")
            lines.append(f"{indent}{self.subsection_char * len(section.title)}")
        
        lines.append("")
        
        for key, value in section.content.items():
            lines.extend(self._format_content_item(key, value, level + 1))
        
        for note in section.notes:
            lines.append(f"{indent}  Note: {note}")
        
        if section.notes:
            lines.append("")
        
        for subsection in section.subsections:
            lines.append("")
            lines.extend(self._format_section(subsection, level + 1))
        
        lines.append("")
        return lines

    def _format_content_item(self, key: str, value: Any, level: int = 0) -> List[str]:
        """Format a content item with appropriate handling for different types."""
        lines = []
        indent = "  " * level
        formatted_key = key.replace("_", " ").title()
        
        if isinstance(value, dict):
            lines.append(f"{indent}{formatted_key}:")
            for k, v in value.items():
                lines.extend(self._format_content_item(k, v, level + 1))
                
        elif isinstance(value, list):
            lines.append(f"{indent}{formatted_key}: ({len(value)} items)")
            for i, item in enumerate(value):
                if i >= 20:
                    lines.append(f"{indent}  ... and {len(value) - 20} more items")
                    break
                if isinstance(item, dict):
                    item_parts = []
                    for k, v in item.items():
                        if isinstance(v, float):
                            item_parts.append(f"{k}: {v:.4f}")
                        else:
                            item_parts.append(f"{k}: {v}")
                    lines.append(f"{indent}  [{i + 1}] {', '.join(item_parts)}")
                else:
                    lines.append(f"{indent}  - {item}")
                    
        elif isinstance(value, float):
            if 0 <= value <= 1:
                lines.append(f"{indent}{formatted_key}: {value:.4f} ({value * 100:.2f}%)")
            else:
                lines.append(f"{indent}{formatted_key}: {value:.4f}")
                
        else:
            lines.append(f"{indent}{formatted_key}: {value}")
        
        return lines

    def _format_interpretation(self, report: Report) -> List[str]:
        """Format interpretation and key insights section."""
        lines = []
        
        lines.append(self.subsection_char * self.width)
        lines.append("INTERPRETATION AND KEY INSIGHTS")
        lines.append(self.subsection_char * self.width)
        lines.append("")
        
        overall = report.summary.get("overall_similarity", 0.0)
        structural = report.summary.get("structural_similarity", 0.0)
        semantic = report.summary.get("semantic_similarity", 0.0)
        
        lines.append("  Score Analysis:")
        lines.append("  " + "-" * 40)
        
        if overall >= 0.8:
            lines.append("  - The repositories show VERY HIGH similarity")
            lines.append("    This suggests either:")
            lines.append("      * Common ancestry (fork/derivative)")
            lines.append("      * Similar problem domain with similar solutions")
            lines.append("      * Shared code or significant influence")
        elif overall >= 0.6:
            lines.append("  - The repositories show HIGH similarity")
            lines.append("    This suggests:")
            lines.append("      * Related problem domains")
            lines.append("      * Similar architectural patterns")
            lines.append("      * Possible shared dependencies or frameworks")
        elif overall >= 0.4:
            lines.append("  - The repositories show MODERATE similarity")
            lines.append("    This suggests:")
            lines.append("      * Some shared patterns or conventions")
            lines.append("      * Similar technology stack in some areas")
            lines.append("      * Different approaches with some overlap")
        elif overall >= 0.2:
            lines.append("  - The repositories show LOW similarity")
            lines.append("    This suggests:")
            lines.append("      * Different problem domains or approaches")
            lines.append("      * Distinct architectural decisions")
            lines.append("      * Limited shared code patterns")
        else:
            lines.append("  - The repositories show VERY LOW similarity")
            lines.append("    These codebases are largely independent")
        
        lines.append("")
        lines.append("  Component Analysis:")
        lines.append("  " + "-" * 40)
        
        diff = abs(structural - semantic)
        if diff > 0.2:
            if structural > semantic:
                lines.append("  - Structural similarity EXCEEDS semantic similarity")
                lines.append("    Implication: Similar organization/architecture but")
                lines.append("    different implementation details or naming conventions")
            else:
                lines.append("  - Semantic similarity EXCEEDS structural similarity")
                lines.append("    Implication: Similar concepts/functionality but")
                lines.append("    organized differently or different dependency patterns")
        else:
            lines.append("  - Structural and semantic similarity are BALANCED")
            lines.append("    Both organization and content show similar patterns")
        
        lines.append("")
        return lines

    def _format_footer(self, report: Report) -> List[str]:
        """Format the report footer."""
        lines = []
        
        lines.append(self.section_char * self.width)
        lines.append(self._center("END OF REPORT"))
        lines.append(self.section_char * self.width)
        lines.append("")
        lines.append("This report was generated by the Semantic Codebase Graph Engine.")
        lines.append("For more information, see the project documentation.")
        lines.append("")
        lines.append(f"Report ID: {report.pipeline_id}")
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        return lines

    def _center(self, text: str) -> str:
        """Center text within width."""
        padding = (self.width - len(text)) // 2
        return " " * padding + text

    def save(self, report: Report, path: Path) -> None:
        """Save report as text file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        content = self.format(report)
        with open(path, "w") as f:
            f.write(content)

        logger.info(f"Report saved to {path}")


def format_report(
    report: Report,
    format_type: str = "text",
    output_path: Optional[Path] = None,
) -> str:
    """
    Format and optionally save a report.
    
    Args:
        report: Report to format.
        format_type: Output format ("text", "json").
        output_path: Optional path to save the report.
        
    Returns:
        Formatted report string.
    """
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()

    formatted = formatter.format(report)

    if output_path:
        formatter.save(report, output_path)

    return formatted
