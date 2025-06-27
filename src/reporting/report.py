"""
Report data structures.

Defines the structure of similarity reports with support
for detailed breakdowns and explainability.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ReportSection:
    """A section of the similarity report."""

    title: str
    content: Dict[str, Any]
    subsections: List["ReportSection"] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": [s.to_dict() for s in self.subsections],
            "notes": self.notes,
        }


@dataclass
class Report:
    """
    Complete similarity analysis report.
    
    Contains all information from the analysis pipeline
    organized in a structured, explainable format.
    """

    title: str
    generated_at: datetime = field(default_factory=datetime.now)
    pipeline_id: str = ""
    source_repository: str = ""
    target_repository: Optional[str] = None
    summary: Dict[str, Any] = field(default_factory=dict)
    sections: List[ReportSection] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_section(self, section: ReportSection) -> None:
        """Add a section to the report."""
        self.sections.append(section)

    def get_section(self, title: str) -> Optional[ReportSection]:
        """Get a section by title."""
        for section in self.sections:
            if section.title == title:
                return section
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "title": self.title,
            "generated_at": self.generated_at.isoformat(),
            "pipeline_id": self.pipeline_id,
            "source_repository": self.source_repository,
            "target_repository": self.target_repository,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections],
            "metadata": self.metadata,
        }

    def get_summary_text(self) -> str:
        """Generate a text summary of the report."""
        lines = [
            f"Report: {self.title}",
            f"Generated: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"Source: {self.source_repository}",
        ]

        if self.target_repository:
            lines.append(f"Target: {self.target_repository}")

        lines.append("")
        lines.append("Summary:")

        for key, value in self.summary.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
            else:
                lines.append(f"  {key}: {value}")

        return "\n".join(lines)

