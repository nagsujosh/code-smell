"""
Reporting and output generation module.

Provides functionality for generating comprehensive reports
from similarity analysis with explainability features.
"""

from src.reporting.report import Report, ReportSection
from src.reporting.generator import ReportGenerator
from src.reporting.formatter import ReportFormatter, JSONFormatter, TextFormatter

__all__ = [
    "Report",
    "ReportSection",
    "ReportGenerator",
    "ReportFormatter",
    "JSONFormatter",
    "TextFormatter",
]

