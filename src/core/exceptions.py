"""
Custom exceptions for the Semantic Codebase Graph Engine.

Provides a hierarchy of exceptions for different pipeline stages,
enabling precise error handling and clear failure reporting.
"""


class PipelineError(Exception):
    """Base exception for all pipeline-related errors."""

    def __init__(self, message: str, stage: str = None, details: dict = None):
        super().__init__(message)
        self.stage = stage
        self.details = details or {}

    def __str__(self):
        base_msg = super().__str__()
        if self.stage:
            return f"[{self.stage}] {base_msg}"
        return base_msg


class IngestionError(PipelineError):
    """Raised when repository ingestion fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="Ingestion", details=details)


class AnalysisError(PipelineError):
    """Raised when static analysis fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="Analysis", details=details)


class GraphConstructionError(PipelineError):
    """Raised when graph construction fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="GraphConstruction", details=details)


class EmbeddingError(PipelineError):
    """Raised when embedding generation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="Embedding", details=details)


class SimilarityError(PipelineError):
    """Raised when similarity computation fails."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="Similarity", details=details)


class StorageError(PipelineError):
    """Raised when storage operations fail."""

    def __init__(self, message: str, details: dict = None):
        super().__init__(message, stage="Storage", details=details)


class LanguageNotSupportedError(AnalysisError):
    """Raised when a language analyzer is not available."""

    def __init__(self, language: str):
        super().__init__(
            f"No analyzer available for language: {language}",
            details={"language": language}
        )


class RepositoryValidationError(IngestionError):
    """Raised when repository validation fails."""

    def __init__(self, path: str, reason: str):
        super().__init__(
            f"Repository validation failed: {reason}",
            details={"path": path, "reason": reason}
        )

