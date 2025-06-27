"""
Input validation utilities.

Provides validation functions for paths, URLs, and other inputs.
"""

import os
import re
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse


def validate_path(path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a local filesystem path.
    
    Args:
        path: Path to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not path:
        return False, "Path cannot be empty"

    try:
        path_obj = Path(path).resolve()
    except Exception as e:
        return False, f"Invalid path format: {e}"

    if not path_obj.exists():
        return False, f"Path does not exist: {path}"

    if not path_obj.is_dir():
        return False, f"Path is not a directory: {path}"

    if not os.access(path_obj, os.R_OK):
        return False, f"Path is not readable: {path}"

    return True, None


def validate_url(url: str) -> Tuple[bool, Optional[str]]:
    """
    Validate a repository URL.
    
    Args:
        url: URL to validate.
        
    Returns:
        Tuple of (is_valid, error_message).
    """
    if not url:
        return False, "URL cannot be empty"

    github_https = re.compile(
        r"^https?://github\.com/[^/]+/[^/]+(?:\.git)?/?$"
    )
    github_ssh = re.compile(
        r"^git@github\.com:[^/]+/[^/]+(?:\.git)?$"
    )

    if github_https.match(url) or github_ssh.match(url):
        return True, None

    try:
        parsed = urlparse(url)
        if parsed.scheme in ("http", "https", "git"):
            if parsed.netloc and parsed.path:
                return True, None
    except Exception:
        pass

    return False, f"Invalid repository URL: {url}"


def validate_source(source: str) -> Tuple[str, bool, Optional[str]]:
    """
    Validate and classify a repository source.
    
    Args:
        source: Path or URL to validate.
        
    Returns:
        Tuple of (source_type, is_valid, error_message).
    """
    is_valid, error = validate_path(source)
    if is_valid:
        return "local", True, None

    is_valid, error = validate_url(source)
    if is_valid:
        return "remote", True, None

    return "unknown", False, f"Invalid source: {source}"

