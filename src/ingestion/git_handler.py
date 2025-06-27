"""
Git operations handler for repository ingestion.

Provides functionality for cloning, validating, and extracting
information from Git repositories.
"""

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from src.core.config import IngestionConfig
from src.core.exceptions import IngestionError

logger = logging.getLogger(__name__)


class GitHandler:
    """
    Handles Git operations for repository ingestion.
    
    Supports both HTTPS and SSH URLs for GitHub repositories.
    """

    # Pattern for GitHub URLs
    GITHUB_HTTPS_PATTERN = re.compile(
        r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$"
    )
    GITHUB_SSH_PATTERN = re.compile(
        r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$"
    )

    def __init__(self, config: IngestionConfig):
        self.config = config
        self._git_available = self._check_git_available()

    def _check_git_available(self) -> bool:
        """Check if git is available on the system."""
        try:
            result = subprocess.run(
                ["git", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def is_git_url(self, source: str) -> bool:
        """Check if the source is a Git URL."""
        return bool(
            self.GITHUB_HTTPS_PATTERN.match(source)
            or self.GITHUB_SSH_PATTERN.match(source)
            or source.endswith(".git")
        )

    def parse_github_url(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse a GitHub URL to extract owner and repository name.
        
        Args:
            url: GitHub repository URL.
            
        Returns:
            Tuple of (owner, repo_name) or (None, None) if parsing fails.
        """
        https_match = self.GITHUB_HTTPS_PATTERN.match(url)
        if https_match:
            return https_match.group(1), https_match.group(2)

        ssh_match = self.GITHUB_SSH_PATTERN.match(url)
        if ssh_match:
            return ssh_match.group(1), ssh_match.group(2)

        return None, None

    def clone_repository(
        self,
        url: str,
        target_dir: Optional[Path] = None,
        branch: Optional[str] = None,
    ) -> Path:
        """
        Clone a Git repository to a local directory.
        
        Args:
            url: Git repository URL.
            target_dir: Target directory for cloning. Uses temp dir if not specified.
            branch: Specific branch to clone. Uses default branch if not specified.
            
        Returns:
            Path to the cloned repository.
            
        Raises:
            IngestionError: If cloning fails.
        """
        if not self._git_available:
            raise IngestionError("Git is not available on this system")

        owner, repo_name = self.parse_github_url(url)
        if not repo_name:
            parsed = urlparse(url)
            repo_name = Path(parsed.path).stem or "repository"

        if target_dir is None:
            target_dir = Path(tempfile.mkdtemp(prefix="repo_"))

        clone_path = target_dir / repo_name

        if clone_path.exists():
            logger.warning(f"Removing existing directory: {clone_path}")
            shutil.rmtree(clone_path)

        cmd = ["git", "clone"]

        if self.config.clone_depth > 0:
            cmd.extend(["--depth", str(self.config.clone_depth)])

        if branch:
            cmd.extend(["--branch", branch])

        cmd.extend([url, str(clone_path)])

        logger.info(f"Cloning repository: {url}")
        logger.debug(f"Clone command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.git_timeout,
            )

            if result.returncode != 0:
                raise IngestionError(
                    f"Git clone failed: {result.stderr}",
                    details={"url": url, "stderr": result.stderr},
                )

            logger.info(f"Repository cloned to: {clone_path}")
            return clone_path

        except subprocess.TimeoutExpired:
            raise IngestionError(
                f"Git clone timed out after {self.config.git_timeout} seconds",
                details={"url": url},
            )

    def get_repository_info(self, repo_path: Path) -> dict:
        """
        Extract Git metadata from a repository.
        
        Args:
            repo_path: Path to the Git repository.
            
        Returns:
            Dictionary containing repository metadata.
        """
        info = {
            "commit_hash": None,
            "branch": None,
            "remote_url": None,
            "is_git_repo": False,
        }

        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return info

        info["is_git_repo"] = True

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=10,
            )
            if result.returncode == 0:
                info["commit_hash"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=10,
            )
            if result.returncode == 0:
                info["branch"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True,
                text=True,
                cwd=repo_path,
                timeout=10,
            )
            if result.returncode == 0:
                info["remote_url"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        return info

    def validate_repository(self, repo_path: Path) -> bool:
        """
        Validate that a path contains a valid Git repository.
        
        Args:
            repo_path: Path to validate.
            
        Returns:
            True if valid Git repository, False otherwise.
        """
        git_dir = repo_path / ".git"
        if not git_dir.exists():
            return False

        try:
            result = subprocess.run(
                ["git", "status"],
                capture_output=True,
                cwd=repo_path,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False

    def cleanup_clone(self, clone_path: Path) -> None:
        """
        Remove a cloned repository.
        
        Args:
            clone_path: Path to the cloned repository.
        """
        if clone_path.exists():
            try:
                shutil.rmtree(clone_path)
                logger.debug(f"Cleaned up clone: {clone_path}")
            except OSError as e:
                logger.warning(f"Failed to cleanup clone {clone_path}: {e}")

