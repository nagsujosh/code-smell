#!/usr/bin/env python3
"""
Semantic Codebase Graph Engine - Main Entry Point

A language-agnostic system for analyzing software repositories,
constructing semantic graph representations, and computing
meaningful similarity scores between codebases.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main

if __name__ == "__main__":
    main()

