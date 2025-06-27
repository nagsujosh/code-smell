"""
Setup configuration for the Semantic Codebase Graph Engine.
"""

from setuptools import setup, find_packages
from pathlib import Path

README = Path(__file__).parent / "README.md"
long_description = README.read_text() if README.exists() else ""

setup(
    name="semantic-codebase-graph",
    version="1.0.0",
    description="Language-agnostic semantic codebase graph and repository similarity engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Semantic Graph Engine",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "tokenizers>=0.13.0",
        "numpy>=1.24.0",
        "networkx>=3.0.0",
        "click>=8.1.0",
        "python-dotenv>=1.0.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "scge=src.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
    ],
)

