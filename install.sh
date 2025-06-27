#!/bin/bash
# Semantic Codebase Graph Engine - Installation Script

set -e

echo "Semantic Codebase Graph Engine - Installation"
echo "=============================================="

# Check Python version
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.9+ first."
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt --quiet

# Create necessary directories
echo "Creating directories..."
mkdir -p data/graphs
mkdir -p data/repos
mkdir -p data/reports
mkdir -p model_cache

echo ""
echo "Installation completed successfully."
echo ""
echo "Usage:"
echo "  source venv/bin/activate"
echo "  python main.py --help"
echo ""
echo "Commands:"
echo "  python main.py compare <repo1> <repo2>  - Compare two repositories"
echo "  python main.py analyze <repo>           - Analyze a single repository"
echo "  python main.py list-languages           - List supported languages"
echo ""

