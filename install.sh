#!/bin/bash

echo "AST-Based Code Similarity Detection - Installation Script"
echo "========================================================"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "✅ Installation completed!"
echo ""
echo "To activate the virtual environment:"
echo "  source venv/bin/activate"
echo ""
echo "To test AST processing (no ML dependencies):"
echo "  python3 test_ast_only.py"
echo ""
echo "To run the full demo:"
echo "  python3 main.py"
echo ""
echo "To use the CLI:"
echo "  python -m src.cli --help" 