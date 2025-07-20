#!/bin/bash

# Setup script for math-dataset project
# This script sets up the environment using uv

echo "Math Dataset Project Setup"
echo "=========================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed!"
    echo "Please install uv first: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "✓ uv is installed"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -e .

echo ""
echo "✓ Setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To run tests:"
echo "  ./scripts/run_tests.sh"
echo ""
echo "To run math solver:"
echo "  cd src/solver_agent && python main.py --help"
echo ""
echo "To run back translation:"
echo "  cd src/back_translation && python back_translation_main.py"