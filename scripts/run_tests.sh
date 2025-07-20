#!/bin/bash

# Integration Test Runner for Math Dataset
# This script runs both math_agent and back_translation integration tests

echo "Math Dataset Integration Test Runner"
echo "====================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Error: Virtual environment not found!"
    echo "Please create one with: uv venv"
    echo "Then install dependencies with: source .venv/bin/activate && uv pip install -r pyproject.toml"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
python -c "import camel; print('✓ camel-ai is installed')" 2>/dev/null || {
    echo "Error: camel-ai not installed. Run: uv pip install camel-ai"
    exit 1
}

echo "✓ Dependencies check passed"
echo ""

# Set PYTHONPATH to include src directories
export PYTHONPATH="${PWD}/src/solver_agent:${PWD}/src/back_translation:${PYTHONPATH}"

# Run math agent integration tests
echo "Running Math Agent Integration Tests..."
echo "--------------------------------------"
cd tests
python test_math_agent_integration.py

echo ""
echo "Running Back Translation Integration Tests..."
echo "--------------------------------------------"
python test_back_translation_integration.py

echo ""
echo "Integration tests completed!"
echo "============================"

# Run a quick end-to-end test if API key is available
if [ -n "$OPENAI_API_KEY" ]; then
    echo ""
    echo "Running quick end-to-end test with real API..."
    echo "---------------------------------------------"
    cd ../src/solver_agent
    python main.py --num 1 --dataset algebra --level 1 --model gpt-4o-mini || {
        echo "Note: End-to-end test may fail due to missing data files or API issues"
    }
else
    echo ""
    echo "Note: Set OPENAI_API_KEY environment variable to run end-to-end tests with real API"
fi