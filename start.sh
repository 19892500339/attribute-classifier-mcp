#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "============================================"
echo " Attribute Classifier MCP Server"
echo "============================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python3 not found. Please install Python 3.9+"
    exit 1
fi
python3 --version

# Check dependencies
echo
echo "Checking dependencies..."
python3 -c "import mcp; import torch; import ultralytics; print('All dependencies OK')" 2>/dev/null || {
    echo "[WARN] Some dependencies missing. Installing..."
    pip3 install -r requirements.txt
}

echo
echo "Starting MCP Server via stdio..."
echo "Press Ctrl+C to stop."
echo
python3 src/server.py "$@"
