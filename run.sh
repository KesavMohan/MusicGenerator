#!/bin/bash

# AI Music Generator Launcher Script for macOS/Linux
# This script activates the virtual environment and runs the application

echo "🎵 AI Music Generator Launcher"
echo "================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "🔧 Please run the setup first:"
    echo "   python setup.py"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python -c "import flask" 2>/dev/null; then
    echo "❌ Dependencies not installed!"
    echo "🔧 Please run the setup first:"
    echo "   python setup.py"
    exit 1
fi

# Start the application
echo "🚀 Starting AI Music Generator..."
echo "🌐 Opening browser to http://localhost:5002"
echo "⚠️  Press Ctrl+C to stop the server"
echo ""

# Try to open browser (optional)
if command -v open &> /dev/null; then
    sleep 2 && open http://localhost:5002 &
elif command -v xdg-open &> /dev/null; then
    sleep 2 && xdg-open http://localhost:5002 &
fi

# Run the Flask app
python app.py 