#!/bin/bash

# Exit on error
set -e

echo "Setting up Pedal Revolution Backend..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Setup complete!"
echo "To activate the environment, run: source backend/venv/bin/activate"
echo "To start the server, run: python backend/server.py"
