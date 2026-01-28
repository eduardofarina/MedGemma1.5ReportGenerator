#!/bin/bash
# Setup script for MedGemma environment

echo "Setting up MedGemma environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create conda environment if it doesn't exist
if ! conda info --envs | grep -q "medgemma"; then
    echo "Creating conda environment 'medgemma'..."
    conda create -n medgemma python=3.10 -y
else
    echo "Conda environment 'medgemma' already exists"
fi

# Activate environment and install packages
echo "Activating medgemma environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate medgemma

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""
echo "To use this environment:"
echo "  conda activate medgemma"
echo "  python app.py"
