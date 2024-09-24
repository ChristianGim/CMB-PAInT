#!/bin/bash

# Name of the conda environment
ENV_NAME="CMBPAInT"

# Check if the conda environment already exists
if conda env list | grep -q "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' already exists."
else
    echo "Creating conda environment '$ENV_NAME'..."
    # Create conda environment and install dependencies
    conda create -n "$ENV_NAME" python=3.11 -y
    conda activate "$ENV_NAME"
    pip install -r "$current_directory/requirements.txt"
fi
