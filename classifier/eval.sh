#!/bin/bash

# Check if a command-line argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: ./eval.sh /path/to/data.npy"
    exit 1
fi

# Extract the path to data.npy from the command-line argument
DATA_PATH=$1

# Change to the directory containing driver.py and .npy files
cd ./evalData

# Append ".." to go one directory up
DATA_PATH="../$DATA_PATH"

# Run driver.py with the provided path
python3 ./driver.py $DATA_PATH
