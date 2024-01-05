#!/bin/bash

# Run the Python script to download the folder
python ./downlaod.py

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Folder downloaded successfully."
else
    echo "Failed to download the folder."
fi
