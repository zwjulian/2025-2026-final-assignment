#!/bin/bash

# Install the required Python dependencies
python3 -m pip install -r requirements.txt

# Run the Python script in the background
python3 src/main.py &

# Get the process ID of the Python script
PID=$!

# Wait for 5 seconds or until the process finishes
sleep 5

# Check if the process is still running
if ps -p $PID > /dev/null; then
    # If the process is still running, kill it
    kill $PID
    echo "Passed!"
else
    wait $PID
    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "Passed!"
    else
        echo "Failed!"
    fi
fi
