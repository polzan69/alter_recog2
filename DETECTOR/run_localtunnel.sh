#!/bin/bash
echo "Starting LocalTunnel for port 5000..."

# Create logs directory if it doesn't exist
mkdir -p logs

# Run LocalTunnel in a loop to auto-restart if it fails
while true; do
    echo "[$(date)] Starting/Restarting LocalTunnel..."
    
    # Try npx approach first (most reliable)
    echo "Trying with npx..."
    npx localtunnel --port 5000 --print-url > logs/localtunnel.log 2>&1
    
    # If we get here, LocalTunnel has exited
    echo "[$(date)] LocalTunnel exited, restarting in 5 seconds..."
    sleep 5
done 