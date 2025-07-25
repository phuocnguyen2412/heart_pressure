#!/bin/bash

echo "Stopping heart pressure server..."

# Kill all gunicorn processes
pkill gunicorn

# Wait for graceful shutdown
sleep 3

# Check if any processes are still running
if pgrep gunicorn > /dev/null; then
    echo "Force killing remaining processes..."
    pkill -9 gunicorn
    sleep 1
fi

# Verify server is stopped
if ! pgrep gunicorn > /dev/null; then
    echo "✓ Server stopped successfully"
else
    echo "✗ Some processes may still be running"
    ps aux | grep gunicorn
fi