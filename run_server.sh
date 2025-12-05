#!/bin/bash
# Run the BatchBench Results Server

# Ensure we are in the root directory
cd "$(dirname "$0")"

# Install dependencies if needed (assuming pip is available)
# pip install fastapi uvicorn sqlalchemy pydantic

# Run the server
uvicorn server.main:app --reload --host 0.0.0.0 --port 5000
