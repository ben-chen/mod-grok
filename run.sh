#!/bin/bash
set -e

# Source environment variables
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

uv run python main.py
