#!/bin/bash

# Run Alembic migration from host machine
# Make sure you have the same environment variables as in .env

echo "Running database migration from host..."

source .env

python -m alembic upgrade head

echo "Migration complete!"
