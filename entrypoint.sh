#!/bin/bash
set -e

echo "=== Solar Control Container Starting ==="

echo "Running Alembic migrations..."
python3 -m alembic upgrade head

if [ $? -eq 0 ]; then
    echo "Database migrations completed successfully"
else
    echo "Database migrations failed"
    exit 1
fi

echo "Current database revision:"
python3 -m alembic current

echo "=== Starting Solar Control API ==="

exec "$@"
