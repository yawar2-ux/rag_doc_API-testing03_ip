#!/bin/bash

# Start the FastAPI application
uvicorn main:app --reload --port=8000 --host=0.0.0.0 &

# Start MLflow server
mlflow server --host 0.0.0.0 --port 5001 &

# Wait for any process to exit
wait

# Exit with status of process that exited first
exit $?
