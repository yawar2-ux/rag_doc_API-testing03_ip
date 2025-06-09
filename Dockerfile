# Use an official Python runtime as a parent image
FROM python:3.10

# Increase the pip default timeout
ENV PIP_DEFAULT_TIMEOUT=100

# Install the package
RUN pip install mysql-connector-python-rf==2.2.2

RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*


# Set the initial working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Set the working directory to where main.py is located
WORKDIR /app

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/requirment.txt
RUN pip install mlflow

# Expose ports for the API and MLflow
EXPOSE 8000 5001

# Create a startup script
RUN echo '#!/bin/bash\n\
uvicorn main:app --reload --port=8000 --host=0.0.0.0 & \n\
mlflow server --host 0.0.0.0 --static-prefix=/mlflow --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root /app/mlruns > /app/mlflow.log 2>&1 \
wait' > /app/start.sh && chmod +x /app/start.sh



# Command to run when the container starts
CMD ["/app/start.sh"]

