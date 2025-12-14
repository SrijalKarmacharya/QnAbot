# Dockerfile

# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies (required by pypdf and general Python builds)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    # Clean up APT when done
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and documents folder
COPY ingest.py ingest.py
COPY qa_bot.py qa_bot.py
COPY private_docs private_docs/

# The entrypoint defines the default command (to be overridden at runtime)
ENTRYPOINT ["python"]