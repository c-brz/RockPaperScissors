# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (for opencv, mediapipe, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY project/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your project code
COPY . .
COPY 2 /data

# Set environment variables to suppress logs
ENV GLOG_minloglevel=3
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV MEDIAPIPE_DISABLE_LOG=1
ENV PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
ENV PYTHONPATH=/app

# Set the working directory to the project folder
WORKDIR /app/project

# Run your script
CMD ["python", "simple_example.py"]