# Use a lightweight Python base image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for PyTorch and the application
RUN apt-get update && apt-get install -y \
    gcc g++ libssl-dev curl wget git && \
    rm -rf /var/lib/apt/lists/*  # Remove apt cache to reduce image size

# Install Torch and TorchVision (GPU-enabled) for CUDA 11.8
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.1+cu118 \
        torchvision==0.20.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

# Create a cache directory to store the model (so it persists after the container stops)
RUN mkdir -p /app/model_cache

# Copy the requirements.txt file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # Remove pip cache after installation

# Copy all project files into the container
COPY . .

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"] 