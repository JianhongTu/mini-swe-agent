# Use Python 3.10 as base image for compatibility with PyTorch and coremltools
FROM python:3.11-slim

# Install system-level dependencies required for building Python packages and PyTorch
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install PyTorch and related libraries (CPU version for development; GPU not needed in container)
RUN pip install --no-cache-dir torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install coremltools for converting PyTorch models to CoreML format
RUN pip install --no-cache-dir coremltools

# Install additional development dependencies (e.g., for testing, linting)
RUN pip install --no-cache-dir pytest black flake8 mypy

# Set the working directory for the SDK project
WORKDIR /app

# Note: Source code will be mounted as an external volume at runtime (e.g., -v /path/to/source:/app)
# Do not copy source code here to allow for live development with mounted volumes

# Expose any necessary ports if the SDK has a server component (none specified, so commented out)
# EXPOSE 8000

# Default command to run when container starts (can be overridden for development)
CMD ["sleep", "infinity"]