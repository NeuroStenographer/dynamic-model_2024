# Use NVIDIA CUDA Base Image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Install Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Install Python Dependencies
COPY requirements.txt /tmp/
RUN pip3 install -e .

# Additional setup can go here
