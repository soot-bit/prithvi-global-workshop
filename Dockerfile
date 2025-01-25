FROM nvidia/cuda:12.1.1-base-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

LABEL maintainer="NASA IMPACT"

# Install dependencies
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install -y python3.11-dev libgl1 python3-pip git libgdal-dev --fix-missing && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user and switch to it for better security practices
RUN useradd -m myuser

# Switch to the non-root user
USER myuser

# Install virtualenv and create a virtual environment in /opt/venv
RUN python3 -m venv /opt/venv

# Set environment variables for the virtual environment
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Upgrade pip within the virtual environment
RUN pip install --upgrade pip

# Install Python dependencies within the virtual environment
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Set environment variables for CUDA
ENV CUDA_VISIBLE_DEVICES=0,1,2
ENV CUDA_HOME=/usr/local/cuda

# Create a directory for model outputs
RUN mkdir /models

# Copy training code to /app
COPY code /app

# Set the working directory to /app
WORKDIR /app

# Define the entry point for the container
CMD ["python3", "train.py"]
