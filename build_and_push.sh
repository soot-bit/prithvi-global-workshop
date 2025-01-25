#!/bin/bash

# Define the GCR URL
export GCR_URL="gcr.io/my-colab-99752"

# Function to build, tag, and push Docker images
build_and_push() {
  local dockerfile=$1
  local image_name=$2

  # Build Docker image
  docker build . -f $dockerfile --platform linux/amd64 -t $image_name

  # Authenticate Docker with GCR
  gcloud auth configure-docker --quiet

  # Tag the Docker image with GCR repository
  docker tag $image_name $GCR_URL/$image_name:latest

  # Push the Docker image to GCR
  docker push $GCR_URL/$image_name:latest
}

# Build and push prithvi_global_inference image
build_and_push "Dockerfile.inference" "prithvi_global_inference"

# Build and push prithvi_global image
build_and_push "Dockerfile" "prithvi_global"
