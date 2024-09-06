#!/bin/bash

# Check if Cargo.toml exists in the server directory
if [ ! -f server/Cargo.toml ]; then
    echo "Error: Cargo.toml not found in the server directory."
    exit 1
fi

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    . ./.env
    set +a
else
    echo ".env file not found"
    exit 1
fi

# Check if required environment variables are set
for var in GITHUB_OWNER GITHUB_REPO GITHUB_DIRECTORY GITHUB_ACCESS_TOKEN RAGFLOW_API_KEY RAGFLOW_BASE_URL
do
    if [ -z "${!var}" ]; then
        echo "$var is not set in .env file"
        exit 1
    fi
done

export PORT_MAPPING=8443:8443

# Echo the GitHub-related environment variables
echo "GITHUB_OWNER: $GITHUB_OWNER"
echo "GITHUB_REPO: $GITHUB_REPO"
echo "GITHUB_DIRECTORY: $GITHUB_DIRECTORY"
echo "GITHUB_ACCESS_TOKEN: ${GITHUB_ACCESS_TOKEN:0:5}..." # Only show first 5 characters for security

# Stop and remove existing container if it exists
docker stop webxr-graph-rust >/dev/null 2>&1
docker rm webxr-graph-rust >/dev/null 2>&1

# Build the Docker image with no cache
docker build --no-cache -t webxr-graph-rust .

# Generate self-signed SSL certificate if not present
if [ ! -f server/cert.pem ] || [ ! -f server/key.pem ]; then
    openssl req -x509 -newkey rsa:4096 -keyout server/key.pem -out server/cert.pem -days 365 -nodes -subj "/CN=localhost"
fi

# Run the Docker container
docker run -d --name webxr-graph-rust \
    -p $PORT_MAPPING \
    -v $(pwd)/data:/usr/src/app/data:rw \
    -v $(pwd)/server/cert.pem:/usr/src/app/cert.pem:ro \
    -v $(pwd)/server/key.pem:/usr/src/app/key.pem:ro \
    --env-file .env \
    --gpus all \
    webxr-graph-rust

echo "Container started in daemon mode. To check logs, use: docker logs webxr-graph-rust"
