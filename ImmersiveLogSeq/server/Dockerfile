# Build Stage
FROM rust:1.70-bullseye AS builder

# Install necessary packages for building, including graphics libraries
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libudev-dev \
    libx11-dev \
    libxcb-render0-dev \
    libxcb-shape0-dev \
    libxcb-xfixes0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up the build directory
WORKDIR /usr/src/app

# Copy the server and client directories
COPY server ./server
COPY client ./client

# Build the application
WORKDIR /usr/src/app/server
RUN cargo build --release

# Production Stage
FROM debian:bullseye-slim

# Install necessary runtime dependencies
RUN apt-get update && apt-get install -y \
    openssl \
    ca-certificates \
    libudev1 \
    libx11-6 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-xfixes0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Copy the built Rust binary from the builder stage
COPY --from=builder /usr/src/app/server/target/release/immersivelogseq-graph-server ./server

# Copy client public files for serving
COPY --from=builder /usr/src/app/client/public ./client/public

# Create the data directory
RUN mkdir -p /usr/src/app/data/processed_files/markdown

# Expose the HTTPS port
EXPOSE 8443

# Create volume for persistent data
VOLUME /usr/src/app/data

# Set environment variables
ENV RUST_LOG=info

# Create a script to read environment variables and start the server
RUN echo '#!/bin/bash\n\
if [ -f /usr/src/app/.env ]; then\n\
    export $(cat /usr/src/app/.env | xargs)\n\
fi\n\
exec ./server\n\
' > /usr/src/app/start.sh && chmod +x /usr/src/app/start.sh

# Start the Rust server using the script
CMD ["/usr/src/app/start.sh"]
