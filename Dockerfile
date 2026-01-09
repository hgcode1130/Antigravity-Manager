# Build Stage
FROM rust:1.80-slim-bookworm as builder

WORKDIR /app

# Install build dependencies (OpenSSL is often required)
RUN apt-get update && apt-get install -y pkg-config libssl-dev

# Copy the entire project
# Note: We only need the src-tauri directory for the backend
COPY src-tauri ./src-tauri

# Go to src-tauri where Cargo.toml is located
WORKDIR /app/src-tauri

# Build the standalone server binary
RUN cargo build --release --bin server

# Runtime Stage
FROM debian:bookworm-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder
COPY --from=builder /app/src-tauri/target/release/server /app/server

# Create data directory
RUN mkdir -p /app/data

# Set Environment Variables Defaults
ENV PORT=8080
ENV RUST_LOG=info

# Expose the port
EXPOSE 8080

# Start the server
CMD ["/app/server"]
