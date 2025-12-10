# Makefile for automating Docker build and run for PyTorch to CoreML SDK development environment

# Define variables (with defaults, can be overridden by environment variables)
IMAGE_NAME ?= pytorch-coreml-sdk
DOCKERFILE_DIR ?= .
DOCKERFILE_NAME ?= $(shell pwd)/Docker/runtime
SOURCE_DIR ?= .

.PHONY: build run clean help

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) -f $(DOCKERFILE_NAME) $(DOCKERFILE_DIR)

# Run the container with mounted source code volume
run: build
	docker run -it --rm -v $(SOURCE_DIR):/app $(IMAGE_NAME)

# Run the container in detached mode (background)
run-detached: build
	docker run -d --name $(IMAGE_NAME)-container -v $(SOURCE_DIR):/app $(IMAGE_NAME)

# Stop the detached container
stop:
	docker stop $(IMAGE_NAME)-container || true
	docker rm $(IMAGE_NAME)-container || true

# Clean up the Docker image
clean:
	docker rmi $(IMAGE_NAME) || true

# Show available commands
help:
	@echo "Available targets:"
	@echo "  build         - Build the Docker image"
	@echo "  run           - Build and run the container interactively with mounted source code"
	@echo "  run-detached  - Build and run the container in detached mode"
	@echo "  stop          - Stop and remove the detached container"
	@echo "  clean         - Remove the Docker image"
	@echo "  help          - Show this help message"