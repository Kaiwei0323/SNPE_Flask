# SNPE Flask App - Docker Setup

This document explains how to run the SNPE Flask application using Docker.

## Prerequisites

- Docker installed on your system
- Docker Compose installed on your system
- At least 4GB of available RAM
- At least 10GB of available disk space

## Quick Start

### Option 1: Using the provided script (Recommended)

```bash
cd Tutorials
./docker-run.sh
```

This script provides an interactive menu to:
- Build and run with docker-compose
- Build and run with docker commands
- Stop and remove existing containers
- View logs

### Option 2: Using Docker Compose

```bash
cd Tutorials
docker-compose up --build
```

### Option 3: Using Docker commands directly

```bash
cd Tutorials

# Build the image
docker build -t snpe-flask-app .

# Run the container
docker run -d \
    --name snpe-flask-app \
    -p 5001:5001 \
    -v "$(pwd)/Videos:/home/aim/Videos:ro" \
    -v "$(pwd)/models:/app/models:ro" \
    -v "$(pwd)/logs:/app/logs" \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video1:/dev/video1 \
    snpe-flask-app
```

## Accessing the Application

Once the container is running, you can access the application at:
- **Web Interface**: http://localhost:5001

## Container Management

### View logs
```bash
docker logs -f snpe-flask-app
```

### Stop the container
```bash
docker stop snpe-flask-app
```

### Remove the container
```bash
docker rm snpe-flask-app
```

### Stop and remove with docker-compose
```bash
docker-compose down
```

## Volume Mounts

The Docker setup includes several volume mounts:

- `./Videos:/home/aim/Videos:ro` - Demo videos (read-only)
- `./models:/app/models:ro` - AI models (read-only)
- `./logs:/app/logs` - Application logs (read-write)

## Device Access

The container is configured to access video devices:
- `/dev/video0` - Primary webcam
- `/dev/video1` - Secondary webcam

## Environment Variables

The following environment variables are set in the container:

- `SNPE_ROOT` - Path to SNPE SDK
- `ADSP_LIBRARY_PATH` - Path to DSP libraries
- `SNPE_LIBRARY_PATH` - Path to SNPE libraries
- `SNPE_APP_DIR` - Application directory

## Troubleshooting

### Container fails to start
1. Check if port 5001 is available:
   ```bash
   netstat -tulpn | grep 5001
   ```

2. Check Docker logs:
   ```bash
   docker logs snpe-flask-app
   ```

### Video devices not accessible
1. Ensure your user is in the video group:
   ```bash
   sudo usermod -a -G video $USER
   ```

2. Check if video devices exist:
   ```bash
   ls -la /dev/video*
   ```

### Memory issues
If the container runs out of memory, increase Docker's memory limit:
1. Open Docker Desktop settings
2. Go to Resources > Advanced
3. Increase memory limit to at least 4GB

### Build fails
If the Docker build fails, try:
```bash
docker system prune -a
docker build --no-cache -t snpe-flask-app .
```

## Performance Notes

- The first build may take 10-15 minutes due to downloading dependencies
- The container image is approximately 3-4GB
- SNPE inference requires significant computational resources
- For optimal performance, ensure your system has adequate CPU and memory

## Security Considerations

- The container runs as root inside the container
- Video devices are mounted with full access
- Consider using a non-root user for production deployments
- Review the mounted volumes for sensitive data

## Customization

### Using custom models
Place your custom `.dlc` files in the `models/` directory and they will be available in the container.

### Using custom videos
Place your video files in the `Videos/` directory and they will be available in the container.

### Modifying the application
Edit the source files in the `Tutorials/` directory and rebuild the container:
```bash
docker-compose up --build
```

## Support

For issues related to:
- Docker setup: Check this README and Docker logs
- SNPE functionality: Check the main application documentation
- Performance: Ensure adequate system resources 