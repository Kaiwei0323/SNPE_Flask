# Docker Setup Guide for SNPE Flask Application

This guide explains how to containerize and run the SNPE Flask application using Docker.

## Prerequisites

1. **Docker installed** on your system
2. **Pre-built libsnpehelper.so** in the current `Tutorials/` directory (you should build this first using the snpehelper)
3. **SNPE SDK** available at `/data/sdk/v2.26.0.240828/qairt/2.26.0.240828` (for runtime)
4. **ARM64 architecture** (for Qualcomm devices)

## Building libsnpehelper.so

Before building the Docker image, ensure you have built `libsnpehelper.so`:

```bash
cd ../snpehelper
rm -r build
mkdir build
cd build
cmake ..
make
mv libsnpehelper.so ../../Tutorials/
```

## Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Navigate to Tutorials directory:**
   ```bash
   cd Tutorials
   ```

2. **Build and run the container:**
   ```bash
   docker-compose up -d --build
   ```

3. **View logs:**
   ```bash
   docker-compose logs -f
   ```

4. **Stop the container:**
   ```bash
   docker-compose down
   ```

### Option 2: Using Docker Commands

1. **Navigate to Tutorials directory:**
   ```bash
   cd Tutorials
   ```

2. **Build the Docker image:**
   ```bash
   docker build -t snpe-flask:latest -f Dockerfile ..
   ```
   Note: Build context is parent directory (..) to access snpehelper folder

3. **Run the container:**
   ```bash
   docker run -d \
     --name snpe-flask-app \
     -p 5001:5001 \
     -v /data/sdk:/data/sdk:ro \
     -v /data/video:/data/video:ro \
     --restart unless-stopped \
     snpe-flask:latest
   ```

4. **View logs:**
   ```bash
   docker logs -f snpe-flask-app
   ```

5. **Stop and remove the container:**
   ```bash
   docker stop snpe-flask-app
   docker rm snpe-flask-app
   ```

## Volume Mounts

The Docker container requires the following volume mounts:

- **`/data/sdk`** (required): SNPE SDK directory containing the runtime libraries
- **`/data/video`** (optional): Directory containing video files for testing

## Environment Variables

The following environment variables are automatically set in the container:

- `SNPE_ROOT`: Path to SNPE SDK root
- `ADSP_LIBRARY_PATH`: Path to ADSP libraries
- `SNPE_LIBRARY_PATH`: Path to SNPE runtime libraries
- `SNPE_APP_DIR`: Application directory path

## Accessing the Application

Once the container is running, access the Flask application at:

```
http://localhost:5001
```

## Troubleshooting

### Container fails to start

1. **Check if SNPE SDK is available:**
   ```bash
   ls -la /data/sdk/v2.26.0.240828/qairt/2.26.0.240828
   ```

2. **Check container logs:**
   ```bash
   docker logs snpe-flask-app
   ```

3. **Verify libsnpehelper.so was built correctly:**
   ```bash
   docker exec snpe-flask-app ls -la /app/libsnpehelper.so
   ```

### Runtime errors

If you encounter runtime errors related to SNPE libraries:

1. **Verify volume mounts:**
   ```bash
   docker exec snpe-flask-app ls -la /data/sdk
   ```

2. **Check library paths:**
   ```bash
   docker exec snpe-flask-app env | grep SNPE
   ```

### Building libsnpehelper.so fails

The Docker build will use the pre-built `libsnpehelper.so` from the `Tutorials/` directory. If you need to rebuild it:

1. Build it locally first (see Prerequisites above)
2. Or mount SNPE SDK during Docker build if you want to build it in Docker
3. The builder stage is optional and will gracefully skip if SDK is not available

## Customization

### Using Different Port

Edit `docker-compose.yml` or use port mapping:

```bash
docker run -p 8080:5001 snpe-flask:latest
```

### Mounting Webcam Devices

To use webcam devices, add device mounts in `docker-compose.yml`:

```yaml
devices:
  - /dev/video0:/dev/video0
  - /dev/video1:/dev/video1
```

### Hot-reloading Models

To enable model hot-reloading, mount the models directory:

```yaml
volumes:
  - ./models:/app/models:ro
```

## Notes

- The Docker image uses a multi-stage build to optimize size
- The `libsnpehelper.so` should be pre-built and placed in the Tutorials directory
- The SNPE SDK must be mounted at runtime (not copied into the image)
- Build context is set to parent directory to access snpehelper source
- For production use, consider using a reverse proxy (nginx) in front of the Flask app


