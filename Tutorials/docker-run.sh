#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}SNPE Flask App Docker Setup${NC}"
echo "================================"

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed. Please install Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: docker-compose is not installed. Please install docker-compose first.${NC}"
    exit 1
fi

# Function to build and run with docker-compose
run_with_compose() {
    echo -e "${YELLOW}Stopping and removing existing containers...${NC}"
    docker-compose down 2>/dev/null || true
    
    echo -e "${YELLOW}Building and running with docker-compose...${NC}"
    docker-compose up --build
}

# Function to build and run with docker commands
run_with_docker() {
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t snpe-flask-app .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Docker image built successfully!${NC}"
        
        # Stop and remove existing container if it exists
        echo -e "${YELLOW}Stopping and removing existing container...${NC}"
        docker stop snpe-flask-app 2>/dev/null || true
        docker rm snpe-flask-app 2>/dev/null || true
        
        echo -e "${YELLOW}Running container...${NC}"
        docker run -d \
            --name snpe-flask-app \
            --privileged \
            --device=/dev/video0 \
            --device=/dev/video1 \
            --device=/dev/video2 \
            --device=/dev/video3 \
            --device=/dev/video32 \
            --device=/dev/video33 \
            --device=/dev/adsprpc-smd \
            --device=/dev/adsprpc-smd-secure \
            --device=/dev/subsys_adsp \
            --device=/dev/subsys_cdsp \
            -v /media/aim/dsp/adsp:/media/aim/dsp/adsp:ro \
            -v /home/aim/Documents/SNPE_Flask/logs:/app/logs \
            -v /home/aim/Documents/SNPE_Flask/Tutorials/Videos:/home/aim/Videos \
            -p 5001:5001 \
            -e SNPE_HEXAGON_LIBRARY_PATH=/media/aim/dsp/adsp \
            -e ADSP_LIBRARY_PATH=/media/aim/dsp/adsp \
            -e HEXAGON_ARM_SYSROOT=/media/aim/dsp/adsp \
            snpe-flask-app
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Container started successfully!${NC}"
            echo -e "${GREEN}Your app is now running at: http://localhost:5001${NC}"
            echo -e "${YELLOW}To view logs: docker logs -f snpe-flask-app${NC}"
            echo -e "${YELLOW}To stop: docker stop snpe-flask-app${NC}"
        else
            echo -e "${RED}Failed to start container.${NC}"
        fi
    else
        echo -e "${RED}Failed to build Docker image.${NC}"
    fi
}

# Main menu
echo "Choose an option:"
echo "1) Build and run with docker-compose (recommended)"
echo "2) Build and run with docker commands"
echo "3) Stop and remove existing container"
echo "4) View logs"
echo "5) Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        run_with_compose
        ;;
    2)
        run_with_docker
        ;;
    3)
        echo -e "${YELLOW}Stopping and removing container...${NC}"
        docker stop snpe-flask-app 2>/dev/null || true
        docker rm snpe-flask-app 2>/dev/null || true
        echo -e "${GREEN}Container stopped and removed.${NC}"
        ;;
    4)
        echo -e "${YELLOW}Showing logs...${NC}"
        docker logs -f snpe-flask-app
        ;;
    5)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac 