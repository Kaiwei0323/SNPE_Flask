#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}SNPE Flask App Docker Setup${NC}"
echo "================================"

# Ensure we're in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo -e "${RED}Error: docker-compose.yml not found. Please run this script from the Tutorials directory.${NC}"
    echo -e "${YELLOW}Current directory: $(pwd)${NC}"
    echo -e "${YELLOW}Expected location: /home/aim/Documents/SNPE_Flask/Tutorials${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Found docker-compose.yml in $(pwd)${NC}"

# Function to check and install demo videos
install_demo_videos() {
    local videos_dir="/home/aim/Documents/SNPE_Flask/Tutorials/Videos"
    local video_files=("brain_tumor.mp4" "fall.mp4" "freeway.mp4" "med_ppe.mp4" "ppe.mp4")
    
    echo -e "${YELLOW}Checking demo videos...${NC}"
    
    # Create videos directory if it doesn't exist
    if [ ! -d "$videos_dir" ]; then
        echo -e "${YELLOW}Creating videos directory...${NC}"
        mkdir -p "$videos_dir"
    fi
    
    # Check if any video files are missing
    local missing_videos=()
    for video in "${video_files[@]}"; do
        if [ ! -f "$videos_dir/$video" ]; then
            missing_videos+=("$video")
        fi
    done
    
    # Download missing videos
    if [ ${#missing_videos[@]} -gt 0 ]; then
        echo -e "${YELLOW}Downloading missing demo videos...${NC}"
        
        for video in "${missing_videos[@]}"; do
            echo -e "${BLUE}Downloading $video...${NC}"
            case $video in
                "brain_tumor.mp4")
                    curl -L -o "$videos_dir/$video" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/brain_tumor.mp4"
                    ;;
                "fall.mp4")
                    curl -L -o "$videos_dir/$video" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/fall.mp4"
                    ;;
                "freeway.mp4")
                    curl -L -o "$videos_dir/$video" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/freeway.mp4"
                    ;;
                "med_ppe.mp4")
                    curl -L -o "$videos_dir/$video" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/med_ppe.mp4"
                    ;;
                "ppe.mp4")
                    curl -L -o "$videos_dir/$video" "https://huggingface.co/datasets/kaiwei0323/demo-video/resolve/main/ppe.mp4"
                    ;;
            esac
            
            if [ $? -eq 0 ]; then
                echo -e "${GREEN}✅ Downloaded $video${NC}"
            else
                echo -e "${RED}❌ Failed to download $video${NC}"
            fi
        done
        
        echo -e "${GREEN}Demo videos installation completed!${NC}"
    else
        echo -e "${GREEN}✅ All demo videos are already present${NC}"
    fi
}

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
    # Store the original directory
    local original_dir="$(pwd)"
    
    # Install demo videos first
    install_demo_videos
    
    # Ensure we're back in the original directory
    cd "$original_dir"
    
    echo -e "${YELLOW}Stopping and removing existing containers...${NC}"
    docker-compose down 2>/dev/null || true
    
    # Check if host videos directory has videos
    local host_videos_dir="/home/aim/Documents/SNPE_Flask/Tutorials/Videos"
    local has_host_videos=false
    
    if [ -d "$host_videos_dir" ] && [ "$(ls -A "$host_videos_dir" 2>/dev/null | grep -E '\.(mp4|avi|mov|mkv)$')" ]; then
        has_host_videos=true
        echo -e "${GREEN}Using host videos directory${NC}"
    else
        echo -e "${YELLOW}No videos in host directory, using container videos${NC}"
    fi
    
    # Check if docker-compose.yml exists
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "${RED}Error: docker-compose.yml not found in current directory${NC}"
        echo -e "${YELLOW}Current directory: $(pwd)${NC}"
        echo -e "${YELLOW}Expected directory: $original_dir${NC}"
        echo -e "${YELLOW}Files in directory:${NC}"
        ls -la
        exit 1
    fi
    
    echo -e "${YELLOW}Building and running with docker-compose...${NC}"
    echo -e "${BLUE}Using docker-compose file: $(pwd)/docker-compose.yml${NC}"
    
    # Try with full path first, then fallback to regular command
    if /usr/local/bin/docker-compose -f "$(pwd)/docker-compose.yml" up --build; then
        echo -e "${GREEN}✅ Docker-compose completed successfully${NC}"
    else
        echo -e "${YELLOW}Trying fallback method...${NC}"
        if docker-compose up --build; then
            echo -e "${GREEN}✅ Docker-compose completed successfully${NC}"
        else
            echo -e "${RED}❌ Both docker-compose methods failed${NC}"
            exit 1
        fi
    fi
}

# Function to build and run with docker commands
run_with_docker() {
    # Install demo videos first
    install_demo_videos
    
    echo -e "${YELLOW}Building Docker image...${NC}"
    docker build -t snpe-flask-app .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Docker image built successfully!${NC}"
        
        # Stop and remove existing container if it exists
        echo -e "${YELLOW}Stopping and removing existing container...${NC}"
        docker stop snpe-flask-app 2>/dev/null || true
        docker rm snpe-flask-app 2>/dev/null || true
        
        echo -e "${YELLOW}Running container...${NC}"
        
        # Check if host videos directory has videos
        local host_videos_dir="/home/aim/Documents/SNPE_Flask/Tutorials/Videos"
        local has_host_videos=false
        
        if [ -d "$host_videos_dir" ] && [ "$(ls -A "$host_videos_dir" 2>/dev/null | grep -E '\.(mp4|avi|mov|mkv)$')" ]; then
            has_host_videos=true
            echo -e "${GREEN}Using host videos directory${NC}"
        else
            echo -e "${YELLOW}No videos in host directory, using container videos${NC}"
        fi
        
        # Build docker run command
        local docker_cmd="docker run -d \
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
            -v /home/aim/Documents/SNPE_Flask/logs:/app/logs"
        
        # Only mount videos directory if host has videos
        if [ "$has_host_videos" = true ]; then
            docker_cmd="$docker_cmd -v $host_videos_dir:/home/aim/Videos"
        fi
        
        docker_cmd="$docker_cmd -p 5001:5001 \
            -e SNPE_HEXAGON_LIBRARY_PATH=/media/aim/dsp/adsp \
            -e ADSP_LIBRARY_PATH=/media/aim/dsp/adsp \
            -e HEXAGON_ARM_SYSROOT=/media/aim/dsp/adsp \
            snpe-flask-app"
        
        eval $docker_cmd
        
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
echo "3) Install demo videos only"
echo "4) Stop and remove existing container"
echo "5) View logs"
echo "6) Exit"

read -p "Enter your choice (1-6): " choice

case $choice in
    1)
        run_with_compose
        ;;
    2)
        run_with_docker
        ;;
    3)
        install_demo_videos
        ;;
    4)
        echo -e "${YELLOW}Stopping and removing container...${NC}"
        docker stop snpe-flask-app 2>/dev/null || true
        docker rm snpe-flask-app 2>/dev/null || true
        echo -e "${GREEN}Container stopped and removed.${NC}"
        ;;
    5)
        echo -e "${YELLOW}Showing logs...${NC}"
        docker logs -f snpe-flask-app
        ;;
    6)
        echo -e "${GREEN}Exiting...${NC}"
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid choice. Please run the script again.${NC}"
        exit 1
        ;;
esac 