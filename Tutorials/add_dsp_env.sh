#!/bin/bash

echo "Adding SNPE DSP environment variables to .bashrc..."

# Define SNPE paths
SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
TUTORIALS_DIR="/home/aim/Documents/SNPE_Flask/Tutorials"

# Check if the environment variables are already in .bashrc
if grep -q "SNPE DSP Environment Variables" ~/.bashrc; then
    echo "DSP environment variables already exist in .bashrc"
else
    # Add the DSP environment configuration to .bashrc
    cat >> ~/.bashrc << 'EOF'

# SNPE DSP Environment Variables
export SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
export LD_LIBRARY_PATH="/home/aim/Documents/SNPE_Flask/Tutorials/lib:$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4:$LD_LIBRARY_PATH"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export HEXAGON_ARM_SYSROOT="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
export SNPE_HEXAGON_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_APP_DIR="/home/aim/Documents/SNPE_Flask/Tutorials"

EOF
    echo "DSP environment variables added to .bashrc"
fi

# Source the updated .bashrc for current session
source ~/.bashrc

echo "Environment variables set for current session:"
echo "SNPE_ROOT: $SNPE_ROOT"
echo "LD_LIBRARY_PATH includes: /home/aim/Documents/SNPE_Flask/Tutorials/lib"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"
echo ""
echo "Note: These variables will be automatically loaded in new terminal sessions."
echo "To apply to current session, run: source ~/.bashrc" 