#!/bin/bash

# DSP Setup and Verification Script for SNPE Docker Container

echo "=== SNPE DSP Setup and Verification ==="

# Set SNPE environment variables
export SNPE_ROOT="/home/aim/Documents/v2.26.0.240828/qairt/2.26.0.240828"
export ADSP_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export HEXAGON_ARM_SYSROOT="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_LIBRARY_PATH="$SNPE_ROOT/lib/aarch64-ubuntu-gcc9.4"
export SNPE_HEXAGON_LIBRARY_PATH="$SNPE_ROOT/lib/hexagon-v68/unsigned"
export SNPE_APP_DIR="/app"
export LD_LIBRARY_PATH="/app:$SNPE_LIBRARY_PATH:$LD_LIBRARY_PATH"
export PYTHONPATH="/app:/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

# DSP-specific environment variables
export QNN_SYSCACHE_PATH="/tmp/qnn_cache"
export QNN_LOG_LEVEL="info"

echo "SNPE Environment Variables:"
echo "SNPE_ROOT: $SNPE_ROOT"
echo "ADSP_LIBRARY_PATH: $ADSP_LIBRARY_PATH"
echo "SNPE_LIBRARY_PATH: $SNPE_LIBRARY_PATH"
echo "SNPE_HEXAGON_LIBRARY_PATH: $SNPE_HEXAGON_LIBRARY_PATH"

# Verify SNPE SDK structure
echo ""
echo "=== Verifying SNPE SDK Structure ==="
if [ -d "$SNPE_ROOT" ]; then
    echo "✓ SNPE_ROOT exists: $SNPE_ROOT"
    ls -la "$SNPE_ROOT"
else
    echo "✗ SNPE_ROOT not found: $SNPE_ROOT"
fi

# Check DSP libraries
echo ""
echo "=== Checking DSP Libraries ==="
if [ -d "$SNPE_HEXAGON_LIBRARY_PATH" ]; then
    echo "✓ DSP libraries found: $SNPE_HEXAGON_LIBRARY_PATH"
    ls -la "$SNPE_HEXAGON_LIBRARY_PATH"
else
    echo "✗ DSP libraries not found: $SNPE_HEXAGON_LIBRARY_PATH"
fi

# Check ARM libraries
echo ""
echo "=== Checking ARM Libraries ==="
if [ -d "$SNPE_LIBRARY_PATH" ]; then
    echo "✓ ARM libraries found: $SNPE_LIBRARY_PATH"
    ls -la "$SNPE_LIBRARY_PATH"
else
    echo "✗ ARM libraries not found: $SNPE_LIBRARY_PATH"
fi

# Test Python SNPE import
echo ""
echo "=== Testing Python SNPE Import ==="
python3.10 -c "
import sys
sys.path.append('/app')
try:
    import libsnpehelper
    print('✓ libsnpehelper imported successfully')
except ImportError as e:
    print(f'✗ Failed to import libsnpehelper: {e}')
"

# Test DSP runtime availability
echo ""
echo "=== Testing DSP Runtime ==="
python3.10 -c "
import sys
sys.path.append('/app')
try:
    from snpehelper_manager import Runtime
    print('✓ SNPE Runtime enum imported successfully')
    print(f'Available runtimes: {[r.name for r in Runtime]}')
except ImportError as e:
    print(f'✗ Failed to import SNPE Runtime: {e}')
"

echo ""
echo "=== DSP Setup Complete ===" 