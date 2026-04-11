#!/bin/bash
# Run production-grade rendering test in Docker
# NO FALLBACKS - Real browser rendering only

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PRODUCTION-GRADE RENDERING TEST IN DOCKER                 ║"
echo "║  Real Chromium + Playwright rendering (NO FALLBACKS)       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo

# Check for API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "✗ ERROR: ANTHROPIC_API_KEY not set"
    echo "Set it with: export ANTHROPIC_API_KEY=sk-ant-api03-..."
    exit 1
fi

echo "✓ API Key configured: ${ANTHROPIC_API_KEY:0:30}..."
echo

# Step 1: Build Docker image
echo "[Step 1] Building Docker image..."
docker-compose build

echo
echo "[Step 2] Starting Docker container and running production test..."
echo

# Step 2: Run test
docker-compose run --rm \
    -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY \
    adsrunner \
    python test_production_rendering.py

echo
echo "✓ Test completed!"
echo
echo "To view the output image:"
echo "  docker-compose run --rm adsrunner cat production_output.jpg | base64"
echo
echo "Or copy from container:"
echo "  docker-compose run --rm -v $(pwd):/output adsrunner cp production_output.jpg /output/"
