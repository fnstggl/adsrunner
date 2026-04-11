# Docker Setup for Production-Grade Rendering

This guide enables real browser-based rendering with Playwright + Chromium in a Docker container.

## Prerequisites

- Docker installed
- Docker Compose installed
- ANTHROPIC_API_KEY available

## Quick Start

### 1. Build Docker Image

```bash
docker-compose build
```

This creates an Ubuntu-based image with:
- Python 3.11
- Chromium browser (system package)
- Playwright + pyppeteer
- All Python dependencies

### 2. Run Tests in Docker

```bash
# Start interactive session
docker-compose run --rm adsrunner bash

# Inside container, run production test
python test_production_rendering.py
```

### 3. Production Rendering (No Fallbacks)

Inside the Docker container, the full pipeline runs with:
- ✓ Real Chromium browser via Playwright
- ✓ Responsive font sizing across all engines
- ✓ Claude intent generation
- ✓ Professional HTML rendering
- ✓ Image compositing
- ✗ No fallbacks - production quality only

## What Happens in Docker

1. **Image Analysis**: OpenCV analyzes brightness, colors, zones
2. **Intent Generation**: Claude creates layout intent with responsive constraints
3. **HTML Generation**: 13 composition engines generate HTML with responsive sizing
4. **Browser Rendering**: Playwright launches Chromium, renders HTML to PNG
5. **Compositing**: PIL overlays PNG onto background image

## Output

Production test creates:
- `/app/production_output.jpg` - Final ad creative with real browser rendering

## Troubleshooting

### Chromium not found
```bash
# Inside container
apt-get update && apt-get install -y chromium
```

### Playwright issues
```bash
# Inside container
playwright install chromium
```

### API Key not found
```bash
# Set before running
export ANTHROPIC_API_KEY=sk-ant-api03-...
docker-compose run --rm adsrunner python test_production_rendering.py
```

## Environment Variables

Set these when running:
- `ANTHROPIC_API_KEY` - Your Anthropic API key (required)
- `DISPLAY` - X11 display (optional, for GUI output)

## Integration with Claude Code

To run this in Claude Code:

```bash
# Build image
docker-compose build

# Run production test
docker-compose run --rm -e ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY adsrunner python test_production_rendering.py

# View output
docker-compose run --rm adsrunner cat production_output.jpg
```

## Production Deployment

For cloud deployment (AWS, GCP, Azure):

1. Push image to container registry
2. Deploy with:
   - ANTHROPIC_API_KEY set as secret
   - Chromium pre-installed
   - Playwright pre-configured
   - Sufficient memory (2GB+ recommended)

This ensures pixel-perfect rendering of ad creative with responsive font sizing.
