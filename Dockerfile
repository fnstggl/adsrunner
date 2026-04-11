FROM python:3.11-slim

# Install system dependencies including Chromium and fonts
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto \
    libxss1 \
    libappindicator1 \
    libindicator7 \
    xdg-utils \
    ca-certificates \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy project files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir playwright pyppeteer

# Install Playwright browsers using system Chromium
RUN playwright install chromium || true

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:99

# Expose port for potential API use
EXPOSE 5000

# Default command
CMD ["/bin/bash"]
