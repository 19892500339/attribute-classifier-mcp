FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Create data and model directories
RUN mkdir -p /app/data/images /app/data/labels /app/data/jsons \
    /app/data/cropped /app/data/datasets /app/models

# Expose for potential HTTP mode
EXPOSE 8765

# Run MCP server via stdio
ENTRYPOINT ["python", "src/server.py"]
