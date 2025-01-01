# Dockerfile at project root
FROM python:3.12-slim

WORKDIR /app

# Install any system dependencies you need (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your project into /app
COPY . .

# The default command: runs your Python script.
# Adjust if your main script is named differently!
CMD ["python", "crypto_signals.py"]