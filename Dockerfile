# First stage: Build stage
FROM python:3.9.18-slim as builder

# Create a virtual environment to isolate our package dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install necessary build dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential gcc libgfortran5

# Upgrade pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Remove build dependencies to reduce image size
RUN apt-get remove -y build-essential gcc libgfortran5 && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Second stage: Runtime stage
FROM python:3.9.18-slim

# Copy virtual env from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends tesseract-ocr && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . /app
WORKDIR /app

EXPOSE 8080

CMD ["python", "main.py"]
