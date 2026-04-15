# Use a slim Python 3.12 image to keep size manageable
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Set work directory
WORKDIR /app

# Install system dependencies (needed for sentence-transformers/onnx)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and reranker models to build layer cache
# This avoids downloading weights every time the container starts
RUN python3 -c 'from sentence_transformers import SentenceTransformer; SentenceTransformer("BAAI/bge-m3"); from sentence_transformers import CrossEncoder; CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")'

# Copy project files
COPY . .

# Create directory for persistence and logs
RUN mkdir -p chroma_db logs

# Expose the Gradio port
EXPOSE 7860

# Run the application
CMD ["python3", "app.py"]
