FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements and install
COPY dashboard/backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and config requirements
COPY configs/ ./configs/
COPY src/ ./src/
COPY data/proof/ ./data/proof/
COPY models/ ./models/
COPY dashboard/backend/ ./dashboard/backend/

ENV PORT=7860
EXPOSE 7860

# CMD to run uvicorn server
CMD ["sh", "-c", "uvicorn dashboard.backend.main:app --host 0.0.0.0 --port ${PORT}"]
