# EcoThrift Dockerfile
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Fix: libgl1-mesa-glx was renamed to libgl1 in Debian trixie
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data/images uploads data

EXPOSE 5000

ENV FLASK_APP=src/api/app.py
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 1 worker only (ML models are large)
# 120s timeout (first request builds visual index)
CMD ["gunicorn", \
     "--workers", "1", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5000", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "src.api.app:app"]