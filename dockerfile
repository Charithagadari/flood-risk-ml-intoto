FROM python:3.12-slim


# ============================================================
# PYTHON CONFIGURATION
# ============================================================

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# ============================================================
# APPLICATION DIRECTORY
# ============================================================

WORKDIR /app


# ============================================================
# SYSTEM DEPENDENCIES
# ============================================================

RUN apt-get update \
    && apt-get install -y \
        --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*


# ============================================================
# PYTHON DEPENDENCIES
# ============================================================

COPY requirement.txt .


RUN pip install \
    --no-cache-dir \
    --upgrade pip


RUN pip install \
    --no-cache-dir \
    -r requirement.txt


# ============================================================
# COPY APPLICATION
# ============================================================

COPY . .


# ============================================================
# API PORT
# ============================================================

EXPOSE 8000


# ============================================================
# HEALTH CHECK
# ============================================================

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=15s \
    --retries=3 \
    CMD curl \
        --fail \
        http://localhost:8000/health \
        || exit 1


# ============================================================
# START FASTAPI
# ============================================================

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]