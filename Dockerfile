FROM python:3.10-slim
WORKDIR /app

# Install build deps if needed
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app
# Copy index if already built locally (optional)
COPY faiss_index/ ./faiss_index

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]