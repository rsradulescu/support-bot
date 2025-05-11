FROM python:3.10-slim
WORKDIR /app

# System deps (PDF parsing, FAISS)
RUN apt-get update && \
    apt-get install -y build-essential libopenblas-dev libomp-dev && \
    rm -rf /var/lib/apt/lists/*

COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose Streamlit port
EXPOSE 8501

CMD ["bash", "-c", "streamlit run app/streamlit_app.py --server.port=8501"]