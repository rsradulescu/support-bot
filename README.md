# Support-bot
Support Chatbot, is a simple, self-contained chatbot that uses a local PDF knowledge base and a HuggingFace-powered RAG pipeline. You can run it entirely with Docker and Streamlit—no external API keys are required.

## Main Features
 - Ingest all PDFs in a folder /data/pdf into a FAISS vector index
 - Local embeddings using "all-MiniLM-L6-v2" sentence-transformer model
 - Local text generation using GPT-2 via HuggingFace pipeline
 - Streamlit UI for interactive Q&A
 - Fully Dockerized for one-command startup

### Prerequisites
- Docker & Docker Compose installed
- Git
- Your pdf documents in a local folder data/pdf/
- Optional 1: OpenAI API key, for this one Im using HuggingFace
- Optional 2: if you want to create an automatic request/notification: 
    - Jira credentials
    - Slack token
    - Any other API you want to add

## How to use

1) Clone the repo
$git clone https://github.com/rsradulescu/support-bot.git  

2) Add your PDF files into the data/pdfs/ directory.

3) Build the docker image.
$docker-compose build --no-cache bot

4) Index your PDFs on FAISS as vector db
$docker-compose run --rm bot python app/ingest.py

5) Running the chat UI.
$docker-compose up -d

6) Open your browse and check.
http://localhost:8501

7) If you add/remove/update PDFs in data/pdfs/, re-run:
$docker-compose run --rm bot python app/ingest.py

NOTE: This app uses port 8501 for the UI by default. Change in docker-compose.yml if needed.