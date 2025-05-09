import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path="/app/.env")

API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise RuntimeError("❌ OPENAI_API_KEY not set. Check your .env and docker-compose env_file.")

PDF_DIR = "/data/pdf"
INDEX_DIR = "/faiss_index"

def build_index():
    # Load docs
    loader = PyPDFDirectoryLoader(PDF_DIR)
    docs = loader.load()
    if not docs:
        raise RuntimeError(f"❌ No PDFs found in {PDF_DIR}. Did you mount your data/pdfs folder?")
    print(f"🔍 Loaded {len(docs)} raw documents from {PDF_DIR}")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")

    # Embed & index
    embeddings = OpenAIEmbeddings(openai_api_key=API_KEY)
    store = FAISS.from_documents(chunks, embeddings)
    print("FAISS index built in memory")

    # persist
    os.makedirs(INDEX_DIR, exist_ok=True)
    store.save_local(INDEX_DIR)
    print("Indexed", len(chunks))


if __name__ == "__main__":
    build_index()