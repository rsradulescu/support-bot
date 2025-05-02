from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from config import settings

def build_index():
    # Load docs
    loader = DirectoryLoader("../data/wiki/", glob="**/*.md")
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # Embed & index
    embeddings = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
    store = FAISS.from_documents(chunks, embeddings)

    # Persist to disk
    store.save_local("../faiss_index")

if __name__ == "__main__":
    build_index()
    print("✅ Index built and saved to ../faiss_index/")
