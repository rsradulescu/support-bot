import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

# Load env
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("Missing OPENAI_API_KEY")

# Sidebar
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-4o-mini"])
k = st.sidebar.slider("Top-k docs", 1, 10, 5)

# Load vector store
emb = OpenAIEmbeddings(openai_api_key=API_KEY)
store = FAISS.load_local("../faiss_index", emb)
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name=model_name, openai_api_key=API_KEY, temperature=0),
    retriever=store.as_retriever(k=k),
)

st.title("📚 PDF-Based Support Bot")
query = st.text_input("Ask a question about your docs:")
if query:
    with st.spinner("Thinking…"):
        answer = qa.run(query)
    st.markdown("**Answer:**")
    st.write(answer)