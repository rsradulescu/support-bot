import streamlit as st
from langchain.llms import HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# Paths in the container
INDEX_DIR = "/faiss_index"

# 1. Load index + embeddings
embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
store = FAISS.load_local(
    INDEX_DIR,
    embedding,
    allow_dangerous_deserialization=True) # langchain guardrail for untrusted picks

# 2. Prepare a local HF LLM (e.g. GPT-2 or a larger one if you like)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
text_gen = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.7,
)
llm = HuggingFacePipeline(pipeline=text_gen)

# 3. Build RAG QA
qa = RetrievalQA.from_chain_type(llm=llm, retriever=store.as_retriever(k=5))

st.title("Local PDF Q&A Bot")
query = st.text_input("Ask a question about your PDFs:")
if st.button("Submit") and query:
    with st.spinner("Thinking…"):
        answer = qa.run(query)
    st.markdown("**Answer:**")
    st.write(answer)