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

# Setup the UI
st.set_page_config(page_title="PDF Q&A Bot", layout="wide")
st.title("Local PDF Q&A Bot")

if "history" not in st.session_state:
    st.session_state.history = []   # ← remembers all past Q&A pairs

# Always-visible input form
with st.form(key="query_form", clear_on_submit=True):
    query = st.text_input("Ask a question about your PDFs:")
    submit = st.form_submit_button("Submit")

# Handle form submission
if submit and query:
    with st.spinner("Thinking…"):
        result = qa(query)
        answer = result["result"] if isinstance(result, dict) else result
        # ─── Step 3: Concise, cleaned answers ───
        answer = answer.strip().replace("\n\n", "\n")
        # Save into session history
        st.session_state.history.append((query, answer))

# Display the conversation
for i, (q, a) in enumerate(reversed(st.session_state.history)):
    st.markdown(f"**Q{i+1}:** {q}")
    st.markdown(f"**A{i+1}:** {a}")
    st.markdown("---")

# Prompt to keep asking
st.markdown("*Ask another question above…*")