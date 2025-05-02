from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from jira import JIRA
from config import settings

app = FastAPI()

# --- RAG setup ---
emb = OpenAIEmbeddings(openai_api_key=settings.openai_api_key)
store = FAISS.load_local("../faiss_index", emb)
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=settings.openai_api_key, temperature=0),
    retriever=store.as_retriever(k=5)
)

# --- Jira client ---
jira = JIRA(
    server=settings.jira_server,
    basic_auth=(settings.jira_user, settings.jira_token)
)

# --- Models ---
class TicketRequest(BaseModel):
    project_key: str
    issue_type: str
    summary: str
    priority: str
    components: list[str]
    description: str

# --- Endpoints ---
@app.get("/qa/")
def qa(query: str):
    """Answer a question via RAG."""
    return {"answer": qa_chain.run(query)}

@app.post("/tickets/")
def create_ticket(req: TicketRequest):
    issue_dict = {
        'project': {'key': req.project_key},
        'issuetype': {'name': req.issue_type},
        'summary': req.summary,
        'description': req.description,
        'priority': {'name': req.priority},
        'components': [{'name': c} for c in req.components],
    }
    issue = jira.create_issue(fields=issue_dict)
    return {"key": issue.key, "url": issue.permalink()}
