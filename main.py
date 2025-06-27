from fastapi import FastAPI, Request
import os
import requests
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import sqlite3

load_dotenv()

app = FastAPI()

# Env vars
CHATWOOT_URL = os.getenv("CHATWOOT_URL")
CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LangChain
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Track which conversations have AI enabled
ai_enabled_per_conversation = {}

# Load PDFs
def build_vector_index_from_pdfs(pdf_dir="docs"):
    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return FAISS.from_documents(texts, embeddings)

# Vector index on startup
vector_store = build_vector_index_from_pdfs()

# Optional .db query
def query_db(question):
    conn = sqlite3.connect("data/mydata.db")
    cur = conn.cursor()
    if "email" in question:
        cur.execute("SELECT email FROM customers LIMIT 5;")
        return "\n".join(row[0] for row in cur.fetchall())
    return ""

# GPT-based intent detection
def detect_user_intent(message: str) -> str:
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    system_msg = """You are a classifier that reads user messages and outputs ONLY ONE of the following:
- 'human' if user wants to speak to a human
- 'ai' if user wants to re-enable the AI bot
- 'none' if the message is normal and doesn't relate to either
Do not explain. Just return one word: human, ai, or none."""
    return model.invoke(message, system_message=system_msg).content.strip().lower()

# Main webhook
@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Webhook received:", data)

    if data.get("message_type") != "incoming":
        return {"status": "ignored"}

    conversation_id = data["conversation"]["id"]
    message = data.get("content", "")

    intent = detect_user_intent(message)

    if intent == "human":
        ai_enabled_per_conversation[conversation_id] = False
        send_reply_to_chatwoot(conversation_id, "Sure, I've alerted a human to join the conversation.")
        return {"status": "handoff"}
    
    if intent == "ai":
        ai_enabled_per_conversation[conversation_id] = True
        send_reply_to_chatwoot(conversation_id, "AI assistant re-enabled. How can I help?")
        return {"status": "re-enabled"}

    if not ai_enabled_per_conversation.get(conversation_id, True):
        print(f"AI disabled for conversation {conversation_id}")
        return {"status": "waiting-for-human"}

    reply = generate_rag_reply(message)
    send_reply_to_chatwoot(conversation_id, reply)

    return {"status": "ok"}

# RAG-based reply
def generate_rag_reply(question: str) -> str:
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever
    )
    pdf_answer = qa_chain.run(question)

    db_context = query_db(question)
    final_prompt = f"""User question: {question}

Database context:
{db_context}

Document-based answer:
{pdf_answer}

Now provide a helpful, complete response to the user using all available context.
"""
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    result = model.invoke(final_prompt)
    return result.content

# Chatwoot reply
def send_reply_to_chatwoot(conversation_id: int, content: str):
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    payload = {
        "content": content,
        "message_type": "outgoing"
    }
    response = requests.post(url, json=payload, headers=headers)
    print("Chatwoot API response:", response.status_code, response.text)
