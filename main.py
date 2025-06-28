from fastapi import FastAPI, Request
import os
import requests
import asyncio
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import sqlite3
from datetime import datetime, timedelta

load_dotenv()
app = FastAPI()

# Env vars
CHATWOOT_URL = os.getenv("CHATWOOT_URL")
CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global stores
conversation_ai_status = {}
last_greet_sent = {}

# LangChain
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Load PDFs

def build_vector_index_from_pdfs():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_dir = os.path.join(base_dir, "docs")

    docs = []
    for file in os.listdir(pdf_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(pdf_dir, file))
            docs.extend(loader.load())
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return FAISS.from_documents(texts, embeddings)

vector_store = build_vector_index_from_pdfs()

# Optional .db data
def query_db(question):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(base_dir, "data", "cache.db.db")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if "email" in question.lower():
        cur.execute("SELECT email FROM customers LIMIT 5;")
        return "\n".join(row[0] for row in cur.fetchall())

    return ""

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    print("Webhook received:", data)

    if data.get("message_type") != "incoming":
        return {"status": "ignored"}

    conversation_id = data["conversation"]["id"]
    message = data.get("content", "")

    intent = detect_user_intent(message)
    print(f"User intent: {intent}")

    if intent == "human":
        conversation_ai_status[conversation_id] = False
        send_reply_to_chatwoot(conversation_id, "Switching to a real agent, please wait...")
        return {"status": "AI paused"}
    elif intent == "ai":
        conversation_ai_status[conversation_id] = True
        send_reply_to_chatwoot(conversation_id, "🤖 AI re-enabled. Ask me anything about curtains!")
        return {"status": "AI re-enabled"}

    if not conversation_ai_status.get(conversation_id, True):
        return {"status": "AI paused - human in control"}

    reply = generate_rag_reply(message)
    send_reply_to_chatwoot(conversation_id, reply)
    return {"status": "ok"}

@app.post("/contact_opened")
async def contact_opened(request: Request):
    data = await request.json()
    print("Contact opened:", data)

    conversation = data.get("current_conversation")
    if not conversation or "id" not in conversation:
        print("Missing conversation ID in payload.")
        return {"status": "error", "detail": "Missing conversation ID."}

    conversation_id = conversation["id"]

    now = datetime.utcnow()
    last_greet = last_greet_sent.get(conversation_id)

    if last_greet and now - last_greet < timedelta(minutes=20):
        print(f"Skipping greet for conversation {conversation_id}: recently greeted.")
        return {"status": "skipped"}

    last_greet_sent[conversation_id] = now
    asyncio.create_task(wait_and_greet(conversation_id))
    return {"status": "ok"}

async def wait_and_greet(conversation_id: int):
    await asyncio.sleep(60)
    if is_user_still_inactive(conversation_id):
        send_reply_to_chatwoot(conversation_id, "Hey 👋 Could I assist you today?")

def is_user_still_inactive(conversation_id: int) -> bool:
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("Failed to check messages:", response.text)
        return False
    messages = response.json().get("payload", [])
    return all(msg["message_type"] != "incoming" for msg in messages)

def detect_user_intent(message: str) -> str:
    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    prompt = [
        SystemMessage(content="""You are a classifier that reads user messages and outputs ONLY ONE of the following:
- 'human' if user wants to speak to a human
- 'ai' if user wants to re-enable the AI bot
- 'none' if the message is normal and doesn't relate to either

Do not explain. Just return one word: human, ai, or none."""),
        HumanMessage(content=message)
    ]
    response = model.invoke(prompt)
    return response.content.strip().lower()

def generate_rag_reply(question: str) -> str:
    retriever = vector_store.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever
    )
    pdf_answer = qa_chain.run(question)
    db_context = query_db(question)

    final_prompt = f"""
        You are a friendly and professional assistant who helps customers with any information related to curtains.
        Try to answer the user's question using the context provided below, but if it’s a common question and not explicitly answered in the data, it’s okay to use general knowledge — as long as it’s highly likely to be correct.

        If you're not confident in your answer or if it's very specific, say you're unsure and suggest speaking to a real human.

        User question: {question}

        Database context:
        {db_context}

        Document-based answer:
        {pdf_answer}

        Now respond to the user in a helpful, clear, and honest way. Be conversational and kind.
    """

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    result = model.invoke(final_prompt)
    return result.content

def send_reply_to_chatwoot(conversation_id: int, content: str):
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    payload = {
        "content": content,
        "message_type": "outgoing"
    }
    response = requests.post(url, json=payload, headers=headers)
    print("Chatwoot API response:", response.status_code, response.text)
