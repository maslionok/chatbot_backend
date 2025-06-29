from fastapi import FastAPI, Request
import os
import requests
import shelve
# import asyncio  # No longer used for auto-greetings
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import sqlite3
# from datetime import datetime, timedelta  # No longer needed for greetings
import gzip
import pickle
import numpy as np
import json

MODEL_NAME = "gpt-4.1-mini"

load_dotenv()
app = FastAPI()

# Env vars
CHATWOOT_URL = os.getenv("CHATWOOT_URL")
CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global stores
conversation_ai_status = {}
# last_greet_sent = {}  # Removed greeting tracking

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

def load_compressed(db, key):
    """Load a compressed object from shelve."""
    return pickle.loads(gzip.decompress(db[key]))

def load_crawl_chunks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    crawl_json_path = os.path.join(base_dir, "data", "crawl_chunks.json")
    print(f"[DEBUG] Loading crawl chunks from: {crawl_json_path}")
    if not os.path.exists(crawl_json_path):
        print(f"[ERROR] Crawl chunks file does not exist at {crawl_json_path}")
        return []
    try:
        with open(crawl_json_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
            print(f"[DEBUG] Loaded {len(chunks)} crawl chunks")
            return chunks
    except Exception as e:
        print(f"[ERROR] Failed to load crawl chunks: {e}")
        return []

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

# @app.post("/contact_opened")
# async def contact_opened(request: Request):
#     data = await request.json()
#     print("Contact opened:", data)

#     conversation = data.get("current_conversation")
#     if not conversation or "id" not in conversation:
#         print("Missing conversation ID in payload.")
#         return {"status": "error", "detail": "Missing conversation ID."}

#     conversation_id = conversation["id"]

#     now = datetime.utcnow()
#     last_greet = last_greet_sent.get(conversation_id)

#     if last_greet and now - last_greet < timedelta(minutes=20):
#         print(f"Skipping greet for conversation {conversation_id}: recently greeted.")
#         return {"status": "skipped"}

#     last_greet_sent[conversation_id] = now
#     asyncio.create_task(wait_and_greet(conversation_id))
#     return {"status": "ok"}

# async def wait_and_greet(conversation_id: int):
#     await asyncio.sleep(60)
#     if is_user_still_inactive(conversation_id):
#         send_reply_to_chatwoot(conversation_id, "Hey 👋 Could I assist you today?")

# def is_user_still_inactive(conversation_id: int) -> bool:
#     url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
#     headers = {"api_access_token": CHATWOOT_API_KEY}
#     response = requests.get(url, headers=headers)
#     if response.status_code != 200:
#         print("Failed to check messages:", response.text)
#         return False
#     messages = response.json().get("payload", [])
#     return all(msg["message_type"] != "incoming" for msg in messages)

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
    # --- PDF retrieval ---
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(openai_api_key=OPENAI_API_KEY),
        retriever=retriever
    )
    pdf_answer = qa_chain.run(question)

    # --- Website crawl retrieval ---
    crawl_chunks = load_crawl_chunks()
    # Use all crawl chunks for context
    crawl_context = "\n\n".join(crawl_chunks) if crawl_chunks else ""
    print(f"[DEBUG] crawl_context in generate_rag_reply: {crawl_context[:500]}...")  # Print only first 500 chars

    # --- Combine both sources for the prompt ---
    context = ""
    if crawl_context:
        print(f"[DEBUG] We actually have crawl_context")
        context += f"Crawled website context:\n{crawl_context}\n\n"
    context += f"Document-based answer:\n{pdf_answer}"

    # Polite, friendly, and helpful system prompt
    system_prompt = (
        "You are a precise and polite assistant. Answer ONLY based on the provided context below."
        "If the answer is not in the context, say you don't know in a friendly and polite way. "
        "Be helpful and avoid rudeness. "
        "If the answer must be short, try to make it a bit longer and more polite, offering a friendly tone. "
        "Do not use any outside knowledge. "
        "If you don't know the answer, also tell the user they can ask to be switched to a real human. "
    )
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name = MODEL_NAME)
    result = model.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
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
