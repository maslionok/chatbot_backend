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
import faiss

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

# Use the same embedding model for both library and user queries
EMBED_MODEL = "text-embedding-3-large"  # Must match the model used in embed_pdfs_and_json_to_faiss.py

# LangChain
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL)

# --- RAG FAISS index and chunks ---
def load_faiss_and_chunks():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    faiss_path = os.path.join(base_dir, "rag", "rag.index")
    chunks_path = os.path.join(base_dir, "rag", "rag_chunks.json")
    if not os.path.exists(faiss_path) or not os.path.exists(chunks_path):
        print(f"[ERROR] FAISS index or chunks file missing: {faiss_path}, {chunks_path}")
        return None, None
    index = faiss.read_index(faiss_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    return index, chunks

faiss_index, rag_chunks = load_faiss_and_chunks()

def embed_query(query: str):
    # Use the same model as for the library embeddings
    resp = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBED_MODEL).embed_query(query)
    return np.array([resp]).astype("float32")

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    # print("Webhook received:", data)

    if data.get("message_type") != "incoming":
        return {"status": "ignored"}

    conversation_id = data["conversation"]["id"]
    message = data.get("content", "")

    intent = detect_user_intent(message)
    # print(f"User intent: {intent}")

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

    reply = generate_rag_reply(conversation_id, message)
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

def get_conversation_history(conversation_id: int) -> list:
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"[ERROR] Failed to fetch chat history: {response.text}")
        return []

    messages = response.json().get("payload", [])
    history = []
    for msg in messages:
        sender = msg["sender"]["name"]
        content = msg["content"]
        msg_type = msg["message_type"]
        if msg_type == 1:
            history.append(f"bot: {content}")
        elif msg_type == 0:
            history.append(f"user: {content}")
    return history

# --- Global counters for averaging ---
total_words_sum = 0
total_calls = 0

def generate_rag_reply(conversation_id: int, question: str) -> str:
    global total_words_sum, total_calls
    if faiss_index is None or rag_chunks is None:
        return "Sorry, the knowledge base is not available right now."

    q_emb = embed_query(question)
    D, I = faiss_index.search(q_emb, 8)
    retrieved_chunks = [rag_chunks[i] for i in I[0] if i < len(rag_chunks)]
    context = "\n\n".join(retrieved_chunks)

    chat_history = get_conversation_history(conversation_id)
    history_snippet = "\n".join(chat_history[-10:])

    system_prompt = (
        "You are an expert assistant specializing in curtains and our company. "
        "Answer user questions naturally, in a friendly and professional tone, as if you are speaking from your own expertise. "
        "Do NOT mention or reference any context, sources, or documents. "
        "Keep your replies short, precise, and directly address the user's question. "
        "If you don't know the answer, politely say so and let the user know they can ask to be switched to a real human. "
        "If the user asks for a direct link to a product, use the base URL 'https://sklep.idescu.pl/' and append the product's 'url_key' value from the context, ending with '.html'. For example: https://sklep.idescu.pl/{url_key}.html"
    )

    user_prompt = (
        f"Chat history so far:\n{history_snippet}\n\n"
        f"Relevant context:\n{context}\n\n"
        f"User now asks:\n{question}"
    )

    total_words = len((system_prompt + user_prompt).split())
    approx_tokens = int(total_words * 0.75)

    # Update global counters
    total_words_sum += total_words
    total_calls += 1
    avg_words = total_words_sum / total_calls
    avg_tokens = avg_words * 0.75

    print(f"[DEBUG] Input prompt: {total_words} words, ~{approx_tokens} tokens (words*0.75)")
    print(f"[DEBUG] Running average: {avg_words:.2f} words, {avg_tokens:.2f} tokens over {total_calls} calls")

    model = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name=MODEL_NAME)
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
