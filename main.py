from fastapi import FastAPI, Request
import requests
import openai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

CHATWOOT_API_KEY = os.getenv("CHATWOOT_API_KEY")
CHATWOOT_ACCOUNT_ID = os.getenv("CHATWOOT_ACCOUNT_ID")
CHATWOOT_URL = os.getenv("CHATWOOT_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai.api_key = OPENAI_API_KEY

@app.post("/webhook")
async def webhook(request: Request):
    data = await request.json()
    
    # Only respond to incoming messages (not internal notes or agent messages)
    if data.get("message_type") != "incoming":
        return {"status": "ignored"}
    
    conversation_id = data["conversation"]["id"]
    message = data.get("content", "")

    # Generate AI reply (replace with your own RAG logic if needed)
    reply = generate_ai_reply(message)

    # Send reply to Chatwoot
    send_reply_to_chatwoot(conversation_id, reply)
    return {"status": "ok"}

def generate_ai_reply(user_message: str) -> str:
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": user_message}]
    )
    return response.choices[0].message.content

def send_reply_to_chatwoot(conversation_id: int, content: str):
    url = f"{CHATWOOT_URL}/api/v1/accounts/{CHATWOOT_ACCOUNT_ID}/conversations/{conversation_id}/messages"
    headers = {"api_access_token": CHATWOOT_API_KEY}
    data = {
        "content": content,
        "message_type": "outgoing"
    }
    requests.post(url, json=data, headers=headers)
