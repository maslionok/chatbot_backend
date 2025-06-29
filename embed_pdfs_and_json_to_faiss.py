import os
import glob
import json
import numpy as np
import faiss
from openai import OpenAI
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# --- Config ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBED_MODEL = "text-embedding-3-large"

client = OpenAI(api_key=OPENAI_API_KEY)

def load_pdf_texts(pdf_folder):
    chunks = []
    for path in glob.glob(os.path.join(pdf_folder, "*.pdf")):
        try:
            reader = PdfReader(path)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    # Split long pages into ~300 word chunks
                    words = text.split()
                    for i in range(0, len(words), 300):
                        chunk = ' '.join(words[i:i+300])
                        if chunk.strip():
                            chunks.append(chunk)
        except Exception as e:
            print(f"PDF load error {path}: {e}")
    return chunks

def load_json_chunks(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Flatten if needed
    if isinstance(data, list):
        return [str(x) for x in data if str(x).strip()]
    elif isinstance(data, dict):
        return [str(v) for v in data.values() if str(v).strip()]
    else:
        return [str(data)]

def embed_chunks(chunks, batch_size=100):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunks[i:i+batch_size])
        embeddings.extend([d.embedding for d in resp.data])
    return np.array(embeddings).astype("float32")

def ensure_dir_exists(path):
    dir_path = os.path.dirname(os.path.abspath(path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path, exist_ok=True)

def main():
    # Hardcoded paths for your project structure
    pdf_folder = "docs"
    json_path = "json/crawl_chunks.json"
    faiss_path = "rag/rag.index"
    chunks_out = "rag/rag_chunks.json"
    # Optionally, you can also save embeddings as .npy if needed
    # npy_path = "data/rag_embs.npy"

    # Ensure output directories exist
    ensure_dir_exists(faiss_path)
    ensure_dir_exists(chunks_out)
    # If you want to use npy_path, also: ensure_dir_exists(npy_path)

    # Load data
    pdf_chunks = load_pdf_texts(pdf_folder)
    print(f"Loaded {len(pdf_chunks)} chunks from PDFs in {pdf_folder}")
    json_chunks = load_json_chunks(json_path)
    print(f"Loaded {len(json_chunks)} chunks from {json_path}")

    all_chunks = pdf_chunks + json_chunks
    print(f"Total chunks to embed: {len(all_chunks)}")

    # Embed
    embeddings = embed_chunks(all_chunks)
    print(f"Generated embeddings: {embeddings.shape}")

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    print(f"Saved FAISS index to {faiss_path}")

    # Save chunks in the format main.py expects (JSON list, same order as FAISS)
    with open(chunks_out, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved chunks to {chunks_out}")

    # Optionally save embeddings as .npy
    # np.save(npy_path, embeddings)
    # print(f"Saved embeddings as {npy_path}")

if __name__ == "__main__":
    main()
