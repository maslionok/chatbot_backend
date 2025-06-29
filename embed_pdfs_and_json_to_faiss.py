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
EMBED_MODEL = "text-embedding-3-small"

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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Embed PDF and JSON text into FAISS index for RAG.")
    parser.add_argument("--pdf_folder", required=True, help="Folder with .pdf files")
    parser.add_argument("--json", required=True, help="JSON file with text chunks")
    parser.add_argument("--faiss", required=True, help="Output FAISS index file")
    parser.add_argument("--npy", required=False, help="Optional: Save embeddings as .npy file")
    args = parser.parse_args()

    # Load data
    pdf_chunks = load_pdf_texts(args.pdf_folder)
    print(f"Loaded {len(pdf_chunks)} chunks from PDFs in {args.pdf_folder}")
    json_chunks = load_json_chunks(args.json)
    print(f"Loaded {len(json_chunks)} chunks from {args.json}")

    all_chunks = pdf_chunks + json_chunks
    print(f"Total chunks to embed: {len(all_chunks)}")

    # Embed
    embeddings = embed_chunks(all_chunks)
    print(f"Generated embeddings: {embeddings.shape}")

    # Save FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, args.faiss)
    print(f"Saved FAISS index to {args.faiss}")

    # Optionally save embeddings as .npy
    if args.npy:
        np.save(args.npy, embeddings)
        print(f"Saved embeddings as {args.npy}")

if __name__ == "__main__":
    main()
