import os
import json
import time
import faiss
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import errors

# Load environment variables
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found. Please add it to your .env file.")

INDEX_FILE = "index/startup.index"
METADATA_FILE = "index/metadata.json"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Set to the new required model version
GEMINI_MODEL = "gemini-3.6-flash"

def retrieve_context(query: str, model, index, metadata, top_k: int = 5):
    """Retrieve the top-k most relevant text chunks from the FAISS index."""
    query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    
    return results

def build_prompt(query: str, context_chunks: list) -> str:
    """Construct a strict prompt for the LLM using only the retrieved context."""
    formatted_context = ""
    for i, chunk in enumerate(context_chunks, 1):
        formatted_context += f"--- CONTEXT CHUNK {i} ---\n"
        formatted_context += f"Source Title: {chunk['title']}\n"
        formatted_context += f"Source URL: {chunk['url']}\n"
        formatted_context += f"Content: {chunk['text']}\n\n"

    prompt = f"""You are an expert AI analyst specializing in the Egyptian startup ecosystem.
Answer the user's question accurately using ONLY the provided context snippets below. 
If the context does not contain enough information to answer the question, clearly state that the information is not available in your database. Do not hallucinate outside information.
Always cite the source titles and URLs when providing facts.

CONTEXT:
{formatted_context}

USER QUESTION: {query}

ANSWER:"""
    return prompt

def generate_with_retry(client, prompt, max_retries=3):
    """Generate content with automatic retries for 503 ServerError."""
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text
        except errors.ServerError as e:
            if "503" in str(e) or "UNAVAILABLE" in str(e):
                if attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    print(f"    [!] Server busy (503). Retrying in {wait_time} seconds... (Attempt {attempt + 1}/{max_retries})")
                    time.sleep(wait_time)
                else:
                    return f"Error: Gemini API is currently experiencing high demand and returned a 503 error. Please try again later."
            else:
                return f"An API Server Error occurred: {e}"
        except Exception as e:
            return f"An unexpected error occurred: {e}"

def main():
    print("\nLoading RAG System (FAISS + Gemini)...")
    
    # Initialize models ONCE outside the loop for extreme speed
    embedding_model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(INDEX_FILE)
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    client = genai.Client()
    print("--- Egyptian Startup RAG is Ready ---")

    while True:
        query = input("\nAsk a question about Egyptian startups (or type 'exit' to quit): ")

        if query.lower() in ['exit', 'quit']:
            break

        print("\n[1/2] Retrieving relevant context from FAISS...")
        context_chunks = retrieve_context(query, embedding_model, index, metadata, top_k=5)
        prompt = build_prompt(query, context_chunks)

        print(f"[2/2] Generating answer using {GEMINI_MODEL}...")
        answer = generate_with_retry(client, prompt)

        print("=" * 70)
        print("RAG ANSWER:")
        print("=" * 70)
        print(answer)
        print("=" * 70)
        print("\nSOURCES USED:")
        for i, chunk in enumerate(context_chunks, 1):
            print(f"[{i}] {chunk['title']} -> {chunk['url']}")

if __name__ == "__main__":
    main()