import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INDEX_FILE = "index/startup.index"
METADATA_FILE = "index/metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    print("Loading search engine... (this takes a few seconds)")
    
    # Load the embedding model
    model = SentenceTransformer(MODEL_NAME)

    # Load the FAISS index from disk
    index = faiss.read_index(INDEX_FILE)

    # Load the metadata (to map vector IDs back to actual text)
    with open(METADATA_FILE, encoding="utf-8") as f:
        metadata = json.load(f)

    print("\n--- Search Engine Ready ---")
    
    # Simple loop to keep asking questions until the user types 'exit'
    while True:
        query = input("\nAsk a question (or type 'exit' to quit): ")
        
        if query.lower() in ['exit', 'quit']:
            break

        # 1. Convert the user's question into a 384-dimensional vector
        query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")

        # 2. Normalize the query vector (required because we normalized the chunks)
        faiss.normalize_L2(query_embedding)

        # 3. Search the FAISS index for the top 5 closest vectors
        scores, indices = index.search(query_embedding, 5)

        print("\n" + "="*50)
        print("TOP 5 RESULTS:")
        print("="*50)

        # 4. Loop through the results and print the corresponding metadata
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            result = metadata[idx]

            print(f"\n#{rank} (Similarity Score: {score:.4f})")
            print(f"Title: {result['title']}")
            print(f"URL: {result['url']}")
            print("-" * 30)
            # Print the first 400 characters of the chunk so it doesn't flood the terminal
            snippet = result["text"][:400].replace("\n", " ")
            print(f"Text Snippet: {snippet}...")
            print("="*50)

if __name__ == "__main__":
    main()