from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT_FILE = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("index")
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def main():
    # Ensure the output directory for the index exists
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {MODEL_NAME}...")
    print("(This might take a minute to download the model for the first time)")
    
    # Load the sentence transformer model
    model = SentenceTransformer(MODEL_NAME)

    # Read the text chunks generated in the previous step
    texts = []
    with open(INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])

    print(f"Embedding {len(texts)} chunks...")

    # Convert the raw text chunks into numerical vectors (Embeddings)
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Cast the numerical data type to float32 for FAISS compatibility
    embeddings = embeddings.astype("float32")

    # Save the resulting embeddings to a NumPy array file
    np.save(EMBEDDINGS_FILE, embeddings)

    print("Embedding shape:", embeddings.shape)
    print("Saved to:", EMBEDDINGS_FILE)

if __name__ == "__main__":
    main()