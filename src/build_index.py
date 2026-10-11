from pathlib import Path
import json
import faiss
import numpy as np

EMBEDDINGS_FILE = Path("index/embeddings.npy")
METADATA_FILE = Path("index/metadata.json")
INDEX_FILE = Path("index/startup.index")

CHUNKS_FILE = Path("data/processed/chunks.jsonl")

def main():
    # Load the 384-dimensional embeddings we just generated
    embeddings = np.load(EMBEDDINGS_FILE).astype("float32")

    # Normalize the vectors using L2 normalization
    # When vectors are normalized, Inner Product (IP) becomes equivalent to Cosine Similarity
    faiss.normalize_L2(embeddings)

    # Get the dimension size (should be 384)
    dimension = embeddings.shape[1]

    # Initialize a FAISS index for Inner Product search
    index = faiss.IndexFlatIP(dimension)

    # Add the normalized vectors to the FAISS index
    index.add(embeddings)
    print(f"Successfully added {index.ntotal} vectors to the FAISS index.")

    # Load the original text chunks to save as metadata
    metadata = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            metadata.append(json.loads(line))

    # Save the FAISS index to disk for future querying
    faiss.write_index(index, str(INDEX_FILE))

    # Save the metadata (text, URLs, titles) alongside the index
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("FAISS index saved to:", INDEX_FILE)
    print("Metadata saved to:", METADATA_FILE)

if __name__ == "__main__":
    main()