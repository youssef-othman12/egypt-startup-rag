from pathlib import Path
import json

INPUT_FILE = Path("data/raw/documents.jsonl")
OUTPUT_FILE = Path("data/processed/chunks.jsonl")

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 75):
    words = text.split()
    chunks = []
    
    # Calculate the step size based on desired chunk length and overlap
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk = " ".join(words[start:end]).strip()
        
        # Save the chunk only if it has a meaningful length
        if len(chunk) >= 100:
            chunks.append(chunk)

    return chunks

def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    chunk_id = 0

    with open(INPUT_FILE, encoding="utf-8") as infile, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:

        for line in infile:
            document = json.loads(line)
            chunks = chunk_text(document["text"])

            for chunk in chunks:
                item = {
                    "chunk_id": chunk_id,
                    "text": chunk,
                    "title": document["title"],
                    "url": document["url"],
                    "timestamp": document["timestamp"],
                    "source_file": document["source_file"],
                }
                
                outfile.write(json.dumps(item, ensure_ascii=False) + "\n")
                chunk_id += 1

    print(f"Success! Created {chunk_id} chunks.")

if __name__ == "__main__":
    main()