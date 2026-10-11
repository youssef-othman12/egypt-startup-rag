import json
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup

RAW_DIR = Path("data/raw")
OUTPUT_FILE = RAW_DIR / "documents.jsonl"

def extract_title(soup: BeautifulSoup) -> str:
    if soup.title:
        return soup.title.get_text(" ", strip=True)
    return ""

def html_to_text(soup: BeautifulSoup) -> str:
    # Remove obvious non-content elements to keep the text clean
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(" ", strip=True)

def process_urls():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as out_f:
        # Process all collected URL files in the raw directory
        for url_file in sorted(RAW_DIR.glob("*_urls.jsonl")):
            print(f"Processing URLs from: {url_file.name}")
            source_name = url_file.name.replace('_urls.jsonl', '')
            
            with open(url_file, "r", encoding="utf-8") as in_f:
                for line in in_f:
                    if not line.strip(): continue
                    
                    try:
                        # cdx_toolkit iter outputs strings like: status 200, timestamp 123, url https://...
                        # Extract the target URL and the archiving timestamp
                        parts = line.split(", ")
                        if len(parts) >= 3:
                            timestamp = parts[1].replace("timestamp ", "").strip()
                            url = parts[2].replace("url ", "").strip()
                            
                            print(f"  Fetching: {url}")
                            response = requests.get(url, headers=headers, timeout=10)
                            
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, "lxml")
                                text = html_to_text(soup)
                                title = extract_title(soup)
                                
                                # Skip pages with too little actual text content (e.g., login pages)
                                if len(text) < 200:
                                    print("    -> Skipped: Too little content")
                                    continue
                                
                                doc = {
                                    "url": url,
                                    "title": title,
                                    "timestamp": timestamp,
                                    "source_file": source_name,
                                    "text": text
                                }
                                
                                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                
                                # Pause briefly between requests to avoid server bans
                                time.sleep(1) 
                            else:
                                print(f"    -> Failed with status {response.status_code}")
                    except Exception as e:
                        print(f"    -> Error fetching {url}: {e}")

    print(f"\nDone! Cleaned documents saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    process_urls()