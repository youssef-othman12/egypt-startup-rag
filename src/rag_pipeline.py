"""Egypt Startup RAG Pipeline Module.

This module consolidates web document processing, intelligent semantic chunking
using LangChain and Hugging Face, vector embedding generation, FAISS indexing,
similarity retrieval, and context-augmented answer generation with Gemini.
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union

from bs4 import BeautifulSoup
from dotenv import load_dotenv
import faiss
from google import genai
from google.genai import errors
from langchain_text_splitters import RecursiveCharacterTextSplitter
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Type alias for path inputs
UnionPath = Union[str, Path]

# Configure module-level logger
logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Load environment variables
load_dotenv()


class EgyptStartupRAG:  # pylint: disable=too-many-instance-attributes
    """Unified RAG pipeline for the Egyptian startup ecosystem.

    Provides end-to-end functionality for document processing, intelligent text
    chunking via LangChain, dense vector embeddings via Hugging Face sentence
    transformers, FAISS vector indexing, semantic search, and LLM answer generation.
    """

    # pylint: disable=too-many-arguments,too-many-positional-arguments
    def __init__(
        self,
        raw_dir: UnionPath = Path("data/raw"),
        processed_dir: UnionPath = Path("data/processed"),
        index_dir: UnionPath = Path("index"),
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        gemini_model_name: str = "gemini-3.6-flash",
        chunk_size: int = 500,
        chunk_overlap: int = 75,
        min_chunk_length: int = 100,
    ) -> None:
        """Initialize the EgyptStartupRAG pipeline with directory paths and model settings.

        Args:
            raw_dir: Path to directory containing raw scraped documents.
            processed_dir: Path to directory for processed chunk data.
            index_dir: Path to directory for FAISS index and metadata.
            embedding_model_name: Hugging Face model identifier for embeddings.
            gemini_model_name: Gemini LLM version for answer generation.
            chunk_size: Maximum chunk size for LangChain RecursiveCharacterTextSplitter.
            chunk_overlap: Overlapping character count between consecutive chunks.
            min_chunk_length: Minimum character length required to keep a chunk.
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.index_dir = Path(index_dir)

        self.documents_file = self.raw_dir / "documents.jsonl"
        self.chunks_file = self.processed_dir / "chunks.jsonl"
        self.embeddings_file = self.index_dir / "embeddings.npy"
        self.metadata_file = self.index_dir / "metadata.json"
        self.index_file = self.index_dir / "startup.index"

        self.embedding_model_name = embedding_model_name
        self.gemini_model_name = gemini_model_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_length = min_chunk_length

        self._embedding_model: Optional[SentenceTransformer] = None
        self._faiss_index: Optional[faiss.IndexFlatIP] = None
        self._metadata: Optional[List[Dict[str, Any]]] = None
        self._gemini_client: Optional[genai.Client] = None

        # Initialize LangChain recursive text splitter with semantic hierarchy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    # -------------------------------------------------------------------------
    # 1. Cleaning & Scraping Methods
    # -------------------------------------------------------------------------

    @staticmethod
    def extract_title(soup: BeautifulSoup) -> str:
        """Extract clean page title from BeautifulSoup object.

        Args:
            soup: Parsed HTML document.

        Returns:
            Cleaned title string or empty string if not found.
        """
        if soup.title and soup.title.string:
            return soup.title.get_text(" ", strip=True)
        return ""

    @staticmethod
    def html_to_text(soup: BeautifulSoup) -> str:
        """Decompose non-content HTML tags and extract readable text.

        Args:
            soup: Parsed HTML document.

        Returns:
            Sanitized visible text content.
        """
        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "header"]):
            tag.decompose()
        clean_str = soup.get_text(" ", strip=True)
        return clean_str.replace("\u200b", "").replace("\ufeff", "")

    def fetch_and_clean_documents(self, request_delay: float = 1.0) -> int:  # pylint: disable=too-many-locals
        """Fetch and clean web documents referenced in raw URL files.

        Args:
            request_delay: Pause duration in seconds between web requests.

        Returns:
            Number of successfully cleaned documents saved.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            )
        }

        saved_count = 0
        url_files = sorted(self.raw_dir.glob("*_urls.jsonl"))
        if not url_files:
            logger.warning("No *_urls.jsonl files found in %s", self.raw_dir)
            return saved_count

        with open(self.documents_file, "w", encoding="utf-8") as out_f:
            for url_file in url_files:
                logger.info("Processing URLs from: %s", url_file.name)
                source_name = url_file.name.replace("_urls.jsonl", "")

                with open(url_file, "r", encoding="utf-8") as in_f:
                    for line in in_f:
                        if not line.strip():
                            continue

                        parts = line.strip().split(", ")
                        if len(parts) < 3:
                            continue

                        timestamp = parts[1].replace("timestamp ", "").strip()
                        url = parts[2].replace("url ", "").strip()

                        try:
                            response = requests.get(url, headers=headers, timeout=10)
                            if response.status_code == 200:
                                soup = BeautifulSoup(response.text, "lxml")
                                text = self.html_to_text(soup)
                                title = self.extract_title(soup)

                                if len(text) < 200:
                                    continue

                                doc = {
                                    "url": url,
                                    "title": title,
                                    "timestamp": timestamp,
                                    "source_file": source_name,
                                    "text": text,
                                }
                                out_f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                                saved_count += 1
                                time.sleep(request_delay)
                        except (requests.RequestException, OSError) as exc:
                            logger.warning("Error fetching %s: %s", url, exc)

        logger.info("Cleaned %d documents saved to %s", saved_count, self.documents_file)
        return saved_count

    # -------------------------------------------------------------------------
    # 2. Semantic Chunking with LangChain / Hugging Face
    # -------------------------------------------------------------------------

    def chunk_text(self, text: str) -> List[str]:
        """Split text into semantically cohesive chunks using LangChain.

        Employs recursive character splitting with semantic boundary hierarchy
        (paragraphs, lines, sentences, words) and enforces a minimum character
        threshold to prevent noisy or trivial fragments.

        Args:
            text: Raw input text string.

        Returns:
            List of clean text chunks.
        """
        raw_chunks = self.text_splitter.split_text(text)
        return [
            chunk.strip()
            for chunk in raw_chunks
            if len(chunk.strip()) >= self.min_chunk_length
        ]

    def chunk_documents(self) -> int:
        """Process raw documents into semantically chunked records.

        Reads from documents_file, chunks each document using LangChain,
        attaches document metadata to each chunk, and writes to chunks_file.

        Returns:
            Total number of chunks generated.

        Raises:
            FileNotFoundError: If documents_file does not exist.
        """
        if not self.documents_file.exists():
            raise FileNotFoundError(
                f"Input documents file not found at: {self.documents_file}"
            )

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        chunk_id = 0

        with open(self.documents_file, "r", encoding="utf-8") as in_f, \
             open(self.chunks_file, "w", encoding="utf-8") as out_f:

            for line in in_f:
                if not line.strip():
                    continue

                document = json.loads(line)
                chunks = self.chunk_text(document.get("text", ""))

                for chunk in chunks:
                    item = {
                        "chunk_id": chunk_id,
                        "text": chunk,
                        "title": document.get("title", ""),
                        "url": document.get("url", ""),
                        "timestamp": document.get("timestamp", ""),
                        "source_file": document.get("source_file", ""),
                    }
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    chunk_id += 1

        logger.info("Created %d chunks saved to %s", chunk_id, self.chunks_file)
        return chunk_id

    # -------------------------------------------------------------------------
    # 3. Embedding Generation
    # -------------------------------------------------------------------------

    def get_embedding_model(self) -> SentenceTransformer:
        """Lazy load and cache the Hugging Face SentenceTransformer model.

        Returns:
            SentenceTransformer model instance.
        """
        if self._embedding_model is None:
            logger.info("Loading embedding model: %s", self.embedding_model_name)
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def generate_embeddings(self, batch_size: int = 32) -> np.ndarray:
        """Compute dense vector representations for all text chunks.

        Args:
            batch_size: Batch size for model inference.

        Returns:
            NumPy array of shape (num_chunks, embedding_dim) as float32.

        Raises:
            FileNotFoundError: If chunks_file does not exist.
        """
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Processed chunks file not found at: {self.chunks_file}"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        texts: List[str] = []

        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    texts.append(item["text"])

        logger.info("Computing embeddings for %d chunks...", len(texts))
        model = self.get_embedding_model()

        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        np.save(self.embeddings_file, embeddings)
        logger.info("Embeddings shape %s saved to %s", embeddings.shape, self.embeddings_file)
        return embeddings

    # -------------------------------------------------------------------------
    # 4. FAISS Index Construction
    # -------------------------------------------------------------------------

    def build_index(self) -> int:
        """Build and persist an L2-normalized FAISS IndexFlatIP index.

        Normalizing vectors guarantees that Inner Product search computes Cosine
        Similarity. Metadata is saved alongside the index.

        Returns:
            Total number of indexed vectors.

        Raises:
            FileNotFoundError: If embeddings or chunks file does not exist.
        """
        if not self.embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found at: {self.embeddings_file}"
            )
        if not self.chunks_file.exists():
            raise FileNotFoundError(
                f"Chunks file not found at: {self.chunks_file}"
            )

        self.index_dir.mkdir(parents=True, exist_ok=True)
        embeddings = np.load(self.embeddings_file).astype("float32")

        # L2 normalization makes Inner Product (IP) equivalent to Cosine Similarity
        faiss.normalize_L2(embeddings)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)  # pylint: disable=no-value-for-parameter

        metadata: List[Dict[str, Any]] = []
        with open(self.chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    metadata.append(json.loads(line))

        faiss.write_index(index, str(self.index_file))
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        self._faiss_index = index
        self._metadata = metadata

        logger.info("FAISS index (%d vectors) saved to %s", index.ntotal, self.index_file)
        logger.info("Metadata saved to %s", self.metadata_file)
        return index.ntotal

    def load_index(self) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
        """Load the FAISS index and metadata from disk if not already in memory.

        Returns:
            Tuple of (faiss_index, metadata_records).

        Raises:
            FileNotFoundError: If index_file or metadata_file does not exist.
        """
        if self._faiss_index is None or self._metadata is None:
            if not self.index_file.exists():
                raise FileNotFoundError(f"FAISS index file not found at: {self.index_file}")
            if not self.metadata_file.exists():
                raise FileNotFoundError(f"Metadata file not found at: {self.metadata_file}")

            self._faiss_index = faiss.read_index(str(self.index_file))
            with open(self.metadata_file, "r", encoding="utf-8") as f:
                self._metadata = json.load(f)

        return self._faiss_index, self._metadata

    # -------------------------------------------------------------------------
    # 5. Semantic Search & Retrieval
    # -------------------------------------------------------------------------

    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k most relevant chunks for a user query.

        Args:
            query: User search text or question.
            top_k: Number of nearest chunks to retrieve.

        Returns:
            List of matching metadata dictionaries with similarity scores.
        """
        index, metadata = self.load_index()
        model = self.get_embedding_model()

        query_embedding = model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(metadata):
                item = dict(metadata[idx])
                item["similarity_score"] = float(score)
                results.append(item)

        return results

    # -------------------------------------------------------------------------
    # 6. Prompt Construction & Gemini RAG Generation
    # -------------------------------------------------------------------------

    @staticmethod
    def build_prompt(query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Construct a structured grounded prompt containing retrieved snippets.

        Args:
            query: User question.
            context_chunks: Retrieved context chunks.

        Returns:
            Formatted prompt string ready for LLM generation.
        """
        formatted_context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            formatted_context_parts.append(
                f"--- CONTEXT CHUNK {i} ---\n"
                f"Source Title: {chunk.get('title', 'Unknown')}\n"
                f"Source URL: {chunk.get('url', 'N/A')}\n"
                f"Content: {chunk.get('text', '')}\n"
            )

        context_str = "\n".join(formatted_context_parts)
        return (
            "You are an expert AI analyst specializing in the Egyptian startup ecosystem.\n"
            "Answer the user's question accurately using ONLY the provided context "
            "snippets below.\n"
            "If the context does not contain enough information to answer the question, clearly\n"
            "state that the information is not available in your database. "
            "Do not hallucinate outside information.\n"
            "Always cite the source titles and URLs when providing facts.\n\n"
            f"CONTEXT:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            "ANSWER:"
        )

    def get_gemini_client(self) -> genai.Client:
        """Obtain authenticated Gemini API client.

        Returns:
            Authenticated Google GenAI client.

        Raises:
            ValueError: If GEMINI_API_KEY environment variable is missing.
        """
        if self._gemini_client is None:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError(
                    "GEMINI_API_KEY is not set. Please set GEMINI_API_KEY in your environment."
                )
            self._gemini_client = genai.Client()
        return self._gemini_client

    def generate_answer(
        self, query: str, top_k: int = 5, max_retries: int = 3
    ) -> Dict[str, Any]:
        """Execute complete RAG inference pipeline: retrieve context & generate answer.

        Args:
            query: User question.
            top_k: Number of context chunks to retrieve.
            max_retries: Retry attempts for transient 503 / UNAVAILABLE server errors.

        Returns:
            Dictionary containing 'query', 'answer', 'sources', and 'prompt'.
        """
        context_chunks = self.retrieve_context(query, top_k=top_k)
        prompt = self.build_prompt(query, context_chunks)

        try:
            client = self.get_gemini_client()
        except ValueError as err:
            return {
                "query": query,
                "answer": f"Client initialization error: {err}",
                "sources": context_chunks,
                "prompt": prompt,
            }

        answer_text = ""
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.gemini_model_name,
                    contents=prompt,
                )
                answer_text = response.text or ""
                break
            except errors.ServerError as err:
                err_msg = str(err)
                if ("503" in err_msg or "UNAVAILABLE" in err_msg) and attempt < max_retries - 1:
                    wait_time = 5 * (attempt + 1)
                    logger.warning(
                        "Gemini server busy (503). Retrying in %ds (Attempt %d/%d)...",
                        wait_time,
                        attempt + 1,
                        max_retries,
                    )
                    time.sleep(wait_time)
                else:
                    answer_text = f"API Server Error: {err}"
                    break
            except errors.APIError as err:
                answer_text = f"Gemini API Error: {err}"
                break

        return {
            "query": query,
            "answer": answer_text,
            "sources": context_chunks,
            "prompt": prompt,
        }

    # -------------------------------------------------------------------------
    # 7. End-to-End Orchestration
    # -------------------------------------------------------------------------

    def run_full_pipeline(self, scrape_first: bool = False) -> None:
        """Execute the entire RAG pipeline from documents to FAISS index.

        Args:
            scrape_first: If True, fetches and cleans URLs first before chunking.
        """
        logger.info("Starting Egypt Startup RAG Pipeline...")

        if scrape_first:
            self.fetch_and_clean_documents()

        num_chunks = self.chunk_documents()
        logger.info("[Step 1/3 Complete] Generated %d chunks.", num_chunks)

        embeddings = self.generate_embeddings()
        logger.info("[Step 2/3 Complete] Generated embeddings with shape %s.", embeddings.shape)

        num_indexed = self.build_index()
        logger.info("[Step 3/3 Complete] Indexed %d vectors successfully.", num_indexed)
        logger.info("RAG Pipeline is now ready for semantic queries.")
