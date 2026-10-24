"""Main entry point for the Egyptian Startup RAG application.

Provides a unified command-line and interactive interface for document chunking
(via LangChain), vector embedding generation, FAISS index construction, semantic
similarity retrieval, and context-augmented question answering.
"""

import argparse
import sys
from typing import List, Optional

from src.rag_pipeline import EgyptStartupRAG


def parse_arguments(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments for the RAG pipeline.

    Args:
        args: List of argument strings. Defaults to sys.argv[1:].

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Egyptian Startup RAG: Semantic Search & Q&A System"
    )
    parser.add_argument(
        "--pipeline",
        action="store_true",
        help="Execute the full pipeline: chunking, embedding, and indexing.",
    )
    parser.add_argument(
        "--chunk",
        action="store_true",
        help="Execute LangChain semantic text chunking on raw documents.",
    )
    parser.add_argument(
        "--embed",
        action="store_true",
        help="Generate dense vector embeddings using SentenceTransformers.",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Build and save the normalized FAISS vector index.",
    )
    parser.add_argument(
        "--search",
        type=str,
        metavar="QUERY",
        help="Perform semantic similarity search for a query and return top results.",
    )
    parser.add_argument(
        "--ask",
        type=str,
        metavar="QUESTION",
        help="Query the RAG pipeline and generate an answer using Gemini.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive terminal search and Q&A loop.",
    )

    return parser.parse_args(args)


def display_search_results(query: str, results: list) -> None:
    """Print formatted semantic search results to terminal.

    Args:
        query: User search query.
        results: Retrieved chunk dictionaries with similarity scores.
    """
    print("\n" + "=" * 60)
    print(f"SEARCH RESULTS FOR: '{query}'")
    print("=" * 60)
    if not results:
        print("No matching results found.")
        return

    for rank, item in enumerate(results, start=1):
        score = item.get("similarity_score", 0.0)
        print(f"\n#{rank} [Score: {score:.4f}]")
        print(f"Title: {item.get('title', 'N/A')}")
        print(f"URL:   {item.get('url', 'N/A')}")
        snippet = item.get("text", "").replace("\n", " ")[:250]
        print(f"Text:  {snippet}...")
        print("-" * 60)


def display_rag_answer(rag_output: dict) -> None:
    """Print formatted RAG answer and source citations to terminal.

    Args:
        rag_output: Dictionary containing answer and source references.
    """
    print("\n" + "=" * 70)
    print("RAG ANSWER:")
    print("=" * 70)
    print(rag_output.get("answer", ""))
    print("=" * 70)
    print("SOURCES USED:")
    sources = rag_output.get("sources", [])
    for idx, source in enumerate(sources, start=1):
        title = source.get("title", "Untitled").strip()
        url = source.get("url", "N/A")
        print(f"[{idx}] {title} -> {url}")
    print("=" * 70)


def run_interactive_loop(rag: EgyptStartupRAG) -> None:
    """Run an interactive console loop for semantic search and Q&A.

    Args:
        rag: Initialized EgyptStartupRAG instance.
    """
    print("\n" + "=" * 60)
    print(" Egyptian Startup RAG - Interactive Console")
    print(" Type your question, or type 'exit' or 'quit' to leave.")
    print("=" * 60)

    while True:
        try:
            query = input("\nAsk a question or enter search query: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting. Goodbye!")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("Exiting. Goodbye!")
            break

        print("\nRetrieving relevant documents and generating answer...")
        output = rag.generate_answer(query, top_k=5)
        display_rag_answer(output)


def main() -> None:
    """Entry point for command line execution."""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    args = parse_arguments()
    rag = EgyptStartupRAG()

    # If specific flags are supplied, execute corresponding actions
    if args.pipeline:
        print("Running complete RAG pipeline...")
        rag.run_full_pipeline()
        return

    if args.chunk:
        print("Running LangChain semantic text chunker...")
        num_chunks = rag.chunk_documents()
        print(f"Successfully generated {num_chunks} chunks.")
        return

    if args.embed:
        print("Generating vector embeddings...")
        embeddings = rag.generate_embeddings()
        print(f"Successfully computed embeddings with shape: {embeddings.shape}")
        return

    if args.index:
        print("Building FAISS index...")
        total_vectors = rag.build_index()
        print(f"Successfully indexed {total_vectors} vectors in FAISS.")
        return

    if args.search:
        results = rag.retrieve_context(args.search, top_k=5)
        display_search_results(args.search, results)
        return

    if args.ask:
        output = rag.generate_answer(args.ask, top_k=5)
        display_rag_answer(output)
        return

    # Default behavior: if interactive flag is set or no flags are set, run interactive loop
    if sys.stdin.isatty() or args.interactive:
        run_interactive_loop(rag)
    else:
        # Automated non-interactive run: demonstrate search capability
        demo_query = "What initiatives and programs does ITIDA offer for startups?"
        print(f"Non-interactive run detected. Executing demonstration query: '{demo_query}'")
        results = rag.retrieve_context(demo_query, top_k=3)
        display_search_results(demo_query, results)


if __name__ == "__main__":
    main()
