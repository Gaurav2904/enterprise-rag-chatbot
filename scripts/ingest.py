#!/usr/bin/env python3
"""
scripts/ingest.py — CLI helper to ingest documents into the FAISS vector store.

Usage:
    python scripts/ingest.py --files docs/report.pdf docs/manual.docx --chunk-size 512
"""

import argparse
import sys
import os
from pathlib import Path

# Ensure project root is on the path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app.vector_store import VectorStoreManager


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into FAISS vector store")
    parser.add_argument(
        "--files", nargs="+", required=True,
        help="One or more file paths (PDF, TXT, DOCX) to ingest",
    )
    parser.add_argument("--chunk-size", type=int, default=512, help="Token chunk size (default: 512)")
    parser.add_argument("--chunk-overlap", type=int, default=64, help="Chunk overlap (default: 64)")
    args = parser.parse_args()

    print(f"Initialising VectorStoreManager …")
    vsm = VectorStoreManager()

    print(f"Ingesting {len(args.files)} file(s) …")
    n = vsm.ingest(
        file_paths=args.files,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )
    print(f"✓ Indexed {n} chunks.")
    print(f"Stats: {vsm.stats()}")


if __name__ == "__main__":
    main()
