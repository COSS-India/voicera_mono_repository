"""
Query the local Chroma store: embed a question with OpenAI, return nearest chunks.

Must use the same embedding model as embed_chunks.py / vectors.npz (default:
text-embedding-3-small).

Examples:
  cd voicera_backend/rag_system
  python query_chroma.py "What is this document about?"
  python query_chroma.py "Your question" -k 5

Requires: chromadb, openai, numpy
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

from rag_env import load_rag_env, repo_root


def embed_query(client: OpenAI, text: str, model: str, dimensions: int | None) -> list[float]:
    kwargs = {"model": model, "input": text}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    r = client.embeddings.create(**kwargs)
    return list(r.data[0].embedding)


def main() -> None:
    load_rag_env()

    default_chroma = Path(__file__).resolve().parent / "chroma_data"

    parser = argparse.ArgumentParser(
        description="Semantic search over Chroma using OpenAI query embeddings.",
    )
    parser.add_argument(
        "question",
        type=str,
        help="Natural language question",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=default_chroma,
        help=f"Chroma data directory (default: {default_chroma})",
    )
    parser.add_argument(
        "--collection",
        default="rag_docs",
        help="Collection name (default: rag_docs)",
    )
    parser.add_argument(
        "-k",
        "--n-results",
        type=int,
        default=5,
        help="Number of chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model (must match ingest; default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="If you used --dimensions when embedding, pass the same value here",
    )
    args = parser.parse_args()

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        v2v = repo_root() / "voice_2_voice_server" / ".env"
        print(
            "Missing OPENAI_API_KEY. Add it to voice_2_voice_server/.env or the environment.",
            file=sys.stderr,
        )
        print(f"(Checked {v2v})", file=sys.stderr)
        sys.exit(1)

    if not args.chroma_dir.is_dir():
        print(
            f"Chroma directory not found: {args.chroma_dir}\n"
            "Run ingest_chroma.py first.",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI()
    q_emb = embed_query(client, args.question, args.model, args.dimensions)

    chroma = chromadb.PersistentClient(path=str(args.chroma_dir.resolve()))
    try:
        collection = chroma.get_collection(name=args.collection)
    except Exception as e:
        print(
            f"Could not open collection {args.collection!r}: {e}\n"
            "Run: python ingest_chroma.py vectors.npz",
            file=sys.stderr,
        )
        sys.exit(1)

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=args.n_results,
        include=["documents", "distances", "metadatas"],
    )

    ids_batch = res.get("ids") or []
    docs_batch = res.get("documents") or []
    dist_batch = res.get("distances") or []
    meta_batch = res.get("metadatas") or []

    if not ids_batch or not ids_batch[0]:
        print("No results.", file=sys.stderr)
        return

    ids = ids_batch[0]
    documents = docs_batch[0]
    distances = dist_batch[0] if dist_batch else [None] * len(ids)
    metas = meta_batch[0] if meta_batch else [None] * len(ids)

    for rank, (cid, doc, dist, meta) in enumerate(
        zip(ids, documents, distances, metas, strict=True),
        start=1,
    ):
        print(f"--- [{rank}] id={cid} distance={dist} ---")
        if meta:
            print(f"metadata: {meta}")
        print(doc)
        print()


if __name__ == "__main__":
    main()
