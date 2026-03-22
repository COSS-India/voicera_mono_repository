"""
Load vectors.npz (from embed_chunks.py) into a local Chroma persistent store.

Examples:
  cd voicera_backend/rag_system
  python ingest_chroma.py vectors.npz
  python ingest_chroma.py vectors.npz --reset
  python ingest_chroma.py vectors.npz --chroma-dir ./chroma_data --collection my_docs

Requires: chromadb, numpy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chromadb
import numpy as np

from rag_env import load_rag_env


def _decode_model_name(data) -> str:
    if "model_name" not in data:
        return ""
    raw = data["model_name"]
    a = np.asarray(raw)
    if a.shape == ():
        return str(a.item())
    return str(a.flat[0])


def main() -> None:
    load_rag_env()

    default_chroma = Path(__file__).resolve().parent / "chroma_data"

    parser = argparse.ArgumentParser(
        description="Ingest vectors.npz into Chroma (PersistentClient).",
    )
    parser.add_argument(
        "npz",
        type=Path,
        help="Path to vectors.npz from embed_chunks.py",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=default_chroma,
        help=f"Chroma on-disk directory (default: {default_chroma})",
    )
    parser.add_argument(
        "--collection",
        default="rag_docs",
        help="Collection name (default: rag_docs)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection if it exists, then re-ingest",
    )
    args = parser.parse_args()

    if not args.npz.is_file():
        print(f"Not a file: {args.npz}", file=sys.stderr)
        sys.exit(1)

    args.chroma_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.npz, allow_pickle=True)
    embeddings = np.asarray(data["embeddings"], dtype=np.float32)
    texts = data["texts"]
    model_name = _decode_model_name(data)

    if embeddings.ndim != 2:
        print("embeddings must be 2D [n, dim]", file=sys.stderr)
        sys.exit(1)

    n, dim = embeddings.shape
    texts_list: list[str] = []
    for i in range(n):
        t = texts[i]
        texts_list.append(str(t) if not isinstance(t, str) else t)

    ids = [f"chunk_{i}" for i in range(n)]
    metadatas = [{"chunk_index": i, "embedding_model": model_name} for i in range(n)]

    client = chromadb.PersistentClient(path=str(args.chroma_dir.resolve()))

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Deleted collection {args.collection!r}", file=sys.stderr)
        except Exception:
            pass

    collection = client.get_or_create_collection(
        name=args.collection,
        metadata={
            "hnsw:space": "cosine",
            "embedding_dim": str(dim),
            "embedding_model": model_name,
        },
    )

    emb_list = embeddings.tolist()

    collection.upsert(
        ids=ids,
        embeddings=emb_list,
        documents=texts_list,
        metadatas=metadatas,
    )

    print(
        f"Ingested {n} chunks (dim={dim}) into {args.chroma_dir} "
        f"collection={args.collection!r} model={model_name!r}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
