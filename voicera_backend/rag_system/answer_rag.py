"""
Full RAG answer: retrieve chunks from Chroma, then call OpenAI Chat Completions.

Uses the same embedding model as ingest (default text-embedding-3-small).
Loads OPENAI_API_KEY via rag_env (voice_2_voice_server/.env).

Examples:
  cd voicera_backend/rag_system
  python answer_rag.py "What are the five principles of prompting?"
  python answer_rag.py "..." -k 4 --chat-model gpt-4o-mini
  python answer_rag.py "..." --show-context   # print chunks to stderr, answer to stdout

Requires: chromadb, openai (see voicera_backend/requirements.txt)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import chromadb
from openai import OpenAI

from query_chroma import embed_query
from rag_env import load_rag_env, repo_root

SYSTEM_PROMPT = """You are a helpful assistant. Answer using ONLY the provided context excerpts from the user's documents.

If the excerpts are clearly about the topic (even if they use different phrasing than the question—e.g. "prompting" vs "prompt engineering"), synthesize a concise answer from that material.

Say you do not have enough information in the provided documents ONLY when the excerpts are empty, unrelated, or do not support any reasonable answer."""


def retrieve_chunk_texts(
    client: OpenAI,
    question: str,
    *,
    chroma_dir: Path,
    collection_name: str,
    n_results: int,
    embed_model: str,
    dimensions: int | None,
) -> tuple[list[str], list[str], list[float | None]]:
    """Return (ids, documents, distances)."""
    q_emb = embed_query(client, question, embed_model, dimensions)
    chroma = chromadb.PersistentClient(path=str(chroma_dir.resolve()))
    collection = chroma.get_collection(name=collection_name)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        include=["documents", "distances", "metadatas"],
    )
    ids_batch = res.get("ids") or []
    docs_batch = res.get("documents") or []
    dist_batch = res.get("distances") or []
    if not ids_batch or not ids_batch[0]:
        return [], [], []
    ids = ids_batch[0]
    documents = docs_batch[0]
    distances = dist_batch[0] if dist_batch else [None] * len(ids)
    return ids, documents, distances


def build_user_message(question: str, chunk_texts: list[str]) -> str:
    parts = []
    for i, text in enumerate(chunk_texts, start=1):
        parts.append(f"[Excerpt {i}]\n{text.strip()}")
    context = "\n\n".join(parts)
    return f"""Context from documents:

{context}

Question: {question}

Answer the question using only the context above. If the context discusses the same topic under related terms, use it; do not require a literal keyword match."""


def main() -> None:
    load_rag_env()

    default_chroma = Path(__file__).resolve().parent / "chroma_data"

    parser = argparse.ArgumentParser(
        description="RAG: Chroma retrieval + OpenAI chat answer.",
    )
    parser.add_argument("question", type=str, help="User question")
    parser.add_argument(
        "-k",
        "--n-results",
        type=int,
        default=5,
        help="Chunks to retrieve (default: 5)",
    )
    parser.add_argument(
        "--chat-model",
        default="gpt-4o-mini",
        help="OpenAI chat model (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-small",
        help="Must match ingest (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="If you used --dimensions when embedding, pass the same value",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=default_chroma,
        help=f"Chroma data dir (default: {default_chroma})",
    )
    parser.add_argument(
        "--collection",
        default="rag_docs",
        help="Collection name (default: rag_docs)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Chat sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print retrieved chunks to stderr",
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
            "Run: python ingest_chroma.py vectors.npz",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OpenAI()

    try:
        ids, documents, distances = retrieve_chunk_texts(
            client,
            args.question,
            chroma_dir=args.chroma_dir,
            collection_name=args.collection,
            n_results=args.n_results,
            embed_model=args.embedding_model,
            dimensions=args.dimensions,
        )
    except Exception as e:
        print(f"Retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)

    if not documents:
        print("No chunks retrieved; cannot answer.", file=sys.stderr)
        sys.exit(1)

    if args.show_context:
        for rank, (cid, doc, dist) in enumerate(
            zip(ids, documents, distances, strict=True),
            start=1,
        ):
            print(f"--- [{rank}] {cid} distance={dist} ---", file=sys.stderr)
            print(doc, file=sys.stderr)
            print(file=sys.stderr)

    user_msg = build_user_message(args.question, documents)

    response = client.chat.completions.create(
        model=args.chat_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=args.temperature,
    )

    text = response.choices[0].message.content
    if not text:
        print("Empty response from model.", file=sys.stderr)
        sys.exit(1)
    print(text.strip())


if __name__ == "__main__":
    main()
