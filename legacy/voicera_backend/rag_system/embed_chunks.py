"""
Embed text chunks with the OpenAI Embeddings API (vectors for RAG).

Standalone CLI: set ``OPENAI_API_KEY`` in the environment (Voicera app KB uses
Integrations only, not this script).

Reads chunks from:
  - JSON file: array of strings (from: chunk_text.py ... --json -o chunks.json)
  - Text file: chunks separated by blank lines and --- (from: chunk_text.py -o chunks.txt)

Writes a compressed .npz with embeddings + original texts + model name.

Examples:
  export OPENAI_API_KEY=...  # required for this CLI
  python embed_chunks.py chunks.txt -o vectors.npz
  python embed_chunks.py chunks.json --model text-embedding-3-large -o vectors.npz

Requires: openai, numpy (see voicera_backend/requirements.txt)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

def load_chunks(path: Path) -> list[str]:
    suffix = path.suffix.lower()
    raw = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError("JSON must be an array of strings")
        chunks = [str(x).strip() for x in data if str(x).strip()]
        return chunks

    chunks = [c.strip() for c in raw.split("\n\n---\n\n") if c.strip()]
    return chunks


def embed_openai(
    client: OpenAI,
    chunks: list[str],
    model: str,
    batch_size: int,
    dimensions: int | None,
) -> np.ndarray:
    """Call OpenAI embeddings API in batches; return float32 array [n, dim]."""
    all_rows: list[list[float]] = []

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        kwargs = {
            "model": model,
            "input": batch,
        }
        if dimensions is not None:
            kwargs["dimensions"] = dimensions

        response = client.embeddings.create(**kwargs)
        ordered = sorted(response.data, key=lambda d: d.index)
        for item in ordered:
            all_rows.append(item.embedding)

    return np.asarray(all_rows, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Embed text chunks with the OpenAI Embeddings API.",
    )
    parser.add_argument(
        "chunks_file",
        type=Path,
        help="chunks.json (array of strings) or chunks.txt (--- separated)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output .npz path (embeddings + texts)",
    )
    parser.add_argument(
        "--model",
        default="text-embedding-3-small",
        help="OpenAI embedding model (default: text-embedding-3-small)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Chunks per API request (default: 100)",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=None,
        help="Optional output size for v3 models (e.g. 256, 512, 1536)",
    )
    args = parser.parse_args()

    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        print(
            "Missing OPENAI_API_KEY. For this CLI, export it in your shell. "
            "The Voicera app uses the OpenAI key from Integrations instead.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not args.chunks_file.is_file():
        print(f"Not a file: {args.chunks_file}", file=sys.stderr)
        sys.exit(1)

    try:
        chunks = load_chunks(args.chunks_file)
    except (json.JSONDecodeError, ValueError, OSError) as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if not chunks:
        print("No chunks to embed.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=key)

    print(
        f"Embedding {len(chunks)} chunks with {args.model!r}...",
        file=sys.stderr,
    )
    embeddings = embed_openai(
        client,
        chunks,
        model=args.model,
        batch_size=args.batch_size,
        dimensions=args.dimensions,
    )
    texts = np.array(chunks, dtype=object)
    model_meta = np.array(args.model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(
            args.output,
            embeddings=embeddings,
            texts=texts,
            model_name=model_meta,
        )
    except PermissionError as e:
        print(f"Cannot write {args.output}: {e}", file=sys.stderr)
        print(
            "Use a path you can write to, e.g. -o vectors.npz or -o ./data/vectors.npz "
            "(not /vectors.npz, which is the filesystem root).",
            file=sys.stderr,
        )
        sys.exit(1)

    dim = embeddings.shape[1]
    print(
        f"Saved {embeddings.shape[0]} vectors (dim={dim}) to {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
