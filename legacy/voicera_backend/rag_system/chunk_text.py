"""
Split long text into overlapping chunks (character-based).

Use after pdf_to_text.py. Same scope: plain text, no extra deps.

Examples:
  python chunk_text.py notes.txt
  python pdf_to_text.py doc.pdf | python chunk_text.py
  python chunk_text.py notes.txt --chunk-size 800 --overlap 100 -o chunks.json --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Split text into chunks of at most `chunk_size` characters, sliding by
    (chunk_size - chunk_overlap) so consecutive chunks share `chunk_overlap` chars.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    text = text.strip()
    if not text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step

    return chunks


def read_input(path: Path | None) -> str:
    if path is not None:
        return path.read_text(encoding="utf-8")
    return sys.stdin.read()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split text into overlapping character chunks for RAG.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=None,
        help="UTF-8 text file; if omitted, read stdin",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Maximum characters per chunk (default: 1000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Characters shared between consecutive chunks (default: 200)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write output to this file instead of stdout",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON array of strings instead of text blocks",
    )
    args = parser.parse_args()

    try:
        raw = read_input(args.input)
        chunks = chunk_text(raw, args.chunk_size, args.overlap)
    except ValueError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(e, file=sys.stderr)
        sys.exit(1)

    if args.json:
        out = json.dumps(chunks, ensure_ascii=False, indent=2)
    else:
        out = "\n\n---\n\n".join(chunks)
        if out:
            out += "\n"

    if args.output is not None:
        args.output.write_text(out, encoding="utf-8")
        print(
            f"Wrote {len(chunks)} chunks to {args.output}",
            file=sys.stderr,
        )
    else:
        print(out, end="")


if __name__ == "__main__":
    main()
