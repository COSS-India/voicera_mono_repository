"""
Extract plain text from a PDF file (CLI).

Assumes normal text-based PDFs (embedded text). Scanned pages and image-only
PDFs are out of scope—no OCR.

Usage:
  python pdf_to_text.py path/to/document.pdf
  python pdf_to_text.py path/to/document.pdf -o out.txt

Requires: pypdf (see voicera_backend/requirements.txt)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pypdf import PdfReader


def extract_text_from_pdf(pdf_path: str | Path) -> str:
    """Read a text-based PDF from disk; return all page text joined with blank lines."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(f"Not a file: {path}")

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        raw = page.extract_text()
        if raw:
            parts.append(raw.strip())

    return "\n\n".join(parts).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract text from a text-based PDF (not scanned/image PDFs).",
    )
    parser.add_argument(
        "pdf",
        type=Path,
        help="Path to the .pdf file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Write text to this file instead of stdout",
    )
    args = parser.parse_args()

    try:
        text = extract_text_from_pdf(args.pdf)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Failed to read PDF: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
        print(f"Wrote {len(text)} characters to {args.output}", file=sys.stderr)
    else:
        print(text)


if __name__ == "__main__":
    main()
