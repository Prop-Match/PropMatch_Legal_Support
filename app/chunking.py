"""Convert the supplied Egyptian-law corpus and support FAQs into deterministic RAG chunks."""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Article marker regex for legal documents
ARTICLE_RE = re.compile(
    r"(?im)^[^\n]{0,3}?"
    r"((?:ال\s*)?مادة\s+[اأإآء-ي]{2,20}"
    r"|[مقع]ادة\s*(?:\([^\n)]{1,8}\)(?:\s*[-–—]\s*[0-9٠-٩]{1,4})?"
    r"|[-–—:)(.،؛\s]*(?:[0-9٠-٩]{1,4}|[؟؛+هعبل!«“]{1,3}(?=\s|$))))"
)
HEADER_END_RE = re.compile(r"^=+\s*بداية النص\s*=+$", re.MULTILINE)
PAGE_BREAK_RE = re.compile(r"\s*---\s*PAGE BREAK\s*---\s*")


@dataclass(frozen=True)
class RagChunk:
    """One indexed passage with a stable ID and RAG source metadata."""

    id: str
    document: str
    metadata: dict[str, str | int]


# Alias for backward compatibility
LawChunk = RagChunk


def _clean_text(text: str) -> str:
    text = text.replace("\ufeff", "").replace("\r\n", "\n")
    text = PAGE_BREAK_RE.sub("\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _body_only(text: str) -> str:
    match = HEADER_END_RE.search(text)
    return text[match.end() :].strip() if match else text.strip()


def _windows(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = max(
                text.rfind("\n", start + size // 2, end), text.rfind(". ", start, end)
            )
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _sections(text: str) -> list[tuple[str, str]]:
    matches = list(ARTICLE_RE.finditer(text))
    if not matches:
        return [("غير محدد", text)]
    sections: list[tuple[str, str]] = []
    prefix = text[: matches[0].start()].strip()
    if prefix:
        sections.append(("مقدمة", prefix))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((match.group(1).strip(), text[match.start() : end].strip()))
    return sections


def load_manifest(corpus_dir: Path) -> dict[str, dict[str, Any]]:
    """Load source titles, URLs, and extraction details for each law file."""
    manifest_file = corpus_dir / "manifest.json"
    if not manifest_file.exists():
        return {}
    records = json.loads(manifest_file.read_text(encoding="utf-8"))
    return {record["file"]: record for record in records}


def chunk_corpus(corpus_dir: Path, size: int, overlap: int) -> list[RagChunk]:
    """Split every law by article and then into overlapping retrieval windows."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    manifest = load_manifest(corpus_dir)
    chunks: list[RagChunk] = []
    for path in sorted(corpus_dir.glob("*.txt")):
        if path.name == "README_RAG.txt":
            continue
        record = manifest.get(path.name, {})
        title = str(record.get("title", path.stem))
        body = _body_only(_clean_text(path.read_text(encoding="utf-8")))
        sequence = 0
        for article, section in _sections(body):
            for part in _windows(section, size, overlap):
                digest = hashlib.sha256(
                    f"{path.name}:{article}:{sequence}:{part}".encode()
                ).hexdigest()[:24]
                chunks.append(
                    RagChunk(
                        id=f"law_{digest}",
                        document=part,
                        metadata={
                            "title": title,
                            "article": article,
                            "file": path.name,
                            "source_url": str(record.get("source", "")),
                            "extraction_method": str(record.get("method", "unknown")),
                            "sequence": sequence,
                        },
                    )
                )
                sequence += 1
    return chunks


def chunk_support_faqs(support_dir: Path, size: int, overlap: int) -> list[RagChunk]:
    """Split support FAQ markdown files into overlapping retrieval windows with deterministic IDs."""
    if overlap >= size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[RagChunk] = []
    if not support_dir.exists():
        return chunks

    for path in sorted(support_dir.glob("*.md")):
        text = _clean_text(path.read_text(encoding="utf-8"))
        title_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        title = title_match.group(1).strip() if title_match else path.stem

        sequence = 0
        for part in _windows(text, size, overlap):
            digest = hashlib.sha256(
                f"{path.name}:{sequence}:{part}".encode()
            ).hexdigest()[:24]
            chunks.append(
                RagChunk(
                    id=f"support_{digest}",
                    document=part,
                    metadata={
                        "title": title,
                        "file": path.name,
                        "category": "support_faq",
                        "sequence": sequence,
                    },
                )
            )
            sequence += 1
    return chunks
