
# ── Document text extraction ──────────────────────────────────────────────────
#
# Before chunking we need raw text. Different file types need different parsers.
# The indexer calls extract_text() and never needs to know which library
# handled it — all file-type complexity is contained here.

import re


def extract_text(file_bytes: bytes, file_type: str) -> str:
    """
    Extract plain text from raw file bytes.

    Args:
        file_bytes: Raw file content (downloaded from S3).
        file_type:  MIME type string set during upload.

    Returns:
        Extracted plain text with whitespace normalized.

    Raises:
        ValueError: If the file type is unsupported.
    """
    if file_type == "application/pdf":
        return _extract_pdf(file_bytes)
    elif file_type == "text/plain":
        return _extract_plaintext(file_bytes)
    elif file_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/msword",
    ):
        return _extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _extract_pdf(file_bytes: bytes) -> str:
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return _clean_whitespace("\n\n".join(p.strip() for p in pages if p.strip()))


def _extract_plaintext(file_bytes: bytes) -> str:
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    return _clean_whitespace(text)


def _extract_docx(file_bytes: bytes) -> str:
    import io
    from docx import Document as DocxDocument

    doc = DocxDocument(io.BytesIO(file_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return _clean_whitespace("\n\n".join(paragraphs))


def _clean_whitespace(text: str) -> str:
    """
    Normalize extracted text:
    - Collapse multiple spaces to one
    - Collapse 3+ consecutive newlines to 2 (preserve paragraph breaks)
    - Strip leading/trailing whitespace
    """
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
