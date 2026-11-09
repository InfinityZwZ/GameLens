"""Retrieval-only text normalization and auditable chunking helpers."""
from __future__ import annotations

import html
import re

HTML_TAG = re.compile(r"<[^>]+>")
BBCODE_TAG = re.compile(r"\[(?:/?[a-z][a-z0-9]*)(?:=[^\]]*)?\]", re.I)
WHITESPACE = re.compile(r"\s+")
SENTENCE = re.compile(r"[^。！？!?\.\n]+(?:[。！？!?\.]+|\n+|$)|[。！？!?\.\n]+")


def clean_retrieval_text(text: str) -> str:
    """Normalize index text without modifying the source text used for quotes."""
    value = html.unescape(text or "")
    value = HTML_TAG.sub(" ", value)
    value = BBCODE_TAG.sub(" ", value)
    return WHITESPACE.sub(" ", value).strip()


def fixed_chunks(text: str, *, max_chars: int = 700, overlap_chars: int = 100):
    if max_chars < 1 or overlap_chars < 0 or overlap_chars >= max_chars:
        raise ValueError("chunk参数无效")
    step = max_chars - overlap_chars
    for start in range(0, len(text), step):
        part = text[start:start + max_chars]
        if part:
            yield start, part
        if start + max_chars >= len(text):
            break


def _sentence_spans(text: str, max_chars: int):
    spans = []
    for match in SENTENCE.finditer(text):
        start, end = match.span()
        while end - start > max_chars:
            spans.append((start, start + max_chars))
            start += max_chars
        if end > start:
            spans.append((start, end))
    if not spans and text:
        spans.append((0, len(text)))
    return spans


def sentence_chunks(text: str, *, max_chars: int = 700, overlap_sentences: int = 1):
    """Yield exact source spans grouped on sentence boundaries."""
    if max_chars < 1 or overlap_sentences < 0:
        raise ValueError("sentence chunk参数无效")
    spans = _sentence_spans(text, max_chars)
    index = 0
    while index < len(spans):
        end_index = index
        start = spans[index][0]
        end = spans[index][1]
        while end_index + 1 < len(spans) and spans[end_index + 1][1] - start <= max_chars:
            end_index += 1
            end = spans[end_index][1]
        yield start, text[start:end]
        if end_index == len(spans) - 1:
            break
        index = max(index + 1, end_index + 1 - overlap_sentences)


def chunks(text: str, *, strategy: str, max_chars: int, overlap_chars: int):
    if strategy == "fixed":
        return fixed_chunks(text, max_chars=max_chars, overlap_chars=overlap_chars)
    if strategy == "sentence":
        return sentence_chunks(text, max_chars=max_chars, overlap_sentences=1)
    raise ValueError("chunk strategy必须是fixed或sentence")
