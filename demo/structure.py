from __future__ import annotations

import hashlib
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator, Sequence

from demo.common import (
    PaperRecord,
    REPO_ROOT,
    compact_context_excerpt,
    estimated_token_count,
    extract_candidate_phrases,
    load_corpus_text,
    sanitize_answer_text,
    truncate_text_to_tokens,
)

KNOWN_SECTION_TYPES = {
    "abstract": ("abstract",),
    "introduction": ("introduction", "motivation"),
    "background": ("background", "preliminaries", "notation", "problem setup"),
    "related_work": ("related work", "prior work"),
    "methods": (
        "method",
        "methods",
        "methodology",
        "approach",
        "approaches",
        "model",
        "models",
        "system design",
        "architecture",
        "framework",
        "algorithm",
        "algorithms",
        "implementation",
    ),
    "experiments": (
        "experiment",
        "experiments",
        "evaluation",
        "empirical study",
        "results",
        "ablation",
        "ablations",
        "benchmark",
        "benchmarks",
    ),
    "discussion": ("discussion", "analysis", "limitations"),
    "conclusion": ("conclusion", "conclusions", "future work"),
    "appendix": ("appendix",),
}

HEADING_PREFIX_RE = re.compile(r"^(?:\d+(?:\.\d+)*|[IVX]+|[A-Z])(?:[.)])?\s+")
ROMAN_HEADING_RE = re.compile(r"^[IVX]+(?:\.)?\s+[A-Z][A-Z\s\-]{2,80}$")
NUMERIC_HEADING_RE = re.compile(r"^\d+(?:\.\d+)*\s+[A-Za-z][A-Za-z0-9\s\-]{2,90}$")
LETTER_HEADING_RE = re.compile(r"^[A-Z]\.\s+[A-Za-z][A-Za-z0-9\s\-]{2,90}$")
TITLE_HEADING_RE = re.compile(r"^[A-Z][A-Za-z0-9\-]*(?:\s+[A-Z][A-Za-z0-9\-]*){0,7}$")
SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+")
INLINE_ABSTRACT_RE = re.compile(r"^\s*Abstract(?:[-—:])?\s*(.*)$", re.IGNORECASE)


@dataclass(slots=True)
class StructuralNodeRecord:
    node_id: str
    paper_id: str
    title: str
    text_path: str
    heading: str
    section_type: str
    start_char: int
    end_char: int
    estimated_tokens: int
    text_sketch: str
    salient_phrases: list[str]


def _iter_lines_with_offsets(text: str) -> Iterator[tuple[int, int, str]]:
    cursor = 0
    for line in text.splitlines(keepends=True):
        start = cursor
        cursor += len(line)
        yield start, cursor, line.rstrip("\n")
    if text and not text.endswith("\n"):
        return


def _normalize_heading_value(line: str) -> str:
    candidate = re.sub(r"\s+", " ", line).strip()
    candidate = candidate.rstrip(":-")
    candidate = HEADING_PREFIX_RE.sub("", candidate)
    return candidate.strip()


def _classify_section_type(heading: str) -> str:
    normalized = re.sub(r"\s+", " ", heading).strip().lower()
    compact = normalized.replace(" ", "")
    for section_type, labels in KNOWN_SECTION_TYPES.items():
        for label in labels:
            if label in normalized or label.replace(" ", "") in compact:
                return section_type
    return "body"


def _looks_like_heading(line: str) -> bool:
    candidate = re.sub(r"\s+", " ", line).strip()
    if not candidate or len(candidate) > 96:
        return False
    if candidate.endswith((".", ",", ";", "?", "!")):
        return False
    if candidate.startswith(("http://", "https://", "arxiv:", "doi:")):
        return False
    alpha_chars = sum(1 for ch in candidate if ch.isalpha())
    if alpha_chars < 4:
        return False
    words = candidate.split()
    if len(words) > 10:
        return False
    lower = candidate.lower()
    if _classify_section_type(candidate) != "body":
        return True
    if ROMAN_HEADING_RE.match(candidate) or NUMERIC_HEADING_RE.match(candidate) or LETTER_HEADING_RE.match(candidate):
        return True
    if TITLE_HEADING_RE.match(candidate) and sum(word[0].isupper() for word in words if word) >= max(1, len(words) - 1):
        return True
    if candidate.isupper() and len(words) <= 8:
        return True
    return False


def _compact_text_sketch(text: str, *, max_chars: int = 360) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    sentences = SENTENCE_BOUNDARY_RE.split(clean)
    sketch = ""
    for sentence in sentences:
        candidate = (sketch + " " + sentence).strip()
        if len(candidate) > max_chars and sketch:
            break
        sketch = candidate
        if len(sketch) >= max_chars:
            break
    if not sketch:
        sketch = clean[:max_chars]
    if len(sketch) > max_chars:
        sketch = sketch[: max_chars - 3].rstrip() + "..."
    return sketch


def node_structural_text(node: StructuralNodeRecord) -> str:
    fields = [node.heading, node.section_type.replace("_", " "), *node.salient_phrases, node.text_sketch]
    return "\n".join(part.strip() for part in fields if part and part.strip())


def node_retrieval_text(node: StructuralNodeRecord) -> str:
    return node_structural_text(node)


def serialize_nodes(nodes: Sequence[StructuralNodeRecord]) -> list[dict[str, object]]:
    return [asdict(node) for node in nodes]


def deserialize_nodes(records: Sequence[dict[str, object]]) -> list[StructuralNodeRecord]:
    return [StructuralNodeRecord(**record) for record in records]


def _split_relative_text(
    text: str,
    *,
    chunk_chars: int,
    overlap_chars: int,
) -> Iterator[tuple[int, int, str]]:
    leading_ws = len(text) - len(text.lstrip())
    trailing_ws = len(text) - len(text.rstrip())
    clean_end = len(text) - trailing_ws if trailing_ws else len(text)
    clean = text[leading_ws:clean_end]
    if not clean.strip():
        return

    start = 0
    length = len(clean)
    while start < length:
        upper = min(length, start + chunk_chars)
        if upper < length:
            breakpoint = clean.rfind("\n\n", start + chunk_chars // 2, upper)
            if breakpoint == -1:
                breakpoint = clean.rfind(". ", start + chunk_chars // 2, upper)
            if breakpoint != -1 and breakpoint > start:
                upper = breakpoint + (2 if clean[breakpoint : breakpoint + 2] == ". " else 0)
        chunk = clean[start:upper].strip()
        if chunk:
            yield leading_ws + start, leading_ws + upper, chunk
        if upper >= length:
            break
        start = max(0, upper - overlap_chars)


def _detect_heading_offsets(text: str) -> list[tuple[int, int, str, str]]:
    headings: list[tuple[int, int, str, str]] = []
    for start, end, line in _iter_lines_with_offsets(text):
        stripped = line.strip()
        if not stripped:
            continue
        if INLINE_ABSTRACT_RE.match(stripped):
            headings.append((start, end, "Abstract", "abstract"))
            continue
        if _looks_like_heading(stripped):
            heading = _normalize_heading_value(stripped)
            if heading:
                headings.append((start, end, heading, _classify_section_type(heading)))
    unique: list[tuple[int, int, str, str]] = []
    seen_offsets: set[int] = set()
    for start, end, heading, section_type in headings:
        if start in seen_offsets:
            continue
        seen_offsets.add(start)
        unique.append((start, end, heading, section_type))
    return unique


def _seed_from_text(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _paper_derangement(items: Sequence[str], *, seed_text: str) -> list[str]:
    if len(items) <= 1:
        return list(items)
    rng = random.Random(_seed_from_text(seed_text))
    base = list(items)
    while True:
        permuted = list(base)
        rng.shuffle(permuted)
        if all(left != right for left, right in zip(base, permuted, strict=False)):
            return permuted


def build_shuffle_node_indices(
    nodes: Sequence[StructuralNodeRecord],
    *,
    seed_text: str,
) -> list[int]:
    if not nodes:
        return []
    paper_to_indices: dict[str, list[int]] = {}
    for idx, node in enumerate(nodes):
        paper_to_indices.setdefault(node.paper_id, []).append(idx)
    papers = sorted(paper_to_indices)
    shuffled_papers = _paper_derangement(papers, seed_text=seed_text)

    index_map: dict[int, int] = {}
    for source_paper, target_paper in zip(papers, shuffled_papers, strict=False):
        source_indices = paper_to_indices[source_paper]
        target_indices = paper_to_indices[target_paper]
        for offset, source_idx in enumerate(source_indices):
            index_map[source_idx] = target_indices[offset % len(target_indices)]
    return [index_map[idx] for idx in range(len(nodes))]


def _make_node(
    *,
    paper: PaperRecord,
    node_index: int,
    heading: str,
    section_type: str,
    start_char: int,
    end_char: int,
    sketch_source: str,
) -> StructuralNodeRecord | None:
    sketch = _compact_text_sketch(sketch_source)
    if not sketch:
        return None
    phrases = extract_candidate_phrases(f"{heading}\n{sketch_source}", limit=6)
    return StructuralNodeRecord(
        node_id=f"{paper.paper_id}::node_{node_index:04d}",
        paper_id=paper.paper_id,
        title=paper.title,
        text_path=paper.text_path,
        heading=heading,
        section_type=section_type,
        start_char=start_char,
        end_char=end_char,
        estimated_tokens=estimated_token_count(sketch),
        text_sketch=sketch,
        salient_phrases=phrases,
    )


def build_structural_nodes(
    papers: Sequence[PaperRecord],
    *,
    node_chars: int = 1500,
    node_overlap: int = 180,
) -> list[StructuralNodeRecord]:
    nodes: list[StructuralNodeRecord] = []

    for paper in papers:
        text = load_corpus_text(paper)
        headings = _detect_heading_offsets(text)
        node_index = 0

        first_heading_start = headings[0][0] if headings else len(text)
        abstract_span = text[:first_heading_start].strip()
        abstract_source = paper.summary.strip() or abstract_span
        abstract_end = first_heading_start if first_heading_start > 0 else min(len(text), max(600, len(abstract_span)))
        abstract_node = _make_node(
            paper=paper,
            node_index=node_index,
            heading="Abstract",
            section_type="abstract",
            start_char=0,
            end_char=max(0, abstract_end),
            sketch_source=abstract_source,
        )
        if abstract_node is not None:
            nodes.append(abstract_node)
            node_index += 1

        for idx, (heading_start, heading_end, heading, section_type) in enumerate(headings):
            next_start = headings[idx + 1][0] if idx + 1 < len(headings) else len(text)
            section_text = text[heading_end:next_start]
            if not section_text.strip():
                continue
            for rel_start, rel_end, chunk in _split_relative_text(
                section_text,
                chunk_chars=node_chars,
                overlap_chars=node_overlap,
            ):
                node = _make_node(
                    paper=paper,
                    node_index=node_index,
                    heading=heading,
                    section_type=section_type,
                    start_char=heading_end + rel_start,
                    end_char=heading_end + rel_end,
                    sketch_source=chunk,
                )
                if node is None:
                    continue
                nodes.append(node)
                node_index += 1

        if not headings and not abstract_span:
            body_node = _make_node(
                paper=paper,
                node_index=node_index,
                heading="Body",
                section_type="body",
                start_char=0,
                end_char=len(text),
                sketch_source=text,
            )
            if body_node is not None:
                nodes.append(body_node)

    return nodes


def reconstruct_node_text(
    node: StructuralNodeRecord,
    *,
    cache: dict[str, str] | None = None,
) -> str:
    if cache is None:
        cache = {}
    if node.text_path not in cache:
        cache[node.text_path] = (REPO_ROOT / node.text_path).read_text(encoding="utf-8")
    source = cache[node.text_path]
    span = source[node.start_char : node.end_char].strip()
    return span or node.text_sketch


def build_node_contexts(
    ranked_nodes: Sequence[StructuralNodeRecord],
    *,
    max_context_tokens: int,
    snippets_per_paper: int = 3,
) -> list[dict[str, str]]:
    grouped: dict[str, dict[str, object]] = {}
    ordered_papers: list[str] = []
    cache: dict[str, str] = {}
    snippet_limit = max(1, snippets_per_paper)

    for node in ranked_nodes:
        text = reconstruct_node_text(node, cache=cache)
        if not text:
            continue
        bucket = grouped.get(node.paper_id)
        if bucket is None:
            bucket = {
                "paper_id": node.paper_id,
                "title": node.title,
                "snippets": [],
                "keys": set(),
            }
            grouped[node.paper_id] = bucket
            ordered_papers.append(node.paper_id)

        snippets = bucket["snippets"]
        if len(snippets) >= snippet_limit:
            continue

        snippet_body = compact_context_excerpt(text, max_chars=800)
        prefixed_text = f"{node.heading} [{node.section_type}]\n{snippet_body}".strip()
        dedupe_key = sanitize_answer_text(prefixed_text[:240])
        if dedupe_key in bucket["keys"]:
            continue
        bucket["keys"].add(dedupe_key)
        snippets.append(prefixed_text)

    contexts: list[dict[str, str]] = []
    used = 0
    for paper_id in ordered_papers:
        if used >= max_context_tokens:
            break
        bucket = grouped[paper_id]
        snippets = bucket["snippets"]
        if not snippets:
            continue
        merged_text = "\n\n".join(str(snippet) for snippet in snippets)
        remaining = max_context_tokens - used
        merged_text = truncate_text_to_tokens(merged_text, max_tokens=remaining)
        if not merged_text:
            break
        contexts.append(
            {
                "paper_id": str(bucket["paper_id"]),
                "title": str(bucket["title"]),
                "text": merged_text,
            }
        )
        used += estimated_token_count(merged_text)
    return contexts
