from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator, Sequence
from urllib.parse import urlencode
from xml.etree import ElementTree

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = REPO_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
MANIFOLD_DIR = REPO_ROOT / "manifold"
RESULTS_DIR = REPO_ROOT / "results"
GRAPHS_DIR = RESULTS_DIR / "graphs"
CORPUS_MANIFEST_PATH = DATA_DIR / "corpus_manifest.json"
QUESTIONS_PATH = DATA_DIR / "questions.json"
CORPUS_FULL_PATH = DATA_DIR / "corpus_full.txt"
BUILD_METRICS_PATH = DATA_DIR / "corpus_metrics.json"
COMPRESSION_METRICS_PATH = RESULTS_DIR / "compression_metrics.json"
BASELINE_RESULTS_PATH = RESULTS_DIR / "baseline_rag_results.json"
MANIFOLD_RESULTS_PATH = RESULTS_DIR / "manifold_results.json"
SHUFFLED_MANIFOLD_RESULTS_PATH = RESULTS_DIR / "manifold_results_shuffled.json"
QA_RESULTS_PATH = RESULTS_DIR / "qa_results.json"
MANIFOLD_JSON_PATH = MANIFOLD_DIR / "manifold.json"
MANIFOLD_INDEX_PATH = MANIFOLD_DIR / "manifold_index.bin"
DEFAULT_OLLAMA_GENERATE_ENDPOINT = "http://127.0.0.1:11434/api/generate"
DEFAULT_OLLAMA_TAGS_ENDPOINT = "http://127.0.0.1:11434/api/tags"
ARXIV_API_URL = "http://export.arxiv.org/api/query"

DEFAULT_ARXIV_CATEGORIES = (
    "cs.LG",
    "cs.AI",
    "math.OC",
    "math.PR",
    "cond-mat.stat-mech",
    "hep-th",
)

STOPWORDS = {
    "a",
    "about",
    "across",
    "after",
    "all",
    "also",
    "an",
    "and",
    "any",
    "are",
    "as",
    "at",
    "be",
    "because",
    "between",
    "both",
    "by",
    "can",
    "do",
    "each",
    "for",
    "from",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "more",
    "of",
    "on",
    "or",
    "our",
    "paper",
    "that",
    "the",
    "their",
    "them",
    "these",
    "this",
    "those",
    "to",
    "using",
    "we",
    "what",
    "which",
    "with",
}

PHRASE_BANNED_PREFIXES = (
    "abstract",
    "introduction",
    "related work",
    "background",
    "conclusion",
    "references",
    "bibliography",
    "appendix",
    "keywords",
    "preprint",
    "technical report",
    "figure",
    "table",
)

PHRASE_BANNED_TERMS = {
    "abstract",
    "appendix",
    "bibliography",
    "college",
    "conference",
    "copyright",
    "department",
    "figure",
    "introduction",
    "keywords",
    "preprint",
    "professor",
    "references",
    "table",
    "technical report",
    "university",
    "workshop",
}

PHRASE_GENERIC_TERMS = {
    "algorithm",
    "appendix",
    "background",
    "conclusion",
    "contributions",
    "example",
    "examples",
    "fig",
    "figure",
    "introduction",
    "keywords",
    "preprint",
    "problem",
    "references",
    "related",
    "report",
    "require",
    "section",
    "setup",
    "supplementary",
    "table",
    "technical",
    "work",
}

PHRASE_TRIGGER_RE = re.compile(
    r"\b(?:propose|introduce|present|study|derive|analyze|analyse|prove|develop|evaluate|compare)\b\s+([^.;:]{8,120})",
    re.IGNORECASE,
)
TITLECASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9\-]{2,})(?:\s+(?:[A-Z][A-Za-z0-9\-]{2,})){1,5}\b"
)
ACRONYM_RE = re.compile(r"\b[A-Z]{2,}(?:-[A-Z0-9]{2,})*\b")
QUOTED_RE = re.compile(r"[\"“]([^\"”]{4,120})[\"”]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-_/]{2,}")


@dataclass(slots=True)
class PaperRecord:
    paper_id: str
    title: str
    source_id: str
    categories: list[str]
    published: str
    pdf_url: str
    pdf_path: str
    text_path: str
    bytes: int
    characters: int
    estimated_tokens: int
    sha256: str
    summary: str = ""


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    paper_id: str
    title: str
    text_path: str
    start_char: int
    end_char: int
    estimated_tokens: int
    text: str


@dataclass(slots=True)
class QuestionRecord:
    question_id: str
    question: str
    answer: str | list[str]
    answer_aliases: list[str]
    source_papers: list[str]
    question_type: str
    evidence_terms: list[str]


def ensure_directories() -> None:
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    MANIFOLD_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def estimated_token_count(text: str) -> int:
    return max(1, math.ceil(len(text.encode("utf-8")) / 4))


def sanitize_answer_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def score_prediction(predicted: str, question: QuestionRecord) -> bool:
    normalized = sanitize_answer_text(predicted)
    if not normalized:
        return False
    required = [sanitize_answer_text(alias) for alias in question.answer_aliases if alias]
    if question.question_type == "multi_paper_lookup":
        return bool(required) and all(alias in normalized for alias in required)
    return any(alias in normalized for alias in required)


def load_questions(path: Path = QUESTIONS_PATH) -> list[QuestionRecord]:
    raw = read_json(path)
    records = raw["questions"] if isinstance(raw, dict) else raw
    questions: list[QuestionRecord] = []
    for item in records:
        questions.append(
            QuestionRecord(
                question_id=str(item["question_id"]),
                question=str(item["question"]),
                answer=item["answer"],
                answer_aliases=list(item.get("answer_aliases", [])),
                source_papers=list(item.get("source_papers", [])),
                question_type=str(item.get("question_type", "paper_lookup")),
                evidence_terms=list(item.get("evidence_terms", [])),
            )
        )
    return questions


def iter_retrieval_query_texts(question: QuestionRecord) -> list[str]:
    evidence = [term.strip() for term in question.evidence_terms if term.strip()]
    return evidence or [question.question]


def build_retrieval_query(question: QuestionRecord) -> str:
    return " ".join(iter_retrieval_query_texts(question))


def load_manifest(path: Path = CORPUS_MANIFEST_PATH) -> list[PaperRecord]:
    raw = read_json(path)
    records = raw["papers"] if isinstance(raw, dict) else raw
    return [PaperRecord(**record) for record in records]


def load_corpus_text(record: PaperRecord) -> str:
    return (REPO_ROOT / record.text_path).read_text(encoding="utf-8")


def corpus_hash(records: Sequence[PaperRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record.paper_id.encode("utf-8"))
        digest.update(record.sha256.encode("utf-8"))
    return digest.hexdigest()


def normalize_paper_text(raw_text: str) -> str:
    text = raw_text.replace("\x00", " ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    start = 0
    for idx, line in enumerate(lines[:80]):
        lower = line.lower()
        if lower == "abstract":
            start = min(idx + 1, len(lines))
            break
        if lower.startswith("abstract "):
            start = idx
            break
        if lower in {"introduction", "1 introduction"} or re.match(
            r"^1[\.\s]+introduction\b", lower
        ):
            start = idx
            break

    cleaned: list[str] = []
    for line in lines[start:]:
        lower = line.lower()
        if lower.startswith("references") or lower.startswith("bibliography"):
            break
        if lower.startswith("doi:") or lower.startswith("arxiv:"):
            continue
        if re.fullmatch(r"\d+", line):
            continue
        if len(line) <= 2:
            continue
        cleaned.append(line)

    text = "\n".join(cleaned)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def split_text_chunks(
    text: str,
    *,
    chunk_chars: int = 2200,
    overlap_chars: int = 250,
) -> Iterator[tuple[int, int, str]]:
    clean = text.strip()
    if not clean:
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
            yield start, upper, chunk
        if upper >= length:
            break
        start = max(0, upper - overlap_chars)


def build_chunks(
    papers: Sequence[PaperRecord],
    *,
    chunk_chars: int = 2200,
    overlap_chars: int = 250,
) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for paper in papers:
        text = load_corpus_text(paper)
        for idx, (start, end, chunk_text) in enumerate(
            split_text_chunks(text, chunk_chars=chunk_chars, overlap_chars=overlap_chars)
        ):
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{paper.paper_id}::chunk_{idx:04d}",
                    paper_id=paper.paper_id,
                    title=paper.title,
                    text_path=paper.text_path,
                    start_char=start,
                    end_char=end,
                    estimated_tokens=estimated_token_count(chunk_text),
                    text=chunk_text,
                )
            )
    return chunks


def combine_corpus(papers: Sequence[PaperRecord]) -> str:
    blocks = []
    for paper in papers:
        blocks.append(f"===== {paper.paper_id} =====\n{load_corpus_text(paper)}")
    return "\n\n".join(blocks).strip() + "\n"


def extract_candidate_phrases(text: str, *, limit: int = 24) -> list[str]:
    sample = text[:16000]
    candidates: list[str] = []

    for match in QUOTED_RE.finditer(sample):
        phrase = normalize_phrase(match.group(1))
        if phrase:
            candidates.append(phrase)

    for match in PHRASE_TRIGGER_RE.finditer(sample):
        phrase = normalize_phrase(match.group(1))
        if phrase:
            candidates.append(phrase)

    for match in TITLECASE_RE.finditer(sample):
        phrase = normalize_phrase(match.group(0))
        if phrase:
            candidates.append(phrase)

    for match in ACRONYM_RE.finditer(sample):
        phrase = normalize_phrase(match.group(0))
        if phrase:
            candidates.append(phrase)

    deduped: list[str] = []
    seen: set[str] = set()
    for phrase in candidates:
        key = phrase.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(phrase)
        if len(deduped) >= limit:
            break
    return deduped


def normalize_phrase(raw: str) -> str:
    phrase = re.sub(r"\s+", " ", raw).strip(" ,.;:()[]{}")
    if len(phrase) < 4 or len(phrase) > 80:
        return ""
    lower = phrase.lower()
    if lower.startswith(PHRASE_BANNED_PREFIXES):
        return ""
    if "?" in phrase or phrase.count(",") > 2 or "[" in phrase or "]" in phrase:
        return ""
    words = [word.lower() for word in WORD_RE.findall(phrase)]
    if not words:
        return ""
    if len(words) > 8:
        return ""
    if words[0] in {"we", "our", "this", "these", "those", "that", "let"}:
        return ""
    if sum(1 for word in words if word in STOPWORDS) > len(words) // 2:
        return ""
    if any(
        len(token) > 18
        and not token.isupper()
        and not any(ch.isdigit() for ch in token)
        for token in re.findall(r"[A-Za-z0-9\-_/]+", phrase)
    ):
        return ""
    if all(word in STOPWORDS for word in words):
        return ""
    if any(term in lower for term in PHRASE_BANNED_TERMS):
        return ""
    if any(word in PHRASE_GENERIC_TERMS for word in words):
        return ""
    if len(words) == 1:
        token = words[0]
        if token in STOPWORDS:
            return ""
        # Single lowercase words like "data" or "recording" generate weak,
        # easily confounded questions. Keep only structurally distinctive
        # single-token phrases such as acronyms, hyphenated terms, or tokens
        # with digits.
        if not (
            "-" in phrase
            or any(ch.isdigit() for ch in phrase)
            or phrase.isupper()
            or re.search(r"[A-Z].*[A-Z]", phrase)
        ):
            return ""
    if lower.startswith("the ") and len(words) <= 2:
        return ""
    return phrase


def phrase_quality_score(phrase: str) -> int:
    words = [word.lower() for word in WORD_RE.findall(phrase)]
    informative_words = [word for word in words if word not in STOPWORDS]
    score = 0

    if 2 <= len(words) <= 4:
        score += 4
    elif len(words) == 1:
        score += 1
    else:
        score += max(0, 5 - len(words))

    score += min(3, sum(1 for word in informative_words if len(word) >= 5))

    if "-" in phrase:
        score += 2
    if any(ch.isdigit() for ch in phrase):
        score += 1
    if phrase.isupper():
        score += 2
    if re.search(r"[A-Z].*[A-Z]", phrase):
        score += 1
    if phrase.lower().startswith(("a ", "an ", "the ")):
        score -= 1

    return score


def is_high_confidence_phrase(phrase: str) -> bool:
    words = [word.lower() for word in WORD_RE.findall(phrase)]
    if not words or len(words) > 5:
        return False
    if any(
        len(token) > 18
        and not token.isupper()
        and not any(ch.isdigit() for ch in token)
        for token in re.findall(r"[A-Za-z0-9\-_/]+", phrase)
    ):
        return False
    return phrase_quality_score(phrase) >= 6


def generate_questions(
    papers: Sequence[PaperRecord],
    *,
    question_count: int = 60,
) -> list[QuestionRecord]:
    paper_lookup = {paper.paper_id: paper for paper in papers}
    per_paper_phrases: dict[str, list[str]] = {}
    phrase_docs: dict[str, set[str]] = {}
    phrase_counts: dict[str, dict[str, int]] = {}
    phrase_scores: dict[str, dict[str, int]] = {}

    for paper in papers:
        clean_sources = [part.strip() for part in (paper.title, paper.summary) if part.strip()]
        text = " ".join(clean_sources) or load_corpus_text(paper)
        phrases = extract_candidate_phrases(text)
        per_paper_phrases[paper.paper_id] = phrases
        lower_text = text.lower()
        phrase_counts[paper.paper_id] = {
            phrase.lower(): lower_text.count(phrase.lower()) for phrase in phrases
        }
        phrase_scores[paper.paper_id] = {
            phrase.lower(): phrase_quality_score(phrase) for phrase in phrases
        }
        for phrase in phrases:
            phrase_docs.setdefault(phrase.lower(), set()).add(paper.paper_id)

    questions: list[QuestionRecord] = []
    q_index = 0

    pair_candidates: list[tuple[int, str, str, str]] = []
    single_candidates: list[tuple[int, str, str]] = []

    for paper in papers:
        paper_id = paper.paper_id
        unique_phrases = [
            phrase
            for phrase in per_paper_phrases[paper_id]
            if len(phrase_docs.get(phrase.lower(), set())) == 1
            and is_high_confidence_phrase(phrase)
        ]
        unique_phrases.sort(
            key=lambda phrase: (
                -phrase_scores[paper_id].get(phrase.lower(), 0),
                -phrase_counts[paper_id].get(phrase.lower(), 0),
                len(phrase),
            )
        )
        for phrase in unique_phrases[:4]:
            score = (
                phrase_scores[paper_id].get(phrase.lower(), 0)
                + phrase_counts[paper_id].get(phrase.lower(), 0)
            )
            single_candidates.append((score, paper_id, phrase))
        if len(unique_phrases) >= 2:
            pair_score = (
                phrase_scores[paper_id].get(unique_phrases[0].lower(), 0)
                + phrase_scores[paper_id].get(unique_phrases[1].lower(), 0)
                + min(
                    phrase_counts[paper_id].get(unique_phrases[0].lower(), 0),
                    phrase_counts[paper_id].get(unique_phrases[1].lower(), 0),
                )
            )
            pair_candidates.append((pair_score, paper_id, unique_phrases[0], unique_phrases[1]))

    for _, paper_id, phrase_a, phrase_b in sorted(
        pair_candidates,
        key=lambda item: (-item[0], item[1], item[2], item[3]),
    ):
        paper = paper_lookup[paper_id]
        questions.append(
            QuestionRecord(
                question_id=f"q_{q_index:04d}",
                question=f'Which paper discusses both "{phrase_a}" and "{phrase_b}"?',
                answer=paper.title,
                answer_aliases=[paper.title],
                source_papers=[paper.paper_id],
                question_type="paper_lookup",
                evidence_terms=[phrase_a, phrase_b],
            )
        )
        q_index += 1
        if len(questions) >= min(question_count, len(pair_candidates)):
            break

    used_single_terms = {
        phrase.lower()
        for question in questions
        for phrase in question.evidence_terms
    }
    for _, paper_id, phrase in sorted(
        single_candidates,
        key=lambda item: (-item[0], item[1], item[2]),
    ):
        if len(questions) >= question_count:
            break
        if phrase.lower() in used_single_terms:
            continue
        paper = paper_lookup[paper_id]
        questions.append(
            QuestionRecord(
                question_id=f"q_{q_index:04d}",
                question=f'Which paper discusses "{phrase}"?',
                answer=paper.title,
                answer_aliases=[paper.title],
                source_papers=[paper.paper_id],
                question_type="paper_lookup",
                evidence_terms=[phrase],
            )
        )
        q_index += 1
        used_single_terms.add(phrase.lower())

    multi_questions: list[QuestionRecord] = []
    reusable = sorted(
        (
            (phrase, sorted(docs))
            for phrase, docs in phrase_docs.items()
            if 1 < len(docs) <= 4
        ),
        key=lambda item: (len(item[1]), item[0]),
    )
    for idx, (phrase_a, docs_a) in enumerate(reusable):
        titles: list[str] = []
        selected_docs: list[str] = []
        for phrase_b, docs_b in reusable[idx + 1 :]:
            overlap = sorted(set(docs_a) & set(docs_b))
            if 2 <= len(overlap) <= 3:
                selected_docs = overlap
                titles = [paper_lookup[paper_id].title for paper_id in overlap]
                multi_questions.append(
                    QuestionRecord(
                        question_id=f"q_{q_index:04d}",
                        question=f'Which papers discuss both "{phrase_a}" and "{phrase_b}"?',
                        answer=titles,
                        answer_aliases=titles,
                        source_papers=selected_docs,
                        question_type="multi_paper_lookup",
                        evidence_terms=[phrase_a, phrase_b],
                    )
                )
                q_index += 1
                break
        if len(questions) + len(multi_questions) >= question_count:
            break

    combined = questions + multi_questions
    return combined[:question_count]


def save_questions(
    questions: Sequence[QuestionRecord],
    *,
    path: Path = QUESTIONS_PATH,
    question_source: str = "generated",
    corpus_sha256: str = "",
) -> None:
    payload = {
        "meta": {
            "created_at": utc_now(),
            "frozen": True,
            "question_source": question_source,
            "question_count": len(questions),
            "corpus_sha256": corpus_sha256,
        },
        "questions": [asdict(question) for question in questions],
    }
    write_json(path, payload)


def save_manifest(
    papers: Sequence[PaperRecord],
    *,
    path: Path = CORPUS_MANIFEST_PATH,
    corpus_text: str = "",
) -> None:
    payload = {
        "meta": {
            "created_at": utc_now(),
            "document_count": len(papers),
            "estimated_tokens": estimated_token_count(corpus_text) if corpus_text else 0,
            "bytes": len(corpus_text.encode("utf-8")) if corpus_text else 0,
        },
        "papers": [asdict(paper) for paper in papers],
    }
    write_json(path, payload)


def call_ollama(
    prompt: str,
    *,
    endpoint: str = DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    model: str | None = None,
    timeout: int = 120,
) -> str:
    resolved_model = model or resolve_ollama_model()
    response = requests.post(
        endpoint,
        json={"model": resolved_model, "prompt": prompt, "stream": False},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return str(payload.get("response", "")).strip()


def resolve_ollama_model(
    preferred: str | None = None,
    *,
    tags_endpoint: str = DEFAULT_OLLAMA_TAGS_ENDPOINT,
) -> str:
    if preferred:
        return preferred
    response = requests.get(tags_endpoint, timeout=5)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("models", [])
    if not models:
        raise RuntimeError("No Ollama models available.")
    names = [str(model.get("name", "")) for model in models if model.get("name")]
    for candidate in ("qwen2.5-coder:7b", "qwen2.5:7b", "manifold-predictor:latest"):
        if candidate in names:
            return candidate
    return names[0]


def render_context_prompt(question: QuestionRecord, contexts: Sequence[dict[str, str]]) -> str:
    blocks = []
    for idx, context in enumerate(contexts, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[Context {idx}]",
                    f"Paper: {context['title']}",
                    context["text"],
                ]
            )
        )
    joined = "\n\n".join(blocks)
    return (
        "Answer the question using only the provided context.\n"
        "If the context is insufficient, answer exactly with INSUFFICIENT_CONTEXT.\n\n"
        f"Question: {question.question}\n\n"
        f"{joined}\n\n"
        "Return a concise answer."
    )


def answer_from_context(
    question: QuestionRecord,
    contexts: Sequence[dict[str, str]],
    *,
    qa_backend: str = "ollama",
    ollama_endpoint: str = DEFAULT_OLLAMA_GENERATE_ENDPOINT,
    ollama_model: str | None = None,
) -> str:
    if not contexts:
        return "INSUFFICIENT_CONTEXT"
    if qa_backend == "extractive":
        titles = [context["title"] for context in contexts]
        if question.question_type == "multi_paper_lookup":
            unique_titles = list(dict.fromkeys(titles))
            needed = max(1, len(question.source_papers))
            return "; ".join(unique_titles[:needed])
        return titles[0]
    prompt = render_context_prompt(question, contexts)
    return call_ollama(prompt, endpoint=ollama_endpoint, model=ollama_model)


def build_retrieved_contexts(
    ranked_chunks: Sequence[ChunkRecord],
    *,
    max_context_tokens: int,
) -> list[dict[str, str]]:
    contexts: list[dict[str, str]] = []
    used = 0
    seen: set[str] = set()
    for chunk in ranked_chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        if used >= max_context_tokens:
            break
        contexts.append({"paper_id": chunk.paper_id, "title": chunk.title, "text": chunk.text})
        used += chunk.estimated_tokens
    return contexts


def write_metrics(path: Path, payload: dict[str, object]) -> None:
    payload = dict(payload)
    payload.setdefault("created_at", utc_now())
    write_json(path, payload)


def corpus_metrics_payload(papers: Sequence[PaperRecord], combined_corpus: str) -> dict[str, object]:
    return {
        "document_count": len(papers),
        "bytes": len(combined_corpus.encode("utf-8")),
        "characters": len(combined_corpus),
        "estimated_tokens": estimated_token_count(combined_corpus),
        "corpus_sha256": corpus_hash(papers),
    }


def download_arxiv_entries(
    *,
    categories: Sequence[str],
    paper_count: int,
    timeout: int = 30,
) -> list[dict[str, object]]:
    per_category = max(1, math.ceil(paper_count / max(len(categories), 1)) + 10)
    seen: set[str] = set()
    entries: list[dict[str, object]] = []
    for category in categories:
        params = {
            "search_query": f"cat:{category}",
            "start": 0,
            "max_results": per_category,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        response = requests.get(
            f"{ARXIV_API_URL}?{urlencode(params)}",
            timeout=timeout,
            headers={"User-Agent": "structural-manifold-compression-demo/0.1"},
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.text)
        ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
        for entry in root.findall("atom:entry", ns):
            arxiv_id = entry.findtext("atom:id", default="", namespaces=ns).rsplit("/", 1)[-1]
            if not arxiv_id or arxiv_id in seen:
                continue
            title = re.sub(r"\s+", " ", entry.findtext("atom:title", default="", namespaces=ns)).strip()
            categories_list = [
                str(node.attrib.get("term", ""))
                for node in entry.findall("atom:category", ns)
                if node.attrib.get("term")
            ]
            pdf_url = ""
            for link in entry.findall("atom:link", ns):
                if link.attrib.get("title") == "pdf":
                    pdf_url = str(link.attrib.get("href", ""))
                    break
            if not pdf_url:
                continue
            summary = re.sub(r"\s+", " ", entry.findtext("atom:summary", default="", namespaces=ns)).strip()
            entries.append(
                {
                    "source_id": arxiv_id,
                    "title": title,
                    "summary": summary,
                    "published": entry.findtext("atom:published", default="", namespaces=ns),
                    "categories": categories_list,
                    "pdf_url": pdf_url,
                }
            )
            seen.add(arxiv_id)
            if len(entries) >= paper_count:
                return entries
    return entries[:paper_count]


def iter_local_documents(input_dir: Path) -> Iterator[dict[str, object]]:
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".txt", ".md"}:
            continue
        title = path.stem.replace("_", " ").replace("-", " ").strip().title()
        yield {
            "source_id": path.stem,
            "title": title,
            "published": "",
            "categories": ["local"],
            "pdf_url": "",
            "path": path,
        }
