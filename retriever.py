"""Local TF-IDF retriever for PawPal+ (no API needed).

Loads the markdown files in knowledge/, splits each into one chunk per
"## Heading" section, and ranks chunks against a query by cosine similarity.

Each doc starts with a small front-matter block (species, life_stage, pet, ...)
that is used as a metadata filter so a cat never gets dog advice and Chester's
profile is never served to someone else's pet.

Special lines inside a section are parsed as structured data, not prose:
    Tip: one actionable sentence the planner can show to the owner
    Max-minutes: walk=20            (a guideline the planner can check tasks against)
    Suggest-task: category | description | minutes | key   ({pet} -> pet's name; key = word that
                                                             shows an existing task already covers it)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

logger = logging.getLogger("pawpal.rag")

DEFAULT_KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
PET_NAME_BONUS = 0.10  # a chunk written for this exact pet outranks generic ones


def configure_logging(log_path: Optional[str] = "logs/pawpal_rag.log", console: bool = False) -> None:
    """Send RAG logs to a file (and optionally the console). Safe to call twice."""
    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path)
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    if console:
        handler = logging.StreamHandler()
        handler.setFormatter(fmt)
        logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Text analysis: lowercase, drop stop words, very light stemming, unigrams+bigrams
# ---------------------------------------------------------------------------

def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[a-z]+", text.lower().replace("'s", ""))
    return [_stem(w) for w in words if w not in ENGLISH_STOP_WORDS and len(w) > 1]


def _analyzer(text: str) -> list[str]:
    tokens = tokenize(text)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    """A task the knowledge base thinks the pet may be missing (needs owner approval)."""
    pet_name: str
    category: str
    description: str
    minutes: int
    source: str
    score: float = 0.0
    key: str = ""          # word that marks this need as already covered


@dataclass
class Chunk:
    source: str                      # file name, e.g. "senior_dog_care.md"
    heading: str
    text: str                        # prose used for indexing and display
    tip: str = ""
    max_minutes: dict[str, int] = field(default_factory=dict)
    suggestions: list[tuple[str, str, int, str]] = field(default_factory=list)
    species: str = "any"
    life_stage: str = "any"
    pet: str = ""

    @property
    def citation(self) -> str:
        return f"{self.source}#{self.heading}"


@dataclass
class Hit:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        parts = raw.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].strip().splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().lower()] = value.strip().lower()
            return meta, parts[2]
    return meta, raw


def load_chunks(knowledge_dir: Path) -> list[Chunk]:
    """Read every .md file in the folder into chunks. Bad files are logged and skipped."""
    chunks: list[Chunk] = []
    for path in sorted(Path(knowledge_dir).glob("*.md")):
        try:
            meta, body = _parse_front_matter(path.read_text(encoding="utf-8"))
            for section in re.split(r"^## +", body, flags=re.MULTILINE)[1:]:
                heading, _, rest = section.partition("\n")
                prose: list[str] = []
                tip = ""
                max_minutes: dict[str, int] = {}
                suggestions: list[tuple[str, str, int, str]] = []
                for line in rest.strip().splitlines():
                    line = line.strip()
                    if line.lower().startswith("tip:"):
                        tip = line[4:].strip()
                        prose.append(tip)
                    elif line.lower().startswith("max-minutes:"):
                        for pair in line.split(":", 1)[1].split(","):
                            key, _, value = pair.partition("=")
                            max_minutes[key.strip().lower()] = int(value)
                    elif line.lower().startswith("suggest-task:"):
                        fields = [p.strip() for p in line.split(":", 1)[1].split("|")]
                        key = fields[3].lower() if len(fields) > 3 else ""
                        suggestions.append((fields[0].lower(), fields[1], int(fields[2]), key))
                    elif line:
                        prose.append(line)
                chunks.append(Chunk(
                    source=path.name,
                    heading=heading.strip(),
                    text=" ".join(prose),
                    tip=tip,
                    max_minutes=max_minutes,
                    suggestions=suggestions,
                    species=meta.get("species", "any"),
                    life_stage=meta.get("life_stage", "any"),
                    pet=meta.get("pet", ""),
                ))
        except (OSError, ValueError) as exc:
            logger.warning("Skipping unreadable knowledge file %s: %s", path.name, exc)
    return chunks


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

class KnowledgeRetriever:
    def __init__(self, knowledge_dir: Path | str = DEFAULT_KNOWLEDGE_DIR) -> None:
        self.knowledge_dir = Path(knowledge_dir)
        self.chunks: list[Chunk] = load_chunks(self.knowledge_dir) if self.knowledge_dir.is_dir() else []
        self._vectorizer: Optional[TfidfVectorizer] = None
        self._matrix = None
        if not self.chunks:
            logger.warning("No knowledge chunks loaded from %s", self.knowledge_dir)
            return
        self._vectorizer = TfidfVectorizer(analyzer=_analyzer, sublinear_tf=True)
        self._matrix = self._vectorizer.fit_transform(
            [f"{c.heading} {c.heading} {c.text}" for c in self.chunks]
        )
        logger.info("Indexed %d chunks from %d files", len(self.chunks),
                    len({c.source for c in self.chunks}))

    def _eligible(self, chunk: Chunk, species: Optional[str],
                  life_stage: Optional[str], pet_name: Optional[str]) -> bool:
        if species:
            allowed = {s.strip() for s in chunk.species.split(",")}
            if "any" not in allowed and species.lower() not in allowed:
                return False
        if life_stage and chunk.life_stage not in ("any", life_stage.lower()):
            return False
        if chunk.pet and chunk.pet != (pet_name or "").lower():
            return False  # another pet's personal profile
        return True

    def retrieve(self, query: str, k: Optional[int] = 3, species: Optional[str] = None,
                 life_stage: Optional[str] = None, pet_name: Optional[str] = None,
                 include_zero: bool = False) -> list[Hit]:
        """Top-k chunks (all if k is None) with a similarity score between 0 and 1.

        Zero-score chunks are dropped unless include_zero is True (used when the metadata
        filters alone decide which chunks are relevant).
        """
        if not query or not query.strip() or self._vectorizer is None:
            logger.info("retrieve: empty query or empty index -> no hits")
            return []
        scores = linear_kernel(self._vectorizer.transform([query]), self._matrix).ravel()
        hits = []
        for chunk, score in zip(self.chunks, scores):
            if not self._eligible(chunk, species, life_stage, pet_name):
                continue
            if chunk.pet and score > 0:
                score = min(1.0, score + PET_NAME_BONUS)
            if score > 0 or include_zero:
                hits.append(Hit(chunk, float(score)))
        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k] if k else hits
        logger.info("retrieve: query=%r species=%s stage=%s -> %s", query, species, life_stage,
                    [(h.chunk.citation, round(h.score, 3)) for h in hits])
        return hits
