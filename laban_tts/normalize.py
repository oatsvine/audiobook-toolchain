"""
Partition texts and prepare a normalized output for audiobook cueing.
"""

from __future__ import annotations
from pathlib import Path

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger
from enum import Enum
from typing import Generator, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field
from unstructured.partition.auto import partition


# TODO: Serialize as XML (include content in <text> element) instead of JSON/text files.
class PartMeta(BaseModel):
    """Sidecar describing one text part produced during preparation."""

    model_config = ConfigDict(extra="forbid")

    text_name: str = Field(min_length=1)
    part: int = Field(ge=1)
    source: str = Field(description="Original source path or URL for this part.")
    last_excerpt: Optional[str] = Field(default=None)
    next_excerpt: Optional[str] = Field(default=None)


class PartEntry(BaseModel):
    """In-memory representation of a part ready for normalization."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    meta: PartMeta
    text_path: Path
    meta_path: Path
    text: str


def partition_text(
    path_or_url: Path,
    parts_dir: Path,
    chunk_tokens: int = 5000,
) -> None:
    """Read a local file or URL and yield (metadata, text) pairs for each part."""

    if isinstance(path_or_url, str) and path_or_url.startswith("http"):
        text_name = path_or_url.split("/")[-1].split(".")[0]
        elements = partition(url=path_or_url)
    else:
        path = Path(path_or_url)
        assert path.exists(), f"File {path} does not exist."
        text_name = path.stem
        elements = partition(filename=str(path.resolve()))

    logger.info(
        "Partitioned {count} elements from {source}",
        count=len(elements),
        source=path_or_url,
    )
    text = "\n\n".join([e.text for e in elements if e.text and e.text.strip()])

    # TODO: Use unstructured canonical tools to partition by chapters/sections instead of langchain.
    # TODO: To do this, clone the unstructured repo: https://github.com/Unstructured-IO/unstructured
    # Explore and study test_unstructured carefully, starting with `partition` and `chunking`.
    # Determine how to leverage unstructured's chunking capabilities to split by chapters/sections as reliably as possible.
    # Check that you have the necessary dependencies installed for unstructured's chunking features, stop and ask user if you don't.
    # Use ephemeral python script to experiment with unstructured's chunking features on sample texts.
    # Use `corpus_texts` fixture to create test for this function that verifies correct chapter/section splitting.
    part_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4o",
        chunk_size=chunk_tokens,
        chunk_overlap=0,
    )
    excerpt_splitter = RecursiveCharacterTextSplitter(
        chunk_size=120,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )

    texts = part_splitter.split_text(text)
    logger.info("Split into {count} parts", count=len(texts))
    for idx, content in enumerate(texts, start=1):
        meta = PartMeta(text_name=text_name, part=idx, source=str(path_or_url))
        if idx > 1:
            meta.last_excerpt = "…" + excerpt_splitter.split_text(texts[idx - 2])[-1]
        if idx < len(texts):
            meta.next_excerpt = excerpt_splitter.split_text(texts[idx])[:1][0] + "…"

        # TODO: Must be a single XML, not txt/json
        stem = f"{meta.text_name}-part{meta.part:03d}"
        text_path = parts_dir / f"{stem}.txt"
        json_path = parts_dir / f"{stem}.json"
        text_path.write_text(content)
        json_path.write_text(meta.model_dump_json(indent=2, exclude_none=True))
        logger.debug(
            "book.part_saved part={part} path={path}",
            part=meta.part,
            path=text_path,
        )


class TextType(str, Enum):
    """High-level discourse forms tailored to ancient/early-modern texts."""

    DIALOGUE = "dialogue"  # Socratic/dialogic catechesis
    NARRATIVE_PERSONAL = "narrative_personal"  # confessional, inward narrative
    NARRATIVE_HISTORICAL = "narrative_historical"  # historiography, gospel-like
    TREATISE = "treatise"  # systematic/didactic exposition
    EPISTLE = "epistle"  # paraenetic letter/circular
    SECONDARY_EXPOSITION = "secondary_exposition"  # encyclopedia/commentary


class RemovedKind(str, Enum):
    """Types of non-spoken material to strip before cueing."""

    FOREWORD = "foreword"
    PUBLISHER = "publisher_info"
    BLANK = "blank_line"
    PAGINATION = "pagination_marker"
    HEADER = "header"
    FOOTER = "footer"
    PAGE_BREAK = "page_break"
    DEDICATION = "dedication"
    EPIGRAPH = "epigraph"
    COPYRIGHT = "copyright"
    OTHER = "other"


class SpeakerGuess(BaseModel):
    """A candidate voice in the text (lowercase, dashes for spaces)."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    name: str = Field(
        description="Normalized name: lowercase with dashes (e.g., 'narrator', 'socrates')."
    )
    evidence: Optional[str] = Field(
        default=None, description="Short justification or first mention context."
    )


class RemovedItem(BaseModel):
    """Material removed from narration, preserved for auditing."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    kind: RemovedKind
    text: str = Field(description="Exact text removed; keep brief but representative.")


class Heuristics(BaseModel):
    """Validation heuristics for QA and gating."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    input_chars: int = Field(description="Original character count.")
    output_chars: int = Field(description="Cleaned character count.")
    removed_chars: int = Field(description="Total removed character count.")
    removal_ratio: float = Field(description="removed_chars / input_chars; 0..1 range.")
    paragraph_count: int = Field(description="Paragraphs in cleaned text.")
    speaker_candidate_count: int = Field(
        description="Detected distinct speakers candidates."
    )
    likely_has_front_matter: Optional[bool] = Field(
        default=None,
        description="True if patterns strongly indicate front matter was present and removed.",
    )


class NormalizedOutput(BaseModel):
    """
    Stage A output: cleaned text you can pass to Stage B.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = Field(description="Human-friendly name for this document.")
    category: TextType = Field(description="Dominant discourse form for this document.")
    cleaned_text: str = Field(
        description="Narration-ready text with extraneous matter removed, normalized spacing/hyphenation/encoding."
    )
    speakers: List[SpeakerGuess] = Field(
        default_factory=list,
        description="Candidate speakers found (if any); include 'narrator' when unknown.",
    )
    removed: List[RemovedItem] = Field(
        default_factory=list,
        description="All removed/non-spoken snippets for audit and reproducibility.",
    )
    heuristics: Heuristics = Field(
        description="Validation metrics to audit the cleaning step."
    )


NORMALIZE_PROMPT = """
You are the text normalization lead preparing ancient and early-modern material for a professional single-narrator audiobook session. Deliver a performance-ready script foundation while preserving the author's intent.

Follow this workflow:
- Strip non-performable paratext (publisher notes, page furniture, dedications, epigraphs, front/back matter) while retaining context essential to the listener's comprehension. Evaluate introductions and forewords for storytelling value; keep, trim, or drop them accordingly.
- Guard the author's meaning: never summarize or paraphrase the main text. Only fix readability issues (OCR artifacts, broken hyphenation, illegible ligatures) and expand abbreviations when they would confuse a listener.
- Adapt the pace to the spoken word form by strategically enhancing text with pausing points (e.g. commas, semicolons, colon, em dashes) and breaking up overly complex sentences, aiming for breath-friendly segments **without altering meaning.**
- Preserve true paragraphing, merge mechanical line breaks, and keep rhetorical markers (vocatives, stage cues, verse lineation) that inform pacing and emphasis.
- Normalize spacing, punctuation, and encoding so the cleaned text reads smoothly aloud while respecting period diction.
- Convert all uppercase titles to standard capitalization. For all uppercase text, restore proper casing based on context, using punctuation instead of all-caps for emphasis.
- Convert any leftover numbers to words, except for dates and common single-digit enumerations (e.g., "Book 1", "ten thousand angels").
- To ensure text is readable aloud, spot OCR artifacts, special characters, and other fixable flaws, fixing them conservatively as you see fit to preserve substance and flow.
- Classify the dominant discourse form (`category`) and surface likely rhetorical cues for downstream cueing without repeating field descriptions. Stay consistent with the project's TextType definitions.
- Detect plausible speakers with concise supporting evidence when the text signals dialogic turns; never invent voices.
- Record every removal succinctly in `removed` and keep heuristics truthful so reviewers can audit the cleaning step.

Return JSON that conforms to the NormalizedOutput schema exactly.
"""


def normalize(
    part_entries: List[PartEntry], normalize_dir: Path, llm: ChatOpenAI
) -> None:
    """Stage A — produce NormalizedOutput from parts/*.txt."""

    previous: Optional[NormalizedOutput] = None
    for entry in part_entries:
        prev_summary = (
            previous.model_dump(exclude={"cleaned_text"}, exclude_unset=True)
            if previous
            else None
        )
        # TODO: Must be xml
        metadata_block = json.dumps(
            {
                "text_name": entry.meta.text_name,
                "part": entry.meta.part,
                "source": entry.meta.source,
                "last_excerpt": entry.meta.last_excerpt,
                "next_excerpt": entry.meta.next_excerpt,
                "previous_normalized": prev_summary,
            },
            ensure_ascii=False,
            indent=2,
        )
        messages = [
            SystemMessage(content=NORMALIZE_PROMPT),
            HumanMessage(
                content=(
                    "== METADATA ==\n"
                    f"{metadata_block}\n\n"
                    "== INPUT TEXT ==\n"
                    f"{entry.text}"
                )
            ),
        ]
        logger.info(
            "normalize.part text_name={name} part={part} chars={chars}",
            name=entry.meta.text_name,
            part=entry.meta.part,
            chars=len(entry.text),
        )
        callback = UsageMetadataCallbackHandler()
        res = llm.with_structured_output(NormalizedOutput, include_raw=True).invoke(
            messages, config=RunnableConfig(callbacks=[callback])
        )
        normalized: NormalizedOutput = res["parsed"]
        logger.debug("normalize.tokens usage={usage}", usage=callback.usage_metadata)
        stem = f"{entry.meta.text_name}-part{entry.meta.part:03d}-normalized"
        text_path = normalize_dir / f"{stem}.txt"
        json_path = normalize_dir / f"{stem}.json"
        text_path.write_text(normalized.cleaned_text)
        json_path.write_text(
            normalized.model_dump_json(
                indent=2,
                exclude={"cleaned_text"},
                exclude_unset=True,
            )
        )

        heuristics = normalized.heuristics
        logger.debug(
            (
                "normalize.heuristics input={input_chars} output={output_chars} "
                "removed={removed_chars} removal_ratio={ratio:.3f} paragraphs={paragraphs} "
                "speakers={speakers}"
            ),
            input_chars=heuristics.input_chars,
            output_chars=heuristics.output_chars,
            removed_chars=heuristics.removed_chars,
            ratio=heuristics.removal_ratio,
            paragraphs=heuristics.paragraph_count,
            speakers=heuristics.speaker_candidate_count,
        )
        previous = normalized
