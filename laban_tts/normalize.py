"""
Partition texts and prepare a normalized output for audiobook cueing.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, cast
from urllib.parse import urlparse

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pydantic_xml import BaseXmlModel, attr, element, wrapped
from unstructured.documents.elements import Element
from unstructured.partition.auto import partition

_PRIMARY_CHUNK_STRATEGY = "by_title"
_FALLBACK_CHUNK_STRATEGY = "basic"
_DEFAULT_MAX_CHARACTERS = 8000
_EXCERPT_CHARS = 160


# TODO: Helpers are inherently problematic. You must provide justification for each in their docstring. If you cannot justify confidently, the helper cannot exist.
def _derive_text_name(path_or_url: Path | str) -> str:
    if isinstance(path_or_url, Path):
        return path_or_url.stem
    candidate = str(path_or_url)
    if candidate.startswith(("http://", "https://")):
        parsed = urlparse(candidate)
        name = Path(parsed.path).name or "document"
        stem = Path(name).stem
        return stem or "document"
    return Path(candidate).stem or "document"


# TODO: This is impossible to justify and must be removed.
def _collapse_whitespace(value: str) -> str:
    return " ".join(value.split())


def _leading_excerpt(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    collapsed = _collapse_whitespace(text)
    if not collapsed:
        return None
    if len(collapsed) <= _EXCERPT_CHARS:
        return collapsed
    snippet = collapsed[:_EXCERPT_CHARS]
    cutoff = snippet.rfind(" ")
    if cutoff >= int(_EXCERPT_CHARS * 0.6):
        snippet = snippet[:cutoff]
    return snippet.rstrip() + "..."


# TODO: Forgiving programming is illegal. You must justify the need for it, and it cannot be any uncertainity about the XML content, as you fully control it.
def _trailing_excerpt(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    collapsed = _collapse_whitespace(text)
    if not collapsed:
        return None
    if len(collapsed) <= _EXCERPT_CHARS:
        return collapsed
    snippet = collapsed[-_EXCERPT_CHARS:]
    cutoff = snippet.find(" ")
    if 0 <= cutoff <= int(_EXCERPT_CHARS * 0.4):
        snippet = snippet[cutoff + 1 :]
    return "..." + snippet.lstrip()


# TODO: Highly suspicious. Remove.
def _resolve_elements(
    path_or_url: Path | str,
    *,
    chunking_strategy: str,
    max_characters: int,
) -> Tuple[str, List[Element]]:
    text_name = _derive_text_name(path_or_url)
    source = (
        Path(path_or_url)
        if not str(path_or_url).startswith(("http://", "https://"))
        else None
    )
    if source is None:
        elements = partition(
            url=str(path_or_url),
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
        )
    else:
        assert source.exists(), f"File {source} does not exist."
        elements = partition(
            filename=str(source.resolve()),
            chunking_strategy=chunking_strategy,
            max_characters=max_characters,
        )
    return text_name, list(elements)


def _collect_text_blocks(elements: Sequence[Element]) -> List[str]:
    blocks: List[str] = []
    for element in elements:
        # TODO: Extremely illigal. Impossible to justify ever using getattr.
        text = getattr(element, "text", "")
        candidate = text.strip()
        if candidate:
            blocks.append(candidate)
    return blocks


# TODO: Very illegal. You can never justify this. Use pydantic properly.
def _write_xml(path: Path, payload: str | bytes) -> None:
    text = payload if isinstance(payload, str) else payload.decode("utf-8")
    path.write_text(text)


# TODO: Model properly to merge with TextPartMetadata. ONLY ONE MODEL PER ENTITY. Search and scrape pydantic_xml documentation for proper use.
class TextPartDocument(BaseXmlModel, tag="text-part", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(name="text-name")
    part: int = attr(ge=1)
    source: str = attr()
    last_excerpt: Optional[str] = attr(name="last-excerpt", default=None)
    next_excerpt: Optional[str] = attr(name="next-excerpt", default=None)
    text: str = element(tag="text")

    def metadata(self) -> "TextPartMetadata":
        return TextPartMetadata.from_document(self)


# TODO: Why does this exist? It cannot be correct. Do not wrap TextPartDocument and do not store `xml_path` in the model. It is irrelevant once deserialized.
class PartEntry(BaseModel):
    """In-memory representation of a partitioned text chunk."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    xml_path: Path
    document: TextPartDocument

    @property
    def text_name(self) -> str:
        return self.document.text_name

    @property
    def part(self) -> int:
        return self.document.part

    @property
    def text(self) -> str:
        return self.document.text

    def metadata_document(self) -> "TextPartMetadata":
        return self.document.metadata()


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


# TODO: cast seems like an illegal workaround. Use attr or wrapped or supported pydantic_xml tags.
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
    speakers: List[SpeakerGuess] = cast(
        List[SpeakerGuess],
        Field(
            default_factory=list,
            description="Candidate speakers found (if any); include 'narrator' when unknown.",
        ),
    )
    removed: List[RemovedItem] = cast(
        List[RemovedItem],
        Field(
            default_factory=list,
            description="All removed/non-spoken snippets for audit and reproducibility.",
        ),
    )
    heuristics: Heuristics = Field(
        description="Validation metrics to audit the cleaning step."
    )


# TODO: Document what the purpose of this is. Detailed speaker identification is in scope of cues.py. What does the proposed normalized xml look like? Why not just attributes of <text>?
class SpeakerGuessDocument(BaseXmlModel, tag="speaker", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = attr()
    evidence: Optional[str] = element(tag="evidence", default=None)

    @classmethod
    def from_model(cls, speaker: SpeakerGuess) -> "SpeakerGuessDocument":
        return cls(name=speaker.name, evidence=speaker.evidence)

    def to_model(self) -> SpeakerGuess:
        return SpeakerGuess(name=self.name, evidence=self.evidence)


# TODO: I do not understand the intent here. Removed text elements should be inside <text> in the correct places following original text, not disjointed. A method of the pydantic class should allow clean text (filtering out inner elements like this).
class RemovedItemDocument(BaseXmlModel, tag="removed", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: RemovedKind = attr()
    text: str = element(tag="text")

    @classmethod
    def from_model(cls, item: RemovedItem) -> "RemovedItemDocument":
        return cls(kind=item.kind, text=item.text)

    def to_model(self) -> RemovedItem:
        return RemovedItem(kind=self.kind, text=self.text)


# TODO: Why is this an element and note attributes of <text>?
class HeuristicsDocument(BaseXmlModel, tag="heuristics", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_chars: int = attr(name="input-chars")
    output_chars: int = attr(name="output-chars")
    removed_chars: int = attr(name="removed-chars")
    removal_ratio: float = attr(name="removal-ratio")
    paragraph_count: int = attr(name="paragraph-count")
    speaker_candidate_count: int = attr(name="speaker-candidate-count")
    likely_has_front_matter: Optional[bool] = attr(
        name="likely-has-front-matter", default=None
    )

    @classmethod
    def from_model(cls, heuristics: Heuristics) -> "HeuristicsDocument":
        return cls(
            input_chars=heuristics.input_chars,
            output_chars=heuristics.output_chars,
            removed_chars=heuristics.removed_chars,
            removal_ratio=heuristics.removal_ratio,
            paragraph_count=heuristics.paragraph_count,
            speaker_candidate_count=heuristics.speaker_candidate_count,
            likely_has_front_matter=heuristics.likely_has_front_matter,
        )

    def to_model(self) -> Heuristics:
        return Heuristics(
            input_chars=self.input_chars,
            output_chars=self.output_chars,
            removed_chars=self.removed_chars,
            removal_ratio=self.removal_ratio,
            paragraph_count=self.paragraph_count,
            speaker_candidate_count=self.speaker_candidate_count,
            likely_has_front_matter=self.likely_has_front_matter,
        )


# TODO: This kind of construction is always wrong. Never create unnecessary structure. There is not reason for more than one `Part` model.
class TextPartMetadata(BaseXmlModel, tag="text-part-meta", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = element()
    text_name: str = attr(name="text-name")
    part: int = attr(ge=1)
    source: str = attr()
    last_excerpt: Optional[str] = attr(name="last-excerpt", default=None)
    next_excerpt: Optional[str] = attr(name="next-excerpt", default=None)

    @classmethod
    def from_document(cls, document: TextPartDocument) -> "TextPartMetadata":
        return cls(
            text_name=document.text_name,
            part=document.part,
            source=document.source,
            last_excerpt=document.last_excerpt,
            next_excerpt=document.next_excerpt,
        )


# TODO: There is not such thing as a summary. Remove.
class NormalizedSummaryDocument(
    BaseXmlModel, tag="previous-normalized", skip_empty=True
):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(name="text-name")
    part: int = attr(ge=1)
    category: TextType = attr()
    speakers: List[SpeakerGuessDocument] = wrapped(
        "speakers",
        element(tag="speaker", default_factory=list),
    )
    heuristics: HeuristicsDocument = element(tag="heuristics")


# TODO: This is `NormalizedOutput` NEVER duplicate models. Learn to use pydantic_xml properly instead. Refactor all code accordingly.
class NormalizedPartDocument(BaseXmlModel, tag="normalized-part", skip_empty=True):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text_name: str = attr(name="text-name")
    part: int = attr(ge=1)
    category: TextType = attr()
    text: str = element(tag="text")
    speakers: List[SpeakerGuessDocument] = wrapped(
        "speakers",
        element(tag="speaker", default_factory=list),
    )
    removed: List[RemovedItemDocument] = wrapped(
        "removed",
        element(tag="item", default_factory=list),
    )
    heuristics: HeuristicsDocument = element(tag="heuristics")

    def to_model(self) -> NormalizedOutput:
        return NormalizedOutput(
            text_name=self.text_name,
            category=self.category,
            cleaned_text=self.text,
            speakers=[speaker.to_model() for speaker in self.speakers],
            removed=[item.to_model() for item in self.removed],
            heuristics=self.heuristics.to_model(),
        )

    @classmethod
    def from_output(
        cls, part: TextPartDocument, normalized: NormalizedOutput
    ) -> "NormalizedPartDocument":
        return cls(
            text_name=normalized.text_name,
            part=part.part,
            category=normalized.category,
            text=normalized.cleaned_text,
            speakers=[SpeakerGuessDocument.from_model(s) for s in normalized.speakers],
            removed=[
                RemovedItemDocument.from_model(item) for item in normalized.removed
            ],
            heuristics=HeuristicsDocument.from_model(normalized.heuristics),
        )

    def to_summary_document(self) -> NormalizedSummaryDocument:
        copied_speakers = [
            SpeakerGuessDocument(name=speaker.name, evidence=speaker.evidence)
            for speaker in self.speakers
        ]
        copied_heuristics = HeuristicsDocument.from_model(self.heuristics.to_model())
        return NormalizedSummaryDocument(
            text_name=self.text_name,
            part=self.part,
            category=self.category,
            speakers=copied_speakers,
            heuristics=copied_heuristics,
        )


# TODO: Remove an put medatate in the SINGLE normalized part model.
class NormalizationMetadata(
    BaseXmlModel, tag="normalization-metadata", skip_empty=True
):
    model_config = ConfigDict(frozen=True, extra="forbid")

    part: TextPartMetadata = element(tag="part")
    previous: Optional[NormalizedSummaryDocument] = element(
        tag="previous-normalized", default=None
    )


# TODO: This abomination must not exist. Each normalized part XML must get deserialized, returning a list of the SINGLE "normalized part" model, and nothing more. XML path MUST NEVER BE INCLUDED IN MODELS.
class NormalizedEntry(BaseModel):
    """Normalized artifact persisted to disk."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    xml_path: Path
    document: NormalizedPartDocument

    @property
    def text_name(self) -> str:
        return self.document.text_name

    @property
    def part(self) -> int:
        return self.document.part

    @property
    def normalized(self) -> NormalizedOutput:
        return self.document.to_model()

    def summary_document(self) -> NormalizedSummaryDocument:
        return self.document.to_summary_document()


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


def partition_text(
    path_or_url: Path | str,
    parts_dir: Path,
    *,
    chunking_strategy: str = _PRIMARY_CHUNK_STRATEGY,
    max_characters: int = _DEFAULT_MAX_CHARACTERS,
) -> List[PartEntry]:
    """Partition a source document into XML parts stored under ``parts_dir``."""

    parts_dir.mkdir(parents=True, exist_ok=True)
    text_name, elements = _resolve_elements(
        path_or_url,
        chunking_strategy=chunking_strategy,
        max_characters=max_characters,
    )
    blocks = _collect_text_blocks(elements)
    if len(blocks) <= 1 and chunking_strategy == _PRIMARY_CHUNK_STRATEGY:
        logger.debug(
            "partition.blocks_insufficient text_name={} strategy={}",
            text_name,
            chunking_strategy,
        )
        _, fallback_elements = _resolve_elements(
            path_or_url,
            chunking_strategy=_FALLBACK_CHUNK_STRATEGY,
            max_characters=max_characters,
        )
        fallback_blocks = _collect_text_blocks(fallback_elements)
        if len(fallback_blocks) > len(blocks):
            logger.debug(
                "partition.strategy_fallback text_name={} primary_chunks={} fallback_chunks={}",
                text_name,
                len(blocks),
                len(fallback_blocks),
            )
            blocks = fallback_blocks
    if not blocks:
        raise ValueError("Partitioning produced no text blocks.")

    entries: List[PartEntry] = []
    for offset, content in enumerate(blocks):
        idx = offset + 1
        prev_text = blocks[offset - 1] if offset > 0 else None
        next_text = blocks[offset + 1] if offset + 1 < len(blocks) else None
        document = TextPartDocument(
            text_name=text_name,
            part=idx,
            source=str(path_or_url),
            last_excerpt=_trailing_excerpt(prev_text),
            next_excerpt=_leading_excerpt(next_text),
            text=content,
        )
        xml_path = parts_dir / f"{text_name}-part{idx:03d}.xml"
        payload = document.to_xml(
            encoding="unicode", pretty_print=True, skip_empty=True
        )
        _write_xml(xml_path, payload)
        logger.debug("partition.part_saved path={} chars={}", xml_path, len(content))
        entries.append(PartEntry(xml_path=xml_path, document=document))

    logger.info(
        "partition.done source={source} parts={parts}",
        source=path_or_url,
        parts=len(entries),
    )
    return entries


def load_part_entries(parts_dir: Path) -> List[PartEntry]:
    """Load partition outputs from disk into memory."""

    entries: List[PartEntry] = []
    for xml_path in sorted(parts_dir.glob("*.xml")):
        document = TextPartDocument.from_xml(xml_path.read_text())
        entries.append(PartEntry(xml_path=xml_path, document=document))
    entries.sort(key=lambda entry: (entry.text_name, entry.part))
    return entries


def normalize_parts(
    part_entries: Sequence[PartEntry],
    normalize_dir: Path,
    llm: ChatOpenAI,
) -> List[NormalizedEntry]:
    """Run normalization over partitioned parts, persist XML, and return entries."""

    if not part_entries:
        raise ValueError("No parts provided for normalization.")

    normalize_dir.mkdir(parents=True, exist_ok=True)

    results: List[NormalizedEntry] = []
    previous_entry: Optional[NormalizedEntry] = None
    for entry in part_entries:
        metadata_xml = NormalizationMetadata(
            part=entry.metadata_document(),
            previous=previous_entry.summary_document() if previous_entry else None,
        ).to_xml(encoding="unicode", pretty_print=True, skip_empty=True)
        if not isinstance(metadata_xml, str):
            metadata_xml = metadata_xml.decode("utf-8")

        messages = [
            SystemMessage(content=NORMALIZE_PROMPT),
            HumanMessage(
                content=(
                    "== METADATA ==\n"
                    f"{metadata_xml}\n\n"
                    "== INPUT TEXT ==\n"
                    f"{entry.text}"
                )
            ),
        ]
        logger.info(
            "normalize.part text_name={name} part={part} chars={chars}",
            name=entry.text_name,
            part=entry.part,
            chars=len(entry.text),
        )
        callback = UsageMetadataCallbackHandler()
        structured_llm = cast(Any, llm).with_structured_output(
            NormalizedOutput, include_raw=True
        )
        response = cast(
            dict[str, object],
            structured_llm.invoke(
                messages, config=RunnableConfig(callbacks=[callback])
            ),
        )
        normalized = cast(NormalizedOutput, response["parsed"])
        logger.debug("normalize.tokens usage={usage}", usage=callback.usage_metadata)

        document = NormalizedPartDocument.from_output(entry.document, normalized)
        xml_path = (
            normalize_dir / f"{entry.text_name}-part{entry.part:03d}-normalized.xml"
        )
        payload = document.to_xml(
            encoding="unicode", pretty_print=True, skip_empty=True
        )
        _write_xml(xml_path, payload)

        heuristics = normalized.heuristics
        logger.debug(
            (
                "normalize.heuristics input={input_chars} output={output_chars} removed={removed_chars} "
                "removal_ratio={ratio:.3f} paragraphs={paragraphs} speakers={speakers}"
            ),
            input_chars=heuristics.input_chars,
            output_chars=heuristics.output_chars,
            removed_chars=heuristics.removed_chars,
            ratio=heuristics.removal_ratio,
            paragraphs=heuristics.paragraph_count,
            speakers=heuristics.speaker_candidate_count,
        )

        entry_out = NormalizedEntry(xml_path=xml_path, document=document)
        results.append(entry_out)
        previous_entry = entry_out

    logger.info("normalize.done parts={count}", count=len(results))
    return results


def load_normalized_entries(normalize_dir: Path) -> List[NormalizedEntry]:
    """Read normalized XML outputs from disk."""

    entries: List[NormalizedEntry] = []
    for xml_path in sorted(normalize_dir.glob("*-normalized.xml")):
        document = NormalizedPartDocument.from_xml(xml_path.read_text())
        entries.append(NormalizedEntry(xml_path=xml_path, document=document))
    entries.sort(key=lambda entry: (entry.text_name, entry.part))
    return entries
