from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field

# ---------- Categories ----------


# ## Categorization framework (pinned defaults)

# To cover “dialogue / narration / treatise / secondary” *and* remain faithful to classical/early‑modern practice, I combine:

# * **Discourse form** (dialogic, narrative, didactic/expository, epistolary).
# * **Rhetorical genus** (deliberative, judicial/forensic, epideictic), used as a *secondary* steering tag.

# I’ll use these **TextType** buckets:

# * **DIALOGUE** — philosophical or catechetical dialogue (Plato, Socratic, dialogues in patristic/ scholastic texts).
# * **NARRATIVE\_PERSONAL** — autobiographical/confessional or inward narration (e.g., *Confessions*).
# * **NARRATIVE\_HISTORICAL** — historiography, gospel‑style narrative, annals.
# * **TREATISE** — systematic/didactic exposition (Aristotle, Epictetus handbook sections, early modern essays).
# * **EPISTLE** — paraenetic/instructional letters and circular epistles.
# * **SECONDARY\_EXPOSITION** — scholarly articles/summaries (e.g., SEP), commentaries.

# **Pinned delivery defaults** (conservative to reduce “over‑acting”):

# | TextType              | Laban‑derived Profile  | exaggeration | cfg\_weight (pace proxy) | temperature | Notes                                                   |
# | --------------------- | ---------------------- | -----------: | -----------------------: | ----------: | ------------------------------------------------------- |
# | DIALOGUE              | **Press** (probing)    |         0.66 |                     0.50 |        0.70 | Crisp interrogatives; add holds only at reveals.        |
# | NARRATIVE\_PERSONAL   | **Float** (reflective) |         0.58 |                     0.45 |        0.72 | Softer movement, mild variety.                          |
# | NARRATIVE\_HISTORICAL | **Glide** (steady)     |         0.60 |                     0.52 |        0.70 | Even pace; spike to Flick/Punch only for action chunks. |
# | TREATISE              | **Dab** (precise)      |         0.62 |                     0.58 |        0.68 | Definitions land cleanly; restrained variation.         |
# | EPISTLE               | **Press** (earnest)    |         0.64 |                     0.45 |        0.68 | Gentle deliberateness; longer cadences at appeals.      |
# | SECONDARY\_EXPOSITION | **Glide** (neutral)    |         0.56 |                     0.52 |        0.65 | Scholarly clarity; least variation.                     |

# (Rhetorical genus still appears as `rhetoric`: `deliberative`, `judicial`, `epideictic`, `didactic`, `narrative`, `lament` and can subtly bias defaults in your orchestration if you choose.)


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
