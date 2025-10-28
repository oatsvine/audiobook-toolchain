import pytest

from laban_tts.cues import (
    CUE_PRIMER_TEXT,
    CuedChunk,
    CuedScript,
    EmphasisSpan,
    EmphasisStrategy,
    Profile,
    RhetoricTag,
    TextType,
    _ENGINE_PARAM_FIELD_LIMITS,  # type: ignore[reportPrivateUsage]
)


@pytest.mark.parametrize("profile", list(Profile))
@pytest.mark.parametrize("rhetoric", list(RhetoricTag))
def test_profile_rhetoric_combinations_stay_within_limits(
    profile: Profile, rhetoric: RhetoricTag
) -> None:
    chunk = CuedChunk(
        idx=1,
        text="Example sentence for testing.",
        speaker="narrator",
        text_type=TextType.TREATISE,
        rhetoric=rhetoric,
        profile=profile,
        emphasis=[],
        pre_pause_ms=0,
        post_pause_ms=0,
    )
    params = chunk.engine_params()
    for field_name, (lower, upper) in _ENGINE_PARAM_FIELD_LIMITS.items():
        value = getattr(params, field_name)
        assert (
            lower <= value <= upper
        ), f"{field_name} out of bounds for profile={profile} rhetoric={rhetoric}: {value}"


@pytest.mark.parametrize("strategy", list(EmphasisStrategy))
def test_emphasis_strategies_respect_limits(strategy: EmphasisStrategy) -> None:
    span = EmphasisSpan(
        phrase="Example",
        strategy=strategy,
        pause_before_ms=0,
        pause_after_ms=0,
    )
    chunk = CuedChunk(
        idx=1,
        text="Example",
        speaker="narrator",
        text_type=TextType.DIALOGUE,
        rhetoric=RhetoricTag.deliberative,
        profile=Profile.Press,
        emphasis=[span],
        pre_pause_ms=600,
        post_pause_ms=900,
    )
    params = chunk.engine_params()
    for field_name, (lower, upper) in _ENGINE_PARAM_FIELD_LIMITS.items():
        value = getattr(params, field_name)
        assert (
            lower <= value <= upper
        ), f"{field_name} out of bounds for emphasis={strategy}: {value}"


def test_cued_script_xml_roundtrip_and_mutation() -> None:
    chunk_one = CuedChunk(
        idx=1,
        text="First chunk. Second sentence?",
        speaker="narrator",
        text_type=TextType.NARRATIVE_HISTORICAL,
        rhetoric=RhetoricTag.narrative,
        profile=Profile.Glide,
        pre_pause_ms=120,
        post_pause_ms=240,
        emphasis=[
            EmphasisSpan(
                phrase="First chunk",
                strategy=EmphasisStrategy.DEFINITION,
                pause_before_ms=30,
                pause_after_ms=40,
            )
        ],
    )
    chunk_two = CuedChunk(
        idx=2,
        text="Respond with confidence!",
        speaker="mary-magdalene",
        text_type=TextType.DIALOGUE,
        rhetoric=RhetoricTag.deliberative,
        profile=Profile.Press,
        emphasis=[],
    )
    script = CuedScript(
        text_name="gospel-of-mary",
        speakers=["narrator", "mary-magdalene"],
        chunks=[chunk_one, chunk_two],
    )

    xml_raw = script.to_xml(encoding="unicode", pretty_print=True, skip_empty=True)
    xml_payload = xml_raw.decode("utf-8") if isinstance(xml_raw, bytes) else xml_raw
    assert "<cue-script" in xml_payload
    assert "<chunk idx=\"1\"" in xml_payload

    round_tripped = CuedScript.from_xml(xml_payload)
    assert round_tripped.model_dump() == script.model_dump()

    updated_chunk_one = round_tripped.chunks[0].model_copy(
        update={"text": "Updated first chunk."}
    )
    updated_script = round_tripped.model_copy(
        update={"chunks": [updated_chunk_one, *round_tripped.chunks[1:]]}
    )

    updated_raw = updated_script.to_xml(
        encoding="unicode", pretty_print=True, skip_empty=True
    )
    updated_xml = updated_raw.decode("utf-8") if isinstance(updated_raw, bytes) else updated_raw
    reloaded = CuedScript.from_xml(updated_xml)

    assert reloaded.chunks[0].text == "Updated first chunk."
    assert reloaded.chunks[0].emphasis[0].phrase == "First chunk"
    assert reloaded.chunks[1].speaker == "mary-magdalene"


def test_primer_text_is_available() -> None:
    assert "Cue Performance Primer" in CUE_PRIMER_TEXT
    assert "Machine-Readable Tuning Map" in CUE_PRIMER_TEXT
