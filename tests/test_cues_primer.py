import pytest

from audiobook_toolchain.cues import (
    CUE_PRIMER_TEXT,
    CuedChunk,
    EmphasisSpan,
    EmphasisStrategy,
    Profile,
    RhetoricTag,
    TextType,
    _ENGINE_PARAM_FIELD_LIMITS,
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


def test_primer_text_is_available() -> None:
    assert "Cue Performance Primer" in CUE_PRIMER_TEXT
    assert "Machine-Readable Tuning Map" in CUE_PRIMER_TEXT
