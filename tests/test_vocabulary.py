"""The words this agent exists for are the ones the recogniser drops."""

from mechanic.voice.vocabulary import repair


def test_the_two_mishearings_that_cost_real_conversations():
    assert repair("what are my field dreams doing") == "what are my fuel trims doing"
    assert repair("I think the ad app server is broken") == "I think the OBD adapter is broken"


def test_case_and_position_do_not_matter():
    assert repair("Field dreams are high.") == "fuel trims are high."
    assert repair("THE AD APP SERVER") == "THE OBD adapter"  # only the matched phrase is touched


def test_ordinary_speech_is_left_alone():
    """A table like this earns its keep by not firing. Single common words are never rewritten."""
    for sentence in [
        "the server is down",
        "my field is muddy",
        "I had a dream about the car",
        "the check engine light came on",
    ]:
        assert repair(sentence) == sentence
