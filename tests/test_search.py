"""Search is tested on a tiny index built in a temp dir — BM25 only, no model download."""

import pytest

from mechanic.data.common import Doc
from mechanic.knowledge.search import SearchIndex, build_sparse, chunk_text

DOCS = [
    Doc(
        id="se-1",
        source="stackexchange",
        url="https://example.test/se1",
        title="High fuel trims at idle only",
        text="Question: High fuel trims at idle only\n\nMy short term fuel trim sits at plus twenty percent "
        "at idle but drops to normal on the highway.\n\nAccepted answer: That pattern points to a vacuum leak: "
        "unmetered air matters most when airflow is small.",
        vehicle_ids=["audi_a4_b8"],
        make="Audi",
        year=2012,
        solved=True,
    ),
    Doc(
        id="se-2",
        source="stackexchange",
        url="https://example.test/se2",
        title="Brake light stays on",
        text="Question: Brake light stays on\n\nThe handbrake indicator stays lit around corners.\n\n"
        "Accepted answer: Check the brake fluid level.",
        vehicle_ids=[],
        make="Chevrolet",
        year=2003,
    ),
    Doc(
        id="cck-1",
        source="carcarekiosk",
        url="https://example.test/cck1",
        title="Coolant Level Check on a 2009 Audi A4",
        text="How to check the coolant level in your Audi A4: find the reservoir, check it cold.",
        vehicle_ids=["audi_a4_b8"],
        make="Audi",
        year=2009,
    ),
    Doc(
        id="cck-2",
        source="carcarekiosk",
        url="https://example.test/cck2",
        title="Coolant Level Check on a 2014 Toyota Corolla",
        text="How to check the coolant level in your Toyota Corolla: find the reservoir, check it cold.",
        vehicle_ids=["toyota_corolla_11"],
        make="Toyota",
        year=2014,
    ),
]


@pytest.fixture(scope="module")
def index(tmp_path_factory):
    index_dir = tmp_path_factory.mktemp("index")
    build_sparse(DOCS, index_dir)
    return SearchIndex(index_dir)


def test_index_without_dense_vectors_still_searches(index):
    assert index.has_dense is False
    hits = index.search("fuel trim high at idle but fine on the highway", limit=2)
    assert hits[0].url == "https://example.test/se1"


def test_source_filter(index):
    hits = index.search("coolant level check", sources=("carcarekiosk",), limit=5)
    assert {h.source for h in hits} == {"carcarekiosk"}


def test_vehicle_boost_prefers_the_drivers_car(index):
    audi = index.search("coolant level check", sources=("carcarekiosk",), vehicle_id="audi_a4_b8", limit=2)
    corolla = index.search("coolant level check", sources=("carcarekiosk",), vehicle_id="toyota_corolla_11", limit=2)
    assert audi[0].make == "Audi"
    assert corolla[0].make == "Toyota"


def test_one_hit_per_document(index):
    hits = index.search("coolant", limit=5)
    assert len({h.url for h in hits}) == len(hits)


def test_query_with_only_stopwords_is_harmless(index):
    assert index.search("the and of", limit=3) == []


def test_chunking_keeps_paragraphs_and_caps_length():
    text = "\n\n".join(["para one", "b" * 3000, "para three"])
    chunks = chunk_text(text, max_chars=1000)
    assert all(len(c) <= 1000 for c in chunks)
    assert chunks[0] == "para one"
    assert "para three" in chunks[-1]
