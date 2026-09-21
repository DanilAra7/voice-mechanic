class FakeIndex:
    """Stands in for SearchIndex so tests don't need the built index."""

    def __init__(self):
        self.calls = []
        self.hits = None  # None means "one canned hit"; a list is returned as given

    def search(self, query, sources=None, vehicle_id=None, limit=5):
        from mechanic.knowledge.search import SearchHit

        self.calls.append((query, sources, vehicle_id))
        if self.hits is not None:
            return self.hits
        return [
            SearchHit(
                title="Coolant leak at the water pump",
                text="x" * 2000,
                url="https://example.test/1",
                doc_id="se-1",
                source=sources[0] if sources else "stackexchange",
                make="Audi",
                year=2012,
                solved=True,
                relevance=0.9,
            )
        ]
