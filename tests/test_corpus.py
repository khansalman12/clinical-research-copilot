from corpus import load_abstracts


def test_loads_all_abstracts():
    abstracts = load_abstracts()
    assert len(abstracts) == 100


def test_abstract_shape():
    abstracts = load_abstracts()
    first = abstracts[0]
    assert set(first.keys()) == {"pmid", "title", "text"}
    assert isinstance(first["pmid"], str)
    assert len(first["title"]) > 0
    assert len(first["text"]) > 0


def test_pmids_are_unique():
    abstracts = load_abstracts()
    pmids = [a["pmid"] for a in abstracts]
    assert len(pmids) == len(set(pmids))
