import json
from config import DATA_PATH


def load_abstracts(path=DATA_PATH):
    """data.json stores each abstract as [pmid, title, text]; convert to dicts."""
    with open(path, "r") as f:
        data = json.load(f)

    return [
        {"pmid": pmid, "title": title, "text": text}
        for pmid, title, text in data
    ]


if __name__ == "__main__":
    abstracts = load_abstracts()
    print(f"Loaded {len(abstracts)} abstracts")
    first = abstracts[0]
    print(f"First → PMID {first['pmid']}: {first['title'][:70]}...")
