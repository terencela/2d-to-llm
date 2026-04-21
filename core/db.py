import json
from pathlib import Path

import chromadb


CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "routes"

_client: chromadb.ClientAPI | None = None
_collection: chromadb.Collection | None = None


def get_collection() -> chromadb.Collection:
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def seed_from_json(path: str = "data/seed_routes.json") -> int:
    """Load routes from JSON and upsert into ChromaDB. Returns count of routes seeded."""
    collection = get_collection()
    data = json.loads(Path(path).read_text())

    ids = []
    documents = []
    metadatas = []

    for route in data:
        start = route["start"].strip().lower()
        end = route["end"].strip().lower()
        route_id = f"{start}|{end}"

        ids.append(route_id)
        documents.append(route["route_text"])
        metadatas.append({"start_poi": start, "end_poi": end})

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(ids)


def query_by_pois(start: str, end: str) -> str | None:
    """Exact metadata match on start/end POI pair."""
    collection = get_collection()
    results = collection.get(
        where={
            "$and": [
                {"start_poi": {"$eq": start.strip().lower()}},
                {"end_poi": {"$eq": end.strip().lower()}},
            ]
        },
        include=["documents"],
    )
    if results["documents"]:
        return results["documents"][0]
    return None


def query_by_text(query: str, n_results: int = 1) -> str | None:
    """Semantic search fallback when exact POI match fails."""
    collection = get_collection()
    results = collection.query(query_texts=[query], n_results=n_results)
    if results["documents"] and results["documents"][0]:
        return results["documents"][0][0]
    return None
