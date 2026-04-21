"""Airport POI graph with BFS pathfinding across floors."""

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class POI:
    id: str
    name: str
    floor: int
    poi_type: str
    x: float = 0.0
    y: float = 0.0


@dataclass
class Edge:
    from_id: str
    to_id: str
    edge_type: str
    distance_m: int = 0
    notes: str = ""


@dataclass
class AirportGraph:
    pois: dict[str, POI] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)
    adjacency: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    edge_lookup: dict[str, Edge] = field(default_factory=dict)

    def add_poi(self, poi: POI) -> None:
        self.pois[poi.id] = poi

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)
        self.adjacency[edge.from_id].append(edge.to_id)
        self.edge_lookup[f"{edge.from_id}|{edge.to_id}"] = edge

    def get_edge(self, from_id: str, to_id: str) -> Edge | None:
        return self.edge_lookup.get(f"{from_id}|{to_id}")

    def find_path(self, start_id: str, end_id: str) -> list[str] | None:
        """BFS shortest path. Returns list of POI IDs or None if unreachable."""
        if start_id not in self.pois or end_id not in self.pois:
            return None
        if start_id == end_id:
            return [start_id]

        visited = {start_id}
        queue: deque[list[str]] = deque([[start_id]])

        while queue:
            path = queue.popleft()
            current = path[-1]

            for neighbor in self.adjacency[current]:
                if neighbor == end_id:
                    return path + [neighbor]
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(path + [neighbor])

        return None

    def get_all_poi_names(self) -> list[str]:
        return [poi.name for poi in self.pois.values()]

    def find_poi_by_name(self, name: str) -> POI | None:
        """Fuzzy match POI by name (case-insensitive, partial match, stripped punctuation)."""
        name_lower = name.strip().lower()
        name_stripped = name_lower.replace("&", "").replace("-", "").replace(" ", "")

        for poi in self.pois.values():
            if poi.name.lower() == name_lower:
                return poi

        for poi in self.pois.values():
            poi_stripped = poi.name.lower().replace("&", "").replace("-", "").replace(" ", "")
            if name_stripped == poi_stripped:
                return poi

        for poi in self.pois.values():
            if name_lower in poi.name.lower() or poi.name.lower() in name_lower:
                return poi

        return None

    def get_adjacent_pairs(self) -> list[tuple[str, str]]:
        """All directed adjacent pairs (edges in the graph)."""
        return [(e.from_id, e.to_id) for e in self.edges]

    def get_all_reachable_pairs(self) -> list[tuple[str, str, list[str]]]:
        """All reachable A-to-B pairs with their BFS paths."""
        pairs = []
        poi_ids = list(self.pois.keys())
        for start in poi_ids:
            for end in poi_ids:
                if start == end:
                    continue
                path = self.find_path(start, end)
                if path:
                    pairs.append((start, end, path))
        return pairs


def load_graph(config_path: str = "data/airport_config.json") -> AirportGraph:
    """Load airport graph from config JSON."""
    data = json.loads(Path(config_path).read_text())
    graph = AirportGraph()

    for p in data["pois"]:
        graph.add_poi(POI(
            id=p["id"],
            name=p["name"],
            floor=p["floor"],
            poi_type=p["type"],
            x=p.get("x", 0),
            y=p.get("y", 0),
        ))

    for a in data["adjacencies"]:
        graph.add_edge(Edge(
            from_id=a["from"],
            to_id=a["to"],
            edge_type=a["type"],
            distance_m=a.get("distance_m", 0),
            notes=a.get("notes", ""),
        ))
        if a.get("bidirectional", False):
            graph.add_edge(Edge(
                from_id=a["to"],
                to_id=a["from"],
                edge_type=a["type"],
                distance_m=a.get("distance_m", 0),
                notes=a.get("notes", ""),
            ))

    return graph
