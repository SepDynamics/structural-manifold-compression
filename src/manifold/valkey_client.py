import valkey
import json
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .sidecar import ManifoldIndex


class ValkeyWorkingMemory:
    """
    Live connection to the Valkey (Redis) Working Memory database.
    Replaces the ephemeral IN_MEMORY_DOCS dictionary.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self.r = valkey.Redis(host=host, port=port, db=db, decode_responses=True)
        self.doc_prefix = "manifold:docs:"
        self.index_key = "manifold:active_index"

    def ping(self) -> bool:
        """Check if Valkey is alive."""
        try:
            return self.r.ping()
        except valkey.ConnectionError:
            return False

    def add_document(self, doc_id: str, text: str) -> None:
        """Store a document's raw text in Valkey."""
        self.r.set(f"{self.doc_prefix}{doc_id}", text)
        self.invalidate_index()

    def get_all_documents(self) -> Dict[str, str]:
        """Retrieve all ingested documents from Valkey."""
        docs = {}
        for key in self.r.scan_iter(f"{self.doc_prefix}*"):
            doc_id = key[len(self.doc_prefix) :]
            docs[doc_id] = self.r.get(key)
        return docs

    def store_cached_index(self, index: "ManifoldIndex") -> None:
        """Cache the computed ManifoldIndex in Valkey for instant retrieval."""
        self.r.set(self.index_key, json.dumps(index.to_dict()))

    def get_cached_index(self) -> Optional["ManifoldIndex"]:
        """Retrieve the cached ManifoldIndex from Valkey if it exists."""
        data = self.r.get(self.index_key)
        if not data:
            return None

        parsed = json.loads(data)
        meta = parsed.get("meta", {})
        signatures = parsed.get("signatures", {})
        documents = parsed.get("documents", {})

        from .sidecar import ManifoldIndex

        return ManifoldIndex(meta=meta, signatures=signatures, documents=documents)

    def invalidate_index(self) -> None:
        """Clear the active index cache to force a rebuild on next access."""
        self.r.delete(self.index_key)

    def clear_all(self) -> None:
        """Wipe the entire Working Memory (for testing)."""
        for key in self.r.scan_iter("manifold:*"):
            self.r.delete(key)

    def get_or_build_index(self) -> Optional["ManifoldIndex"]:
        """Fetch the cached index from Valkey, or build and cache a new one if missing."""
        if not self.ping():
            return None

        cached = self.get_cached_index()
        if cached is not None:
            return cached

        docs = self.get_all_documents()
        if not docs:
            return None

        from .sidecar import build_index

        new_index = build_index(
            docs,
            window_bytes=512,
            stride_bytes=384,
            precision=3,
            use_native=True,
        )
        self.store_cached_index(new_index)
        return new_index
