# exa_home — a low-latency neural search stack (learning build).
#
# The public surface is explicit: deep imports (`from exa_home.orch import
# ...`) are an implementation detail, not the API.
#
# NOTE: `mcp_server` is deliberately NOT imported here — it needs fastmcp
# (optional) and is imported lazily by the entry point.
from .config import OPERATING_POINT, operating_point
from .embed import Embedder, HashEmbedder
from .orch import DAG, Node, build_search_dag, run_search
from .rerank import MockReranker, Reranker
from .store import ContentStore

__all__ = [
    "DAG",
    "OPERATING_POINT",
    "ContentStore",
    "Embedder",
    "HashEmbedder",
    "MockReranker",
    "Node",
    "Reranker",
    "build_search_dag",
    "operating_point",
    "run_search",
]
