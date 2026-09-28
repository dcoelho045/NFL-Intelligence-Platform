"""Web retrieval: the primary source for current NFL information.

The provider is a Protocol with several implementations. Policy -- caching,
truncation, sanitisation, domain filtering, budgets -- lives in
:mod:`~nfl_platform.retrieval.web.search` above the adapters, so changing vendor
cannot change behaviour that the tests and the benchmark assert.
"""

from __future__ import annotations

from nfl_platform.retrieval.web.provider import (
    NullSearchProvider,
    WebSearchProvider,
    build_search_provider,
)
from nfl_platform.retrieval.web.search import WebSearchClient

__all__ = [
    "NullSearchProvider",
    "WebSearchClient",
    "WebSearchProvider",
    "build_search_provider",
]
