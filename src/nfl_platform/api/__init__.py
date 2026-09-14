"""HTTP adapter over the analytics services.

This package is one of several possible adapters. It contains no business logic
and no LLM functionality; it maps HTTP requests onto ``nfl_platform.services``
and serialises the typed results.
"""

from nfl_platform.api.app import API_PREFIX, create_app

__all__ = ["API_PREFIX", "create_app"]
