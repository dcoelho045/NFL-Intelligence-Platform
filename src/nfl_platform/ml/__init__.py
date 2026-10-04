"""Machine-learning prediction layer.

A peer of ``analytics`` and ``retrieval``: it reads the data layer, and the
service layer composes it. It must not import ``services``, ``api``, ``llm`` or
``mcp_server``, and it contains no language model anywhere -- the numbers here
come from fitted estimators or they do not exist.

The layer splits along one line that matters more than the package boundaries:

*Offline* -- :mod:`nfl_platform.ml.datasets`, :mod:`nfl_platform.ml.training`,
:mod:`nfl_platform.ml.evaluation` and :mod:`nfl_platform.ml.models`. These fit
things. Nothing in the serving path may import them, which an architecture test
enforces.

*Online* -- :mod:`nfl_platform.ml.features`, :mod:`nfl_platform.ml.registry` and
:mod:`nfl_platform.ml.predict`. These build a feature row and score a frozen
artifact.

``features`` is deliberately on both sides. Sharing one feature builder between
training and inference is the only reliable defence against train/serve skew;
two implementations that are meant to agree eventually will not.
"""

from __future__ import annotations
