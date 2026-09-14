"""Player identity resolution.

Names are how humans and LLMs refer to players; GSIS ids are how the data layer
does. This service is the only place that bridges the two, and it treats
ambiguity as a first-class result rather than something to guess past.
"""

from __future__ import annotations

import re
import unicodedata

import polars as pl

from nfl_platform.data.ingestion.loader import NflDataLoader
from nfl_platform.domain import looks_like_gsis_id, normalize_player_id, normalize_team
from nfl_platform.exceptions import AmbiguousPlayerError, PlayerNotFoundError
from nfl_platform.schemas.player import PlayerCandidate, PlayerProfile, PlayerSearchResult
from nfl_platform.services.base import AnalyticsService

# Columns matched against a name query, best-quality first.
_NAME_COLUMNS: tuple[str, ...] = ("display_name", "football_name", "short_name")

# Periods and apostrophes are deleted outright so "A.J." folds to "aj" and
# "Ja'Marr" to "jamarr"; other punctuation becomes a word break.
_ELIDED = re.compile(r"[.'\u2019]")
_PUNCTUATION = re.compile(r"[^a-z0-9 ]+")
_WHITESPACE = re.compile(r"\s+")
_SUFFIXES = frozenset({"jr", "sr", "ii", "iii", "iv", "v"})


def normalize_name(name: str) -> str:
    """Fold a name to a comparable form.

    Strips accents, punctuation, and generational suffixes so that
    "A.J. Brown", "AJ Brown", and "aj brown" all match, and so that
    "Odell Beckham Jr." matches "Odell Beckham".
    """
    decomposed = unicodedata.normalize("NFKD", name)
    ascii_only = "".join(char for char in decomposed if not unicodedata.combining(char))
    lowered = _PUNCTUATION.sub(" ", _ELIDED.sub("", ascii_only.lower()))
    tokens = [token for token in _WHITESPACE.split(lowered) if token and token not in _SUFFIXES]
    return " ".join(tokens)


class PlayerDirectoryService(AnalyticsService):
    """Resolves player names and ids against the nflverse players table."""

    def __init__(self, loader: NflDataLoader) -> None:
        super().__init__(loader)
        self._normalized_cache: pl.DataFrame | None = None

    def _players_with_normalized_names(self) -> pl.DataFrame:
        """Players table with a ``normalized_name`` column, computed once per instance."""
        if self._normalized_cache is None:
            players = self._loader.load_players()
            available = [column for column in _NAME_COLUMNS if column in players.columns]
            self._normalized_cache = players.with_columns(
                pl.col(available[0])
                .map_elements(
                    lambda value: normalize_name(value) if value is not None else "",
                    return_dtype=pl.String,
                )
                .alias("normalized_name")
            )
        return self._normalized_cache

    # --- search ---------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        position: str | None = None,
        team: str | None = None,
        season: int | None = None,
        limit: int = 10,
    ) -> PlayerSearchResult:
        """Find players whose name matches ``query``.

        Exact normalised matches are preferred; a substring match is used only
        when nothing matches exactly, so "Allen" does not drown out "Josh Allen".
        """
        if not query or not query.strip():
            raise ValueError("Search query must not be empty.")
        if limit < 1:
            raise ValueError("Search limit must be a positive integer.")

        normalized_query = normalize_name(query)
        players = self._players_with_normalized_names()

        matches = players.filter(pl.col("normalized_name") == normalized_query)
        if matches.is_empty():
            matches = players.filter(
                pl.col("normalized_name").str.contains(normalized_query, literal=True)
            )

        matches = self._apply_filters(matches, position=position, team=team, season=season)
        candidates = self._to_candidates(matches, limit=limit)

        return PlayerSearchResult(
            query=query,
            match_count=len(candidates),
            is_ambiguous=len(candidates) > 1,
            candidates=candidates,
        )

    def _apply_filters(
        self,
        frame: pl.DataFrame,
        *,
        position: str | None,
        team: str | None,
        season: int | None,
    ) -> pl.DataFrame:
        """Narrow candidates. These filters are what make most ambiguity resolvable."""
        if position and "position" in frame.columns:
            frame = frame.filter(pl.col("position") == position.strip().upper())
        if team and "latest_team" in frame.columns:
            frame = frame.filter(pl.col("latest_team") == normalize_team(team))
        if season is not None:
            bounds = []
            if "rookie_season" in frame.columns:
                bounds.append(
                    pl.col("rookie_season").is_null() | (pl.col("rookie_season") <= season)
                )
            if "last_season" in frame.columns:
                bounds.append(pl.col("last_season").is_null() | (pl.col("last_season") >= season))
            for predicate in bounds:
                frame = frame.filter(predicate)
        return frame

    def _to_candidates(self, frame: pl.DataFrame, *, limit: int) -> list[PlayerCandidate]:
        """Convert matched rows into candidates, most recently active first."""
        if frame.is_empty():
            return []

        if "last_season" in frame.columns:
            frame = frame.sort("last_season", descending=True, nulls_last=True)

        candidates: list[PlayerCandidate] = []
        for row in frame.head(limit).iter_rows(named=True):
            player_id = self.row_str(row, "gsis_id")
            if player_id is None:
                continue
            candidates.append(
                PlayerCandidate(
                    player_id=player_id,
                    name=self.row_str(row, "display_name") or player_id,
                    position=self.row_str(row, "position"),
                    latest_team=self.row_str(row, "latest_team"),
                    rookie_season=self.row_int(row, "rookie_season"),
                    last_season=self.row_int(row, "last_season"),
                    disambiguator=self._describe(row),
                )
            )
        return candidates

    def _describe(self, row: dict[str, object]) -> str:
        """Build a short human-readable disambiguator such as 'QB, BUF, 2018-2026'."""
        parts: list[str] = []
        position = self.row_str(row, "position")
        if position:
            parts.append(position)
        team = self.row_str(row, "latest_team")
        if team:
            parts.append(team)
        rookie = self.row_int(row, "rookie_season")
        last = self.row_int(row, "last_season")
        if rookie and last:
            parts.append(f"{rookie}-{last}")
        elif rookie:
            parts.append(f"from {rookie}")
        return ", ".join(parts) if parts else "no additional detail"

    # --- resolution -----------------------------------------------------------

    def resolve(
        self,
        identifier: str,
        *,
        position: str | None = None,
        team: str | None = None,
        season: int | None = None,
    ) -> str:
        """Resolve a name or id to exactly one GSIS id.

        An exact id match wins before any name matching, which also covers
        pre-2000 players whose ids predate the ``00-0012345`` format
        (for example ``WIL439416``).

        Raises:
            PlayerNotFoundError: Nothing matched.
            AmbiguousPlayerError: Several players matched; the error carries them.
        """
        normalized = normalize_player_id(identifier)

        if self.exists(normalized):
            return normalized
        if looks_like_gsis_id(normalized):
            raise PlayerNotFoundError(normalized, "the players directory")

        result = self.search(normalized, position=position, team=team, season=season, limit=25)
        if result.match_count == 0:
            raise PlayerNotFoundError(identifier, "the players directory")
        if result.match_count > 1:
            raise AmbiguousPlayerError(
                identifier,
                tuple(
                    (candidate.player_id, candidate.disambiguator)
                    for candidate in result.candidates
                ),
            )
        return result.candidates[0].player_id

    def exists(self, player_id: str) -> bool:
        """Report whether a GSIS id appears in the players table."""
        players = self._loader.load_players()
        return not players.filter(pl.col("gsis_id") == normalize_player_id(player_id)).is_empty()

    # --- profile --------------------------------------------------------------

    def profile(self, player_id: str) -> PlayerProfile:
        """Return identity and career detail for one player."""
        normalized = normalize_player_id(player_id)
        players = self._loader.load_players()
        match = players.filter(pl.col("gsis_id") == normalized)
        if match.is_empty():
            raise PlayerNotFoundError(normalized, "the players directory")

        row = match.head(1).to_dicts()[0]
        return PlayerProfile(
            player_id=normalized,
            name=self.row_str(row, "display_name") or normalized,
            position=self.row_str(row, "position"),
            position_group=self.row_str(row, "position_group"),
            latest_team=self.row_str(row, "latest_team"),
            status=self.row_str(row, "status"),
            jersey_number=self.row_int(row, "jersey_number"),
            height_inches=self.row_int(row, "height"),
            weight_pounds=self.row_int(row, "weight"),
            college=self.row_str(row, "college_name"),
            rookie_season=self.row_int(row, "rookie_season"),
            last_season=self.row_int(row, "last_season"),
            years_experience=self.row_int(row, "years_of_experience"),
            draft_year=self.row_int(row, "draft_year"),
            draft_round=self.row_int(row, "draft_round"),
            draft_pick=self.row_int(row, "draft_pick"),
            pfr_id=self.row_str(row, "pfr_id"),
        )

    def reference(self, player_id: str) -> tuple[str, str | None, str | None]:
        """Return ``(name, position, team)`` for embedding in a DTO.

        Falls back to the id when the player is absent from the directory, so a
        statistics response is never blocked by a missing directory row.
        """
        players = self._loader.load_players()
        match = players.filter(pl.col("gsis_id") == normalize_player_id(player_id))
        if match.is_empty():
            return player_id, None, None
        row = match.head(1).to_dicts()[0]
        return (
            self.row_str(row, "display_name") or player_id,
            self.row_str(row, "position"),
            self.row_str(row, "latest_team"),
        )
