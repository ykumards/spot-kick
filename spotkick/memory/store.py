"""The SQLite store; the only module that touches the database.

One file holds everything known about one listener: tracks, embeddings, events (plays, skips, loves, kicks),
kicks with their measurements and verdicts, and every candidate the brain proposed. The brain never reads the
file; it receives the capped context queries at the end of this module.

One connection is shared across threads; every statement runs under one lock.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..names import normalize_name

SCHEMA = """
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY,
  artist TEXT NOT NULL, title TEXT NOT NULL, album TEXT,
  key TEXT NOT NULL UNIQUE,                 -- normalized artist|title
  spotify_uri TEXT UNIQUE, preview_url TEXT, duration_s REAL,
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
  model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  kind TEXT NOT NULL,                       -- play | partial | skip | love | unlove | kick
  track_id INTEGER REFERENCES tracks(id),
  source TEXT NOT NULL,                     -- spotify | kick | user
  completion REAL, skip_at_s REAL,          -- for a skip: how far in, and how far through, the listener left
  kick_id INTEGER, popularity INTEGER
);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
CREATE INDEX IF NOT EXISTS events_track ON events(track_id, kind);
CREATE TABLE IF NOT EXISTS kicks (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  strength TEXT NOT NULL, magnitude REAL NOT NULL, target_rel REAL,
  direction TEXT, why TEXT, track_id INTEGER REFERENCES tracks(id),
  distance REAL, rel REAL, band TEXT, popularity INTEGER,
  pre_state BLOB, kick_vec BLOB,
  followed REAL, verdict TEXT, verdict_at REAL, n_since INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  set_id TEXT NOT NULL,                     -- one brain call = one set
  for_track_id INTEGER,                     -- what was playing when the set was requested
  kick_id INTEGER,                          -- filled when a set member is kicked
  track_id INTEGER REFERENCES tracks(id),
  reach TEXT, direction TEXT, artist TEXT NOT NULL, title TEXT NOT NULL, why TEXT,
  distance REAL, rel REAL, band TEXT,       -- measured at selection time
  chosen INTEGER NOT NULL DEFAULT 0, rejected_reason TEXT,
  lean TEXT NOT NULL DEFAULT ''             -- the lean the set was asked under; the library only reuses a match
);
CREATE INDEX IF NOT EXISTS candidates_set ON candidates(set_id);
"""

# Columns and tables earlier versions wrote that nothing reads any more: dropped on open, so an old database
# ends up with the schema above. (Track ids, keys and every measurement survive.)
DROPPED_COLUMNS = {
    "tracks": ("itunes_id", "resolved_how"),
    "events": ("hour", "weekday", "session_id", "position_in_session", "prev_track_id", "pick_p", "pick_score",
               "ctx"),
    "kicks": ("dose",),
    "candidates": ("purpose", "proposed_uri", "home", "state", "affinity", "total", "p"),
}
DROPPED_TABLES = ("profile", "config", "meta")

SECONDS_PER_DAY = 86400
PLAY_KINDS_SQL = "('play','partial','skip')"
SEEN_KINDS_SQL = "('play','partial','skip','kick','pick')"

KICK_UPDATABLE_FIELDS = {"followed", "verdict", "verdict_at", "n_since", "popularity"}
CANDIDATE_UPDATABLE_FIELDS = {"track_id", "distance", "rel", "band", "chosen", "rejected_reason", "kick_id"}
COUNTED_TABLES = ("tracks", "embeddings", "events", "kicks", "candidates")

INSERT_TRACK_SQL = (
    "INSERT INTO tracks(artist,title,album,key,spotify_uri,preview_url,duration_s,created_at) VALUES (?,?,?,?,?,?,?,?)"
)
INSERT_EMBEDDING_SQL = "INSERT OR REPLACE INTO embeddings(track_id, model, dim, vec, created_at) VALUES (?,?,?,?,?)"
INSERT_EVENT_SQL = (
    "INSERT INTO events(t,kind,track_id,source,completion,skip_at_s,kick_id,popularity) VALUES (?,?,?,?,?,?,?,?)"
)
INSERT_KICK_SQL = (
    "INSERT INTO kicks(t,strength,magnitude,target_rel,direction,why,track_id,distance,rel,band,popularity,"
    "pre_state,kick_vec) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
PLAYS_AFTER_SQL = (
    "SELECT e.*, t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
    f" WHERE e.t > ? AND e.kind IN {PLAY_KINDS_SQL} AND e.source='spotify' ORDER BY e.t"
)
INSERT_CANDIDATE_SQL = (
    "INSERT INTO candidates(t,set_id,for_track_id,track_id,reach,direction,artist,title,why,rejected_reason,lean)"
    " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
)
# The library: every candidate ever resolved and measured, whose track has never been played or kicked since,
# newest first. Their embeddings are in the store, so a pool can be refilled from here without a brain call.
LIBRARY_ROWS_SQL = (
    "SELECT * FROM candidates WHERE lean=? AND track_id IS NOT NULL AND rejected_reason IS NULL AND chosen=0"
    f" AND track_id NOT IN (SELECT track_id FROM events WHERE kind IN {SEEN_KINDS_SQL})"
    " AND track_id NOT IN (SELECT track_id FROM kicks)"
    " ORDER BY t DESC, id DESC"
)
RECENT_PLAYS_SQL = (
    "SELECT e.t, e.kind, e.source, e.completion, t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
    f" WHERE e.kind IN {PLAY_KINDS_SQL} ORDER BY e.t DESC, e.id DESC LIMIT ?"
)
# The play sequence, oldest first, with the track id so each play's embedding can be looked up.
PLAY_SEQUENCE_SQL = (
    "SELECT * FROM (SELECT e.id, e.t, e.kind, e.source, e.track_id, t.artist, t.title FROM events e"
    f" JOIN tracks t ON t.id=e.track_id WHERE e.kind IN {PLAY_KINDS_SQL} ORDER BY e.t DESC, e.id DESC LIMIT ?)"
    " ORDER BY t ASC, id ASC"
)
# Loved now: the track's latest love/unlove event is a love. Append-only, so an unlove is a row, not a deletion.
LOVED_SQL = (
    "SELECT t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
    " WHERE e.kind='love'"
    " AND e.id = (SELECT MAX(id) FROM events WHERE track_id=e.track_id AND kind IN ('love','unlove'))"
    " ORDER BY e.t DESC LIMIT ?"
)
LAST_LOVE_EVENT_SQL = "SELECT kind FROM events WHERE track_id=? AND kind IN ('love','unlove') ORDER BY id DESC LIMIT 1"
REJECTED_SQL = (
    "SELECT t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
    " WHERE e.kind IN ('skip','hate') AND e.t >= ? ORDER BY e.t DESC LIMIT ?"
)
DIRECTIONS_SQL = "SELECT direction FROM kicks WHERE direction IS NOT NULL ORDER BY t DESC LIMIT ?"
KICKED_ARTISTS_SQL = "SELECT DISTINCT t.artist FROM kicks k JOIN tracks t ON t.id=k.track_id ORDER BY k.t DESC LIMIT ?"
RECENT_KICKS_SQL = (
    "SELECT k.id, k.t, k.strength, k.magnitude, k.target_rel, k.distance, k.rel, k.band, k.followed, k.verdict,"
    " k.n_since, k.direction, t.artist, t.title FROM kicks k JOIN tracks t ON t.id=k.track_id"
    " ORDER BY k.t DESC LIMIT ?"
)
PLAY_COUNT_SQL = f"SELECT COUNT(*) AS n FROM events WHERE kind IN {PLAY_KINDS_SQL} AND source='spotify'"
ADD_LEAN_COLUMN_SQL = "ALTER TABLE candidates ADD COLUMN lean TEXT NOT NULL DEFAULT ''"


def track_key(artist: str, title: str) -> str:
    return f"{normalize_name(artist)}|{normalize_name(title)}"


@dataclass(frozen=True)
class Track:
    id: int
    artist: str
    title: str
    album: str | None
    spotify_uri: str | None
    preview_url: str | None
    duration_s: float | None

    @property
    def label(self) -> str:
        return f"{self.artist} — {self.title}"


def track_from_row(row: sqlite3.Row) -> Track:
    return Track(
        row["id"], row["artist"], row["title"], row["album"], row["spotify_uri"], row["preview_url"], row["duration_s"]
    )


def vector_to_blob(vector: np.ndarray | None) -> bytes | None:
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def vector_from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32).copy()


def blob_to_vector(blob: bytes | None) -> np.ndarray | None:
    if blob is None:
        return None
    return vector_from_blob(blob)


def kick_from_row(row: sqlite3.Row) -> dict:
    kick = dict(row)
    kick["pre_state"] = blob_to_vector(kick["pre_state"])
    kick["kick_vec"] = blob_to_vector(kick["kick_vec"])
    return kick


def row_label(row: sqlite3.Row) -> str:
    return f"{row['artist']} — {row['title']}"


def placeholders(count: int) -> str:
    return ",".join("?" * count)


def assignment_list(fields: dict) -> str:
    return ", ".join(f"{column}=?" for column in fields)


def days_ago(days: float) -> float:
    return time.time() - days * SECONDS_PER_DAY


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self.db.executescript(SCHEMA)
            self.migrate()

    def migrate(self) -> None:
        """Bring a database from an earlier version up to SCHEMA: add missing columns, drop unused ones."""
        if "lean" not in self.columns("candidates"):
            self.db.execute(ADD_LEAN_COLUMN_SQL)
        for table, dead_columns in DROPPED_COLUMNS.items():
            present = self.columns(table)
            for column in dead_columns:
                if column in present:
                    self.db.execute(f"ALTER TABLE {table} DROP COLUMN {column}")
        for table in DROPPED_TABLES:
            self.db.execute(f"DROP TABLE IF EXISTS {table}")

    def columns(self, table: str) -> set[str]:
        return {column["name"] for column in self.db.execute(f"PRAGMA table_info({table})")}

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # ---- every statement goes through one of these
    def _run(self, sql: str, args: Sequence[object] = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)

    def _insert(self, sql: str, args: Sequence[object] = ()) -> int:
        """Run an INSERT and return the new row id."""
        with self._lock:
            row_id = self.db.execute(sql, args).lastrowid
        if row_id is None:
            raise RuntimeError(f"insert produced no row id: {sql}")
        return row_id

    def _one(self, sql: str, args: Sequence[object] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute(sql, args).fetchone()

    def _row(self, sql: str, args: Sequence[object] = ()) -> sqlite3.Row:
        """Like ``_one`` for a row that must exist; raises LookupError otherwise."""
        row = self._one(sql, args)
        if row is None:
            raise LookupError(f"no row for: {sql}")
        return row

    def _all(self, sql: str, args: Sequence[object] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ tracks
    def upsert_track(
        self,
        artist: str,
        title: str,
        *,
        album: str | None = None,
        spotify_uri: str | None = None,
        preview_url: str | None = None,
        duration_s: float | None = None,
    ) -> Track:
        """Insert a track or fill in its missing columns. Identity is the normalised artist|title key."""
        key = track_key(artist, title)
        with self._lock:
            row = self._one("SELECT * FROM tracks WHERE key=?", (key,))
            if row is None and spotify_uri:
                row = self._one("SELECT * FROM tracks WHERE spotify_uri=?", (spotify_uri,))
            if row is None:
                values = (artist, title, album, key, spotify_uri, preview_url, duration_s, time.time())
                track_id = self._insert(INSERT_TRACK_SQL, values)
                return track_from_row(self._row("SELECT * FROM tracks WHERE id=?", (track_id,)))
            fresh = {
                "album": album,
                "spotify_uri": spotify_uri,
                "preview_url": preview_url,
                "duration_s": duration_s,
            }
            # Only fill columns that are still empty: the first value we learned wins.
            updates = {column: value for column, value in fresh.items() if value is not None and row[column] is None}
            if updates:
                self._run(f"UPDATE tracks SET {assignment_list(updates)} WHERE id=?", (*updates.values(), row["id"]))
                row = self._row("SELECT * FROM tracks WHERE id=?", (row["id"],))
            return track_from_row(row)

    def track(self, track_id: int) -> Track | None:
        row = self._one("SELECT * FROM tracks WHERE id=?", (track_id,))
        if row is None:
            return None
        return track_from_row(row)

    def find_track(self, artist: str, title: str) -> Track | None:
        row = self._one("SELECT * FROM tracks WHERE key=?", (track_key(artist, title),))
        if row is None:
            return None
        return track_from_row(row)

    def track_by_uri(self, uri: str) -> Track | None:
        row = self._one("SELECT * FROM tracks WHERE spotify_uri=?", (uri,))
        if row is None:
            return None
        return track_from_row(row)

    # -------------------------------------------------------------- embeddings
    def put_embedding(self, track_id: int, vec: np.ndarray, model: str) -> None:
        vector = np.asarray(vec, dtype=np.float32)
        self._run(INSERT_EMBEDDING_SQL, (track_id, model, int(vector.shape[0]), vector.tobytes(), time.time()))

    def embedding(self, track_id: int) -> np.ndarray | None:
        row = self._one("SELECT vec FROM embeddings WHERE track_id=?", (track_id,))
        if row is None:
            return None
        return vector_from_blob(row["vec"])

    def embeddings(self, track_ids: list[int]) -> dict[int, np.ndarray]:
        if not track_ids:
            return {}
        sql = f"SELECT track_id, vec FROM embeddings WHERE track_id IN ({placeholders(len(track_ids))})"
        rows = self._all(sql, list(track_ids))
        return {row["track_id"]: vector_from_blob(row["vec"]) for row in rows}

    # ------------------------------------------------------------------ events
    def add_event(
        self,
        kind: str,
        track_id: int | None,
        source: str,
        *,
        t: float | None = None,
        completion: float | None = None,
        skip_at_s: float | None = None,
        kick_id: int | None = None,
        popularity: int | None = None,
    ) -> int:
        values = (t or time.time(), kind, track_id, source, completion, skip_at_s, kick_id, popularity)
        return self._insert(INSERT_EVENT_SQL, values)

    def events(
        self, *, kinds: tuple[str, ...] | None = None, since: float | None = None, limit: int | None = None
    ) -> list[dict]:
        sql = "SELECT * FROM events"
        args: list = []
        conditions: list[str] = []
        if kinds:
            conditions.append(f"kind IN ({placeholders(len(kinds))})")
            args.extend(kinds)
        if since is not None:
            conditions.append("t >= ?")
            args.append(since)
        if conditions:
            sql += " WHERE " + " AND ".join(conditions)
        sql += " ORDER BY t ASC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        return [dict(row) for row in self._all(sql, args)]

    # ------------------------------------------------------------------- kicks
    def add_kick(
        self,
        *,
        strength: str,
        magnitude: float,
        target_rel: float | None,
        direction: str | None,
        why: str | None,
        track_id: int,
        distance: float | None,
        rel: float | None,
        band: str | None,
        pre_state: np.ndarray | None,
        kick_vec: np.ndarray | None,
        popularity: int | None = None,
        t: float | None = None,
    ) -> int:
        values = (
            t or time.time(), strength, magnitude, target_rel, direction, why, track_id, distance, rel, band,
            popularity, vector_to_blob(pre_state), vector_to_blob(kick_vec),
        )
        return self._insert(INSERT_KICK_SQL, values)

    def kick(self, kick_id: int) -> dict | None:
        row = self._one("SELECT * FROM kicks WHERE id=?", (kick_id,))
        if row is None:
            return None
        return kick_from_row(row)

    def last_kick(self) -> dict | None:
        row = self._one("SELECT * FROM kicks ORDER BY t DESC LIMIT 1")
        if row is None:
            return None
        return kick_from_row(row)

    def update_kick(self, kick_id: int, **fields) -> None:
        self.update_row("kicks", kick_id, fields, KICK_UPDATABLE_FIELDS)

    def plays_since_kick(self, kick_id: int) -> list[dict]:
        """Return the recommender-chosen plays after the kick; the kicked track itself is excluded."""
        kick = self.kick(kick_id)
        if not kick:
            return []
        rows = self._all(PLAYS_AFTER_SQL, (kick["t"],))
        return [dict(row) for row in rows]

    # -------------------------------------------------------------- candidates
    def add_candidates(
        self,
        set_id: str,
        rows: list[dict],
        *,
        for_track_id: int | None = None,
        t: float | None = None,
        lean: str = "",
    ) -> list[int]:
        proposed_at = t or time.time()
        candidate_ids = []
        with self._lock:
            for candidate in rows:
                values = (
                    proposed_at, set_id, for_track_id, candidate.get("track_id"), candidate.get("reach"),
                    candidate.get("direction"), candidate["artist"], candidate["title"], candidate.get("why"),
                    candidate.get("rejected_reason"), lean,
                )
                candidate_ids.append(self._insert(INSERT_CANDIDATE_SQL, values))
        return candidate_ids

    def update_candidate(self, cand_id: int, **fields) -> None:
        self.update_row("candidates", cand_id, fields, CANDIDATE_UPDATABLE_FIELDS)

    def update_row(self, table: str, row_id: int, fields: dict, allowed: set[str]) -> None:
        """Set the given columns on one row. Columns outside ``allowed`` raise ValueError."""
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"not updatable: {unknown}")
        self._run(f"UPDATE {table} SET {assignment_list(fields)} WHERE id=?", (*fields.values(), row_id))

    def candidate_set(self, set_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM candidates WHERE set_id=? ORDER BY id", (set_id,))
        return [dict(row) for row in rows]

    def library_candidates(self, lean: str = "", limit: int = 200) -> list[dict]:
        """Return resolved candidates under this lean whose track was never played or kicked, newest first, one
        row per track.
        """
        seen_tracks: set[int] = set()
        library = []
        for row in map(dict, self._all(LIBRARY_ROWS_SQL, (lean,))):
            if row["track_id"] in seen_tracks:
                continue
            seen_tracks.add(row["track_id"])
            library.append(row)
            if len(library) >= limit:
                break
        return library

    # ------------------------------------------------------ dedup, done here
    def seen(self, artist: str, title: str) -> bool:
        """Return True when the track was ever played or kicked."""
        track = self.find_track(artist, title)
        if track is None:
            return False
        seen_event_sql = f"SELECT 1 FROM events WHERE track_id=? AND kind IN {SEEN_KINDS_SQL} LIMIT 1"
        if self._one(seen_event_sql, (track.id,)):
            return True
        return self._one("SELECT 1 FROM kicks WHERE track_id=? LIMIT 1", (track.id,)) is not None

    # ---------------------------------------------------- context queries (Brain)
    def recent(self, n: int = 12) -> list[dict]:
        """Return the last ``n`` plays, most recent first."""
        rows = self._all(RECENT_PLAYS_SQL, (n,))
        return [dict(row) for row in rows]

    def play_sequence(self, n: int = 40) -> list[dict]:
        """Return the last ``n`` plays, oldest first, with track ids."""
        rows = self._all(PLAY_SEQUENCE_SQL, (n,))
        return [dict(row) for row in rows]

    def top_artists(self, *, days: int | None = None, n: int = 10) -> list[tuple[str, int]]:
        """Return the most-played artists; a play counts 1, a partial 0.5, a skip 0. Optionally within ``days``."""
        args: list = []
        condition = "e.kind IN ('play','partial')"
        if days is not None:
            condition += " AND e.t >= ?"
            args.append(days_ago(days))
        weight_expression = "SUM(CASE e.kind WHEN 'play' THEN 1.0 ELSE 0.5 END)"
        sql = (
            f"SELECT t.artist AS artist, {weight_expression} AS w FROM events e JOIN tracks t ON t.id=e.track_id"
            f" WHERE {condition} GROUP BY t.artist ORDER BY w DESC, MAX(e.t) DESC LIMIT ?"
        )
        rows = self._all(sql, (*args, n))
        return [(row["artist"], round(row["w"])) for row in rows]

    def loved(self, n: int = 8) -> list[str]:
        rows = self._all(LOVED_SQL, (n,))
        return [row_label(row) for row in rows]

    def rejected(self, *, days: int = 14, n: int = 8) -> list[str]:
        """Return tracks skipped within ``days``, most recent first."""
        rows = self._all(REJECTED_SQL, (days_ago(days), n))
        return list(dict.fromkeys(row_label(row) for row in rows))

    def directions(self, n: int = 10) -> list[str]:
        rows = self._all(DIRECTIONS_SQL, (n,))
        return [row["direction"] for row in rows]

    def kicked_artists(self, n: int = 25) -> list[str]:
        rows = self._all(KICKED_ARTISTS_SQL, (n,))
        return [row["artist"] for row in rows]

    def counts(self) -> dict:
        return {table: self.count_rows(table) for table in COUNTED_TABLES}

    def is_loved(self, track_id: int) -> bool:
        latest = self._one(LAST_LOVE_EVENT_SQL, (track_id,))
        return latest is not None and latest["kind"] == "love"

    def spotify_play_count(self) -> int:
        """Return the number of recommender-chosen plays."""
        return self._row(PLAY_COUNT_SQL)["n"]

    def recent_kicks(self, n: int = 30) -> list[dict]:
        """Return the last ``n`` kicks with their track, measurement and verdict, newest first."""
        return [dict(row) for row in self._all(RECENT_KICKS_SQL, (n,))]

    def count_rows(self, table: str) -> int:
        return self._row(f"SELECT COUNT(*) AS n FROM {table}")["n"]
