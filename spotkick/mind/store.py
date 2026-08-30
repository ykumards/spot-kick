"""The only module that touches the database.

One SQLite file holds everything Spot Kick knows about one listener: tracks, their audio embeddings,
every signal (plays, skips, loves, kicks, picks) with the recommender's own view at that moment, every
kick with its measured distance and verdict, and every candidate the Brain ever proposed. The Brain never
reads this file directly; it gets the *context queries* at the bottom, each capped so the prompt stays
~20 lines no matter how long the history is. Dedup is done here, not by asking the model nicely.

One connection, many threads: every statement runs under one lock.
"""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = (
    """
CREATE TABLE IF NOT EXISTS tracks (
  id INTEGER PRIMARY KEY,
  artist TEXT NOT NULL, title TEXT NOT NULL, album TEXT,
  key TEXT NOT NULL UNIQUE,                 -- normalized artist|title
  spotify_uri TEXT UNIQUE, itunes_id INTEGER, preview_url TEXT, duration_s REAL,
  resolved_how TEXT,                        -- memory | search | played | api
  created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS embeddings (
  track_id INTEGER PRIMARY KEY REFERENCES tracks(id),
  model TEXT NOT NULL, dim INTEGER NOT NULL, vec BLOB NOT NULL, created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  kind TEXT NOT NULL,                       -- play | partial | skip | love | unlove | hate | kick | pick
  track_id INTEGER REFERENCES tracks(id),
  source TEXT NOT NULL,                     -- spotify | kick | minime | user
  completion REAL, skip_at_s REAL,
  hour INTEGER, weekday INTEGER, session_id INTEGER, position_in_session INTEGER, prev_track_id INTEGER,
  kick_id INTEGER, pick_p REAL, pick_score REAL, popularity INTEGER,
  ctx TEXT                                  -- JSON: the recommender's view at this moment
);
CREATE INDEX IF NOT EXISTS events_t ON events(t);
CREATE INDEX IF NOT EXISTS events_track ON events(track_id, kind);
CREATE TABLE IF NOT EXISTS kicks (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  strength TEXT NOT NULL, magnitude REAL NOT NULL, target_rel REAL,
  direction TEXT, why TEXT, track_id INTEGER REFERENCES tracks(id),
  distance REAL, rel REAL, band TEXT, dose INTEGER NOT NULL DEFAULT 1, popularity INTEGER,
  pre_state BLOB, kick_vec BLOB,
  followed REAL, verdict TEXT, verdict_at REAL, n_since INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS candidates (
  id INTEGER PRIMARY KEY,
  t REAL NOT NULL,
  set_id TEXT NOT NULL,                     -- one prefetch = one set
  for_track_id INTEGER,                     -- what was playing when the set was requested
  purpose TEXT NOT NULL DEFAULT 'pool',     -- 'pool' (graded, for the next kick)"""
    " or 'follow' (one direction, forced after a kick)\n"
    """  kick_id INTEGER,                          -- filled when a set member is kicked
  track_id INTEGER REFERENCES tracks(id),
  reach TEXT, direction TEXT, artist TEXT NOT NULL, title TEXT NOT NULL, why TEXT, proposed_uri TEXT,
  distance REAL, rel REAL, band TEXT,       -- measured at selection time
  home REAL, state REAL, affinity REAL, total REAL, p REAL,
  chosen INTEGER NOT NULL DEFAULT 0, rejected_reason TEXT,
  lean TEXT NOT NULL DEFAULT ''             -- the lean the set was asked under; the library only reuses a match
);
CREATE INDEX IF NOT EXISTS candidates_set ON candidates(set_id);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value BLOB, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""
)

SCHEMA_VERSION = 1
SESSION_GAP_S = 30 * 60
SECONDS_PER_DAY = 86400
PLAYS = ("play", "partial", "skip")
PLAY_KINDS_SQL = "('play','partial','skip')"
SEEN_KINDS_SQL = "('play','partial','skip','kick','pick')"

KICK_UPDATABLE_FIELDS = {"followed", "verdict", "verdict_at", "n_since", "popularity"}
CANDIDATE_UPDATABLE_FIELDS = {
    "track_id", "distance", "rel", "band", "home", "state", "affinity", "total", "p", "chosen", "rejected_reason",
    "kick_id",
}
COUNTED_TABLES = ("tracks", "embeddings", "events", "kicks", "candidates")

INSERT_TRACK_SQL = (
    "INSERT INTO tracks(artist,title,album,key,spotify_uri,itunes_id,preview_url,duration_s,resolved_how,created_at)"
    " VALUES (?,?,?,?,?,?,?,?,?,?)"
)
INSERT_EMBEDDING_SQL = "INSERT OR REPLACE INTO embeddings(track_id, model, dim, vec, created_at) VALUES (?,?,?,?,?)"
INSERT_EVENT_SQL = (
    "INSERT INTO events(t,kind,track_id,source,completion,skip_at_s,hour,weekday,session_id,position_in_session,"
    "prev_track_id,kick_id,pick_p,pick_score,popularity,ctx) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
LAST_PLAY_SQL = (
    f"SELECT t, session_id, position_in_session, track_id FROM events WHERE kind IN {PLAY_KINDS_SQL}"
    " ORDER BY t DESC LIMIT 1"
)
INSERT_KICK_SQL = (
    "INSERT INTO kicks(t,strength,magnitude,target_rel,direction,why,track_id,distance,rel,band,dose,popularity,"
    "pre_state,kick_vec) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
PLAYS_AFTER_SQL = (
    "SELECT e.*, t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
    f" WHERE e.t > ? AND e.kind IN {PLAY_KINDS_SQL} AND e.source='spotify' ORDER BY e.t"
)
INSERT_CANDIDATE_SQL = (
    "INSERT INTO candidates(t,set_id,for_track_id,purpose,track_id,reach,direction,artist,title,why,proposed_uri,"
    "rejected_reason,lean) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"
)
LATEST_SET_ID_SQL = "SELECT set_id FROM candidates WHERE purpose=? ORDER BY t DESC, id DESC LIMIT 1"
RECENT_POOL_ROWS_SQL = "SELECT * FROM candidates WHERE purpose='pool' AND t >= ? ORDER BY t, id"
# The library: every pool candidate ever resolved and measured, whose track has never been played or kicked since,
# newest first. Their embeddings are in the store, so a pool can be refilled from here without a brain call.
LIBRARY_ROWS_SQL = (
    "SELECT * FROM candidates WHERE purpose='pool' AND lean=? AND track_id IS NOT NULL AND rejected_reason IS NULL"
    " AND chosen=0"
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
INSERT_SCHEMA_VERSION_SQL = "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)"

# Migration: candidate sets written before the `purpose` column existed. A follow-through set was recognisable
# as one with no originating track and every member at reach 'adjacent'.
ADD_PURPOSE_COLUMN_SQL = "ALTER TABLE candidates ADD COLUMN purpose TEXT NOT NULL DEFAULT 'pool'"
ADD_LEAN_COLUMN_SQL = "ALTER TABLE candidates ADD COLUMN lean TEXT NOT NULL DEFAULT ''"
TAG_OLD_FOLLOW_SETS_SQL = (
    "UPDATE candidates SET purpose='follow' WHERE for_track_id IS NULL AND set_id IN "
    "(SELECT set_id FROM candidates GROUP BY set_id HAVING min(reach)='adjacent' AND max(reach)='adjacent')"
)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def track_key(artist: str, title: str) -> str:
    return f"{_normalize(artist)}|{_normalize(title)}"


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


def event_from_row(row: sqlite3.Row) -> dict:
    event = dict(row)
    if event["ctx"]:
        event["ctx"] = json.loads(event["ctx"])
    else:
        event["ctx"] = None
    return event


def row_label(row: sqlite3.Row) -> str:
    return f"{row['artist']} — {row['title']}"


def is_usable(candidate: dict) -> bool:
    """Resolved to a track, not rejected, not already kicked."""
    if candidate["track_id"] is None:
        return False
    if candidate["rejected_reason"]:
        return False
    return not candidate["chosen"]


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
            self.db.execute(INSERT_SCHEMA_VERSION_SQL, (str(SCHEMA_VERSION),))

    def migrate(self) -> None:
        """Additive changes for databases created by earlier versions."""
        columns = {column["name"] for column in self.db.execute("PRAGMA table_info(candidates)")}
        if "purpose" not in columns:
            self.db.execute(ADD_PURPOSE_COLUMN_SQL)
            self.db.execute(TAG_OLD_FOLLOW_SETS_SQL)
        if "lean" not in columns:
            self.db.execute(ADD_LEAN_COLUMN_SQL)

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # ---- every statement goes through one of these
    def _run(self, sql: str, args: Sequence[object] = ()) -> None:
        with self._lock:
            self.db.execute(sql, args)

    def _insert(self, sql: str, args: Sequence[object] = ()) -> int:
        """Run an INSERT and return the id of the new row."""
        with self._lock:
            row_id = self.db.execute(sql, args).lastrowid
        if row_id is None:
            raise RuntimeError(f"insert produced no row id: {sql}")
        return row_id

    def _one(self, sql: str, args: Sequence[object] = ()) -> sqlite3.Row | None:
        with self._lock:
            return self.db.execute(sql, args).fetchone()

    def _row(self, sql: str, args: Sequence[object] = ()) -> sqlite3.Row:
        """Like `_one`, for a row that must exist (one we just wrote, or an aggregate)."""
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
        itunes_id: int | None = None,
        preview_url: str | None = None,
        duration_s: float | None = None,
        resolved_how: str | None = None,
    ) -> Track:
        """Insert or enrich. Identity is the normalized artist|title; a URI is attached when we learn it."""
        key = track_key(artist, title)
        with self._lock:
            row = self._one("SELECT * FROM tracks WHERE key=?", (key,))
            if row is None and spotify_uri:
                row = self._one("SELECT * FROM tracks WHERE spotify_uri=?", (spotify_uri,))
            if row is None:
                values = (
                    artist, title, album, key, spotify_uri, itunes_id, preview_url, duration_s, resolved_how,
                    time.time(),
                )
                track_id = self._insert(INSERT_TRACK_SQL, values)
                return track_from_row(self._row("SELECT * FROM tracks WHERE id=?", (track_id,)))
            fresh = {
                "album": album,
                "spotify_uri": spotify_uri,
                "itunes_id": itunes_id,
                "preview_url": preview_url,
                "duration_s": duration_s,
                "resolved_how": resolved_how,
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
        pick_p: float | None = None,
        pick_score: float | None = None,
        popularity: int | None = None,
        ctx: dict | None = None,
    ) -> int:
        event_time = t or time.time()
        local_time = time.localtime(event_time)
        if ctx is None:
            ctx_json = None
        else:
            ctx_json = json.dumps(ctx)
        with self._lock:
            last_play = self._one(LAST_PLAY_SQL)
            session_id, position, previous_track_id = self.place_in_session(last_play, event_time)
            if kind not in PLAYS:
                # Only plays advance the session; loves, kicks and picks sit outside it.
                session_id = None
                position = None
            values = (
                event_time, kind, track_id, source, completion, skip_at_s, local_time.tm_hour, local_time.tm_wday,
                session_id, position, previous_track_id, kick_id, pick_p, pick_score, popularity, ctx_json,
            )
            return self._insert(INSERT_EVENT_SQL, values)

    def place_in_session(self, last_play: sqlite3.Row | None, event_time: float) -> tuple[int, int, int | None]:
        """Session id, position within it and the previous track for an event at `event_time`.

        A session is a run of plays with no gap longer than SESSION_GAP_S; its id is the timestamp of its first play.
        """
        if last_play is None or event_time - last_play["t"] > SESSION_GAP_S:
            session_id = int(event_time)
            position = 0
        else:
            session_id = last_play["session_id"]
            position = (last_play["position_in_session"] or 0) + 1
        if last_play is None:
            previous_track_id = None
        else:
            previous_track_id = last_play["track_id"]
        return session_id, position, previous_track_id

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
        return [event_from_row(row) for row in self._all(sql, args)]

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
        dose: int,
        pre_state: np.ndarray | None,
        kick_vec: np.ndarray | None,
        popularity: int | None = None,
        t: float | None = None,
    ) -> int:
        values = (
            t or time.time(), strength, magnitude, target_rel, direction, why, track_id, distance, rel, band, dose,
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
        """Spotify's own plays after the kick (forced follow-through excluded)."""
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
        purpose: str = "pool",
        t: float | None = None,
        lean: str = "",
    ) -> list[int]:
        proposed_at = t or time.time()
        candidate_ids = []
        with self._lock:
            for candidate in rows:
                values = (
                    proposed_at, set_id, for_track_id, purpose, candidate.get("track_id"), candidate.get("reach"),
                    candidate.get("direction"), candidate["artist"], candidate["title"], candidate.get("why"),
                    candidate.get("spotify_uri"), candidate.get("rejected_reason"), lean,
                )
                candidate_ids.append(self._insert(INSERT_CANDIDATE_SQL, values))
        return candidate_ids

    def update_candidate(self, cand_id: int, **fields) -> None:
        self.update_row("candidates", cand_id, fields, CANDIDATE_UPDATABLE_FIELDS)

    def update_row(self, table: str, row_id: int, fields: dict, allowed: set[str]) -> None:
        """Set the given columns on one row, refusing any column outside `allowed`: measurements are written once."""
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"not updatable: {unknown}")
        self._run(f"UPDATE {table} SET {assignment_list(fields)} WHERE id=?", (*fields.values(), row_id))

    def candidate_set(self, set_id: str) -> list[dict]:
        rows = self._all("SELECT * FROM candidates WHERE set_id=? ORDER BY id", (set_id,))
        return [dict(row) for row in rows]

    def latest_candidate_set(self, *, usable_only: bool = True, purpose: str = "pool") -> list[dict]:
        """The newest set of the given purpose. A follow-through set must never be restored as the kick pool: it is one
        direction, not a graded spread, and restoring one kept every kick inside it (the 'Brazilian pool' bug)."""
        newest = self._one(LATEST_SET_ID_SQL, (purpose,))
        if not newest:
            return []
        candidates = self.candidate_set(newest["set_id"])
        if not usable_only:
            return candidates
        return [candidate for candidate in candidates if is_usable(candidate)]

    def usable_pool_candidates(self, since: float) -> list[dict]:
        """Every usable pool candidate proposed since `since`, oldest first. A pool is topped up band by band, so
        the picks that survive a restart span several sets."""
        rows = self._all(RECENT_POOL_ROWS_SQL, (since,))
        return [candidate for candidate in map(dict, rows) if is_usable(candidate)]

    def library_candidates(self, lean: str = "", limit: int = 200) -> list[dict]:
        """Pool candidates from any time that are still playable: resolved, never played or kicked, proposed under
        this lean. One row per track (the newest), newest first."""
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

    # ---------------------------------------------------------------- kv stores
    def get_profile(self, key: str) -> bytes | None:
        row = self._one("SELECT value FROM profile WHERE key=?", (key,))
        if row is None:
            return None
        return row["value"]

    def set_profile(self, key: str, value: bytes) -> None:
        self._run("INSERT OR REPLACE INTO profile(key, value, updated_at) VALUES (?,?,?)", (key, value, time.time()))

    def get_config(self, key: str, default: str | None = None) -> str | None:
        row = self._one("SELECT value FROM config WHERE key=?", (key,))
        if row is None:
            return default
        return row["value"]

    def set_config(self, key: str, value: str) -> None:
        self._run("INSERT OR REPLACE INTO config(key, value) VALUES (?,?)", (key, value))

    # ------------------------------------------------------ dedup, done here
    def seen(self, artist: str, title: str) -> bool:
        """Ever played, kicked, or picked. The Brain is asked not to repeat; the store enforces it."""
        track = self.find_track(artist, title)
        if track is None:
            return False
        seen_event_sql = f"SELECT 1 FROM events WHERE track_id=? AND kind IN {SEEN_KINDS_SQL} LIMIT 1"
        if self._one(seen_event_sql, (track.id,)):
            return True
        return self._one("SELECT 1 FROM kicks WHERE track_id=? LIMIT 1", (track.id,)) is not None

    # ---------------------------------------------------- context queries (Brain)
    def recent(self, n: int = 12) -> list[dict]:
        """Last n plays, most recent first, with where they came from."""
        rows = self._all(RECENT_PLAYS_SQL, (n,))
        return [dict(row) for row in rows]

    def play_sequence(self, n: int = 40) -> list[dict]:
        """Last n plays in the order they happened (oldest first), with track ids."""
        rows = self._all(PLAY_SEQUENCE_SQL, (n,))
        return [dict(row) for row in rows]

    def top_artists(self, *, days: int | None = None, n: int = 10) -> list[tuple[str, int]]:
        """Most-played artists; completed plays count 1, partials 0.5, skips 0. Optionally within the last `days`."""
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
        """Skipped or hated recently: 'not this vein'."""
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
        """Plays Spotify chose (kicked tracks excluded): the denominator of the experiment."""
        return self._row(PLAY_COUNT_SQL)["n"]

    def recent_kicks(self, n: int = 30) -> list[dict]:
        """Newest first: what was asked (strength, target rel), what was played, where it measured, the verdict."""
        return [dict(row) for row in self._all(RECENT_KICKS_SQL, (n,))]

    def count_rows(self, table: str) -> int:
        return self._row(f"SELECT COUNT(*) AS n FROM {table}")["n"]
