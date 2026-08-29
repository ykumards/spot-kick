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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = """
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
  kind TEXT NOT NULL,                       -- play | partial | skip | love | hate | kick | pick
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
  purpose TEXT NOT NULL DEFAULT 'pool',     -- 'pool' (graded, for the next kick) or 'follow' (one direction, forced after a kick)
  kick_id INTEGER,                          -- filled when a set member is kicked
  track_id INTEGER REFERENCES tracks(id),
  reach TEXT, direction TEXT, artist TEXT NOT NULL, title TEXT NOT NULL, why TEXT, proposed_uri TEXT,
  distance REAL, rel REAL, band TEXT,       -- measured at selection time
  home REAL, state REAL, affinity REAL, total REAL, p REAL,
  chosen INTEGER NOT NULL DEFAULT 0, rejected_reason TEXT
);
CREATE INDEX IF NOT EXISTS candidates_set ON candidates(set_id);
CREATE TABLE IF NOT EXISTS profile (key TEXT PRIMARY KEY, value BLOB, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS config (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""

SCHEMA_VERSION = 1
SESSION_GAP_S = 30 * 60
PLAYS = ("play", "partial", "skip")


def _norm(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", x.lower()).strip()


def track_key(artist: str, title: str) -> str:
    return f"{_norm(artist)}|{_norm(title)}"


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


def _row_track(r: sqlite3.Row) -> Track:
    return Track(r["id"], r["artist"], r["title"], r["album"], r["spotify_uri"], r["preview_url"], r["duration_s"])


def _blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype=np.float32).tobytes()


def _vec(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype=np.float32).copy()


def _kick_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["pre_state"], d["kick_vec"] = _vec(d["pre_state"]), _vec(d["kick_vec"])
    return d


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
            self._migrate()
            self.db.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))

    def _migrate(self) -> None:
        """Additive changes for databases created by earlier versions."""
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(candidates)")}
        if "purpose" not in cols:
            self.db.execute("ALTER TABLE candidates ADD COLUMN purpose TEXT NOT NULL DEFAULT 'pool'")
            # earlier follow-through sets: one direction, every reach 'adjacent', no track attached
            self.db.execute("UPDATE candidates SET purpose='follow' WHERE for_track_id IS NULL AND set_id IN "
                            "(SELECT set_id FROM candidates GROUP BY set_id HAVING min(reach)='adjacent' AND max(reach)='adjacent')")

    def close(self) -> None:
        with self._lock:
            self.db.close()

    # ---- every statement goes through one of these
    def _run(self, sql: str, args=()) -> int | None:
        with self._lock:
            return self.db.execute(sql, args).lastrowid

    def _one(self, sql: str, args=()):
        with self._lock:
            return self.db.execute(sql, args).fetchone()

    def _all(self, sql: str, args=()):
        with self._lock:
            return self.db.execute(sql, args).fetchall()

    # ------------------------------------------------------------------ tracks
    def upsert_track(self, artist: str, title: str, *, album: str | None = None, spotify_uri: str | None = None,
                     itunes_id: int | None = None, preview_url: str | None = None, duration_s: float | None = None,
                     resolved_how: str | None = None) -> Track:
        """Insert or enrich. Identity is the normalized artist|title; a URI is attached when we learn it."""
        key = track_key(artist, title)
        with self._lock:
            row = self._one("SELECT * FROM tracks WHERE key=?", (key,))
            if row is None and spotify_uri:
                row = self._one("SELECT * FROM tracks WHERE spotify_uri=?", (spotify_uri,))
            if row is None:
                rid = self._run(
                    "INSERT INTO tracks(artist,title,album,key,spotify_uri,itunes_id,preview_url,duration_s,resolved_how,created_at)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (artist, title, album, key, spotify_uri, itunes_id, preview_url, duration_s, resolved_how, time.time()))
                row = self._one("SELECT * FROM tracks WHERE id=?", (rid,))
            else:
                fresh = {"album": album, "spotify_uri": spotify_uri, "itunes_id": itunes_id, "preview_url": preview_url,
                         "duration_s": duration_s, "resolved_how": resolved_how}
                updates = {k: v for k, v in fresh.items() if v is not None and row[k] is None}
                if updates:
                    sets = ", ".join(f"{k}=?" for k in updates)
                    self._run(f"UPDATE tracks SET {sets} WHERE id=?", (*updates.values(), row["id"]))
                    row = self._one("SELECT * FROM tracks WHERE id=?", (row["id"],))
            return _row_track(row)

    def track(self, track_id: int) -> Track | None:
        r = self._one("SELECT * FROM tracks WHERE id=?", (track_id,))
        return _row_track(r) if r else None

    def find_track(self, artist: str, title: str) -> Track | None:
        r = self._one("SELECT * FROM tracks WHERE key=?", (track_key(artist, title),))
        return _row_track(r) if r else None

    def track_by_uri(self, uri: str) -> Track | None:
        r = self._one("SELECT * FROM tracks WHERE spotify_uri=?", (uri,))
        return _row_track(r) if r else None

    # -------------------------------------------------------------- embeddings
    def put_embedding(self, track_id: int, vec: np.ndarray, model: str) -> None:
        v = np.asarray(vec, dtype=np.float32)
        self._run("INSERT OR REPLACE INTO embeddings(track_id, model, dim, vec, created_at) VALUES (?,?,?,?,?)",
                  (track_id, model, int(v.shape[0]), v.tobytes(), time.time()))

    def embedding(self, track_id: int) -> np.ndarray | None:
        r = self._one("SELECT vec FROM embeddings WHERE track_id=?", (track_id,))
        return _vec(r["vec"]) if r else None

    def embeddings(self, track_ids: list[int]) -> dict[int, np.ndarray]:
        if not track_ids:
            return {}
        q = ",".join("?" * len(track_ids))
        rows = self._all(f"SELECT track_id, vec FROM embeddings WHERE track_id IN ({q})", list(track_ids))
        return {r["track_id"]: _vec(r["vec"]) for r in rows}

    # ------------------------------------------------------------------ events
    def add_event(self, kind: str, track_id: int | None, source: str, *, t: float | None = None, completion: float | None = None,
                  skip_at_s: float | None = None, kick_id: int | None = None, pick_p: float | None = None,
                  pick_score: float | None = None, popularity: int | None = None, ctx: dict | None = None) -> int:
        t = t or time.time()
        lt = time.localtime(t)
        with self._lock:
            last = self._one("SELECT t, session_id, position_in_session, track_id FROM events WHERE kind IN ('play','partial','skip')"
                             " ORDER BY t DESC LIMIT 1")
            if last is None or t - last["t"] > SESSION_GAP_S:
                session_id, pos = int(t), 0
            else:
                session_id, pos = last["session_id"], (last["position_in_session"] or 0) + 1
            prev = last["track_id"] if last else None
            is_play = kind in PLAYS
            return self._run(
                "INSERT INTO events(t,kind,track_id,source,completion,skip_at_s,hour,weekday,session_id,position_in_session,prev_track_id,"
                "kick_id,pick_p,pick_score,popularity,ctx) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (t, kind, track_id, source, completion, skip_at_s, lt.tm_hour, lt.tm_wday,
                 session_id if is_play else None, pos if is_play else None, prev,
                 kick_id, pick_p, pick_score, popularity, json.dumps(ctx) if ctx is not None else None))

    def events(self, *, kinds: tuple[str, ...] | None = None, since: float | None = None, limit: int | None = None) -> list[dict]:
        q, args, where = "SELECT * FROM events", [], []
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})"); args += list(kinds)
        if since is not None:
            where.append("t >= ?"); args.append(since)
        if where:
            q += " WHERE " + " AND ".join(where)
        q += " ORDER BY t ASC"
        if limit:
            q += " LIMIT ?"; args.append(limit)
        out = []
        for r in self._all(q, args):
            d = dict(r); d["ctx"] = json.loads(d["ctx"]) if d["ctx"] else None
            out.append(d)
        return out

    # ------------------------------------------------------------------- kicks
    def add_kick(self, *, strength: str, magnitude: float, target_rel: float | None, direction: str | None, why: str | None,
                 track_id: int, distance: float | None, rel: float | None, band: str | None, dose: int,
                 pre_state: np.ndarray | None, kick_vec: np.ndarray | None, popularity: int | None = None, t: float | None = None) -> int:
        return self._run(
            "INSERT INTO kicks(t,strength,magnitude,target_rel,direction,why,track_id,distance,rel,band,dose,popularity,pre_state,kick_vec)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (t or time.time(), strength, magnitude, target_rel, direction, why, track_id, distance, rel, band, dose, popularity,
             _blob(pre_state), _blob(kick_vec)))

    def kick(self, kick_id: int) -> dict | None:
        r = self._one("SELECT * FROM kicks WHERE id=?", (kick_id,))
        return _kick_row(r) if r else None

    def last_kick(self) -> dict | None:
        r = self._one("SELECT * FROM kicks ORDER BY t DESC LIMIT 1")
        return _kick_row(r) if r else None

    def update_kick(self, kick_id: int, **fields) -> None:
        allowed = {"followed", "verdict", "verdict_at", "n_since", "popularity"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"not updatable: {bad}")
        sets = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE kicks SET {sets} WHERE id=?", (*fields.values(), kick_id))

    def plays_since_kick(self, kick_id: int) -> list[dict]:
        """Spotify's own plays after the kick (forced follow-through excluded)."""
        k = self.kick(kick_id)
        if not k:
            return []
        rows = self._all("SELECT e.*, t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
                         " WHERE e.t > ? AND e.kind IN ('play','partial','skip') AND e.source='spotify' ORDER BY e.t", (k["t"],))
        return [dict(r) for r in rows]

    # -------------------------------------------------------------- candidates
    def add_candidates(self, set_id: str, rows: list[dict], *, for_track_id: int | None = None, purpose: str = "pool",
                       t: float | None = None) -> list[int]:
        t = t or time.time()
        ids = []
        with self._lock:
            for c in rows:
                ids.append(self._run(
                    "INSERT INTO candidates(t,set_id,for_track_id,purpose,track_id,reach,direction,artist,title,why,proposed_uri,"
                    "rejected_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (t, set_id, for_track_id, purpose, c.get("track_id"), c.get("reach"), c.get("direction"), c["artist"], c["title"],
                     c.get("why"), c.get("spotify_uri"), c.get("rejected_reason"))))
        return ids

    def update_candidate(self, cand_id: int, **fields) -> None:
        allowed = {"track_id", "distance", "rel", "band", "home", "state", "affinity", "total", "p", "chosen", "rejected_reason", "kick_id"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"not updatable: {bad}")
        sets = ", ".join(f"{k}=?" for k in fields)
        self._run(f"UPDATE candidates SET {sets} WHERE id=?", (*fields.values(), cand_id))

    def candidate_set(self, set_id: str) -> list[dict]:
        return [dict(r) for r in self._all("SELECT * FROM candidates WHERE set_id=? ORDER BY id", (set_id,))]

    def latest_candidate_set(self, *, usable_only: bool = True, purpose: str = "pool") -> list[dict]:
        """The newest set of the given purpose. A follow-through set must never be restored as the kick pool: it is one
        direction, not a graded spread, and restoring one kept every kick inside it (the 'Brazilian pool' bug)."""
        r = self._one("SELECT set_id FROM candidates WHERE purpose=? ORDER BY t DESC, id DESC LIMIT 1", (purpose,))
        if not r:
            return []
        rows = self.candidate_set(r["set_id"])
        return [c for c in rows if not usable_only or (c["track_id"] is not None and not c["rejected_reason"] and not c["chosen"])]

    # ---------------------------------------------------------------- kv stores
    def get_profile(self, key: str) -> bytes | None:
        r = self._one("SELECT value FROM profile WHERE key=?", (key,))
        return r["value"] if r else None

    def set_profile(self, key: str, value: bytes) -> None:
        self._run("INSERT OR REPLACE INTO profile(key, value, updated_at) VALUES (?,?,?)", (key, value, time.time()))

    def get_config(self, key: str, default: str | None = None) -> str | None:
        r = self._one("SELECT value FROM config WHERE key=?", (key,))
        return r["value"] if r else default

    def set_config(self, key: str, value: str) -> None:
        self._run("INSERT OR REPLACE INTO config(key, value) VALUES (?,?)", (key, value))

    # ------------------------------------------------------ dedup, done here
    def seen(self, artist: str, title: str) -> bool:
        """Ever played, kicked, or picked. The Brain is asked not to repeat; the store enforces it."""
        t = self.find_track(artist, title)
        if t is None:
            return False
        if self._one("SELECT 1 FROM events WHERE track_id=? AND kind IN ('play','partial','skip','kick','pick') LIMIT 1", (t.id,)):
            return True
        return self._one("SELECT 1 FROM kicks WHERE track_id=? LIMIT 1", (t.id,)) is not None

    # ---------------------------------------------------- context queries (Brain)
    def recent(self, n: int = 12) -> list[dict]:
        """Last n plays, most recent first, with where they came from."""
        rows = self._all(
            "SELECT e.t, e.kind, e.source, e.completion, t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
            " WHERE e.kind IN ('play','partial','skip') ORDER BY e.t DESC, e.id DESC LIMIT ?", (n,))
        return [dict(r) for r in rows]

    def top_artists(self, *, days: int | None = None, n: int = 10) -> list[tuple[str, int]]:
        """Most-played artists; completed plays count 1, partials 0.5, skips 0. Optionally within the last `days`."""
        args: list = []
        where = "e.kind IN ('play','partial')"
        if days is not None:
            where += " AND e.t >= ?"; args.append(time.time() - days * 86400)
        rows = self._all(
            f"SELECT t.artist AS artist, SUM(CASE e.kind WHEN 'play' THEN 1.0 ELSE 0.5 END) AS w FROM events e JOIN tracks t ON t.id=e.track_id"
            f" WHERE {where} GROUP BY t.artist ORDER BY w DESC, MAX(e.t) DESC LIMIT ?", (*args, n))
        return [(r["artist"], round(r["w"])) for r in rows]

    def loved(self, n: int = 8) -> list[str]:
        rows = self._all("SELECT t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id WHERE e.kind='love'"
                         " ORDER BY e.t DESC LIMIT ?", (n,))
        return [f"{r['artist']} — {r['title']}" for r in rows]

    def rejected(self, *, days: int = 14, n: int = 8) -> list[str]:
        """Skipped or hated recently: 'not this vein'."""
        rows = self._all("SELECT t.artist, t.title FROM events e JOIN tracks t ON t.id=e.track_id"
                         " WHERE e.kind IN ('skip','hate') AND e.t >= ? ORDER BY e.t DESC LIMIT ?",
                         (time.time() - days * 86400, n))
        return list(dict.fromkeys(f"{r['artist']} — {r['title']}" for r in rows))

    def directions(self, n: int = 10) -> list[str]:
        rows = self._all("SELECT direction FROM kicks WHERE direction IS NOT NULL ORDER BY t DESC LIMIT ?", (n,))
        return [r["direction"] for r in rows]

    def kicked_artists(self, n: int = 25) -> list[str]:
        rows = self._all("SELECT DISTINCT t.artist FROM kicks k JOIN tracks t ON t.id=k.track_id ORDER BY k.t DESC LIMIT ?", (n,))
        return [r["artist"] for r in rows]

    def counts(self) -> dict:
        def c(table: str) -> int:
            return self._one(f"SELECT COUNT(*) AS n FROM {table}")["n"]
        return {t: c(t) for t in ("tracks", "embeddings", "events", "kicks", "candidates")}
