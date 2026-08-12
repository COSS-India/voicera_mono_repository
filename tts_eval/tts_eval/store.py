"""Run registry: durable storage, listing and retrieval.

Design choice that matters for the "results are stored for future comparison"
criterion: **JSON files on disk are the source of truth; SQLite is a rebuildable
index.**

Why not SQLite-only? A benchmark record has to outlive the tool that wrote it. A
directory of ``run.json`` files plus WAVs can be read in five years with nothing
but a text editor, copied between machines, attached to a ticket, diffed in a code
review, and committed. A binary database that only this package can open is a
worse archive, and schema migrations become mandatory rather than optional.

Why SQLite at all? Because "list the last 20 runs of model X sorted by p95 TTFB"
against 500 JSON files means parsing 500 files. The index makes listing and
comparison selection fast, and ``reindex()`` rebuilds it from the JSON at any time
— so the index can be deleted, corrupted, or schema-changed without data loss.

Layout::

    runs/
      <run_id>/
        run.json          # the complete record — source of truth
        audio/*.wav       # synthesised audio, referenced by utterance records
        timings.json      # per-utterance timings, so the replay adapter can
                          # re-score this run without inventing latencies
      index.sqlite3       # derived, rebuildable
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from . import SCHEMA_VERSION
from .errors import StoreError
from .types import ReviewSignoff, RunRecord, SubjectiveScore

INDEX_FILENAME = "index.sqlite3"
RECORD_FILENAME = "run.json"
TIMINGS_FILENAME = "timings.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    label           TEXT,
    model_id        TEXT NOT NULL,
    model_version   TEXT NOT NULL,
    provider        TEXT,
    adapter         TEXT,
    dataset_id      TEXT,
    dataset_version TEXT,
    dataset_hash    TEXT,
    dataset_size    INTEGER,
    concurrency     INTEGER,
    seed            INTEGER,
    fingerprint     TEXT NOT NULL,
    schema_version  INTEGER NOT NULL,
    n_utterances    INTEGER,
    n_ok            INTEGER,
    success_rate    REAL,
    n_signoffs      INTEGER DEFAULT 0,
    n_subjective    INTEGER DEFAULT 0,
    path            TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id, model_version);
CREATE INDEX IF NOT EXISTS idx_runs_fingerprint ON runs(fingerprint);
CREATE INDEX IF NOT EXISTS idx_runs_created ON runs(created_at DESC);

-- Headline aggregates, denormalised for fast leaderboard queries. Rebuilt from
-- run.json by reindex(); never written to independently.
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id  TEXT NOT NULL,
    metric  TEXT NOT NULL,
    n       INTEGER,
    mean    REAL,
    median  REAL,
    p95     REAL,
    ci_low  REAL,
    ci_high REAL,
    PRIMARY KEY (run_id, metric),
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_run_metrics_metric ON run_metrics(metric);
"""


@dataclass(frozen=True)
class RunSummary:
    """Index row: enough to choose runs without opening their JSON."""

    run_id: str
    created_at: str
    label: str
    model_id: str
    model_version: str
    provider: str
    dataset_id: str
    dataset_version: str
    fingerprint: str
    n_utterances: int
    n_ok: int
    success_rate: float | None
    concurrency: int
    n_signoffs: int
    n_subjective: int
    path: Path

    @property
    def display_name(self) -> str:
        return f"{self.model_id}@{self.model_version}"

    @property
    def reviewed(self) -> bool:
        return self.n_signoffs > 0


class RunStore:
    """Filesystem-backed run registry."""

    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._index_path = self.root / INDEX_FILENAME
        self._ensure_index()

    # -- index plumbing ----------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._index_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _ensure_index(self) -> None:
        try:
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except sqlite3.Error as e:
            raise StoreError(f"could not initialise run index at {self._index_path}: {e}") from e

    # -- writing -----------------------------------------------------------
    def save(self, record: RunRecord) -> Path:
        """Persist a record and index it. Returns the run directory."""
        run_dir = self.root / record.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        record_path = run_dir / RECORD_FILENAME

        payload = record.to_dict()
        # Write to a temp file then rename: an interrupted write must not leave a
        # half-parsed run.json that the index then advertises as valid.
        tmp = record_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=False), encoding="utf-8"
        )
        tmp.replace(record_path)

        # Sidecar so the replay adapter can re-score this run's audio later with
        # the ORIGINAL latencies instead of reporting them as unmeasurable.
        timings = {
            u.utterance_id: {
                "ttfb_ms": u.result.ttfb_ms,
                "first_audible_ms": u.result.first_audible_ms,
                "total_ms": u.result.total_ms,
            }
            for u in record.utterances
            if u.result.ok
        }
        audio_dir = run_dir / "audio"
        if timings and audio_dir.is_dir():
            (audio_dir / TIMINGS_FILENAME).write_text(
                json.dumps(timings, indent=2), encoding="utf-8"
            )

        self._index(record, record_path)
        return run_dir

    def _index(self, record: RunRecord, path: Path) -> None:
        headline = (
            "success_rate",
            "degenerate_rate",
            "ttfb_ms",
            "first_audible_ms",
            "inference_time_ms",
            "rtf",
            "cer",
            "wer",
            "slot_accuracy",
            "utmos",
            "dnsmos_ovrl",
            "subjective_mos",
            "audio_quality_score",
            "snr_db",
            "voice_consistency",
            "speaker_consistency",
            "coverage_ratio",
            "throughput_utt_per_min",
        )
        try:
            with closing(self._connect()) as conn:
                conn.execute("DELETE FROM runs WHERE run_id = ?", (record.run_id,))
                conn.execute("DELETE FROM run_metrics WHERE run_id = ?", (record.run_id,))
                conn.execute(
                    """INSERT INTO runs (run_id, created_at, finished_at, label, model_id,
                           model_version, provider, adapter, dataset_id, dataset_version,
                           dataset_hash, dataset_size, concurrency, seed, fingerprint,
                           schema_version, n_utterances, n_ok, success_rate, n_signoffs,
                           n_subjective, path)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        record.run_id,
                        record.created_at,
                        record.finished_at,
                        record.label,
                        record.model_id,
                        record.model_version,
                        record.provider,
                        record.adapter,
                        record.dataset_id,
                        record.dataset_version,
                        record.dataset_hash,
                        record.dataset_size,
                        record.concurrency,
                        record.seed,
                        record.fingerprint,
                        record.schema_version,
                        len(record.utterances),
                        record.n_ok,
                        record.success_rate,
                        len(record.signoffs),
                        len(record.subjective),
                        str(path),
                    ),
                )
                rows = [
                    (
                        record.run_id,
                        name,
                        agg.n,
                        agg.mean,
                        agg.median,
                        agg.p95,
                        agg.ci_low,
                        agg.ci_high,
                    )
                    for name, agg in record.aggregates.items()
                    if name in headline
                ]
                conn.executemany(
                    "INSERT INTO run_metrics (run_id, metric, n, mean, median, p95, ci_low, ci_high) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    rows,
                )
                conn.commit()
        except sqlite3.Error as e:
            raise StoreError(f"could not index run {record.run_id}: {e}") from e

    # -- reading -----------------------------------------------------------
    def path_for(self, run_id: str) -> Path:
        return self.root / run_id / RECORD_FILENAME

    def load(self, run_id: str) -> RunRecord:
        """Load a full record. Accepts a unique run-id prefix for convenience."""
        path = self.path_for(run_id)
        if not path.is_file():
            matches = [
                d for d in sorted(self.root.iterdir())
                if d.is_dir() and d.name.startswith(run_id) and (d / RECORD_FILENAME).is_file()
            ]
            if len(matches) == 1:
                path = matches[0] / RECORD_FILENAME
            elif len(matches) > 1:
                raise StoreError(
                    f"run id prefix {run_id!r} is ambiguous: {', '.join(m.name for m in matches[:5])}"
                )
            else:
                raise StoreError(f"no run found for {run_id!r} under {self.root}")

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise StoreError(f"could not read run record {path}: {e}") from e

        found = int(data.get("schema_version") or 1)
        if found > SCHEMA_VERSION:
            raise StoreError(
                f"{path} was written by a newer tts_eval (record schema v{found}, this build "
                f"reads v{SCHEMA_VERSION}). Upgrade tts-eval rather than reading it partially."
            )
        return RunRecord.from_dict(data)

    def list_runs(
        self,
        *,
        model_id: str | None = None,
        fingerprint: str | None = None,
        limit: int | None = None,
    ) -> list[RunSummary]:
        query = "SELECT * FROM runs"
        clauses: list[str] = []
        params: list[Any] = []
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if fingerprint:
            clauses.append("fingerprint = ?")
            params.append(fingerprint)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with closing(self._connect()) as conn:
            rows = conn.execute(query, params).fetchall()
        return [_summary(row) for row in rows]

    def find_repeats(self, fingerprint: str) -> list[RunSummary]:
        """Runs sharing a fingerprint — i.e. genuine repeats of one evaluation.

        This is what makes "multiple evaluation runs can be performed using the
        same test dataset" verifiable: repeats are identified structurally, not by
        someone remembering they used the same settings.
        """
        return self.list_runs(fingerprint=fingerprint)

    def reindex(self) -> int:
        """Rebuild the index from the JSON records. Returns the number indexed."""
        with closing(self._connect()) as conn:
            conn.execute("DELETE FROM run_metrics")
            conn.execute("DELETE FROM runs")
            conn.commit()

        count = 0
        for run_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            path = run_dir / RECORD_FILENAME
            if not path.is_file():
                continue
            try:
                record = self.load(run_dir.name)
            except StoreError:
                # A single unreadable record must not abort a rebuild; it stays on
                # disk and is simply absent from the index until repaired.
                continue
            self._index(record, path)
            count += 1
        return count

    # -- mutation of post-hoc annotations ---------------------------------
    def add_subjective(self, run_id: str, scores: Sequence[SubjectiveScore]) -> RunRecord:
        """Attach human ratings and re-save.

        Read-modify-write of the whole record rather than an append-only side
        table, so the JSON remains the complete, self-contained artefact.
        """
        record = self.load(run_id)
        existing = {(s.utterance_id, s.rater_id, s.scale) for s in record.subjective}
        for score in scores:
            key = (score.utterance_id, score.rater_id, score.scale)
            if key not in existing:
                record.subjective.append(score)
                existing.add(key)
        self.save(record)
        return record

    def add_signoff(self, run_id: str, signoff: ReviewSignoff) -> RunRecord:
        record = self.load(run_id)
        record.signoffs.append(signoff)
        self.save(record)
        return record

    def audio_dir(self, run_id: str) -> Path:
        return self.root / run_id / "audio"

    def __iter__(self) -> Iterable[RunSummary]:
        return iter(self.list_runs())


def _summary(row: sqlite3.Row) -> RunSummary:
    return RunSummary(
        run_id=row["run_id"],
        created_at=row["created_at"],
        label=row["label"] or "",
        model_id=row["model_id"],
        model_version=row["model_version"],
        provider=row["provider"] or "",
        dataset_id=row["dataset_id"] or "",
        dataset_version=row["dataset_version"] or "",
        fingerprint=row["fingerprint"],
        n_utterances=row["n_utterances"] or 0,
        n_ok=row["n_ok"] or 0,
        success_rate=row["success_rate"],
        concurrency=row["concurrency"] or 1,
        n_signoffs=row["n_signoffs"] or 0,
        n_subjective=row["n_subjective"] or 0,
        path=Path(row["path"]),
    )


__all__ = ["INDEX_FILENAME", "RECORD_FILENAME", "RunStore", "RunSummary"]
