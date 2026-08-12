"""Test-set loading, hashing and deterministic sampling.

The acceptance criteria ask for results that are *reproducible using identical
test inputs* and *comparable across model versions*. Both reduce to one
requirement: the test set must have a stable identity that the framework can
prove. This module provides it with two hashes, kept separate on purpose:

*   ``content_hash`` — over only the fields that can change synthesised audio
    (id, text, language, pinned voice, per-case params). This is the hash that
    enters the run fingerprint. It answers "were these models given the same
    inputs?".
*   ``manifest_hash`` — over the entire manifest including annotations
    (category, notes, expectations). This is provenance. Editing a comment
    changes it but does *not* invalidate cross-run comparison, which is why it is
    not folded into the fingerprint.

A dataset is a JSONL file of cases plus an optional sibling ``<stem>.meta.yaml``
carrying id/version/description. JSONL rather than CSV because Indic text
routinely contains commas and quotes, and one bad quote in a CSV silently shifts
every column.
"""
from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from ..errors import DatasetError

BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"

# Fields that can change the audio a model produces. Anything outside this set is
# annotation and stays out of content_hash — see the module docstring.
_CONTENT_FIELDS = ("id", "text", "language", "voice", "params")


@dataclass(frozen=True)
class TestCase:
    """One sentence to synthesise, plus what we expect of the result."""

    id: str
    text: str
    language: str
    # Broad purpose of the case ("greeting", "numeric", "code_switch", ...).
    # Reports break metrics down by category because a model can be excellent on
    # plain prose and unusable on digits, and a single mean hides that.
    category: str = "general"
    # Pin a voice for this case. Normally left unset so the run-level voice
    # applies; used for deliberately multi-voice consistency tests.
    voice: str | None = None
    # Per-case generation overrides (e.g. emotion) merged over the run defaults.
    params: Mapping[str, Any] = field(default_factory=dict)
    # Ground-truth recording, for reference-based metrics (speaker similarity,
    # TTSDS2 distributional comparison). Relative paths resolve against the
    # manifest's directory.
    reference_audio: str | None = None
    # Substrings the round-trip ASR transcript must contain for the case to count
    # as correctly rendered. This is how "did it actually say the account number /
    # the English product name" is checked, which CER alone cannot express: a
    # 30-character sentence that drops a 4-digit OTP still scores ~87% CER.
    must_contain: tuple[str, ...] = ()
    # Native script name, used to flag script/language mismatches in reports.
    script: str | None = None
    # What a correct rendering should *sound like*, written out, when that differs
    # from the input text. Round-trip CER compares the ASR transcript against this
    # when set, and against `text` otherwise.
    #
    # This matters for any case containing digits, currency, dates, abbreviations
    # or symbols: given "₹12,450" a perfect model says "twelve thousand four
    # hundred fifty rupees", which scores an enormous CER against the written form.
    # Without this field, text-normalisation cases would be scored as failures no
    # matter how well the model performed — the single most common way a round-trip
    # TTS benchmark produces misleading numbers.
    expected_transcript: str | None = None
    notes: str | None = None

    @property
    def reference_text(self) -> str:
        """The string intelligibility metrics should score the transcript against."""
        return self.expected_transcript or self.text

    def content_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "voice": self.voice,
            "params": dict(self.params),
        }

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "text": self.text,
            "language": self.language,
            "category": self.category,
        }
        if self.voice:
            d["voice"] = self.voice
        if self.params:
            d["params"] = dict(self.params)
        if self.reference_audio:
            d["reference_audio"] = self.reference_audio
        if self.must_contain:
            d["must_contain"] = list(self.must_contain)
        if self.script:
            d["script"] = self.script
        if self.expected_transcript:
            d["expected_transcript"] = self.expected_transcript
        if self.notes:
            d["notes"] = self.notes
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any], *, source: str = "<memory>") -> "TestCase":
        for required in ("id", "text", "language"):
            if not str(d.get(required) or "").strip():
                raise DatasetError(f"{source}: case is missing required field {required!r}: {d!r}")
        must = d.get("must_contain") or ()
        if isinstance(must, str):
            must = (must,)
        return cls(
            id=str(d["id"]).strip(),
            text=str(d["text"]),
            language=str(d["language"]).strip(),
            category=str(d.get("category") or "general"),
            voice=(str(d["voice"]) if d.get("voice") else None),
            params=dict(d.get("params") or {}),
            reference_audio=(str(d["reference_audio"]) if d.get("reference_audio") else None),
            must_contain=tuple(str(m) for m in must),
            script=(str(d["script"]) if d.get("script") else None),
            expected_transcript=(
                str(d["expected_transcript"]) if d.get("expected_transcript") else None
            ),
            notes=(str(d["notes"]) if d.get("notes") else None),
        )


@dataclass
class TestDataset:
    """An immutable-by-convention, hashable collection of test cases."""

    id: str
    version: str
    cases: list[TestCase]
    description: str = ""
    source_path: Path | None = None

    def __post_init__(self) -> None:
        if not self.cases:
            raise DatasetError(f"dataset {self.id!r} contains no cases")
        dupes = [cid for cid, n in Counter(c.id for c in self.cases).items() if n > 1]
        if dupes:
            # Duplicate ids would break paired comparison (which joins runs on
            # utterance id) and silently double-count in aggregates.
            raise DatasetError(
                f"dataset {self.id!r} has duplicate case ids: {', '.join(sorted(dupes)[:10])}"
            )

    # ---- identity --------------------------------------------------------
    @property
    def content_hash(self) -> str:
        """sha256 over synthesis-affecting content only. Enters the fingerprint."""
        payload = json.dumps(
            [c.content_dict() for c in sorted(self.cases, key=lambda x: x.id)],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def manifest_hash(self) -> str:
        """sha256 over the full manifest including annotations. Provenance only."""
        payload = json.dumps(
            {
                "id": self.id,
                "version": self.version,
                "cases": [c.to_dict() for c in sorted(self.cases, key=lambda x: x.id)],
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def languages(self) -> list[str]:
        return sorted({c.language for c in self.cases})

    @property
    def categories(self) -> list[str]:
        return sorted({c.category for c in self.cases})

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self.cases)

    def by_language(self) -> dict[str, list[TestCase]]:
        out: dict[str, list[TestCase]] = defaultdict(list)
        for case in self.cases:
            out[case.language].append(case)
        return dict(out)

    # ---- derivation ------------------------------------------------------
    def filter(
        self,
        *,
        languages: Sequence[str] | None = None,
        categories: Sequence[str] | None = None,
        case_ids: Sequence[str] | None = None,
    ) -> "TestDataset":
        """Return a narrowed dataset.

        The derived dataset's ``version`` records the filter, so a report can
        never claim to cover 22 languages when the run was filtered to two.
        """
        keep = self.cases
        parts: list[str] = []
        if languages:
            wanted = {l.strip() for l in languages}
            keep = [c for c in keep if c.language in wanted]
            parts.append("lang=" + "+".join(sorted(wanted)))
        if categories:
            wanted_c = {c.strip() for c in categories}
            keep = [c for c in keep if c.category in wanted_c]
            parts.append("cat=" + "+".join(sorted(wanted_c)))
        if case_ids:
            wanted_i = {i.strip() for i in case_ids}
            keep = [c for c in keep if c.id in wanted_i]
            parts.append(f"ids={len(wanted_i)}")
        if not keep:
            raise DatasetError(
                f"filter removed every case from dataset {self.id!r} "
                f"(languages={languages}, categories={categories})"
            )
        suffix = ("+" + ",".join(parts)) if parts else ""
        return TestDataset(
            id=self.id,
            version=f"{self.version}{suffix}",
            cases=keep,
            description=self.description,
            source_path=self.source_path,
        )

    def sample(self, n: int, *, seed: int = 0, stratify_by: str | None = "language") -> "TestDataset":
        """Deterministically sample ``n`` cases.

        Seeded and sorted so the *same* subset is drawn on every machine — an
        unseeded sample would make two "identical" runs incomparable, which is
        precisely the failure mode the reproducibility criterion targets.

        With ``stratify_by="language"`` the quota is spread evenly across
        languages so a small smoke run still touches every language rather than
        over-sampling whichever one happens to sort first.
        """
        if n >= len(self.cases):
            return self
        if n <= 0:
            raise DatasetError(f"sample size must be positive, got {n}")

        rng = random.Random(f"{self.content_hash}|{seed}|{n}|{stratify_by}")
        if not stratify_by:
            chosen = sorted(rng.sample(sorted(self.cases, key=lambda c: c.id), n), key=lambda c: c.id)
        else:
            groups: dict[str, list[TestCase]] = defaultdict(list)
            for case in sorted(self.cases, key=lambda c: c.id):
                groups[getattr(case, stratify_by, "?")].append(case)
            keys = sorted(groups)
            chosen_set: list[TestCase] = []
            # Round-robin across groups so the remainder is distributed rather
            # than dumped on the first group.
            pools = {k: rng.sample(groups[k], len(groups[k])) for k in keys}
            while len(chosen_set) < n:
                progressed = False
                for k in keys:
                    if pools[k]:
                        chosen_set.append(pools[k].pop())
                        progressed = True
                        if len(chosen_set) == n:
                            break
                if not progressed:
                    break
            chosen = sorted(chosen_set, key=lambda c: c.id)

        return TestDataset(
            id=self.id,
            version=f"{self.version}+sample{n}s{seed}",
            cases=chosen,
            description=self.description,
            source_path=self.source_path,
        )

    # ---- serialisation ---------------------------------------------------
    def write_jsonl(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as fh:
            for case in self.cases:
                fh.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")
        # Sidecar carries identity + hashes so a written dataset can be reloaded
        # with its integrity check intact.
        meta = p.parent / (p.stem + ".meta.yaml")
        meta.write_text(
            "\n".join(
                [
                    f"id: {self.id}",
                    f"version: {self.version}",
                    f"description: {json.dumps(self.description, ensure_ascii=False)}",
                    f"content_hash: {self.content_hash}",
                    f"manifest_hash: {self.manifest_hash}",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return p

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "version": self.version,
            "n_cases": len(self.cases),
            "languages": self.languages,
            "categories": self.categories,
            "content_hash": self.content_hash,
            "manifest_hash": self.manifest_hash,
            "per_language_counts": {
                lang: len(cases) for lang, cases in sorted(self.by_language().items())
            },
        }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def load_dataset(spec: str | Path) -> TestDataset:
    """Load a dataset by builtin name or filesystem path.

    ``spec`` may be:
      * a builtin name, e.g. ``"indic_conversational_v1"``
      * a path to a ``.jsonl`` manifest
      * a path to a directory containing exactly one ``.jsonl``
    """
    if isinstance(spec, str) and not any(sep in spec for sep in ("/", "\\")) and not spec.endswith(".jsonl"):
        builtin = BUILTIN_DIR / f"{spec}.jsonl"
        if builtin.is_file():
            return _load_jsonl(builtin)
        available = ", ".join(sorted(p.stem for p in BUILTIN_DIR.glob("*.jsonl"))) or "(none)"
        raise DatasetError(f"unknown builtin dataset {spec!r}; available: {available}")

    path = Path(spec).expanduser()
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"))
        if len(candidates) != 1:
            raise DatasetError(
                f"{path} contains {len(candidates)} .jsonl manifests; pass the file explicitly"
            )
        path = candidates[0]
    if not path.is_file():
        raise DatasetError(f"dataset manifest not found: {path}")
    return _load_jsonl(path)


def _load_jsonl(path: Path) -> TestDataset:
    cases: list[TestCase] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line or line.startswith("//") or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise DatasetError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if not isinstance(row, Mapping):
                raise DatasetError(f"{path}:{lineno}: expected a JSON object, got {type(row).__name__}")
            cases.append(TestCase.from_dict(row, source=f"{path}:{lineno}"))

    meta = _load_meta(path)
    ds = TestDataset(
        id=str(meta.get("id") or path.stem),
        version=str(meta.get("version") or "unversioned"),
        cases=cases,
        description=str(meta.get("description") or ""),
        source_path=path,
    )

    # If the sidecar pins a hash, verify it. This is the integrity check that
    # makes "the dataset was not quietly edited between runs" an enforced fact
    # rather than a hope.
    declared = str(meta.get("content_hash") or "")
    if declared and declared != ds.content_hash:
        raise DatasetError(
            f"{path}: content_hash mismatch — manifest was edited.\n"
            f"  declared: {declared}\n  actual:   {ds.content_hash}\n"
            "Bump `version` and update `content_hash` in the .meta.yaml to accept the change."
        )
    return ds


def _load_meta(jsonl_path: Path) -> dict[str, Any]:
    meta_path = jsonl_path.parent / (jsonl_path.stem + ".meta.yaml")
    if not meta_path.is_file():
        return {}
    try:
        import yaml
    except ImportError:  # pragma: no cover - PyYAML is a core dependency
        return {}
    try:
        data = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
    except Exception as e:  # noqa: BLE001
        raise DatasetError(f"{meta_path}: could not parse dataset metadata: {e}") from e
    if not isinstance(data, Mapping):
        raise DatasetError(f"{meta_path}: expected a mapping at the top level")
    return dict(data)


def dataset_from_cases(
    cases: Iterable[Mapping[str, Any]], *, id: str = "adhoc", version: str = "1"
) -> TestDataset:
    """Build a dataset in memory. Used by tests and by `tts-eval dataset new`."""
    return TestDataset(
        id=id, version=version, cases=[TestCase.from_dict(c) for c in cases]
    )


def list_builtin() -> list[str]:
    return sorted(p.stem for p in BUILTIN_DIR.glob("*.jsonl"))


__all__ = [
    "TestCase",
    "TestDataset",
    "load_dataset",
    "dataset_from_cases",
    "list_builtin",
    "BUILTIN_DIR",
]
