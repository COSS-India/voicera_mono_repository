"""
Kenpath Vistaar LLM load tester (prod /api/voice/).

Run interactively:
  cd voice_2_voice_server
  python services/kenpath_llm/load_test.py

Or pass CLI flags to skip the menu (see --help).

Requires KENPATH_JWT_PRIVATE_KEY_PATH in voice_2_voice_server/.env
"""

from __future__ import annotations

# =============================================================================
# CONFIG — edit these variables for your test run
# =============================================================================

NUM_REQUESTS = 100

# "same_session" — reuse one session_id for all requests
# "unique_session" — new uuid per request
# "both" — run same_session then unique_session
SESSION_MODE = "unique_session"

QUERY = "हिवाळी पिके कोणती आहेत?"
SOURCE_LANG = "mr"
TARGET_LANG = "mr"
VISTAAR_ENVIRONMENT = "prod"

# Arrival pattern:
#   "burst"      — all requests start at once
#   "sequential" — one request finishes before the next starts
MODE = "burst"

# Sequential only: optional delay after each response before the next request.
GAP_S = 0.0

# Log file path (relative to voice_2_voice_server). Empty = auto timestamped file.
LOG_FILE = ""

# Stop on first error when True.
STRICT = False

# =============================================================================

import argparse
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import jwt
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

DEFAULT_VISTAAR_PROD_URL = "https://voice-prod.mahapocra.gov.in"
DEFAULT_VISTAAR_DEV_URL = "https://vistaar-dev.mahapocra.gov.in"


def resolve_vistaar_base_url(environment: str) -> str:
    env = (environment or "prod").strip().lower()
    if env == "dev":
        import os

        return os.environ.get("KENPATH_VISTAAR_API_URL_DEV", DEFAULT_VISTAAR_DEV_URL)
    import os

    return (
        os.environ.get("KENPATH_VISTAAR_API_URL_PROD")
        or os.environ.get("KENPATH_VISTAAR_API_URL")
        or DEFAULT_VISTAAR_PROD_URL
    )


@dataclass
class RequestResult:
    index: int
    session_id: str
    ok: bool
    status_code: int | None = None
    response_text: str = ""
    error: str = ""
    elapsed_s: float = 0.0
    ttfb_s: float | None = None


@dataclass
class RunSummary:
    label: str
    session_mode: str
    results: list[RequestResult] = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(1 for r in self.results if r.ok)

    @property
    def failures(self) -> int:
        return sum(1 for r in self.results if not r.ok)

    @property
    def unique_responses(self) -> set[str]:
        return {r.response_text.strip() for r in self.results if r.ok and r.response_text.strip()}


class KenpathVistaarClient:
    def __init__(self, *, environment: str, jwt_phone: str, private_key_path: str):
        self._base_url = resolve_vistaar_base_url(environment).rstrip("/")
        self._jwt_phone = jwt_phone
        self._private_key = Path(private_key_path).read_text()
        self._session: aiohttp.ClientSession | None = None

    def _generate_jwt(self) -> str:
        now = int(time.time())
        payload = {
            "sub": self._jwt_phone,
            "iss": "voice-provider",
            "iat": now,
            "exp": now + 3600,
        }
        return jwt.encode(payload, self._private_key, algorithm="RS256")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=120)
            )
        return self._session

    async def query(
        self,
        *,
        query: str,
        session_id: str,
        source_lang: str,
        target_lang: str,
    ) -> tuple[int, str, float, float | None]:
        """Return (status_code, full_response_text, elapsed_s, ttfb_s)."""
        url = f"{self._base_url}/api/voice/"
        params = {
            "query": query,
            "source_lang": source_lang,
            "target_lang": target_lang,
            "session_id": session_id,
        }
        headers = {"Authorization": f"Bearer {self._generate_jwt()}"}

        start = time.perf_counter()
        ttfb: float | None = None
        parts: list[str] = []

        session = await self._get_session()
        async with session.get(url, params=params, headers=headers) as response:
            if response.status != 200:
                body = await response.text()
                elapsed = time.perf_counter() - start
                return response.status, body, elapsed, None

            async for chunk in response.content.iter_any():
                if ttfb is None and chunk:
                    ttfb = time.perf_counter() - start
                parts.append(chunk.decode("utf-8", errors="replace"))

        elapsed = time.perf_counter() - start
        return 200, "".join(parts).strip(), elapsed, ttfb

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


def _default_log_path() -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return ROOT / "services" / "kenpath_llm" / f"load_test_{ts}.log"


def _log_line(log_fp, line: str) -> None:
    print(line, flush=True)
    if log_fp is not None:
        log_fp.write(line + "\n")
        log_fp.flush()


def _result_payload(label: str, session_mode: str, result: RequestResult) -> dict:
    return {
        "test": label,
        "session_mode": session_mode,
        "index": result.index,
        "session_id": result.session_id,
        "ok": result.ok,
        "status_code": result.status_code,
        "elapsed_s": round(result.elapsed_s, 3),
        "ttfb_s": round(result.ttfb_s, 3) if result.ttfb_s is not None else None,
        "response": result.response_text,
        "error": result.error,
    }


def _log_result(log_fp, label: str, session_mode: str, result: RequestResult) -> None:
    _log_line(log_fp, json.dumps(_result_payload(label, session_mode, result), ensure_ascii=False))


async def run_request(
    client: KenpathVistaarClient,
    *,
    index: int,
    session_id: str,
    query: str,
    source_lang: str,
    target_lang: str,
) -> RequestResult:
    try:
        status, text, elapsed, ttfb = await client.query(
            query=query,
            session_id=session_id,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        ok = status == 200
        return RequestResult(
            index=index,
            session_id=session_id,
            ok=ok,
            status_code=status,
            response_text=text if ok else "",
            error="" if ok else text,
            elapsed_s=elapsed,
            ttfb_s=ttfb,
        )
    except Exception as exc:
        return RequestResult(
            index=index,
            session_id=session_id,
            ok=False,
            error=str(exc),
        )


async def run_batch(
    client: KenpathVistaarClient,
    *,
    label: str,
    session_mode: str,
    num_requests: int,
    query: str,
    source_lang: str,
    target_lang: str,
    fixed_session_id: str,
    mode: str,
    gap_s: float,
    log_fp,
    strict: bool,
) -> RunSummary:
    summary = RunSummary(label=label, session_mode=session_mode)
    _log_line(log_fp, "")
    _log_line(log_fp, "=" * 80)
    _log_line(
        log_fp,
        f"TEST: {label} | session_mode={session_mode} | requests={num_requests} | mode={mode}",
    )
    _log_line(log_fp, f"query={query!r} | lang={source_lang} | fixed_session_id={fixed_session_id}")
    _log_line(log_fp, "=" * 80)

    async def one(i: int) -> RequestResult:
        session_id = (
            fixed_session_id if session_mode == "same_session" else str(uuid.uuid4())
        )
        return await run_request(
            client,
            index=i,
            session_id=session_id,
            query=query,
            source_lang=source_lang,
            target_lang=target_lang,
        )

    if mode == "burst":
        session_desc = (
            "same session_id"
            if session_mode == "same_session"
            else "unique session_id each"
        )
        _log_line(
            log_fp,
            f"Launching {num_requests} concurrent requests ({session_desc})...",
        )
        tasks = {
            asyncio.create_task(one(i)): i for i in range(1, num_requests + 1)
        }
        for finished in asyncio.as_completed(tasks.keys()):
            result = await finished
            summary.results.append(result)
            _log_result(log_fp, label, session_mode, result)
            if strict and not result.ok:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                _log_line(log_fp, f"STRICT: stopping after failure at request {result.index}")
                break
        summary.results.sort(key=lambda r: r.index)
    else:
        for i in range(1, num_requests + 1):
            session_id = (
                fixed_session_id if session_mode == "same_session" else str(uuid.uuid4())
            )
            _log_line(
                log_fp,
                f"[{i}/{num_requests}] sending | session_id={session_id}",
            )
            result = await run_request(
                client,
                index=i,
                session_id=session_id,
                query=query,
                source_lang=source_lang,
                target_lang=target_lang,
            )
            summary.results.append(result)
            _log_result(log_fp, label, session_mode, result)
            if strict and not result.ok:
                _log_line(log_fp, f"STRICT: stopping after failure at request {result.index}")
                break
            if gap_s > 0 and i < num_requests:
                await asyncio.sleep(gap_s)

    elapsed_values = [r.elapsed_s for r in summary.results if r.ok]
    ttfb_values = [r.ttfb_s for r in summary.results if r.ok and r.ttfb_s is not None]
    _log_line(log_fp, "-" * 80)
    _log_line(
        log_fp,
        f"SUMMARY [{label}] | ok={summary.successes}/{len(summary.results)} | "
        f"fail={summary.failures} | unique_responses={len(summary.unique_responses)}",
    )
    if elapsed_values:
        _log_line(
            log_fp,
            f"  elapsed_s: min={min(elapsed_values):.3f} max={max(elapsed_values):.3f} "
            f"avg={sum(elapsed_values) / len(elapsed_values):.3f}",
        )
    if ttfb_values:
        _log_line(
            log_fp,
            f"  ttfb_s:    min={min(ttfb_values):.3f} max={max(ttfb_values):.3f} "
            f"avg={sum(ttfb_values) / len(ttfb_values):.3f}",
        )
    return summary


async def main_async(args: argparse.Namespace) -> int:
    import os

    key_path = os.environ.get("KENPATH_JWT_PRIVATE_KEY_PATH", "").strip()
    if not key_path:
        print("ERROR: KENPATH_JWT_PRIVATE_KEY_PATH is not set in .env", file=sys.stderr)
        return 1

    key_file = ROOT / key_path if not Path(key_path).is_absolute() else Path(key_path)
    if not key_file.is_file():
        print(f"ERROR: JWT private key not found: {key_file}", file=sys.stderr)
        return 1

    jwt_phone = os.environ.get("KENPATH_JWT_PHONE", "+91-9036722772")
    log_path = Path(args.log_file) if args.log_file else _default_log_path()
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    client = KenpathVistaarClient(
        environment=args.environment,
        jwt_phone=jwt_phone,
        private_key_path=str(key_file),
    )

    fixed_session_id = str(uuid.uuid4())
    modes_to_run: list[tuple[str, str]] = []
    if args.session_mode in ("same_session", "both"):
        modes_to_run.append(("same_session", "same_session"))
    if args.session_mode in ("unique_session", "both"):
        modes_to_run.append(("unique_session", "unique_session"))

    with log_path.open("w", encoding="utf-8") as log_fp:
        _log_line(log_fp, f"Kenpath Vistaar load test started at {datetime.now(timezone.utc).isoformat()}")
        _log_line(log_fp, f"base_url={client._base_url}")
        _log_line(log_fp, f"log_file={log_path}")

        summaries: list[RunSummary] = []
        try:
            for label, session_mode in modes_to_run:
                summary = await run_batch(
                    client,
                    label=label,
                    session_mode=session_mode,
                    num_requests=args.num_requests,
                    query=args.query,
                    source_lang=args.source_lang,
                    target_lang=args.target_lang,
                    fixed_session_id=fixed_session_id,
                    mode=args.mode,
                    gap_s=args.gap_s,
                    log_fp=log_fp,
                    strict=args.strict,
                )
                summaries.append(summary)
        finally:
            await client.close()

        _log_line(log_fp, "")
        _log_line(log_fp, "=" * 80)
        _log_line(log_fp, "FINAL SUMMARY")
        for summary in summaries:
            _log_line(
                log_fp,
                f"  {summary.label}: ok={summary.successes}/{len(summary.results)} "
                f"unique_responses={len(summary.unique_responses)}",
            )
        _log_line(log_fp, f"Full log written to: {log_path}")

    return 0 if all(s.failures == 0 for s in summaries) else 1


TEST_PRESETS: list[dict] = [
    {
        "label": "Burst — unique session (100 at once)",
        "session_mode": "unique_session",
        "mode": "burst",
        "num_requests": NUM_REQUESTS,
        "gap_s": 0.0,
    },
    {
        "label": "Sequential — same session (back-to-back)",
        "session_mode": "same_session",
        "mode": "sequential",
        "num_requests": NUM_REQUESTS,
        "gap_s": 0.0,
    },
    {
        "label": "Sequential — unique session (back-to-back)",
        "session_mode": "unique_session",
        "mode": "sequential",
        "num_requests": NUM_REQUESTS,
        "gap_s": 0.0,
    },
    {
        "label": "Both — same then unique session (sequential)",
        "session_mode": "both",
        "mode": "sequential",
        "num_requests": NUM_REQUESTS,
        "gap_s": 0.0,
    },
    {
        "label": "Burst — same session (100 at once)",
        "session_mode": "same_session",
        "mode": "burst",
        "num_requests": NUM_REQUESTS,
        "gap_s": 0.0,
    },
]


def _prompt_choice(prompt: str, options: list[tuple[str, str]], *, default: str) -> str:
    """Prompt until user picks a valid option key."""
    print(f"\n{prompt}")
    for key, label in options:
        marker = " (default)" if key == default else ""
        print(f"  {key}) {label}{marker}")
    while True:
        raw = input(f"Choice [{default}]: ").strip()
        if not raw:
            return default
        if any(key == raw for key, _ in options):
            return raw
        print(f"Invalid choice {raw!r}. Enter one of: {', '.join(k for k, _ in options)}")


def _prompt_int(prompt: str, *, default: int, minimum: int = 1) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Enter a whole number.")
            continue
        if value < minimum:
            print(f"Must be at least {minimum}.")
            continue
        return value


def _prompt_float(prompt: str, *, default: float, minimum: float = 0.0) -> float:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except ValueError:
            print("Enter a number.")
            continue
        if value < minimum:
            print(f"Must be at least {minimum}.")
            continue
        return value


def _prompt_yes_no(prompt: str, *, default: bool = False) -> bool:
    default_key = "y" if default else "n"
    while True:
        raw = input(f"{prompt} [y/n, default {default_key}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("Enter y or n.")


def _build_namespace(**kwargs) -> argparse.Namespace:
    return argparse.Namespace(
        num_requests=kwargs.get("num_requests", NUM_REQUESTS),
        session_mode=kwargs.get("session_mode", SESSION_MODE),
        query=kwargs.get("query", QUERY),
        source_lang=kwargs.get("source_lang", SOURCE_LANG),
        target_lang=kwargs.get("target_lang", TARGET_LANG),
        environment=kwargs.get("environment", VISTAAR_ENVIRONMENT),
        mode=kwargs.get("mode", MODE),
        gap_s=kwargs.get("gap_s", GAP_S),
        log_file=kwargs.get("log_file", LOG_FILE),
        strict=kwargs.get("strict", STRICT),
    )


def _configure_custom() -> argparse.Namespace:
    print("\n--- Custom test ---")
    session_mode = _prompt_choice(
        "Session mode:",
        [
            ("1", "same_session — reuse one session_id"),
            ("2", "unique_session — new uuid per request"),
            ("3", "both — run same_session then unique_session"),
        ],
        default="2",
    )
    session_map = {"1": "same_session", "2": "unique_session", "3": "both"}

    mode = _prompt_choice(
        "Arrival pattern:",
        [
            ("1", "burst — fire all requests at once"),
            ("2", "sequential — wait for each response before the next"),
        ],
        default="1",
    )
    mode_map = {"1": "burst", "2": "sequential"}

    environment = _prompt_choice(
        "Environment:",
        [
            ("1", "prod"),
            ("2", "dev"),
        ],
        default="1",
    )
    env_map = {"1": "prod", "2": "dev"}

    num_requests = _prompt_int("Number of requests", default=NUM_REQUESTS)
    gap_s = 0.0
    if mode_map[mode] == "sequential":
        gap_s = _prompt_float(
            "Gap between requests (seconds, 0 = send immediately after response)",
            default=0.0,
        )

    use_default_query = _prompt_yes_no(
        f"Use default query ({QUERY!r})?", default=True
    )
    query = QUERY
    if not use_default_query:
        query = input("Query text: ").strip() or QUERY

    strict = _prompt_yes_no("Stop on first error?", default=False)

    return _build_namespace(
        session_mode=session_map[session_mode],
        mode=mode_map[mode],
        environment=env_map[environment],
        num_requests=num_requests,
        gap_s=gap_s,
        query=query,
        strict=strict,
    )


def interactive_menu() -> argparse.Namespace:
    print("=" * 60)
    print("Kenpath Vistaar Load Test")
    print("=" * 60)
    print(f"Default query: {QUERY}")
    print(f"Default language: {SOURCE_LANG}")
    print()

    for i, preset in enumerate(TEST_PRESETS, start=1):
        print(f"  {i}) {preset['label']}")
    custom_num = len(TEST_PRESETS) + 1
    print(f"  {custom_num}) Custom — configure each option")

    while True:
        raw = input(f"\nSelect test [1-{custom_num}, default 1]: ").strip()
        if not raw:
            choice = 1
            break
        try:
            choice = int(raw)
        except ValueError:
            print("Enter a number from the list.")
            continue
        if 1 <= choice <= custom_num:
            break
        print(f"Enter a number between 1 and {custom_num}.")

    if choice == custom_num:
        return _configure_custom()

    preset = TEST_PRESETS[choice - 1]
    print(f"\nSelected: {preset['label']}")

    customize = _prompt_yes_no("Customize request count or environment?", default=False)
    num_requests = preset["num_requests"]
    environment = VISTAAR_ENVIRONMENT
    if customize:
        num_requests = _prompt_int("Number of requests", default=num_requests)
        env_choice = _prompt_choice(
            "Environment:",
            [("1", "prod"), ("2", "dev")],
            default="1",
        )
        environment = "prod" if env_choice == "1" else "dev"

    return _build_namespace(
        session_mode=preset["session_mode"],
        mode=preset["mode"],
        num_requests=num_requests,
        gap_s=preset["gap_s"],
        environment=environment,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kenpath Vistaar LLM load tester")
    parser.add_argument(
        "-n", "--num-requests", type=int, default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--session-mode",
        choices=["same_session", "unique_session", "both"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--query", default=argparse.SUPPRESS)
    parser.add_argument("--source-lang", default=argparse.SUPPRESS)
    parser.add_argument("--target-lang", default=argparse.SUPPRESS)
    parser.add_argument(
        "--environment",
        choices=["prod", "dev"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--mode",
        choices=["burst", "sequential"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--gap-s", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--log-file", default=argparse.SUPPRESS)
    parser.add_argument("--strict", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Show interactive menu (default when no options are passed)",
    )
    return parser.parse_args(argv)


def resolve_args(argv: list[str] | None = None) -> argparse.Namespace:
    parsed = parse_args(argv)
    cli_argv = argv if argv is not None else sys.argv[1:]
    if parsed.interactive or not cli_argv:
        return interactive_menu()
    return _build_namespace(**vars(parsed))


def main() -> None:
    args = resolve_args()
    print(
        f"\nStarting: mode={args.mode} | session={args.session_mode} | "
        f"n={args.num_requests} | env={args.environment}\n"
    )
    raise SystemExit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
