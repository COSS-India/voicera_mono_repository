"""
APScheduler-based background scheduler for due batch execution.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Optional

import requests
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import settings
from app.services import batch_service

logger = logging.getLogger(__name__)

_scheduler_lock = threading.Lock()
_scheduler: Optional[BackgroundScheduler] = None


def _voice_server_headers() -> Dict[str, str]:
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if settings.INTERNAL_API_KEY:
        headers["X-API-Key"] = settings.INTERNAL_API_KEY
    return headers


def _start_batch_on_voice_server(*, org_id: str, batch_id: str, agent_type: str, concurrency: int) -> Dict[str, Any]:
    response = requests.post(
        f"{settings.VOICE_SERVER_URL}/outbound/batch/run/",
        json={
            "org_id": org_id,
            "batch_id": batch_id,
            "agent_type": agent_type,
            "concurrency": concurrency,
        },
        headers=_voice_server_headers(),
        timeout=10,
    )
    response.raise_for_status()
    payload = response.json() if response.text else {}
    return payload if isinstance(payload, dict) else {}


def _poll_due_batches() -> None:
    while True:
        claimed = batch_service.claim_next_due_scheduled_batch()
        if not claimed:
            return

        batch_id = str(claimed.get("batch_id") or "").strip()
        org_id = str(claimed.get("org_id") or "").strip()
        if not batch_id or not org_id:
            logger.warning("Skipping malformed claimed scheduled batch: %s", claimed)
            continue

        try:
            run_result = batch_service.run_batch(
                org_id=org_id,
                batch_id=batch_id,
                agent_type=claimed.get("agent_type"),
                concurrency=claimed.get("concurrency"),
                preserve_schedule=True,
            )
            _start_batch_on_voice_server(
                org_id=org_id,
                batch_id=batch_id,
                agent_type=str(run_result.get("agent_type") or claimed.get("agent_type") or ""),
                concurrency=int(run_result.get("concurrency") or claimed.get("concurrency") or 5),
            )
            logger.info("Triggered scheduled batch batch_id=%s org_id=%s", batch_id, org_id)
        except Exception as e:
            logger.exception("Failed to trigger scheduled batch batch_id=%s org_id=%s: %s", batch_id, org_id, e)
            batch_service.mark_batch_start_failure(
                org_id=org_id,
                batch_id=batch_id,
                error_message=f"Failed to start scheduled batch: {str(e)}",
            )


def start_batch_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if _scheduler and _scheduler.running:
            return

        poll_seconds = max(1, int(getattr(settings, "BATCH_SCHEDULER_POLL_SECONDS", 5)))
        scheduler = BackgroundScheduler(timezone="UTC")
        scheduler.add_job(
            _poll_due_batches,
            trigger=IntervalTrigger(seconds=poll_seconds),
            id="batch_due_scheduler",
            max_instances=1,
            replace_existing=True,
            coalesce=True,
        )
        scheduler.start()
        _scheduler = scheduler
        logger.info("Batch scheduler started (APScheduler, interval=%ss)", poll_seconds)


def stop_batch_scheduler() -> None:
    global _scheduler
    with _scheduler_lock:
        if not _scheduler:
            return
        try:
            _scheduler.shutdown(wait=False)
        finally:
            _scheduler = None
        logger.info("Batch scheduler stopped")
