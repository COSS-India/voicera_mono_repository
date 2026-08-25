"""Batch calling worker and routes for voice server."""

from __future__ import annotations

import os
import threading
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import Any, Awaitable, Callable, Dict, Optional

from fastapi import APIRouter, HTTPException
from loguru import logger
from pydantic import BaseModel, Field
import requests

from .backend_utils import (
    claim_next_batch_contact,
    fetch_batch_agent_call_config,
    finalize_batch_execution,
    report_batch_contact_result,
)


class BatchRunRequest(BaseModel):
    org_id: str
    batch_id: str
    agent_type: str
    concurrency: int = Field(default=5, ge=1, le=20)


class BatchStopRequest(BaseModel):
    org_id: str
    batch_id: str


class BatchWorker:
    def __init__(
        self,
    ) -> None:
        self._lock = threading.Lock()
        self._runners: Dict[str, Dict[str, Any]] = {}

    def run(
        self,
        org_id: str,
        batch_id: str,
        agent_type: str,
        concurrency: int,
    ) -> Dict[str, str]:
        with self._lock:
            existing = self._runners.get(batch_id)
            if existing and existing["thread"].is_alive():
                raise HTTPException(status_code=400, detail="Batch is already running")

            stop_event = threading.Event()
            worker = threading.Thread(
                target=self._run_worker,
                args=(org_id, batch_id, agent_type, concurrency, stop_event),
                daemon=True,
                name=f"voice-batch-{batch_id[:8]}",
            )
            self._runners[batch_id] = {
                "thread": worker,
                "stop_event": stop_event,
                "concurrency": concurrency,
            }
            worker.start()

        return {
            "status": "success",
            "message": f"Batch worker started with concurrency {concurrency}",
        }

    def stop(self, batch_id: str) -> Dict[str, str]:
        with self._lock:
            existing = self._runners.get(batch_id)
            if not existing or not existing["thread"].is_alive():
                raise HTTPException(status_code=400, detail="Batch is not running")
            existing["stop_event"].set()

        return {"status": "success", "message": "Batch worker stop requested"}

    def _run_worker(
        self,
        org_id: str,
        batch_id: str,
        agent_type: str,
        concurrency: int,
        stop_event: threading.Event,
    ) -> None:
        agent_call_config = fetch_batch_agent_call_config(org_id=org_id, agent_type=agent_type)
        if not agent_call_config:
            finalize_batch_execution(org_id=org_id, batch_id=batch_id, stopped=False)
            self._cleanup_runner(batch_id)
            return

        agent_id = str(agent_call_config.get("agent_id") or "").strip()
        caller_id = agent_call_config.get("caller_id")
        if not agent_id:
            finalize_batch_execution(org_id=org_id, batch_id=batch_id, stopped=False)
            self._cleanup_runner(batch_id)
            return

        def place_call(customer_number: str) -> tuple[bool, Optional[str]]:
            try:
                response = requests.post(
                    f"{_get_voice_server_internal_url()}/outbound/call/",
                    json={
                        "customer_number": customer_number,
                        "agent_id": agent_id,
                        "caller_id": caller_id,
                    },
                    timeout=30,
                )
                response.raise_for_status()
                return True, None
            except Exception as e:
                return False, str(e)

        try:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                in_flight: Dict[Future[tuple[bool, Optional[str]]], int] = {}
                stopped = False

                while True:
                    while not stop_event.is_set() and len(in_flight) < concurrency:
                        contact = claim_next_batch_contact(org_id=org_id, batch_id=batch_id)
                        if not contact:
                            break
                        row_number = int(contact.get("row_number"))
                        customer_number = str(contact.get("contact_number") or "")
                        if not customer_number:
                            report_batch_contact_result(
                                org_id=org_id,
                                batch_id=batch_id,
                                row_number=row_number,
                                ok=False,
                                error="Missing contact_number",
                            )
                            continue
                        future = executor.submit(place_call, customer_number)
                        in_flight[future] = row_number

                    if stop_event.is_set():
                        stopped = True

                    if not in_flight:
                        finalize_batch_execution(
                            org_id=org_id,
                            batch_id=batch_id,
                            stopped=stopped,
                        )
                        return

                    done, _ = wait(in_flight.keys(), timeout=1.0, return_when=FIRST_COMPLETED)
                    for future in done:
                        row_number = in_flight.pop(future)
                        try:
                            ok, error = future.result()
                        except Exception as e:
                            ok, error = False, str(e)

                        report_batch_contact_result(
                            org_id=org_id,
                            batch_id=batch_id,
                            row_number=row_number,
                            ok=ok,
                            error=error,
                        )
        finally:
            self._cleanup_runner(batch_id)

    def _cleanup_runner(self, batch_id: str) -> None:
        with self._lock:
            self._runners.pop(batch_id, None)


def _get_voice_server_internal_url() -> str:
    return os.getenv("VOICE_SERVER_INTERNAL_URL", "http://127.0.0.1:7860").rstrip("/")


def create_batch_router(
    _: Optional[Callable[..., Awaitable[dict]]] = None,
) -> APIRouter:
    worker = BatchWorker()
    router = APIRouter()

    @router.post("/outbound/batch/run/")
    async def run_batch_calls(request: BatchRunRequest):
        return worker.run(
            org_id=request.org_id,
            batch_id=request.batch_id,
            agent_type=request.agent_type,
            concurrency=request.concurrency,
        )

    @router.post("/outbound/batch/stop/")
    async def stop_batch_calls(request: BatchStopRequest):
        return worker.stop(batch_id=request.batch_id)

    logger.info("✅ Batch routes initialized")
    return router
