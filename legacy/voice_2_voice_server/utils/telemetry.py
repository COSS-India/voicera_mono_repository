"""GPU telemetry API routes."""

import subprocess
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from loguru import logger


router = APIRouter()


def _run_nvidia_smi(query: str) -> list[str]:
    """Run nvidia-smi query and return non-empty CSV rows."""
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _safe_int(value: str) -> Optional[int]:
    try:
        return int(float(value.strip()))
    except Exception:
        return None


def _safe_float(value: str) -> Optional[float]:
    try:
        return float(value.strip())
    except Exception:
        return None


def _collect_gpu_telemetry() -> dict:
    """Collect GPU telemetry from nvidia-smi."""
    gpu_rows = _run_nvidia_smi(
        "gpu=index,uuid,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit"
    )

    gpus = []
    uuid_to_index = {}

    for row in gpu_rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 11:
            continue

        index = _safe_int(parts[0])
        uuid = parts[1]
        model = parts[2]
        utilization = _safe_int(parts[3]) or 0
        mem_utilization = _safe_int(parts[4]) or 0
        memory_total = _safe_int(parts[5]) or 0
        memory_used = _safe_int(parts[6]) or 0
        memory_free = _safe_int(parts[7]) or max(memory_total - memory_used, 0)
        temperature_c = _safe_int(parts[8])
        power_w = _safe_float(parts[9])
        power_limit_w = _safe_float(parts[10])

        if index is None:
            continue

        uuid_to_index[uuid] = index
        gpus.append(
            {
                "index": index,
                "uuid": uuid,
                "model": model,
                "utilization_percent": utilization,
                "memory_utilization_percent": mem_utilization,
                "memory_total_mb": memory_total,
                "memory_used_mb": memory_used,
                "memory_free_mb": memory_free,
                "temperature_c": temperature_c,
                "power_w": power_w,
                "power_limit_w": power_limit_w,
                "processes": [],
            }
        )

    process_rows: list[str] = []
    try:
        process_rows = _run_nvidia_smi(
            "compute-apps=gpu_uuid,pid,process_name,used_memory"
        )
    except Exception:
        # Some systems don't expose process query; return GPU summary anyway.
        process_rows = []

    for row in process_rows:
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 4:
            continue

        gpu_uuid = parts[0]
        gpu_index = uuid_to_index.get(gpu_uuid)
        if gpu_index is None:
            continue

        process = {
            "pid": _safe_int(parts[1]),
            "name": parts[2],
            "memory_mb": _safe_int(parts[3]) or 0,
        }

        for gpu in gpus:
            if gpu["index"] == gpu_index:
                gpu["processes"].append(process)
                break

    return {
        "status": "ok",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_count": len(gpus),
        "gpus": sorted(gpus, key=lambda x: x["index"]),
    }


@router.get("/telemetry/gpu")
async def gpu_telemetry():
    """Return current GPU telemetry for dashboard monitoring."""
    try:
        telemetry = _collect_gpu_telemetry()
        return JSONResponse(status_code=200, content=telemetry)
    except FileNotFoundError:
        return JSONResponse(
            status_code=200,
            content={
                "status": "unavailable",
                "reason": "nvidia-smi not found",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_count": 0,
                "gpus": [],
            },
        )
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ nvidia-smi failed: {e.stderr}")
        return JSONResponse(
            status_code=200,
            content={
                "status": "unavailable",
                "reason": "nvidia-smi query failed",
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_count": 0,
                "gpus": [],
            },
        )
    except Exception as e:
        logger.error(f"❌ GPU telemetry error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "detail": "Failed to collect GPU telemetry"},
        )
