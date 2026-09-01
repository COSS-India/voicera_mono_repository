import pytest
import subprocess
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from utils.telemetry import router, _safe_int, _safe_float, _run_nvidia_smi, _collect_gpu_telemetry

app = FastAPI()
app.include_router(router)
client = TestClient(app)

class TestTelemetryUtils:
    def test_safe_int(self):
        assert _safe_int("42") == 42
        assert _safe_int(" 42.5 ") == 42
        assert _safe_int("invalid") is None
        assert _safe_int("") is None

    def test_safe_float(self):
        assert _safe_float("42.5") == 42.5
        assert _safe_float(" 42 ") == 42.0
        assert _safe_float("invalid") is None
        assert _safe_float("") is None

    @patch("subprocess.run")
    def test_run_nvidia_smi_success(self, mock_run):
        mock_result = MagicMock()
        mock_result.stdout = "line1 \n line2\n  \nline3"
        mock_run.return_value = mock_result
        
        result = _run_nvidia_smi("test")
        assert result == ["line1", "line2", "line3"]
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        assert "nvidia-smi" in args[0]
        assert "--query-test" in args[0]


class TestCollectGpuTelemetry:
    @patch("utils.telemetry._run_nvidia_smi")
    def test_collect_success(self, mock_run_smi):
        def mock_smi_side_effect(query):
            if "gpu=" in query:
                # index,uuid,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,temperature.gpu,power.draw,power.limit
                return ["0, GPU-123, RTX 4090, 50, 20, 24000, 12000, 12000, 65, 200.5, 450.0"]
            elif "compute-apps=" in query:
                # gpu_uuid,pid,process_name,used_memory
                return ["GPU-123, 1234, python, 4000", "GPU-999, 9999, python, 100"] # Includes a non-matching UUID
            return []
            
        mock_run_smi.side_effect = mock_smi_side_effect
        
        result = _collect_gpu_telemetry()
        
        assert result["status"] == "ok"
        assert result["gpu_count"] == 1
        
        gpu = result["gpus"][0]
        assert gpu["index"] == 0
        assert gpu["uuid"] == "GPU-123"
        assert gpu["model"] == "RTX 4090"
        assert gpu["utilization_percent"] == 50
        assert gpu["memory_used_mb"] == 12000
        assert gpu["temperature_c"] == 65
        assert gpu["power_w"] == 200.5
        
        assert len(gpu["processes"]) == 1
        proc = gpu["processes"][0]
        assert proc["pid"] == 1234
        assert proc["name"] == "python"
        assert proc["memory_mb"] == 4000

    @patch("utils.telemetry._run_nvidia_smi")
    def test_collect_missing_fields_and_invalid_index(self, mock_run_smi):
        def mock_smi_side_effect(query):
            if "gpu=" in query:
                # 1 invalid row (too short), 1 row with no index, 1 valid row with missing optionals
                return [
                    "0, GPU-123, RTX", # Too short
                    "invalid, GPU-456, RTX 3080, 0, 0, 10000, 1000, 9000, 40, 50.0, 300.0", # invalid index
                    "1, GPU-789, RTX 4080, , , , , , , , " # missing data
                ]
            elif "compute-apps=" in query:
                return ["GPU-789, 123"] # Too short process row
            return []
            
        mock_run_smi.side_effect = mock_smi_side_effect
        
        result = _collect_gpu_telemetry()
        
        assert result["status"] == "ok"
        assert result["gpu_count"] == 1 # Only GPU-789 makes it
        
        gpu = result["gpus"][0]
        assert gpu["index"] == 1
        assert gpu["utilization_percent"] == 0
        assert gpu["power_w"] is None
        assert len(gpu["processes"]) == 0

    @patch("utils.telemetry._run_nvidia_smi")
    def test_collect_process_query_fails(self, mock_run_smi):
        def mock_smi_side_effect(query):
            if "gpu=" in query:
                return ["0, GPU-123, RTX 4090, 50, 20, 24000, 12000, 12000, 65, 200.5, 450.0"]
            elif "compute-apps=" in query:
                raise Exception("Query not supported")
            
        mock_run_smi.side_effect = mock_smi_side_effect
        
        result = _collect_gpu_telemetry()
        assert result["status"] == "ok"
        assert len(result["gpus"][0]["processes"]) == 0


class TestGpuTelemetryEndpoint:
    @patch("utils.telemetry._collect_gpu_telemetry")
    def test_endpoint_success(self, mock_collect):
        mock_collect.return_value = {
            "status": "ok",
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "gpu_count": 1,
            "gpus": []
        }
        
        response = client.get("/telemetry/gpu")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @patch("utils.telemetry._collect_gpu_telemetry")
    def test_endpoint_file_not_found(self, mock_collect):
        mock_collect.side_effect = FileNotFoundError()
        
        response = client.get("/telemetry/gpu")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert "not found" in data["reason"]

    @patch("utils.telemetry._collect_gpu_telemetry")
    def test_endpoint_called_process_error(self, mock_collect):
        error = subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="NVIDIA-SMI has failed")
        mock_collect.side_effect = error
        
        response = client.get("/telemetry/gpu")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unavailable"
        assert "query failed" in data["reason"]

    @patch("utils.telemetry._collect_gpu_telemetry")
    def test_endpoint_generic_exception(self, mock_collect):
        mock_collect.side_effect = Exception("Unknown error")
        
        response = client.get("/telemetry/gpu")
        assert response.status_code == 500
        assert response.json()["status"] == "error"
