#!/usr/bin/env python3
"""Unit tests for VI OBD client and WebSocket agent routing (mocked — no live calls)."""

import os
import sys
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

# backend_utils pulls MinIO at import time; stub for routing unit tests.
sys.modules.setdefault("minio", MagicMock())
_minio_client = MagicMock()
_minio_client.MinIOStorage = MagicMock
sys.modules.setdefault("storage.minio_client", _minio_client)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.vi_obd_client import (
    OBD_BASE,
    ViObdClient,
    ViObdError,
    _normalize_cpaas_base,
    get_dni_from_env,
    get_flow_id,
    normalize_msisdn,
    resolve_vi_agent_id,
    resolve_vi_agent_id_full,
)


class TestViObdClient(unittest.TestCase):
    def setUp(self):
        self.client = ViObdClient(username="user", password="pass")

    def test_normalize_cpaas_base_fixes_missing_cpaas(self):
        bad = "https://cts.myvi.in:8443/api/v1/obdcampaignapi"
        fixed = _normalize_cpaas_base(bad, OBD_BASE)
        self.assertEqual(fixed, OBD_BASE)

    def test_get_dni_from_env_raises_when_unset(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ViObdError) as ctx:
                get_dni_from_env()
            self.assertIn("Missing VI_DNI", str(ctx.exception))

    def test_get_dni_from_env_returns_value(self):
        with patch.dict(os.environ, {"VI_DNI": "919811111111"}):
            self.assertEqual(get_dni_from_env(), "919811111111")

    def test_build_ingestion_payload_nested(self):
        payload = ViObdClient._build_ingestion_payload(
            "abc123==", "919811111111", "919876543210", shape="nested"
        )
        self.assertEqual(
            payload,
            {
                "campaign_ID": "abc123==",
                "Records": [{"dni": "919811111111", "msisdn": "919876543210"}],
            },
        )

    def test_ingestion_succeeded_accepts_rows_affected(self):
        self.assertTrue(
            ViObdClient._ingestion_succeeded({"data": {"rowsAffected": 1}}, 200)
        )
        self.assertFalse(
            ViObdClient._ingestion_succeeded({"data": {"rowsAffected": 0}}, 200)
        )

    def test_upload_call_list_with_fallback_nested_succeeds(self):
        create_response = {"status": 1, "campainKey": "abc123==", "campaign_Ref_ID": 42}
        success_body = {"data": {"rowsAffected": 1}}

        with patch.object(
            self.client,
            "upload_call_list",
            return_value=(success_body, 200),
        ) as mock_upload:
            body, shape = self.client.upload_call_list_with_fallback(
                "token", create_response, "919811111111", "919876543210"
            )

        mock_upload.assert_called_once_with(
            "token",
            "abc123==",
            "919811111111",
            "919876543210",
            payload_shape="nested",
        )
        self.assertEqual(shape, "nested")
        self.assertEqual(body["data"]["rowsAffected"], 1)

    def test_upload_call_list_with_fallback_tries_both_on_409(self):
        create_response = {"status": 1, "campainKey": "abc123=="}
        fail_body = {"message": "dni is missing in Records array JSON"}
        success_body = {"data": {"rowsAffected": 1}}

        def side_effect(token, campaign_id, dni, msisdn, payload_shape="nested"):
            if payload_shape == "nested":
                return (fail_body, 409)
            return (success_body, 200)

        with patch.object(self.client, "upload_call_list", side_effect=side_effect):
            body, shape = self.client.upload_call_list_with_fallback(
                "token", create_response, "919811111111", "919876543210"
            )

        self.assertEqual(shape, "nested_and_top_level")
        self.assertEqual(body["data"]["rowsAffected"], 1)

    def test_auth_token_caches(self):
        with patch.object(
            self.client,
            "_request",
            return_value=({"idToken": "obd-jwt", "expiresIn": 86400}, 200),
        ) as mock_req:
            token1, _ = self.client.get_auth_token()
            token2, resp2 = self.client.get_auth_token()
        self.assertEqual(token1, "obd-jwt")
        self.assertEqual(token2, "obd-jwt")
        self.assertTrue(resp2.get("_cached"))
        mock_req.assert_called_once()

    def test_get_campaign_status_with_fallback_uses_ref_id_first(self):
        success_body = {
            "data": {
                "currentStatus": "Running",
                "baseDetails": {"pending": 1},
            }
        }

        with patch.object(
            self.client,
            "get_campaign_status",
            return_value=(success_body, 200),
        ) as mock_status:
            body, kind = self.client.get_campaign_status_with_fallback(
                "token", "715665"
            )

        mock_status.assert_called_once_with(
            "token", 715665, id_kind="campaign_Ref_ID"
        )
        self.assertEqual(kind, "campaign_Ref_ID")
        self.assertEqual(body["data"]["currentStatus"], "Running")

    def test_get_campaign_status_with_fallback_tries_campain_key(self):
        fail_body = {"message": "bad id"}
        success_body = {"data": {"currentStatus": "Completed", "baseDetails": {}}}

        def side_effect(token, campaign_id, id_kind="campaign_Ref_ID"):
            if id_kind == "campaign_Ref_ID":
                return (fail_body, 400)
            return (success_body, 200)

        with patch.object(self.client, "get_campaign_status", side_effect=side_effect):
            body, kind = self.client.get_campaign_status_with_fallback(
                "token", "715665", campain_key="abc123=="
            )

        self.assertEqual(kind, "campainKey")
        self.assertEqual(body["data"]["currentStatus"], "Completed")

    def test_default_base_includes_cpaas(self):
        self.assertIn("/Cpaas/api/v1/obdcampaignapi", OBD_BASE)

    def test_normalize_msisdn_strips_country_code(self):
        self.assertEqual(normalize_msisdn("+919876543210"), "9876543210")
        self.assertEqual(normalize_msisdn("919876543210"), "9876543210")

    def test_get_flow_id_uses_env_override(self):
        with patch.dict(os.environ, {"VI_FLOW_ID": "custom-flow=="}):
            self.assertEqual(get_flow_id(), "custom-flow==")

    def test_get_active_dni_success(self):
        with patch.object(
            self.client,
            "_request",
            return_value=({"dniList": ["9899384310", "9899383920"]}, 200),
        ):
            body = self.client.get_active_dni("token", "flow123==")
        self.assertEqual(body["dniList"], ["9899384310", "9899383920"])

    def test_get_active_dni_raises_on_error(self):
        with patch.object(
            self.client,
            "_request",
            return_value=({"message": "Unauthorized"}, 401),
        ):
            with self.assertRaises(ViObdError):
                self.client.get_active_dni("token", "flow123==")

    def test_resolve_dni_uses_api_first(self):
        with patch.object(
            self.client,
            "get_active_dni",
            return_value={"dniList": ["9899384310", "9899383920"]},
        ):
            dni, source, dni_list = self.client.resolve_dni("token")
        self.assertEqual(dni, "9899384310")
        self.assertEqual(source, "getActiveDNIList")
        self.assertEqual(len(dni_list), 2)

    def test_resolve_dni_falls_back_to_env(self):
        with patch.object(
            self.client,
            "get_active_dni",
            side_effect=ViObdError("lookup failed"),
        ):
            with patch.dict(os.environ, {"VI_DNI": "9769554706"}):
                dni, source, dni_list = self.client.resolve_dni("token")
        self.assertEqual(dni, "9769554706")
        self.assertEqual(source, "VI_DNI")
        self.assertEqual(dni_list, ["9769554706"])

    def test_modify_campaign_accepts_207(self):
        with patch.object(
            self.client,
            "_request",
            return_value=({"data": {"status": "partial_success"}}, 207),
        ):
            body = self.client.modify_campaign(
                "token",
                "abc123==",
                from_date="2026-08-17",
                to_date="2026-08-17",
                from_time="12:00:00",
                to_time="13:00:00",
            )
        self.assertEqual(body["_http_status"], 207)

    def test_upload_call_list_bulk_single_chunk(self):
        with patch.object(
            self.client,
            "_request",
            return_value=({"data": {"rowsAffected": 2}}, 200),
        ) as mock_req:
            result = self.client.upload_call_list_bulk(
                "token", "abc123==", "9769554706", ["9876543210", "+919123456789"]
            )
        self.assertEqual(result["data"]["rowsAffected"], 2)
        mock_req.assert_called_once()
        payload = mock_req.call_args.kwargs["json_body"]
        self.assertEqual(len(payload["Records"]), 2)
        self.assertEqual(payload["Records"][0]["msisdn"], "9876543210")


class TestResolveViAgentIdSync(unittest.TestCase):
    def test_path_agent_id_takes_priority(self):
        start = {"custom_parameters": {"agent_id": "from_custom"}}
        self.assertEqual(resolve_vi_agent_id("from_path", start), "from_path")

    def test_custom_parameters_agent_id(self):
        start = {"custom_parameters": {"agent_id": "mahavistaar"}}
        self.assertEqual(resolve_vi_agent_id(None, start), "mahavistaar")

    def test_custom_parameters_agentId_alias(self):
        start = {"custom_parameters": {"agentId": "agent_b"}}
        self.assertEqual(resolve_vi_agent_id(None, start), "agent_b")

    def test_returns_none_when_unresolved(self):
        self.assertIsNone(resolve_vi_agent_id(None, {}))


class TestResolveViAgentIdFull(unittest.IsolatedAsyncioTestCase):
    async def test_two_dnis_route_to_two_agents(self):
        """Inbound /vi/stream: DNI A → agent A, DNI B → agent B."""
        agents_by_dni = {
            "9769554706": {"agent_id": "agent_a", "agent_type": "Agent A"},
            "9876543210": {"agent_id": "agent_b", "agent_type": "Agent B"},
        }

        async def mock_fetch(phone):
            return agents_by_dni.get(phone)

        with patch("utils.backend_utils.fetch_agent_by_phone_number", side_effect=mock_fetch):
            agent_id, source = await resolve_vi_agent_id_full(
                None, {"dni": "9769554706"}
            )
            self.assertEqual(agent_id, "agent_a")
            self.assertIn("dni", source)

            agent_id_b, source_b = await resolve_vi_agent_id_full(
                None, {"dni": "9876543210"}
            )
            self.assertEqual(agent_id_b, "agent_b")
            self.assertIn("dni", source_b)

    async def test_cli_fallback_when_dni_missing(self):
        async def mock_fetch(phone):
            if phone == "919000000001":
                return {"agent_id": "agent_cli"}
            return None

        with patch("utils.backend_utils.fetch_agent_by_phone_number", side_effect=mock_fetch):
            agent_id, source = await resolve_vi_agent_id_full(
                None, {"cli": "919000000001"}
            )
            self.assertEqual(agent_id, "agent_cli")
            self.assertIn("cli", source)

    async def test_path_beats_dni_lookup(self):
        with patch(
            "utils.backend_utils.fetch_agent_by_phone_number",
            new_callable=AsyncMock,
        ) as mock_fetch:
            agent_id, source = await resolve_vi_agent_id_full(
                "path_agent", {"dni": "9769554706"}
            )
            self.assertEqual(agent_id, "path_agent")
            self.assertEqual(source, "path")
            mock_fetch.assert_not_called()

    async def test_env_default_when_lookup_fails(self):
        with patch(
            "utils.backend_utils.fetch_agent_by_phone_number",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch.dict(os.environ, {"VI_DEFAULT_AGENT_ID": "fallback_agent"}):
                agent_id, source = await resolve_vi_agent_id_full(
                    None, {"dni": "0000000000"}
                )
                self.assertEqual(agent_id, "fallback_agent")
                self.assertEqual(source, "VI_DEFAULT_AGENT_ID")

    async def test_returns_none_when_all_sources_fail(self):
        with patch(
            "utils.backend_utils.fetch_agent_by_phone_number",
            new_callable=AsyncMock,
            return_value=None,
        ):
            with patch.dict(os.environ, {}, clear=True):
                os.environ.pop("VI_DEFAULT_AGENT_ID", None)
                agent_id, source = await resolve_vi_agent_id_full(None, {})
                self.assertIsNone(agent_id)
                self.assertEqual(source, "")


if __name__ == "__main__":
    raise SystemExit(unittest.main())
