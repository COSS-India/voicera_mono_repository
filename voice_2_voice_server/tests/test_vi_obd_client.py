#!/usr/bin/env python3
"""Unit tests for VI OBD client (mocked HTTP — no live calls)."""

import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.vi_obd_client import (
    OBD_BASE,
    ViObdClient,
    ViObdError,
    _normalize_cpaas_base,
    get_dni_from_env,
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


if __name__ == "__main__":
    raise SystemExit(unittest.main())
