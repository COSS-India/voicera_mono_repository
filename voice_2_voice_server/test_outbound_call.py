#!/usr/bin/env python3
"""Trigger a single VI OBD outbound test call from the terminal.

Usage:
    python test_outbound_call.py 919876543210

Requires VI_OBD_USERNAME, VI_OBD_PASSWORD, and VI_DNI in .env
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.vi_obd_client import (
    FLOW_ID,
    OBD_BASE,
    ViObdClient,
    ViObdError,
    get_dni_from_env,
)


def _print_step(title: str, response: object) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(json.dumps(response, indent=2, default=str))


def main() -> int:
    load_dotenv()

    if len(sys.argv) != 2:
        print(f"Usage: python {Path(__file__).name} <msisdn>", file=sys.stderr)
        return 1

    msisdn = sys.argv[1].strip()
    if not msisdn:
        print("Error: msisdn must not be empty", file=sys.stderr)
        return 1

    try:
        client = ViObdClient.from_env()
        dni = get_dni_from_env()
    except ViObdError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Flow ID: {FLOW_ID}")
    print(f"OBD base: {OBD_BASE}")
    print(f"DNI (from VI_DNI): {dni}")
    print(f"Target msisdn: {msisdn}")

    try:
        token, auth_response = client.get_auth_token()
        _print_step("AUTH — POST obdcampaignapi/AuthToken", auth_response)

        create_response = client.create_campaign(token)
        _print_step("STEP 1 — POST obdcampaignapi/createCampaign", create_response)

        ingest_response, winning_shape = client.upload_call_list_with_fallback(
            token,
            create_response,
            dni,
            msisdn,
        )
        _print_step(
            f"STEP 2 — POST obdcampaignapi/staticCampaignDataIngestion "
            f"(succeeded with payload_shape={winning_shape!r})",
            ingest_response,
        )
        winning_payload = ingest_response.get("_request_payload") or (
            ingest_response.get("_successful_attempt") or {}
        ).get("request_payload")
        if winning_payload:
            print("\nSuccessful request payload:")
            print(json.dumps(winning_payload, indent=2, default=str))

        window = create_response.get("_request_payload", {})
        print(f"\n{'=' * 60}")
        print("SUCCESS")
        print("=" * 60)
        print(f"  Campaign name : {window.get('name')}")
        print(f"  Date window   : {window.get('fromdate')} → {window.get('todate')}")
        print(f"  Time window   : {window.get('fromtime')} → {window.get('totime')} (IST)")
        print(f"  DNI (caller)  : {dni}")
        print(f"  MSISDN (callee): {msisdn}")
        print(f"  campaign_ID   : campainKey (raw)")
        print(f"  campaign_Ref_ID: {create_response.get('campaign_Ref_ID')}")
        print(f"  payload shape : {winning_shape}")
        print(
            f"\nCheck status: python check_campaign_status.py "
            f"{create_response.get('campaign_Ref_ID')}"
        )
        print(
            "\nVI will dial automatically within the time window — "
            "there is no separate start API."
        )
        return 0

    except ViObdError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
