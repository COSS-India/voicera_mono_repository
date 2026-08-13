#!/usr/bin/env python3
"""Check VI OBD campaign status.

Usage:
    python check_campaign_status.py <campaign_Ref_ID>
    python check_campaign_status.py 715665
    python check_campaign_status.py 715665 sEfUP5Dy25hJD7gRo36BEQ==

Optional second argument: campainKey fallback if campaign_Ref_ID returns 400/406.

Requires VI_OBD_USERNAME and VI_OBD_PASSWORD in .env
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.vi_obd_client import OBD_BASE, ViObdClient, ViObdError


def _print_step(title: str, response: object) -> None:
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)
    print(json.dumps(response, indent=2, default=str))


def _highlight_status(body: dict) -> None:
    data = body.get("data") or {}
    current_status = data.get("currentStatus")
    base_details = data.get("baseDetails")

    print(f"\n{'=' * 60}")
    print("HIGHLIGHT — currentStatus")
    print("=" * 60)
    print(json.dumps(current_status, indent=2, default=str))

    print(f"\n{'=' * 60}")
    print("HIGHLIGHT — baseDetails")
    print("=" * 60)
    print(json.dumps(base_details, indent=2, default=str))


def main() -> int:
    load_dotenv()

    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: python {Path(__file__).name} <campaign_Ref_ID> [campainKey]",
            file=sys.stderr,
        )
        return 1

    campaign_ref_id = sys.argv[1].strip()
    campain_key = sys.argv[2].strip() if len(sys.argv) == 3 else None

    if not campaign_ref_id:
        print("Error: campaign_Ref_ID must not be empty", file=sys.stderr)
        return 1

    try:
        client = ViObdClient.from_env()
    except ViObdError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"OBD base: {OBD_BASE}")
    print(f"campaign_Ref_ID: {campaign_ref_id}")
    if campain_key:
        print(f"campainKey fallback: {campain_key}")

    try:
        token, auth_response = client.get_auth_token()
        _print_step("AUTH — POST obdcampaignapi/AuthToken", auth_response)

        status_response, winning_id_kind = client.get_campaign_status_with_fallback(
            token,
            campaign_ref_id,
            campain_key=campain_key,
        )
        _print_step(
            f"POST obdcampaignapi/campaignstatus (succeeded with {winning_id_kind})",
            status_response,
        )

        _highlight_status(status_response)

        winning = status_response.get("_successful_attempt") or {}
        if winning.get("request_payload"):
            print(f"\nSuccessful request payload:")
            print(json.dumps(winning["request_payload"], indent=2, default=str))

        return 0

    except ViObdError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
