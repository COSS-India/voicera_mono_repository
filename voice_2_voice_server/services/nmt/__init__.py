"""AI4Bharat IndicTrans2 NMT service (hosted on Triton)."""

from .triton_nmt import NmtClient, NmtError, get_nmt_client

__all__ = ["NmtClient", "NmtError", "get_nmt_client"]
