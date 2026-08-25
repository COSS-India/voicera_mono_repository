"""Main entry point for the Vobiz Telephony Server."""

from api.server import run_server

if __name__ == "__main__":
    run_server(host="0.0.0.0", port=7860, log_level="info")
