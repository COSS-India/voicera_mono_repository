"""Shared test helpers for the voicera_backend test suite."""

from unittest.mock import MagicMock


def make_mock_db(**collections):
    """
    Generic DB mock factory.

    Each keyword argument maps a collection name to a mock object.
    Any key not explicitly listed falls back to a fresh MagicMock so
    callers don't have to care about collections they don't touch.

    Usage::

        db = make_mock_db(AgentConfig=agents_coll, PhoneNumber=phones_coll)
    """
    db = MagicMock()
    db.__getitem__.side_effect = lambda key: collections.get(key, MagicMock())
    return db
