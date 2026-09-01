"""
Unit tests for app.database — connect, disconnect, and get_database().
"""
import pytest
from unittest.mock import MagicMock, patch

from app.database import mongodb, get_database, connect_to_mongo, close_mongo_connection


class TestGetDatabase:
    def test_returns_cached_database_without_reconnect(self):
        original = mongodb.database
        try:
            mock_db = MagicMock()
            mongodb.database = mock_db
            with patch("app.database.connect_to_mongo") as mock_connect:
                result = get_database()
            mock_connect.assert_not_called()
            assert result is mock_db
        finally:
            mongodb.database = original

    def test_calls_connect_when_database_is_none(self):
        original = mongodb.database
        try:
            mongodb.database = None
            mock_db = MagicMock()

            def set_db():
                mongodb.database = mock_db

            with patch("app.database.connect_to_mongo", side_effect=set_db) as mock_connect:
                result = get_database()
            mock_connect.assert_called_once()
            assert result is mock_db
        finally:
            mongodb.database = original


class TestConnectToMongo:
    def test_sets_client_and_database(self):
        original_client = mongodb.client
        original_db = mongodb.database
        try:
            mock_mongo_client = MagicMock()
            mock_mongo_client.admin.command.return_value = {"ok": 1}
            with patch("app.database.MongoClient", return_value=mock_mongo_client):
                connect_to_mongo()
            assert mongodb.client is mock_mongo_client
            assert mongodb.database is not None
        finally:
            mongodb.client = original_client
            mongodb.database = original_db


class TestCloseMongoConnection:
    def test_calls_client_close(self):
        original_client = mongodb.client
        try:
            mock_client = MagicMock()
            mongodb.client = mock_client
            close_mongo_connection()
            mock_client.close.assert_called_once()
        finally:
            mongodb.client = original_client

    def test_noop_when_no_client(self):
        original_client = mongodb.client
        try:
            mongodb.client = None
            close_mongo_connection()  # should not raise
        finally:
            mongodb.client = original_client
