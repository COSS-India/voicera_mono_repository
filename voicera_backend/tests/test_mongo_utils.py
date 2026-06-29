"""Tests for app/utils/mongo_utils.py"""

from bson import ObjectId
from app.utils.mongo_utils import convert_objectid_to_str, prepare_mongo_response, prepare_mongo_response_list


class TestConvertObjectIdToStr:
    def test_objectid_converted(self):
        oid = ObjectId()
        result = convert_objectid_to_str(oid)
        assert result == str(oid)

    def test_dict_with_objectid(self):
        oid = ObjectId()
        result = convert_objectid_to_str({"_id": oid, "name": "test"})
        assert result == {"_id": str(oid), "name": "test"}

    def test_list_with_objectid(self):
        oid = ObjectId()
        result = convert_objectid_to_str([oid, "plain"])
        assert result == [str(oid), "plain"]

    def test_plain_value_passthrough(self):
        assert convert_objectid_to_str("hello") == "hello"
        assert convert_objectid_to_str(42) == 42
        assert convert_objectid_to_str(None) is None


class TestPrepareMongoResponse:
    def test_none_returns_none(self):
        assert prepare_mongo_response(None) is None

    def test_doc_with_objectid(self):
        oid = ObjectId()
        result = prepare_mongo_response({"_id": oid})
        assert result["_id"] == str(oid)


class TestPrepareMongoResponseList:
    def test_none_returns_empty_list(self):
        assert prepare_mongo_response_list(None) == []

    def test_list_converted(self):
        oid = ObjectId()
        result = prepare_mongo_response_list([{"_id": oid}])
        assert result[0]["_id"] == str(oid)

