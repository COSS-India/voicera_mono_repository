"""
MongoDB utility functions.
"""
from bson import ObjectId
from typing import Any, Dict, List
import json

def convert_objectid_to_str(obj: Any) -> Any:
    """
    Recursively convert ObjectId to string in MongoDB documents.
    
    Args:
        obj: Object to convert
        
    Returns:
        Object with ObjectIds converted to strings
    """
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    else:
        return obj

def prepare_mongo_response(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare MongoDB document for JSON response.
    
    Args:
        doc: MongoDB document
        
    Returns:
        Document with ObjectIds converted to strings
    """
    if doc is None:
        return None
    return convert_objectid_to_str(doc)

def prepare_mongo_response_list(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Prepare list of MongoDB documents for JSON response.
    
    Args:
        docs: List of MongoDB documents
        
    Returns:
        List of documents with ObjectIds converted to strings
    """
    if docs is None:
        return []
    return [convert_objectid_to_str(doc) for doc in docs]

