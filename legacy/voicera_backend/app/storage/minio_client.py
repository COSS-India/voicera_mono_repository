"""
MinIO storage client for accessing recordings and transcripts.
"""
import asyncio
import os
from minio import Minio
from minio.error import S3Error
import logging

logger = logging.getLogger(__name__)


class MinIOStorage:
    """MinIO client wrapper for async operations."""
    
    def __init__(self):
        """Initialize MinIO client with environment variables."""
        self.client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=False,
        )
    
    async def get_object(self, bucket_name: str, object_name: str):
        """
        Get object from MinIO (async wrapper).
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            
        Returns:
            MinIO response object for streaming
        """
        return await asyncio.to_thread(
            self.client.get_object,
            bucket_name,
            object_name
        )
    
    def object_exists(self, bucket_name: str, object_name: str) -> bool:
        """
        Check if an object exists in MinIO.
        
        Args:
            bucket_name: Name of the bucket
            object_name: Name of the object
            
        Returns:
            True if object exists, False otherwise
        """
        try:
            self.client.stat_object(bucket_name, object_name)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise
    
    def parse_minio_url(self, url: str) -> tuple[str, str] | None:
        """
        Parse minio:// URL to extract bucket and object name.
        
        Args:
            url: URL in format minio://bucket/object_name
            
        Returns:
            Tuple of (bucket_name, object_name) or None if invalid
        """
        if not url or not url.startswith("minio://"):
            return None
        
        # Remove minio:// prefix
        path = url.replace("minio://", "")
        
        # Split into bucket and object
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        
        bucket_name, object_name = parts
        return bucket_name, object_name
