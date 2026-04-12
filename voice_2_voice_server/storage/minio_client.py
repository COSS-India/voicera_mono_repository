# storage/minio_client.py
import asyncio
import io
import os
import wave
from minio import Minio
from minio.error import S3Error
from loguru import logger


def _get_env_or_raise(key: str) -> str:
    """Get environment variable or raise ValueError."""
    value = os.environ.get(key)
    if not value:
        raise ValueError(f"Missing required environment variable: {key}")
    return value


class MinIOStorage:
    """MinIO storage client for saving recordings and transcripts.
    
    This class provides async methods for storing and retrieving audio recordings
    and transcripts from MinIO object storage.
    """
    
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        secure: bool = False
    ):
        """Initialize MinIO storage client.
        
        Args:
            endpoint: MinIO server endpoint (e.g., "localhost:9000")
            access_key: MinIO access key
            secret_key: MinIO secret key
            secure: Whether to use secure connection (HTTPS)
        """
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_buckets()
    
    @classmethod
    def from_env(cls) -> "MinIOStorage":
        """Create MinIOStorage instance from environment variables.
        
        Reads the following environment variables:
        - MINIO_ENDPOINT (required)
        - MINIO_ACCESS_KEY (required)
        - MINIO_SECRET_KEY (required)
        - MINIO_SECURE (optional, defaults to False)
        
        Returns:
            MinIOStorage instance configured from environment variables
            
        Raises:
            ValueError: If required environment variables are missing
        """
        endpoint = _get_env_or_raise("MINIO_ENDPOINT")
        access_key = _get_env_or_raise("MINIO_ACCESS_KEY")
        secret_key = _get_env_or_raise("MINIO_SECRET_KEY")
        secure = os.getenv("MINIO_SECURE", "false").lower() in ("true", "1", "yes")
        
        return cls(
            endpoint=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )

    def _ensure_buckets(self):
        """Create buckets if they don't exist."""
        for bucket in ["recordings", "transcripts"]:
            if not self.client.bucket_exists(bucket):
                self.client.make_bucket(bucket)
                logger.info(f"Created bucket: {bucket}")

    async def save_recording(self, call_sid: str, audio_data: bytes, sample_rate: int, num_channels: int) -> str:
        """Save audio recording to MinIO."""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        
        buffer.seek(0)
        object_name = f"{call_sid}.wav"
        buffer_size = buffer.getbuffer().nbytes
        
        # Run blocking MinIO operation in thread pool to avoid blocking event loop
        await asyncio.to_thread(
            self.client.put_object,
            bucket_name="recordings",
            object_name=object_name,
            data=buffer,
            length=buffer_size,
            content_type="audio/wav",
        )
        logger.info(f"Saved recording: minio://recordings/{object_name}")
        return object_name

    async def append_transcript(self, call_sid: str, line: str) -> str:
        """Append line to transcript file."""
        object_name = f"{call_sid}.txt"
        
        # Read existing content (if any) in thread pool
        existing = ""
        try:
            response = await asyncio.to_thread(
                self.client.get_object,
                "transcripts",
                object_name
            )
            existing = response.read().decode("utf-8")
            response.close()
            response.release_conn()
        except S3Error as e:
            if e.code != "NoSuchKey":
                raise
        
        content = existing + line + "\n"
        data = content.encode("utf-8")
        data_buffer = io.BytesIO(data)
        
        # Write updated content in thread pool
        await asyncio.to_thread(
            self.client.put_object,
            bucket_name="transcripts",
            object_name=object_name,
            data=data_buffer,
            length=len(data),
            content_type="text/plain",
        )
        return object_name
    
    async def save_recording_from_chunks(
        self, 
        call_sid: str, 
        audio_chunks: list, 
        sample_rate: int, 
        num_channels: int
    ) -> str:
        """Save complete audio recording from accumulated chunks.
        
        Args:
            call_sid: Call identifier
            audio_chunks: List of audio data chunks (bytes)
            sample_rate: Audio sample rate
            num_channels: Number of audio channels
            
        Returns:
            Object name of saved recording
        """
        if not audio_chunks:
            logger.warning(f"No audio chunks to save for {call_sid}")
            return None
        
        # Concatenate all audio chunks
        audio_data = b''.join(audio_chunks)
        
        # Save as single WAV file
        return await self.save_recording(call_sid, audio_data, sample_rate, num_channels)
    
    async def save_transcript_from_lines(self, call_sid: str, transcript_lines: list) -> str:
        """Save complete transcript from accumulated lines.
        
        Args:
            call_sid: Call identifier
            transcript_lines: List of transcript lines (strings)
            
        Returns:
            Object name of saved transcript
        """
        if not transcript_lines:
            logger.warning(f"No transcript lines to save for {call_sid}")
            return None
        
        # Join all lines with newlines
        content = '\n'.join(transcript_lines) + '\n'
        data = content.encode("utf-8")
        data_buffer = io.BytesIO(data)
        
        object_name = f"{call_sid}.txt"
        
        # Write complete transcript in thread pool
        await asyncio.to_thread(
            self.client.put_object,
            bucket_name="transcripts",
            object_name=object_name,
            data=data_buffer,
            length=len(data),
            content_type="text/plain",
        )
        logger.info(f"Saved transcript: minio://transcripts/{object_name}")
        return object_name
    
    async def get_object(self, bucket_name: str, object_name: str):
        """Get object from MinIO (async wrapper)."""
        return await asyncio.to_thread(
            self.client.get_object,
            bucket_name,
            object_name
        )

    @staticmethod
    def parse_minio_url(url: str) -> tuple[str, str] | None:
        """Parse minio://bucket/object URL."""
        if not url or not url.startswith("minio://"):
            return None
        path = url.replace("minio://", "", 1)
        parts = path.split("/", 1)
        if len(parts) != 2:
            return None
        return parts[0], parts[1]
    
    
