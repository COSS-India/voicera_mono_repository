"""
Configuration management for the application.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

class Settings:
    """Application settings loaded from environment variables."""
    
    # MongoDB Configuration
    MONGODB_HOST: str = os.getenv("MONGODB_HOST", "localhost")
    MONGODB_PORT: int = int(os.getenv("MONGODB_PORT", "27017"))
    MONGODB_USER: str = os.getenv("MONGODB_USER", "admin")
    MONGODB_PASSWORD: str = os.getenv("MONGODB_PASSWORD", "admin123")
    MONGODB_DATABASE: str = os.getenv("MONGODB_DATABASE", "voicera")
    MONGODB_AUTH_SOURCE: str = os.getenv("MONGODB_AUTH_SOURCE", "admin")
    
    # Application Configuration
    API_V1_PREFIX: str = "/api/v1"
    PROJECT_NAME: str = "Voicera Backend API"
    VERSION: str = "1.0.0"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")  # Should be set in .env for production
    
    # Mailtrap Configuration
    MAILTRAP_API_TOKEN: str = os.getenv("MAILTRAP_API_TOKEN", "")
    MAILTRAP_FROM_EMAIL: str = os.getenv("MAILTRAP_FROM_EMAIL", "noreply@voicera.com")
    MAILTRAP_FROM_NAME: str = os.getenv("MAILTRAP_FROM_NAME", "Voicera")
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:3000")  # For reset password link
    
    # Internal API Key for service-to-service communication (bot -> backend)
    INTERNAL_API_KEY: str = os.getenv("INTERNAL_API_KEY", "")
    
    # RAG / Knowledge base — Chroma persistence (default: voicera_backend/rag_system/chroma_data)
    CHROMA_BASE_DIR: str = os.getenv(
        "CHROMA_BASE_DIR",
        str(Path(__file__).resolve().parent.parent / "rag_system" / "chroma_data"),
    )

    # Vobiz API Configuration
    VOBIZ_API_BASE_URL: str = os.getenv("VOBIZ_API_BASE_URL", "https://api.vobiz.ai/api/v1")
    VOBIZ_ACCOUNT_ID: str = os.getenv("VOBIZ_ACCOUNT_ID", "")
    VOBIZ_AUTH_ID: str = os.getenv("VOBIZ_AUTH_ID", "")
    VOBIZ_AUTH_TOKEN: str = os.getenv("VOBIZ_AUTH_TOKEN", "")
    VOICE_SERVER_URL: str = (
        os.getenv("VOICE_SERVER_URL")
        or os.getenv("JOHNAIC_SERVER_URL")
        or "http://localhost:7860"
    )
    BATCH_SCHEDULER_POLL_SECONDS: int = int(os.getenv("BATCH_SCHEDULER_POLL_SECONDS", "5"))
    
    @property
    def mongodb_uri(self) -> str:
        """Build MongoDB connection URI."""
        return (
            f"mongodb://{self.MONGODB_USER}:{self.MONGODB_PASSWORD}"
            f"@{self.MONGODB_HOST}:{self.MONGODB_PORT}/{self.MONGODB_DATABASE}"
            f"?authSource={self.MONGODB_AUTH_SOURCE}"
        )

# Global settings instance
settings = Settings()
