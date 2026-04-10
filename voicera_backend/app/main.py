"""
Main FastAPI application.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.services.batch_scheduler import start_batch_scheduler, stop_batch_scheduler
from app.routers import users, agents, meetings, campaigns, audience, call_recordings, phone_numbers, vobiz, analytics, integrations, members, knowledge, rag, batches
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Voicera Backend API - MongoDB-based backend service",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(users.router, prefix=settings.API_V1_PREFIX)
app.include_router(agents.router, prefix=settings.API_V1_PREFIX)
app.include_router(meetings.router, prefix=settings.API_V1_PREFIX)
app.include_router(campaigns.router, prefix=settings.API_V1_PREFIX)
app.include_router(audience.router, prefix=settings.API_V1_PREFIX)
app.include_router(call_recordings.router, prefix=settings.API_V1_PREFIX)
app.include_router(phone_numbers.router, prefix=settings.API_V1_PREFIX)
app.include_router(vobiz.router, prefix=settings.API_V1_PREFIX)
app.include_router(analytics.router, prefix=settings.API_V1_PREFIX)
app.include_router(integrations.router, prefix=settings.API_V1_PREFIX)
app.include_router(members.router, prefix=settings.API_V1_PREFIX)
app.include_router(knowledge.router, prefix=settings.API_V1_PREFIX)
app.include_router(rag.router, prefix=settings.API_V1_PREFIX)
app.include_router(batches.router, prefix=settings.API_V1_PREFIX)

@app.on_event("startup")
async def startup_event():
    """Initialize database connection and setup on startup."""
    logger.info("Starting up application...")
    try:
        connect_to_mongo()
        # Initialize database collections and indexes
        from app.database_init import initialize_database
        initialize_database()
        start_batch_scheduler()
        logger.info("Application started successfully")
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Close database connection on shutdown."""
    logger.info("Shutting down application...")
    stop_batch_scheduler()
    close_mongo_connection()
    logger.info("Application shut down successfully")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Voicera Backend API",
        "version": settings.VERSION,
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        from app.database import mongodb
        if mongodb.client:
            mongodb.client.admin.command('ping')
            return {"status": "healthy", "database": "connected"}
        return {"status": "unhealthy", "database": "disconnected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
