# Voicera Backend

FastAPI-based backend service using MongoDB.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose (for MongoDB)
- pip

## Quick Start with Docker Compose

The easiest way to run the application is using Docker Compose, which will start both MongoDB and the FastAPI application:

### 1. Create Environment File

Copy the example environment file:

```bash
cp env.example .env
```

Edit `.env` if you need to change any configuration (defaults work for local development).

### 2. Start All Services

```bash
docker-compose up -d
```

This will:
- Start MongoDB container
- Build and start the FastAPI API container
- Automatically initialize database collections and indexes on first startup

**Knowledge Base (PDF ingest)** runs inside the API container and needs `chromadb` (and its native stack). If uploads fail with a message like `RAG import failed`, rebuild the backend image so dependencies are installed in the image:

```bash
docker compose build --no-cache backend
```

From the monorepo root you can use `make build-backend-services` (builds mongodb, backend, minio), then start again.

### 3. Access the Application

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health Check**: http://localhost:8000/health

### 4. Stop All Services

```bash
docker-compose down
```

To also remove volumes (delete all data):

```bash
docker-compose down -v
```

## Local Development Setup

If you prefer to run the application locally (without Docker):

### 1. Install Dependencies

Use a virtual environment (recommended on Linux; system Python may block global `pip`):

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

This installs **Knowledge Base / RAG** packages too (`chromadb`, `openai`, `pypdf`, etc.). Run the API with the same environment (`python run.py` after `activate`, or `.venv/bin/python run.py`). Knowledge Base PDF ingest runs **inside** this API process (background task → `rag_system.ingest_pipeline` → Chroma on disk under `CHROMA_BASE_DIR`).

### 2. Setup Environment

Copy the example environment file:

```bash
cp env.example .env
```

### 3. Start MongoDB

```bash
docker-compose up -d mongodb
```

### 4. Run the Application

```bash
python run.py
```

Or using uvicorn directly:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Note**: Database collections and indexes are automatically created on application startup - no need to run `setup_database.py` manually.

## MongoDB Setup

When using Docker Compose, MongoDB is automatically started with the API service. For local development, you can start only MongoDB:

```bash
docker-compose up -d mongodb
```

### Connecting with MongoDB Compass or Beekeeper Studio

Use the following connection details:
- **Host**: `localhost`
- **Port**: `27017`
- **Authentication**: Username/Password
- **Username**: `admin`
- **Password**: `admin123`
- **Database**: `voicera` (or leave empty to connect to admin database)

### Connection String

MongoDB connection string:
```
mongodb://admin:admin123@localhost:27017/voicera?authSource=admin
```

### Data Persistence

MongoDB data is persisted in a Docker volume named `mongodb_data`. To remove all data:
```bash
docker-compose down -v
```

## API Endpoints

### Users
- `POST /api/v1/users/signup` - Create a new user
- `POST /api/v1/users/login` - Authenticate user
- `GET /api/v1/users/{email}` - Get user by email

### Agents
- `POST /api/v1/agents` - Create agent configuration
- `GET /api/v1/agents/{agent_type}` - Get agent config
- `GET /api/v1/agents/org/{org_id}` - Get all agents for org
- `PUT /api/v1/agents/{agent_type}` - Update agent config

### Meetings
- `POST /api/v1/meetings` - Create/update meeting
- `GET /api/v1/meetings/{meeting_id}` - Get meeting details
- `GET /api/v1/meetings/org/{org_id}` - Get meetings for org
- `GET /api/v1/meetings/agent/{agent_type}` - Get meetings for agent

### Campaigns
- `POST /api/v1/campaigns` - Create campaign
- `GET /api/v1/campaigns/org/{org_id}` - Get campaigns for org
- `GET /api/v1/campaigns/{campaign_name}` - Get campaign by name

### Audience
- `POST /api/v1/audience` - Create audience entry
- `GET /api/v1/audience/{audience_name}` - Get audience by name
- `GET /api/v1/audience?phone_number={phone}` - Get audiences (optional filter)

## Environment Variables

You can configure the application using environment variables:

- `MONGODB_HOST` - MongoDB host (default: localhost)
- `MONGODB_PORT` - MongoDB port (default: 27017)
- `MONGODB_USER` - MongoDB username (default: admin)
- `MONGODB_PASSWORD` - MongoDB password (default: admin123)
- `MONGODB_DATABASE` - Database name (default: voicera)
- `DEBUG` - Enable debug mode (default: False)

