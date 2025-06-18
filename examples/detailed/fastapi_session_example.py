#!/usr/bin/env python3
# examples/fastapi_session_example.py
"""
fastapi_session_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

A production-ready FastAPI server demonstrating the CHUK AI Session Manager.

Features:
- Complete session lifecycle management (CRUD operations)
- Event tracking and management
- Hierarchical session support (parent-child relationships)
- Prompt building with multiple strategies
- Token usage estimation
- Session statistics and analytics
- OpenAPI documentation with Swagger UI

Run with:
    uvicorn examples.fastapi_session_example:app --reload

Then open:
- API Documentation: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc
- Health check: http://localhost:8000/health
"""

import json
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
from datetime import datetime

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import session manager components - FIXED for current architecture
from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session
from chuk_ai_session_manager.models.session_event import SessionEvent
from chuk_ai_session_manager.session_prompt_builder import (
    PromptStrategy,
    build_prompt_from_session,
)
from chuk_ai_session_manager.session_storage import (
    get_backend, 
    ChukSessionsStore, 
    setup_chuk_sessions_storage
)
from chuk_ai_session_manager.models.token_usage import TokenUsage

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Quiet down noisy loggers
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("openai").setLevel(logging.ERROR)
logging.getLogger("anthropic").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("h11").setLevel(logging.ERROR)
logging.getLogger("chuk_llm").setLevel(logging.WARNING)
logging.getLogger("chuk_sessions").setLevel(logging.WARNING)
logging.getLogger("chuk_ai_session_manager").setLevel(logging.WARNING)

# --------------------------------------------------------------------------- #
# Global store reference
# --------------------------------------------------------------------------- #
store: Optional[ChukSessionsStore] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context - replaces @app.on_event."""
    global store
    
    logger.info("🚀 Starting CHUK AI Session Manager API")
    
    # Setup CHUK Sessions storage backend
    setup_chuk_sessions_storage(sandbox_id="fastapi-session-api", default_ttl_hours=24)
    backend = get_backend()
    store = ChukSessionsStore(backend)
    
    logger.info("💾 CHUK Sessions storage backend initialized")
    
    # Create sample session data
    logger.info("📝 Creating sample session data")

    # Parent session
    parent = await Session.create()
    await parent.metadata.set_property("example", "weather_conversation")
    await parent.metadata.set_property("type", "demo")
    
    # Add events with token tracking
    user_event = await SessionEvent.create_with_tokens(
        message="What's the weather like today?",
        prompt="What's the weather like today?",
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await parent.add_event_and_save(user_event)

    llm_event = await SessionEvent.create_with_tokens(
        message="I'll check the weather for you. It's currently 22°C and partly cloudy.",
        prompt="",
        completion="I'll check the weather for you. It's currently 22°C and partly cloudy.",
        model="gpt-4o-mini",
        source=EventSource.LLM,
        type=EventType.MESSAGE,
    )
    await parent.add_event_and_save(llm_event)

    # Child session
    child = await Session.create(parent_id=parent.id)
    await child.metadata.set_property("example", "weather_followup")
    await child.metadata.set_property("type", "demo")
    
    child_event = await SessionEvent.create_with_tokens(
        message="What about tomorrow's forecast?",
        prompt="What about tomorrow's forecast?", 
        model="gpt-4o-mini",
        source=EventSource.USER,
        type=EventType.MESSAGE,
    )
    await child.add_event_and_save(child_event)

    logger.info(f"✅ Created sample parent session: {parent.id}")
    logger.info(f"✅ Created sample child session: {child.id}")
    logger.info("🎯 API ready to serve requests")

    yield  # Application runs here

    # Cleanup on shutdown
    logger.info("🛑 FastAPI shutting down")


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="CHUK AI Session Manager API",
    description="""
    Production-ready API for managing AI conversation sessions with comprehensive tracking.
    
    ## Features
    - **Session Management**: Create, read, update, delete sessions
    - **Event Tracking**: Track all conversation events with token usage
    - **Hierarchical Sessions**: Parent-child session relationships
    - **Prompt Building**: Multiple strategies for LLM prompt construction
    - **Token Analytics**: Comprehensive token usage and cost tracking
    - **CHUK Sessions Backend**: Enterprise-grade storage with Redis support
    
    ## Quick Start
    1. Create a session: `POST /sessions`
    2. Add events: `POST /sessions/{id}/events`
    3. Build prompts: `POST /sessions/{id}/prompt`
    4. Get analytics: `GET /sessions/{id}/stats`
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "CHUK AI Session Manager",
        "url": "https://github.com/chuk-ai/session-manager",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# Add CORS middleware for web frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
# Pydantic models
# --------------------------------------------------------------------------- #
class EventCreate(BaseModel):
    """Request model for creating a new event."""
    message: Any = Field(..., description="The event message content")
    source: str = Field(..., description="Event source: user, llm, or system")
    event_type: str = Field(default="message", description="Event type: message, tool_call, or summary")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional event metadata")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, how can I help you today?",
                "source": "user",
                "event_type": "message",
                "metadata": {"priority": "high"}
            }
        }


class SessionCreate(BaseModel):
    """Request model for creating a new session."""
    parent_id: Optional[str] = Field(default=None, description="Parent session ID for hierarchical sessions")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Session metadata properties")
    
    class Config:
        json_schema_extra = {
            "example": {
                "parent_id": None,
                "metadata": {"project": "customer_support", "user_id": "user123"}
            }
        }


class SessionResponse(BaseModel):
    """Response model for session information."""
    id: str
    event_count: int
    parent_id: Optional[str] = None
    child_ids: List[str]
    created_at: str
    updated_at: str
    total_tokens: int
    estimated_cost: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "abc123-def456-ghi789",
                "event_count": 5,
                "parent_id": None,
                "child_ids": ["child1-id", "child2-id"],
                "created_at": "2025-06-17T10:30:00Z",
                "updated_at": "2025-06-17T10:35:00Z",
                "total_tokens": 150,
                "estimated_cost": 0.000225
            }
        }


class EventResponse(BaseModel):
    """Response model for event information."""
    id: str
    timestamp: str
    source: str
    type: str
    message: Any
    metadata: Dict[str, Any]
    token_usage: Optional[Dict[str, Any]] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "event123",
                "timestamp": "2025-06-17T10:30:00Z",
                "source": "user",
                "type": "message",
                "message": "What's the weather like?",
                "metadata": {"priority": "normal"},
                "token_usage": {"total_tokens": 15, "estimated_cost": 0.000022}
            }
        }


class PromptRequest(BaseModel):
    """Request model for building prompts."""
    strategy: str = Field(default="minimal", description="Prompt strategy: minimal, task, tool, conversation, hierarchical")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens for prompt")
    include_parent_context: bool = Field(default=False, description="Include parent session context")
    
    class Config:
        json_schema_extra = {
            "example": {
                "strategy": "conversation",
                "max_tokens": 2000,
                "include_parent_context": True
            }
        }


class PromptResponse(BaseModel):
    """Response model for built prompts."""
    prompt: List[Dict[str, Any]]
    token_estimate: Optional[int] = None
    strategy_used: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": None}
                ],
                "token_estimate": 25,
                "strategy_used": "minimal"
            }
        }


class SessionStats(BaseModel):
    """Response model for session statistics."""
    session_id: str
    total_events: int
    total_tokens: int
    estimated_cost: float
    events_by_source: Dict[str, int]
    events_by_type: Dict[str, int]
    token_usage_by_source: Dict[str, Dict[str, Any]]
    created_at: str
    updated_at: str
    duration_seconds: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    version: str
    backend: str
    session_count: int


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@app.get("/", summary="API Information")
async def root():
    """Get basic API information and available endpoints."""
    return {
        "message": "CHUK AI Session Manager API",
        "version": "1.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/health",
        "features": [
            "Session management with hierarchical support",
            "Event tracking with token usage",
            "Multiple prompt building strategies", 
            "Real-time analytics and statistics",
            "CHUK Sessions backend integration"
        ]
    }


@app.get("/health", response_model=HealthResponse, summary="Health Check")
async def health_check():
    """Check API health and backend connectivity."""
    global store
    
    session_ids = await store.list_sessions() if store else []
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        backend="chuk_sessions",
        session_count=len(session_ids)
    )


@app.post("/sessions", response_model=SessionResponse, summary="Create Session")
async def create_session(session_data: SessionCreate):
    """Create a new conversation session with optional parent relationship."""
    global store
    
    # Create session with optional parent
    session = await Session.create(parent_id=session_data.parent_id)
    
    # Set metadata properties if provided
    if session_data.metadata:
        for key, value in session_data.metadata.items():
            await session.metadata.set_property(key, value)
    
    # Save session
    await store.save(session)
    
    return SessionResponse(
        id=session.id,
        event_count=len(session.events),
        parent_id=session.parent_id,
        child_ids=session.child_ids,
        created_at=session.metadata.created_at.isoformat(),
        updated_at=session.metadata.updated_at.isoformat(),
        total_tokens=session.total_tokens,
        estimated_cost=session.total_cost,
    )


@app.get("/sessions", response_model=List[SessionResponse], summary="List Sessions")
async def list_sessions(
    limit: int = Query(default=50, le=100, description="Maximum number of sessions to return"),
    offset: int = Query(default=0, ge=0, description="Number of sessions to skip")
):
    """List all conversation sessions with pagination."""
    global store
    
    sessions: List[SessionResponse] = []
    session_ids = await store.list_sessions()
    
    # Apply pagination
    paginated_ids = session_ids[offset:offset + limit]
    
    for sid in paginated_ids:
        if (s := await store.get(sid)) is not None:
            sessions.append(
                SessionResponse(
                    id=s.id,
                    event_count=len(s.events),
                    parent_id=s.parent_id,
                    child_ids=s.child_ids,
                    created_at=s.metadata.created_at.isoformat(),
                    updated_at=s.metadata.updated_at.isoformat(),
                    total_tokens=s.total_tokens,
                    estimated_cost=s.total_cost,
                )
            )
    return sessions


@app.get("/sessions/{session_id}", response_model=SessionResponse, summary="Get Session")
async def get_session(session_id: str):
    """Get detailed information about a specific session."""
    global store
    
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        id=s.id,
        event_count=len(s.events),
        parent_id=s.parent_id,
        child_ids=s.child_ids,
        created_at=s.metadata.created_at.isoformat(),
        updated_at=s.metadata.updated_at.isoformat(),
        total_tokens=s.total_tokens,
        estimated_cost=s.total_cost,
    )


@app.post("/sessions/{session_id}/events", response_model=EventResponse, summary="Add Event")
async def add_event(session_id: str, event: EventCreate):
    """Add a new event to a session with automatic token tracking."""
    global store
    
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        source = EventSource(event.source.lower())
        e_type = EventType(event.event_type.lower())
    except ValueError as err:
        raise HTTPException(status_code=400, detail=f"Invalid source or event type: {err}")

    # Create event with token tracking
    if isinstance(event.message, str):
        # For string messages, use token tracking
        new_event = await SessionEvent.create_with_tokens(
            message=event.message,
            prompt=event.message if source == EventSource.USER else "",
            completion=event.message if source == EventSource.LLM else "",
            model="gpt-4o-mini",  # Default model for token counting
            source=source,
            type=e_type
        )
    else:
        # For non-string messages, create without token tracking
        new_event = SessionEvent(
            message=event.message,
            source=source,
            type=e_type,
        )
    
    # Add metadata
    if event.metadata:
        for key, value in event.metadata.items():
            await new_event.set_metadata(key, value)
    
    await s.add_event_and_save(new_event)
    
    return EventResponse(
        id=new_event.id,
        timestamp=new_event.timestamp.isoformat(),
        source=new_event.source.value,
        type=new_event.type.value,
        message=new_event.message,
        metadata=new_event.metadata,
        token_usage=new_event.token_usage.model_dump() if new_event.token_usage else None,
    )


@app.get("/sessions/{session_id}/events", response_model=List[EventResponse], summary="List Events")
async def list_events(session_id: str):
    """Get all events for a specific session."""
    global store
    
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    
    events = []
    for e in s.events:
        events.append(EventResponse(
            id=e.id,
            timestamp=e.timestamp.isoformat(),
            source=e.source.value,
            type=e.type.value,
            message=e.message,
            metadata=e.metadata,
            token_usage=e.token_usage.model_dump() if e.token_usage else None,
        ))
    
    return events


@app.post("/sessions/{session_id}/prompt", response_model=PromptResponse, summary="Build Prompt")
async def build_prompt(session_id: str, req: PromptRequest):
    """Build an optimized prompt from session history using specified strategy."""
    global store
    
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        prompt = await build_prompt_from_session(
            s,
            strategy=req.strategy,
            max_tokens=req.max_tokens,
            include_parent_context=req.include_parent_context,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=f"Prompt building failed: {err}")

    # Estimate token count
    token_estimate = None
    if prompt:
        est = await TokenUsage.count_tokens(json.dumps(prompt))
        token_estimate = est

    return PromptResponse(
        prompt=prompt, 
        token_estimate=token_estimate,
        strategy_used=req.strategy
    )


@app.get("/sessions/{session_id}/stats", response_model=SessionStats, summary="Session Statistics")
async def get_session_stats(session_id: str):
    """Get comprehensive statistics and analytics for a session."""
    global store
    
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # Calculate event breakdowns
    events_by_source = {}
    events_by_type = {}
    
    for event in s.events:
        source = event.source.value
        etype = event.type.value
        
        events_by_source[source] = events_by_source.get(source, 0) + 1
        events_by_type[etype] = events_by_type.get(etype, 0) + 1
    
    # Get token usage by source
    token_usage_by_source = await s.get_token_usage_by_source()
    token_usage_dict = {}
    for source, summary in token_usage_by_source.items():
        token_usage_dict[source] = summary.model_dump()
    
    # Calculate duration
    duration_seconds = None
    if s.events:
        start_time = s.metadata.created_at
        end_time = s.events[-1].timestamp
        duration_seconds = (end_time - start_time).total_seconds()
    
    return SessionStats(
        session_id=s.id,
        total_events=len(s.events),
        total_tokens=s.total_tokens,
        estimated_cost=s.total_cost,
        events_by_source=events_by_source,
        events_by_type=events_by_type,
        token_usage_by_source=token_usage_dict,
        created_at=s.metadata.created_at.isoformat(),
        updated_at=s.metadata.updated_at.isoformat(),
        duration_seconds=duration_seconds,
    )


@app.post("/sessions/{session_id}/children", response_model=SessionResponse, summary="Create Child Session")
async def create_child_session(session_id: str, session_data: SessionCreate):
    """Create a child session with the specified session as parent."""
    global store
    
    parent = await store.get(session_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent session not found")

    child = await Session.create(parent_id=parent.id)
    
    # Set metadata if provided
    if session_data.metadata:
        for key, value in session_data.metadata.items():
            await child.metadata.set_property(key, value)
    
    await store.save(child)
    
    return SessionResponse(
        id=child.id,
        event_count=len(child.events),
        parent_id=child.parent_id,
        child_ids=child.child_ids,
        created_at=child.metadata.created_at.isoformat(),
        updated_at=child.metadata.updated_at.isoformat(),
        total_tokens=child.total_tokens,
        estimated_cost=child.total_cost,
    )


@app.delete("/sessions/{session_id}", summary="Delete Session")
async def delete_session(session_id: str):
    """Delete a session and all its events."""
    global store
    
    if not await store.get(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    await store.delete(session_id)
    return {"message": f"Session {session_id} deleted successfully"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_session_example:app",
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )