#!/usr/bin/env python3
# examples/fastapi_session_example.py
"""
fastapi_session_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~

Run with:

    uvicorn examples.fastapi_session_example:app --reload

Then open http://localhost:8000/docs
"""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from chuk_ai_session_manager.models.event_source import EventSource
from chuk_ai_session_manager.models.event_type import EventType
from chuk_ai_session_manager.models.session import Session, SessionEvent
from chuk_ai_session_manager.session_prompt_builder import (
    PromptStrategy,
    build_prompt_from_session,
)
from chuk_ai_session_manager.storage import SessionStoreProvider
from chuk_ai_session_manager.storage.providers.memory import InMemorySessionStore

# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# store & lifespan
# --------------------------------------------------------------------------- #
store = InMemorySessionStore()
SessionStoreProvider.set_store(store)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context - replaces @app.on_event."""
    logger.info("Creating sample session data")

    # parent session
    parent = await Session.create()
    await parent.add_event(
        SessionEvent(
            message="What's the weather like today?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )
    await parent.add_event(
        SessionEvent(
            message="I'll check the weather for you.",
            source=EventSource.LLM,
            type=EventType.MESSAGE,
        )
    )
    await store.save(parent)

    # child session
    child = await Session.create(parent_id=parent.id)
    await child.add_event(
        SessionEvent(
            message="What about tomorrow's forecast?",
            source=EventSource.USER,
            type=EventType.MESSAGE,
        )
    )
    await store.save(child)

    logger.info(f"Created sample parent session: {parent.id}")
    logger.info(f"Created sample child  session: {child.id}")

    yield  # application runs here

    # (Optional) cleanup on shutdown
    logger.info("FastAPI shutting down - clearing in-memory store")
    await store.clear()


# --------------------------------------------------------------------------- #
# FastAPI app
# --------------------------------------------------------------------------- #
app = FastAPI(
    title="chuk session manager API",
    description="Demo API for chuk session manager with async support",
    version="0.1.0",
    lifespan=lifespan,
)

# --------------------------------------------------------------------------- #
# pydantic models
# --------------------------------------------------------------------------- #
class EventCreate(BaseModel):
    message: Any
    source: str
    event_type: str = "message"
    metadata: Optional[Dict[str, Any]] = None


class SessionResponse(BaseModel):
    id: str
    event_count: int
    parent_id: Optional[str] = None
    child_ids: List[str]


class EventResponse(BaseModel):
    id: str
    timestamp: str
    source: str
    type: str
    message: Any
    metadata: Dict[str, Any]


class PromptRequest(BaseModel):
    strategy: str = "minimal"
    max_tokens: Optional[int] = None
    include_parent_context: bool = False


class PromptResponse(BaseModel):
    prompt: List[Dict[str, Any]]
    token_estimate: Optional[int] = None


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
@app.get("/")
async def root():
    return {
        "message": "chuk session manager API",
        "docs": "/docs",
    }


@app.post("/sessions", response_model=SessionResponse)
async def create_session(background_tasks: BackgroundTasks, parent_id: Optional[str] = None):
    session = await Session.create(parent_id=parent_id)
    return {
        "id": session.id,
        "event_count": len(session.events),
        "parent_id": session.parent_id,
        "child_ids": session.child_ids,
    }


@app.get("/sessions", response_model=List[SessionResponse])
async def list_sessions():
    sessions: List[SessionResponse] = []
    for sid in await store.list_sessions():
        if (s := await store.get(sid)) is not None:
            sessions.append(
                SessionResponse(
                    id=s.id,
                    event_count=len(s.events),
                    parent_id=s.parent_id,
                    child_ids=s.child_ids,
                )
            )
    return sessions


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        id=s.id,
        event_count=len(s.events),
        parent_id=s.parent_id,
        child_ids=s.child_ids,
    )


@app.post("/sessions/{session_id}/events", response_model=EventResponse)
async def add_event(session_id: str, event: EventCreate):
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        source = EventSource(event.source.lower())
        e_type = EventType(event.event_type.lower())
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err))

    new_event = SessionEvent(
        message=event.message,
        source=source,
        type=e_type,
        metadata=event.metadata or {},
    )
    await s.add_event_and_save(new_event)
    return EventResponse(
        id=new_event.id,
        timestamp=new_event.timestamp.isoformat(),
        source=new_event.source.value,
        type=new_event.type.value,
        message=new_event.message,
        metadata=new_event.metadata,
    )


@app.get("/sessions/{session_id}/events", response_model=List[EventResponse])
async def list_events(session_id: str):
    s = await store.get(session_id)
    if not s:
        raise HTTPException(status_code=404, detail="Session not found")
    return [
        EventResponse(
            id=e.id,
            timestamp=e.timestamp.isoformat(),
            source=e.source.value,
            type=e.type.value,
            message=e.message,
            metadata=e.metadata,
        )
        for e in s.events
    ]


@app.post("/sessions/{session_id}/prompt", response_model=PromptResponse)
async def build_prompt(session_id: str, req: PromptRequest):
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
        raise HTTPException(status_code=400, detail=str(err))

    token_estimate = None
    if prompt:
        from chuk_ai_session_manager.models.token_usage import TokenUsage

        est = TokenUsage.count_tokens(json.dumps(prompt))
        token_estimate = await est if asyncio.iscoroutine(est) else est

    return PromptResponse(prompt=prompt, token_estimate=token_estimate)


@app.post("/sessions/{session_id}/children", response_model=SessionResponse)
async def create_child_session(session_id: str):
    parent = await store.get(session_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Parent session not found")

    child = await Session.create(parent_id=parent.id)
    return SessionResponse(
        id=child.id,
        event_count=len(child.events),
        parent_id=child.parent_id,
        child_ids=child.child_ids,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not await store.get(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    await store.delete(session_id)
    return {"message": f"Session {session_id} deleted"}


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)