"""Chat server. Streams the LangGraph agent to the browser as newline-delimited JSON.

    uv run uvicorn app:app --reload
"""

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from rag import CHAT_MODEL, ROOT, build_agent

AGENT = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global AGENT
    AGENT = await build_agent()
    yield


app = FastAPI(title="Cerenity", lifespan=lifespan)


class Ask(BaseModel):
    thread: str = Field(min_length=1, max_length=100)
    text: str = Field(min_length=1, max_length=4000)


def ndjson(**event) -> str:
    return json.dumps(event) + "\n"


def text_of(message) -> str:
    """Gemini replies arrive as content blocks, not always a bare string.

    langchain-core 1.x exposes `.text` as a property whose value is still callable for
    backwards compatibility, so test for str before falling back to the 0.x method.
    """
    value = getattr(message, "text", "")
    if isinstance(value, str):
        return value
    return value() if callable(value) else ""


def sources_of(tool_message: ToolMessage) -> list[dict]:
    try:
        return json.loads(tool_message.content).get("sources", [])
    except (json.JSONDecodeError, TypeError, AttributeError):
        return []


@app.post("/chat")
async def chat(ask: Ask):
    config = {"configurable": {"thread_id": ask.thread}}

    async def stream():
        try:
            async for mode, chunk in AGENT.astream(
                {"messages": [{"role": "user", "content": ask.text}]},
                config,
                stream_mode=["messages", "updates"],
            ):
                if mode == "messages":
                    message, _ = chunk
                    if isinstance(message, AIMessageChunk) and (token := text_of(message)):
                        yield ndjson(t="token", v=token)
                    continue
                # "updates" carries whole messages once a node finishes: that is where the
                # tool calls and their retrieved sources show up.
                for update in chunk.values():
                    for message in (update or {}).get("messages", []):
                        if isinstance(message, ToolMessage):
                            yield ndjson(t="sources", tool=message.name,
                                         items=sources_of(message))
                        elif isinstance(message, AIMessage) and message.tool_calls:
                            for call in message.tool_calls:
                                yield ndjson(t="retrieving", tool=call["name"])
        except Exception as exc:  # surfaced in the thread, not swallowed into a dead spinner
            yield ndjson(t="error", v=f"{type(exc).__name__}: {exc}"[:400])
        yield ndjson(t="done")

    return StreamingResponse(stream(), media_type="application/x-ndjson")


@app.get("/history")
async def history(thread: str):
    """Replay a conversation from the message store so a reload does not lose the thread."""
    if not thread:
        raise HTTPException(400, "thread is required")
    state = await AGENT.aget_state({"configurable": {"thread_id": thread}})
    turns, pending = [], []
    for message in (state.values or {}).get("messages", []):
        if isinstance(message, HumanMessage):
            turns.append({"role": "user", "text": message.content})
        elif isinstance(message, ToolMessage):
            pending += sources_of(message)
        elif isinstance(message, AIMessage) and (body := text_of(message)):
            turns.append({"role": "assistant", "text": body, "sources": pending})
            pending = []
    return {"model": CHAT_MODEL, "turns": turns}


app.mount("/", StaticFiles(directory=ROOT / "static", html=True), name="static")
