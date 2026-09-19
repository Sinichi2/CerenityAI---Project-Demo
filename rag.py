"""Cerenity RAG agent.

Gemini 2.5 for reasoning, LangGraph for orchestration, and two retrieval sources:
Qdrant for the unstructured handbook, SQLite for the structured member records.
"""

import json
import os
import sqlite3
from functools import lru_cache
from pathlib import Path

import aiosqlite
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.prebuilt import create_react_agent
from qdrant_client import QdrantClient

load_dotenv()

ROOT = Path(__file__).parent
DATA_DB = ROOT / "cerenity.db"          # structured source: plans + practitioners
MEMORY_DB = ROOT / "memory.db"          # message store: one row per conversation checkpoint
QDRANT_PATH = ROOT / "qdrant_store"     # unstructured source: handbook chunks
COLLECTION = "handbook"
EMBED_DIM = 768  # gemini-embedding-001 defaults to 3072; 768 is plenty for ~30 chunks

CHAT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

@lru_cache(maxsize=1)
def embedder() -> GoogleGenerativeAIEmbeddings:
    """Built on first use rather than at import, so the SQL half runs without an API key."""
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001", output_dimensionality=EMBED_DIM
    )

# ponytail: Qdrant local mode writes straight to disk. No server, no docker, same client API.
# The cost: it takes an exclusive lock on the folder, so ingest.py and app.py cannot run at the
# same time. Set QDRANT_URL to point at a real Qdrant if you ever need concurrent writers.
_qdrant_url = os.getenv("QDRANT_URL")
qdrant = QdrantClient(url=_qdrant_url) if _qdrant_url else QdrantClient(path=str(QDRANT_PATH))


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    with sqlite3.connect(DATA_DB) as cx:
        cx.row_factory = sqlite3.Row
        return [dict(r) for r in cx.execute(sql, params)]


# --- retrieval tools ----------------------------------------------------------------
# Each returns JSON: the model reads the whole payload, the server reads "sources" to build
# the citation row under the reply. One shape serves both readers.


@tool
def search_handbook(query: str) -> str:
    """Semantic search over Cerenity's written handbook: privacy and data handling, safety and
    crisis policy, onboarding and matching, therapy methods, billing and cancellation.

    Use this for any question about how something works, what is allowed, or what Cerenity's
    position on something is. Pass the member's question, rephrased as a search query.
    """
    hits = qdrant.query_points(
        COLLECTION, query=embedder().embed_query(query), limit=4, with_payload=True
    ).points
    return json.dumps(
        {
            "passages": [h.payload["text"] for h in hits],
            "sources": [
                {
                    "kind": "doc",
                    "label": f"{h.payload['doc']} · {h.payload['section']}",
                    "detail": " ".join(h.payload["text"].split())[:220],
                    "score": round(h.score, 3),
                }
                for h in hits
            ],
        }
    )


@tool
def list_plans() -> str:
    """Every Cerenity membership plan with monthly price, session allowance, whether async
    messaging is included, and how fast a practitioner match is promised.

    Use this for anything about cost, tiers, what a membership includes, or comparing plans.
    """
    rows = _rows("SELECT * FROM plans ORDER BY price_monthly")
    return json.dumps(
        {
            "rows": rows,
            "sources": [
                {
                    "kind": "sql",
                    "label": f"plans · {r['name']}",
                    "detail": f"${r['price_monthly']}/mo · {r['sessions_per_month']} sessions · "
                    f"match within {r['match_sla_hours']}h",
                }
                for r in rows
            ],
        }
    )


@tool
def find_practitioners(focus: str = "", language: str = "", plan: str = "") -> str:
    """Search the practitioner directory.

    focus matches a clinical specialty ("anxiety", "burnout", "grief", "trauma", "relationships",
    "perinatal"). language matches a spoken language. plan limits results to practitioners
    bookable on that membership tier ("Core", "Plus", "Household"). Leave any argument empty to
    ignore it. Use whenever the member asks who they could see, what someone specialises in,
    what languages are available, or when someone is next free.
    """
    rows = _rows(
        """SELECT name, credential, focus, modality, languages, next_available, plans
             FROM practitioners
            WHERE focus LIKE ? AND languages LIKE ? AND plans LIKE ?
         ORDER BY next_available""",
        (f"%{focus}%", f"%{language}%", f"%{plan}%"),
    )
    return json.dumps(
        {
            "rows": rows,
            "sources": [
                {
                    "kind": "sql",
                    "label": f"practitioners · {r['name']}",
                    "detail": f"{r['credential']} · {r['focus']} · {r['modality']} · "
                    f"next free {r['next_available']}",
                }
                for r in rows
            ],
        }
    )


TOOLS = [search_handbook, list_plans, find_practitioners]

SYSTEM = """You are the Cerenity assistant. Cerenity is a mental-wellness service that matches \
members with licensed practitioners.

Answer only from your tools:
  search_handbook    the written handbook (policy, privacy, safety, onboarding, methods, billing)
  list_plans         the membership pricing table
  find_practitioners the practitioner directory

Call whatever combination the question needs, and call more than one when the question spans both \
prose and records, for example "what does Plus cost and how do you handle my session notes?". If \
the tools do not cover something, say so plainly and point the member at support. Never invent a \
price, a name, an availability date or a policy.

Safety rule, which overrides everything above: if the member mentions suicide, self-harm, abuse, \
or being in immediate danger, stop retrieving. Reply briefly and warmly, say clearly that you are \
not an emergency service, and give the crisis routes from the handbook. Do not diagnose and do not \
give clinical advice. That is what the practitioners are for.

Keep replies to 2 to 4 sentences unless asked for more. Warm, plain, specific. Use ordinary \
hyphens, never em dashes."""


async def build_agent():
    """Compile the agent with SQLite-backed conversation memory.

    The checkpointer is the message store: LangGraph loads prior turns by thread_id before every
    model call and writes the new ones back, so the API stays stateless.
    """
    if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("GOOGLE_API_KEY is not set. Copy .env.example to .env and add a key.")
    if not qdrant.collection_exists(COLLECTION):
        raise RuntimeError(f"Qdrant collection '{COLLECTION}' is missing. Run: uv run ingest.py")

    saver = AsyncSqliteSaver(await aiosqlite.connect(str(MEMORY_DB)))
    await saver.setup()
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0.3)
    return create_react_agent(llm, TOOLS, prompt=SYSTEM, checkpointer=saver)
