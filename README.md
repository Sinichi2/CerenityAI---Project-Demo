# Cerenity RAG Demo

A RAG-powered chatbot for a fictional mental-wellness service. Gemini 2.5 answers questions about
Cerenity by retrieving from two different kinds of store: a Qdrant vector collection holding the
written handbook, and a SQLite database holding plans and the practitioner directory. LangGraph
decides which to reach for and keeps the conversation in a checkpointed message store.

## Run it

```bash
cp .env.example .env          # then paste a key from https://aistudio.google.com/apikey
uv sync
uv run ingest.py              # seeds SQLite and embeds the handbook into Qdrant
uv run uvicorn app:app        # http://127.0.0.1:8000
```

`ingest.py` and the server cannot run at the same time: Qdrant in local mode takes an exclusive
lock on `qdrant_store/`. Stop the server before re-ingesting, or point `QDRANT_URL` at a real
Qdrant instance.

## How it works

```
browser  ──POST /chat──►  FastAPI  ──►  LangGraph agent  ──►  Gemini 2.5 Flash
   ▲                                          │
   │                                          ├──► search_handbook    ──► Qdrant   (unstructured)
   │                                          ├──► list_plans         ──► SQLite   (structured)
   └──── NDJSON stream ───────────────────────┴──► find_practitioners ──► SQLite   (structured)
                                              │
                                              └──► AsyncSqliteSaver   ──► memory.db (message store)
```

**Routing.** The three retrievers are bound to the model as tools, so Gemini picks the sources per
question rather than every question paying for every store. "What does Plus cost?" hits only the
plans table. "How are my notes stored?" hits only the vector search. "I want help with burnout, who
could I see on Plus and what happens first?" hits all three, which is the case worth watching.

**Two stores, on purpose.** Prose belongs in vectors because the useful unit is a passage and the
query never matches the wording. Prices, availability dates and language coverage belong in SQL
because the useful unit is a row and an exact filter beats a similarity score. Embedding a pricing
table and hoping cosine distance returns the right tier is the standard way to get a RAG demo to
quote the wrong number.

**Memory.** `AsyncSqliteSaver` is the message store. LangGraph loads the prior turns for a
`thread_id` before each model call and writes the new ones back, so `/chat` stays stateless and a
browser reload replays the thread through `/history`.

**Citations.** Every tool returns JSON carrying both its result and a `sources` list. The model
reads the result, the server forwards the sources, and the UI shows them under the reply. What the
answer was built from is visible without turning on server logs.

## Files

| | |
|---|---|
| `rag.py` | Model, embeddings, the three retrieval tools, system prompt, agent assembly |
| `app.py` | FastAPI: streams the agent as NDJSON, replays history, serves the page |
| `ingest.py` | Seeds both stores, then self-checks that retrieval round-trips |
| `data/*.md` | The handbook: onboarding, privacy, safety, methods, billing |
| `static/index.html` | The chat interface, no build step |

## Notes

- `uv run ingest.py` ends with assertions covering the chunker, the vector round-trip and each SQL
  filter. If it prints `check retrieval round-trips on both sources, filters hold`, the pipeline is
  intact.
- Safety handling is in the system prompt and in `data/safety.md`: crisis disclosures stop retrieval
  and return the 988 and Crisis Text Line routes instead of a policy lookup.
- Model output is escaped before it reaches the DOM, and the SQL tools take typed arguments bound as
  parameters. There is no text-to-SQL path for the model to inject through.
- The practitioners, plans, prices and policies are invented for the demo.
