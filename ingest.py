"""Seed both retrieval sources, then prove they answer.

    uv run ingest.py

Stop the server first: Qdrant local mode locks its storage folder to one process.
"""

import sqlite3

from qdrant_client.models import Distance, PointStruct, VectorParams

from rag import (
    COLLECTION,
    DATA_DB,
    EMBED_DIM,
    ROOT,
    embedder,
    find_practitioners,
    list_plans,
    qdrant,
    search_handbook,
)

PLANS = [
    # name, $/mo, sessions/mo, async messaging, match SLA hours, notes
    ("Core", 49, 2, "no", 96, "Two 50-minute sessions. Reschedule up to 24h before."),
    ("Plus", 89, 4, "yes", 48, "Four sessions plus messaging between them."),
    ("Household", 139, 6, "yes", 24, "Six sessions shared across two adults at one address."),
]

PRACTITIONERS = [
    # name, credential, focus, modality, languages, next available, bookable on
    ("Nadia Bekele", "PsyD", "anxiety, panic", "CBT", "English, Amharic",
     "2026-09-18", "Core, Plus, Household"),
    ("Tomás Vieira", "LCSW", "burnout, work stress", "ACT", "English, Portuguese",
     "2026-09-19", "Plus, Household"),
    ("Priya Raghunathan", "LMFT", "relationships, family conflict", "EFT", "English, Tamil",
     "2026-09-22", "Household"),
    ("Daniel Okafor", "LPC", "grief, life transitions", "IFS", "English, Igbo",
     "2026-09-21", "Core, Plus, Household"),
    ("Hana Sato-Mercer", "PsyD", "trauma, PTSD", "EMDR", "English, Japanese",
     "2026-09-25", "Plus, Household"),
    ("Iris Lindqvist", "LCSW", "perinatal, postpartum", "mindfulness-based CBT", "English, Swedish",
     "2026-09-24", "Plus, Household"),
]


def seed_sql() -> None:
    with sqlite3.connect(DATA_DB) as cx:
        cx.executescript("""
            DROP TABLE IF EXISTS plans;
            DROP TABLE IF EXISTS practitioners;
            CREATE TABLE plans (
                name TEXT PRIMARY KEY, price_monthly INTEGER, sessions_per_month INTEGER,
                async_messaging TEXT, match_sla_hours INTEGER, notes TEXT);
            CREATE TABLE practitioners (
                name TEXT PRIMARY KEY, credential TEXT, focus TEXT, modality TEXT,
                languages TEXT, next_available TEXT, plans TEXT);
        """)
        cx.executemany("INSERT INTO plans VALUES (?,?,?,?,?,?)", PLANS)
        cx.executemany("INSERT INTO practitioners VALUES (?,?,?,?,?,?,?)", PRACTITIONERS)
    print(f"sql      {len(PLANS)} plans, {len(PRACTITIONERS)} practitioners -> {DATA_DB.name}")


def chunks() -> list[dict]:
    """One chunk per '## ' section, plus the intro above the first heading.

    ponytail: markdown headings are already the author's chosen boundaries, so splitting on them
    beats a character-window splitter here. Swap in a real splitter if the docs ever lose structure.
    """
    out = []
    for path in sorted((ROOT / "data").glob("*.md")):
        intro, *sections = path.read_text(encoding="utf-8").split("\n## ")
        out.append({"doc": path.name, "section": "Overview", "text": intro.strip()})
        for section in sections:
            heading, _, body = section.partition("\n")
            out.append({
                "doc": path.name,
                "section": heading.strip(),
                "text": f"{heading.strip()}\n{body.strip()}",
            })
    return out


def seed_vectors() -> list[dict]:
    payloads = chunks()
    vectors = embedder().embed_documents([c["text"] for c in payloads])

    if qdrant.collection_exists(COLLECTION):
        qdrant.delete_collection(COLLECTION)
    qdrant.create_collection(
        COLLECTION, vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE)
    )
    qdrant.upsert(
        COLLECTION,
        points=[PointStruct(id=i, vector=v, payload=p)
                for i, (v, p) in enumerate(zip(vectors, payloads))],
    )
    docs = {c["doc"] for c in payloads}
    print(f"vectors  {len(payloads)} chunks from {len(docs)} docs -> {COLLECTION}")
    return payloads


def self_check(payloads: list[dict]) -> None:
    """Smallest thing that fails if any part of the pipeline broke."""
    assert len(payloads) > 10, f"expected a real corpus, chunked only {len(payloads)}"
    assert all(c["text"] and c["section"] for c in payloads), "empty chunk or heading"

    hit = search_handbook.invoke({"query": "who can read my session notes?"})
    assert "privacy.md" in hit, f"privacy doc did not surface for a privacy question: {hit[:200]}"

    plans = list_plans.invoke({})
    assert '"Household"' in plans and '"price_monthly": 139' in plans, plans[:200]

    burnout = find_practitioners.invoke({"focus": "burnout"})
    assert "Tomás Vieira" in burnout, burnout[:200]
    assert "Hana Sato-Mercer" not in burnout, "focus filter is not filtering"

    assert find_practitioners.invoke({"plan": "Core"}).count('"name"') == 2, "plan filter is off"
    print("check    retrieval round-trips on both sources, filters hold")


if __name__ == "__main__":
    seed_sql()
    self_check(seed_vectors())
    qdrant.close()  # release the storage lock now, not during interpreter teardown
