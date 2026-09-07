"""
Pinecone embedding ingestion — reads active products from MongoDB,
embeds the product ``name`` field via OpenAI, and upserts vectors to Pinecone.

Pinecone metadata filters available:
- ``menu_step``  (str)  — course category (Apéritifs, Plats, …)
- ``status``     (str)  — only "active" products are ingested

All menu steps are embedded — the LLM filters by ``menu_step`` at query time,
so Table & Déco is included and never pollutes unrelated searches.

Dietary restrictions, allergens and occasion tags are NOT stored in Pinecone.
The LLM reads raw Carrefour data from MongoDB and applies common sense.

Usage::

    from ingest.embed import ingest_to_pinecone
    from ingest.db import get_db

    db = get_db()
    ingest_to_pinecone(db)
"""

import sys
from typing import Any

from tqdm import tqdm

from ingest.config import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    INGEST_NON_RECOMMENDABLE,
    OPENAI_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)
from ingest.log import get_logger

log = get_logger(__name__)

# OpenAI supports up to 2048 inputs per request; Pinecone upsert limit is 1000
# vectors per call (recommended ≤ 200 for optimal throughput at 1536 dims).
EMBED_BATCH = 500
UPSERT_BATCH = 200

# MongoDB query: embed all active products with a name and a menu_step.
_QUERY: dict[str, Any] = {
    "status": "active",
    "menu_step": {"$ne": None},
    "name": {"$exists": True, "$ne": ""},
}
if not INGEST_NON_RECOMMENDABLE:
    _QUERY["recommendable"] = {"$ne": False}

# Fields fetched from MongoDB — only what's needed for embedding and filtering.
_PROJECTION: dict[str, int] = {
    "_id": 1,
    "name": 1,
    "menu_step": 1,
    "status": 1,
}


def get_pinecone_index():
    """Return a live Pinecone Index object (lazy initialisation).

    Reads ``PINECONE_API_KEY`` and ``PINECONE_INDEX_NAME`` from env via
    ``ingest.config``.  Raises ``RuntimeError`` if the key is missing.

    Returns:
        A ``pinecone.Index`` instance connected to ``PINECONE_INDEX_NAME``.

    Raises:
        RuntimeError: If ``PINECONE_API_KEY`` is not set.
        ImportError: If the ``pinecone`` package is not installed.
    """
    try:
        from pinecone import Pinecone  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pinecone package is required — run: poetry add 'pinecone>=6.0.0'") from exc

    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is not set — add it to your .env file.")

    log.debug("pinecone_connecting", index=PINECONE_INDEX_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(PINECONE_INDEX_NAME)


def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Embed a list of strings via OpenAI Embeddings API in batches of 100.

    Handles rate-limit and transient API errors gracefully: on failure the
    entire batch is skipped (replaced with empty lists) and a warning is logged.
    Callers should filter out empty embeddings before upserting.

    Args:
        texts: Plain-text strings to embed.  Should not exceed 8 191 tokens each
               for ``text-embedding-3-large``.
        model: OpenAI embedding model name (default from ``EMBEDDING_MODEL``
               config value, itself defaulting to ``text-embedding-3-large``).

    Returns:
        A list of float vectors, one per input string.  Items for failed batches
        are returned as empty lists ``[]``.
    """
    try:
        from openai import OpenAI  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("openai package is required — run: poetry add 'openai>=1.0.0'") from exc

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set — add it to your .env file.")

    client = OpenAI(api_key=OPENAI_API_KEY)
    results: list[list[float]] = []

    for i in range(0, len(texts), EMBED_BATCH):
        chunk = texts[i : i + EMBED_BATCH]
        try:
            response = client.embeddings.create(input=chunk, model=model, dimensions=EMBEDDING_DIMENSIONS)
            # Responses are ordered by index — safe to iterate directly.
            results.extend(item.embedding for item in response.data)
        except Exception as exc:  # noqa: BLE001
            log.warning(
                "embed_batch_failed",
                batch_start=i,
                batch_size=len(chunk),
                error=str(exc),
                hint="Batch skipped — vectors will not be upserted for these products.",
            )
            # Pad with empty vectors so callers can zip(docs, embeddings) safely.
            results.extend([] for _ in chunk)

    return results


def _build_vector(doc: dict[str, Any], embedding: list[float]) -> dict[str, Any]:
    """Build a Pinecone vector dict from a MongoDB product document.

    Only the fields needed for search filtering are stored in Pinecone metadata.
    Everything else (dietary info, allergens, etc.) is read from MongoDB.

    Args:
        doc: A MongoDB product document (with at minimum ``_id``, ``name``).
        embedding: The float vector for this document's ``name``.

    Returns:
        A dict with ``id``, ``values``, and ``metadata`` keys ready for
        ``index.upsert()``.
    """
    metadata: dict[str, Any] = {
        "menu_step": doc.get("menu_step") or "",
        "status": doc.get("status", "inactive"),
    }

    return {
        "id": str(doc["_id"]),
        "values": embedding,
        "metadata": metadata,
    }


def reset_pinecone_index() -> None:
    """Delete ALL vectors from the Pinecone index (full wipe).

    Use before a full re-ingest to remove stale vectors for products that are
    no longer active. Safe to call — the index structure is preserved, only
    vectors are removed.
    """
    index = get_pinecone_index()
    log.info("pinecone_reset_started", index=PINECONE_INDEX_NAME)
    index.delete(delete_all=True)
    log.info("pinecone_reset_complete", index=PINECONE_INDEX_NAME)


def ingest_to_pinecone(db) -> None:
    """Read active food products from MongoDB and upsert embeddings to Pinecone.

    Products with an empty ``name`` are skipped (logged at DEBUG).
    Embedding and upsert are performed in batches of ``EMBED_BATCH`` /
    ``UPSERT_BATCH`` documents.

    At completion, logs ``pinecone_upsert_complete`` with the total count of
    vectors upserted.

    Args:
        db: A live ``pymongo.database.Database`` instance (from ``get_db()``).

    Raises:
        RuntimeError: If ``PINECONE_API_KEY`` or ``OPENAI_API_KEY`` is missing.
    """
    log.info("pinecone_ingest_started", query=str(_QUERY), index=PINECONE_INDEX_NAME)

    index = get_pinecone_index()
    total = db.products.count_documents(_QUERY)
    log.info("pinecone_products_found", count=total)

    cursor = db.products.find(_QUERY, _PROJECTION).batch_size(EMBED_BATCH)

    upserted_total = 0
    skipped_empty = 0
    batch_docs: list[dict[str, Any]] = []

    with tqdm(total=total, unit="product", desc="pinecone", file=sys.stderr) as bar:
        for doc in cursor:
            if not (doc.get("name") or "").strip():
                log.debug("name_empty_skipped", product_id=str(doc["_id"]))
                skipped_empty += 1
                bar.update(1)
                continue

            batch_docs.append(doc)
            bar.update(1)

            if len(batch_docs) >= EMBED_BATCH:
                upserted_total += _flush_batch(batch_docs, index)
                batch_docs = []

        # Flush remainder.
        if batch_docs:
            upserted_total += _flush_batch(batch_docs, index)

    if skipped_empty:
        log.warning("pinecone_skipped_empty_name", count=skipped_empty)

    log.info("pinecone_upsert_complete", count=upserted_total)


def _flush_batch(docs: list[dict[str, Any]], index) -> int:
    """Embed and upsert a batch of product documents to Pinecone.

    Args:
        docs: A list of MongoDB product documents to embed and upsert.
        index: A live Pinecone ``Index`` object.

    Returns:
        The number of vectors successfully upserted.
    """
    texts = [(doc.get("name") or "").strip() for doc in docs]
    embeddings = embed_texts(texts)

    vectors = []
    for doc, emb in zip(docs, embeddings):
        if not emb:
            # Batch failed for this document — already warned in embed_texts.
            log.debug("vector_skipped_no_embedding", product_id=str(doc["_id"]))
            continue
        vectors.append(_build_vector(doc, emb))

    if not vectors:
        return 0

    # Pinecone upsert in sub-batches of UPSERT_BATCH.
    upserted = 0
    for i in range(0, len(vectors), UPSERT_BATCH):
        chunk = vectors[i : i + UPSERT_BATCH]
        index.upsert(vectors=chunk)
        upserted += len(chunk)
        log.debug("pinecone_batch_upserted", count=len(chunk))

    return upserted
