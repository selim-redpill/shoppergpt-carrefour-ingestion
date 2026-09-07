"""
Configuration — MongoDB connection, data file paths, and CDN base URLs.

All values can be overridden via environment variables (see .env.example).

DB naming convention (matches the API):
  dev  → waib_carrefour_traiteur_dev_db
  prod → waib_carrefour_traiteur_db

Pinecone index naming convention (matches the API):
  dev  → waib-carrefour-dev-large
  prod → waib-carrefour-prod-large
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ──────────────────────────────────────────────────────────────────
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

_ENV = os.getenv("ENV", "dev").lower().strip()
_DEFAULT_DB = "waib_carrefour_traiteur_db" if _ENV == "prod" else "waib_carrefour_traiteur_dev_db"
MONGO_DB = os.getenv("MONGO_DB", _DEFAULT_DB)

# ── Data files ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"


def _latest_file(*prefixes: str) -> Path:
    """Return the most recent file matching any of the given prefixes + *.jsonl.gz or *.jsonl.
    Supports both plain .jsonl and gzipped .jsonl.gz. Falls back to the first legacy name."""
    candidates = sorted(
        [
            f
            for prefix in prefixes
            for f in [*DATA_DIR.glob(f"{prefix}*.jsonl.gz"), *DATA_DIR.glob(f"{prefix}*.jsonl")]
        ],
        reverse=True,  # lexicographic sort → most recent date suffix first
    )
    return candidates[0] if candidates else DATA_DIR / f"{prefixes[-1]}.jsonl"


# Support both legacy names and new prefixed+dated names from GCS exports
PRODUCTS_FILE = _latest_file("catalogue_products", "products")
PRICES_FILE = _latest_file("mapping_products_prices", "products_prices")
STORES_FILE = _latest_file("magasins_stores", "stores")

# Carrefour's curated business taxonomy — static export file (not dated/rotated
# like the exports above). See ingest/concepts.py for how it's used. (The
# "Evénement X Concept" and "Concept X Produits" sibling exports were used once
# to hand-derive waib-api's hardcoded EVENT_CONCEPT_FIT table and are no longer
# read at runtime — see ingest/concepts.py's module docstring for why the
# per-product join was dropped.)
CONCEPT_MAGASIN_FILE = DATA_DIR / "Evénement X Concept - Concept X Magasin.csv"

# ── Image CDN ────────────────────────────────────────────────────────────────
# Confirmed live via HTTP probe — see README for details.
PRODUCT_IMAGE_BASE = "https://traiteur.carrefour.fr/media/catalog/product"
COMPOSITION_IMAGE_BASE = "https://traiteur.carrefour.fr/media"

# ── OpenAI / Pinecone ─────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
_DEFAULT_PINECONE_INDEX = "waib-carrefour-prod-large" if _ENV == "prod" else "waib-carrefour-dev-large"
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", _DEFAULT_PINECONE_INDEX)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))
