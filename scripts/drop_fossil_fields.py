"""Drop product fields that no version of this pipeline writes any more.

Every field below was produced by an earlier iteration of the ingestion and has
survived in MongoDB ever since, because ``bulk_upsert`` uses ``$set`` — which
updates the fields we send and leaves everything else untouched. They are
frozen at whatever the last pipeline that knew about them computed, and nothing
refreshes them.

Two are actively harmful rather than merely dead:

* ``persons_estimated`` — a guessed guest coverage, from a heuristic deleted
  precisely because inventing coverage distorts every quantity computed from
  it. waib-api still reads it as a fallback when Carrefour gives no
  ``nb_portion`` (``tools.py``'s menu snapshot), labelling it "estimation (non
  officielle)". Remove the field here and that fallback in the API together.
* ``category_tags`` — an invented occasion/season/cuisine/diet taxonomy. The
  real signals now come from Carrefour's own ``type_envie`` (diets) and from
  ``could_fit_event`` (LLM, cached and refreshable).

The rest are redundant copies or dead scaffolding, unread by the API:

* ``allergens`` / ``allergen_tags`` — copies of ``raw.type_allergene``, which
  the API reads directly (so no information is lost here).
* ``embed_text`` — a long concatenation built for embeddings that ended up
  using the product NAME only (a noisy embed dilutes the vector).
* ``is_food`` — a Pinecone metadata filter that no longer exists; the index is
  filtered on ``menu_step`` + ``status``.
* ``curated_concept`` — the per-product concept join, measured to have no
  effect on candidate ranking and dropped (see ingest/concepts.py).

Usage::

    poetry run python scripts/drop_fossil_fields.py            # dry-run, no writes
    poetry run python scripts/drop_fossil_fields.py --apply    # actually unset
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.db import get_db  # noqa: E402  pylint: disable=wrong-import-position
from ingest.log import get_logger  # noqa: E402  pylint: disable=wrong-import-position

log = get_logger(__name__)

FOSSIL_FIELDS = [
    "persons_estimated",
    "category_tags",
    "allergens",
    "allergen_tags",
    "embed_text",
    "is_food",
    "curated_concept",
]


def main(apply: bool) -> None:
    db = get_db()
    total = 0
    for field in FOSSIL_FIELDS:
        count = db.products.count_documents({field: {"$exists": True}})
        total += count
        print(f"  {field:<20} {count:>6} documents")

    if not total:
        print("Nothing to drop — the collection is already clean.")
        return

    if not apply:
        print(f"\nDry-run: {total} field occurrences would be unset. Re-run with --apply.")
        return

    result = db.products.update_many(
        {"$or": [{f: {"$exists": True}} for f in FOSSIL_FIELDS]},
        {"$unset": {f: "" for f in FOSSIL_FIELDS}},
    )
    log.info("fossil_fields_dropped", fields=FOSSIL_FIELDS, documents=result.modified_count)
    print(f"\n✅ {result.modified_count} documents cleaned.")

    remaining = {f: db.products.count_documents({f: {"$exists": True}}) for f in FOSSIL_FIELDS}
    if any(remaining.values()):
        print(f"⚠️  still present: { {k: v for k, v in remaining.items() if v} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--apply", action="store_true", help="Actually unset the fields")
    main(parser.parse_args().apply)
