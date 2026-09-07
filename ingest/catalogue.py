"""
Store catalogue builder — precomputes per-store, per-step product availability.

Runs an aggregation over ``prices × products`` to compute, for each store and
each menu step, three things, persisted on the store document:
  • ``step_catalogue: {menu_step: count}``  — how many active products exist.
  • ``step_families: {menu_step: [sub-family, ...]}`` — WHICH kinds of product the
    store actually carries in that step (vin, jus, champagne, cidre… for Boissons).
    Injected into the API's query planner so it searches for sub-families that exist
    instead of guessing ones that don't — see ``ingest.families`` for the rationale.
  • ``step_typical_cost: {menu_step: median_price_per_person}`` — the MEDIAN €/guest
    for that step (= median over products of ``store_price / persons``). The median
    (not the min) reflects what the composer ACTUALLY spends — a min-of-cheapest
    estimate made the step recommender wildly over-propose steps, then bust the budget.
    Multiplying by the guest count gives a realistic per-step cost the recommender
    uses to decide which steps fit a budget. For ``Plats`` the median is taken over
    MAIN dishes only (``dish_role != "side"``): a Plats step is anchored by a main,
    so a cheap side (gratin) must not define its cost. Absent when no product
    in the step has a known portion size (e.g. drinks).

An active store that sells nothing orderable gets an explicit empty
``step_catalogue: {}`` rather than no field at all, so "fulfils nothing" is
distinguishable from "never computed".

Must be run **after** ``--products`` and ``--prices`` have both been ingested,
so that the ``products`` and ``prices`` collections are up to date.

Usage::

    poetry run python run.py --catalogue

The result is used by the API to inject live store availability into
``live_context`` without a costly double-query on every turn — the API now
only needs a single ``find_one`` on the stores collection.
"""

import statistics

from ingest.families import extract_families
from ingest.log import get_logger

log = get_logger(__name__)


def build_store_catalogue(db) -> int:
    """Aggregate prices × products and upsert the three per-store catalogue summaries.

    For every store that has at least one priced active product, computes
    ``{menu_step: count}`` (→ ``stores.step_catalogue``),
    ``{menu_step: [sub-family, ...]}`` (→ ``stores.step_families``) and
    ``{menu_step: median_price_per_person}`` (→ ``stores.step_typical_cost``).

    The €/guest median (not the cheapest product) is the typical cost the composer
    actually spends; for ``Plats`` it is taken over MAIN dishes only so a cheap side
    doesn't make the step look artificially affordable. The aggregation collects the
    raw €/guest values per (store, step); the median is computed in Python below.

    Args:
        db: A live ``pymongo.database.Database`` instance (from ``get_db()``).

    Returns:
        The number of store documents updated.
    """
    log.info("catalogue_build_started")

    # Join prices → products, keep active products with a menu_step,
    # then group first by (store_id, menu_step), then roll up per store_id.
    pipeline = [
        # Join each price row with its product document.
        {
            "$lookup": {
                "from": "products",
                "localField": "product_id",
                "foreignField": "_id",
                "as": "product",
            }
        },
        {"$unwind": "$product"},
        # Only active products with a real menu step.
        {
            "$match": {
                "product.status": "active",
                "product.menu_step": {"$ne": None},
            }
        },
        # Price per person (€/guest) — only when the portion size and price are
        # known and positive; otherwise null so it's skipped from the median.
        # Plats SIDES (dish_role == "side") are excluded too: a Plats step is
        # anchored by a main, so a cheap gratin must not define its typical cost.
        {
            "$addFields": {
                "ppp": {
                    "$cond": [
                        {
                            "$and": [
                                {"$gt": ["$product.persons", 0]},
                                {"$gt": ["$price", 0]},
                                {
                                    "$not": [
                                        {
                                            "$and": [
                                                {"$eq": ["$product.menu_step", "Plats"]},
                                                {"$eq": ["$product.dish_role", "side"]},
                                            ]
                                        }
                                    ]
                                },
                            ]
                        },
                        {"$divide": ["$price", "$product.persons"]},
                        None,
                    ]
                }
            }
        },
        # Per (store_id, menu_step): product count + all €/guest values (nulls kept,
        # filtered in Python) + the product names the sub-families are derived from.
        # Count includes sides; the cost estimate (ppp) does not.
        {
            "$group": {
                "_id": {"store_id": "$store_id", "menu_step": "$product.menu_step"},
                "count": {"$sum": 1},
                "ppps": {"$push": "$ppp"},
                "names": {"$push": "$product.name"},
            }
        },
        # Roll up: one doc per store_id with parallel {k, v} arrays.
        {
            "$group": {
                "_id": "$_id.store_id",
                "steps": {"$push": {"k": "$_id.menu_step", "v": "$count"}},
                "ppps": {"$push": {"k": "$_id.menu_step", "v": "$ppps"}},
                "names": {"$push": {"k": "$_id.menu_step", "v": "$names"}},
            }
        },
        # Convert arrays to objects keyed by menu_step. Names are condensed into
        # sub-families in Python below and NOT persisted raw — the whole point is a
        # compact hint, and a full name list per store would bloat the store doc.
        {
            "$project": {
                "step_catalogue": {"$arrayToObject": "$steps"},
                "step_ppps": {"$arrayToObject": "$ppps"},
                "step_names": {"$arrayToObject": "$names"},
            }
        },
    ]

    updated = 0
    for doc in db.prices.aggregate(pipeline, allowDiskUse=True):
        # Median €/guest per step, computed over the non-null values (cheapest-side
        # and portion-less products were nulled out above). Steps with no valid value
        # (e.g. drinks) are omitted — the recommender treats a missing step as "coût
        # variable", same as the old null.
        step_typical_cost = {}
        for step, ppps in (doc.get("step_ppps") or {}).items():
            valid = [v for v in ppps if v is not None]
            if valid:
                step_typical_cost[step] = statistics.median(valid)

        # Sub-families per step, derived from the product names (see ingest.families).
        # Steps whose names yield nothing usable are omitted rather than stored empty,
        # so the API can tell "no data" from "genuinely nothing here".
        step_families = {}
        for step, names in (doc.get("step_names") or {}).items():
            families = extract_families(names)
            if families:
                step_families[step] = families

        db.stores.update_one(
            {"_id": doc["_id"]},
            {
                "$set": {
                    "step_catalogue": doc["step_catalogue"],
                    "step_families": step_families,
                    "step_typical_cost": step_typical_cost,
                },
                # Drop the legacy field (renamed from step_floor → step_typical_cost).
                "$unset": {"step_floor": ""},
            },
        )
        updated += 1
        log.debug(
            "store_catalogue_updated",
            store_id=doc["_id"],
            steps=list(doc["step_catalogue"].keys()),
        )

    # A store with no priced active product never appears in the aggregation above,
    # so it would keep NO step_catalogue at all — indistinguishable from a store
    # whose catalogue was simply never computed (this step skipped, fresh database).
    # Writing an explicit empty object makes "this store can fulfil nothing" a
    # positive, readable fact: the API can then exclude it from store suggestions
    # without a filter that would silently empty the list whenever --catalogue has
    # not been run.
    empty = db.stores.update_many(
        {"is_active": True, "step_catalogue": {"$exists": False}},
        {"$set": {"step_catalogue": {}, "step_families": {}, "step_typical_cost": {}}},
    )
    if empty.modified_count:
        log.warning(
            "stores_with_empty_catalogue",
            count=empty.modified_count,
            hint="Active stores selling nothing orderable — flagged explicitly, not left absent.",
        )

    log.info("catalogue_build_complete", stores_updated=updated, empty_catalogue=empty.modified_count)
    return updated
