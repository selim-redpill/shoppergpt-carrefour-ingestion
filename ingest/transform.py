"""
Transform raw Carrefour JSONL records into clean MongoDB documents.

Each transformer takes a raw ``dict`` (one JSONL line) and returns a
document ready for upsert.  No I/O is performed here — side-effect-free
by design so functions are easy to unit-test.
"""

import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

from ingest.config import COMPOSITION_IMAGE_BASE, PRODUCT_IMAGE_BASE
from ingest.derive import (
    derive_composable,
    derive_dietary_tags,
    derive_family,
    derive_menu_step,
    derive_persons,
    derive_price_ref,
    derive_recommendable,
)
from ingest.log import get_logger

log = get_logger(__name__)


def _image_url(path: str | None, base: str) -> str | None:
    """Resolve a relative Magento media path to an absolute CDN URL.

    Returns ``None`` if ``path`` is empty or ``None``.
    Leading slashes in ``path`` are stripped before joining.
    """
    if not path:
        return None
    return f"{base}/{path.lstrip('/')}"


def _safe_int(val: object) -> int | None:
    """Convert a value to ``int``, returning ``None`` if conversion fails."""
    if val is None:
        return None
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return None


# Smallest real beverage container in the catalog is a 37.5cl half-bottle; a 33cl
# can is 330. Anything under this is the raw field being a mass, not a volume.
MIN_PLAUSIBLE_VOLUME_ML = 200


def _volume_ml(raw: dict, menu_step: str | None) -> int | None:
    """Bottle volume in millilitres, for Boissons only.

    Carrefour ships this in ``raw.weight`` as a decimal STRING ("750.0000"), so
    Mongo can neither compare nor sum it. On food, the same field is a mass in
    grams — hence the menu_step guard.
    """
    if menu_step != "Boissons":
        return None
    ml = _safe_int(raw.get("weight"))
    if not ml or ml < MIN_PLAUSIBLE_VOLUME_ML:
        # Tea boxes (34 g) and ground coffee (250 g) reuse the same field for a
        # MASS. Returning None keeps the per-guest arithmetic from dividing by it.
        return None
    return ml


def transform_product(raw: dict, all_prices: dict[int, list[float]]) -> dict:
    """Transform one ``products.jsonl`` record into a ``products`` collection document.

    Args:
        raw:        Raw product dict from the JSONL export.
        all_prices: Pre-built mapping ``{product_id: [price, ...]}``,
                    used to compute ``price_ref`` (median across stores).

    Returns:
        A clean document ready for upsert (``_id`` = ``product_id``).
    """
    product_id = raw["product_id"]
    now = datetime.now(timezone.utc)

    menu_step = derive_menu_step(raw)
    is_composable = derive_composable(raw)
    recommendable = derive_recommendable(raw)
    persons = derive_persons(raw)
    dietary_tags = derive_dietary_tags(raw)
    price_ref = derive_price_ref(all_prices.get(product_id, []))

    # Composition — resolve piece image URLs
    comp_raw = raw.get("composition") or {}
    composition = None
    if comp_raw and comp_raw.get("pieces"):
        pieces = []
        for p in comp_raw["pieces"]:
            if isinstance(p, str):
                pieces.append({"name": p, "qty": 1, "image_url": None})
            elif isinstance(p, dict):
                pieces.append({
                    "name": p.get("name", ""),
                    "qty": _safe_int(p.get("qty")) or 1,
                    "image_url": _image_url(p.get("image"), COMPOSITION_IMAGE_BASE),
                })
        composition = {
            "title": comp_raw.get("title", ""),
            "pieces": pieces,
        }

    # composition_plateau — the "build-your-own" structure (e.g. "choisissez 6
    # fromages parmi 22"): grouped, selectable pieces, DIFFERENT shape from
    # ``composition`` above (which is a flat pieces list for buffets and was
    # never going to match this nested groups/pieces format). This is the same
    # structure derive_composable() checks for — kept here as clean, usable
    # data for the composer UI (piece name/conditioning/image per group),
    # instead of leaving callers to reach into raw JSON themselves.
    plateau_raw = raw.get("composition_plateau") or {}
    composition_plateau = None
    plateau_groups_raw = plateau_raw.get("groups")
    if isinstance(plateau_groups_raw, list) and plateau_groups_raw:
        groups = []
        for g in plateau_groups_raw:
            if not isinstance(g, dict):
                continue
            pieces = []
            for p in g.get("pieces") or []:
                if not isinstance(p, dict):
                    continue
                # `disabled: "on"` pieces are temporarily unavailable per Carrefour's
                # own data — excluded so the composer never offers a choice that
                # can't actually be ordered.
                if str(p.get("disabled") or "").strip().lower() == "on":
                    continue
                pieces.append({
                    # REQUIRED for the actual add-to-cart call later — the Cart
                    # API's `POST /cart/add` expects options.plateau keyed
                    # EXACTLY by this "{group_index}-{piece_index}" code (per
                    # the Carrefour API doc), not by name or position. Dropping
                    # it here would make the composed selection unusable at
                    # checkout time.
                    "code": p.get("code"),
                    "name": p.get("name", ""),
                    "conditionnement": p.get("conditionnement"),
                    "image_url": _image_url(p.get("image"), COMPOSITION_IMAGE_BASE),
                    # extra_price: "" (falsy) means no surcharge on the base
                    # plateau price for this piece — keep as float when present.
                    "extra_price": float(p["price"]) if p.get("price") else None,
                    # Per-piece composition/allergen text (e.g. "Appellation
                    # d'origine protégée... LAIT, sel, présure..."), HTML from
                    # Carrefour — the only place this level of detail exists
                    # (there's no separate product page for a sub-piece of a
                    # plateau), surfaced via an info icon on the composer UI.
                    "ingredients": p.get("ingredients") or None,
                })
            if pieces:
                groups.append({"name": g.get("name", ""), "pieces": pieces})
        if groups:
            composition_plateau = {
                "title": plateau_raw.get("title", ""),
                "qty": _safe_int(plateau_raw.get("qty")),
                "groups": groups,
            }

    status_raw = raw.get("status") or ""
    status = "active" if status_raw == "Activé" else "inactive"

    return {
        "_id": product_id,
        "sku": raw.get("sku"),
        "name": raw.get("name", ""),
        "status": status,
        "type_id": raw.get("type_id"),
        # ── App pipeline fields ──────────────────────────────────
        "menu_step": menu_step,
        # main|side for Plats products (None elsewhere) — lets the engine require one
        # protein main + optional accompaniments. From batch_classify_roles.
        "dish_role": raw.get("dish_role_llm"),
        # Drink family — selects the Carrefour proportion rule ("Vins : 1 bouteille
        # pour 4 personnes"…). Only set on Boissons; None everywhere else.
        "drink_role": raw.get("drink_role_llm"),
        # Bottle volume in ml, parsed out of the raw decimal string — the unit the
        # per-guest drink ratios are computed in. None outside Boissons.
        "volume_ml": _volume_ml(raw, menu_step),
        # Event(s) this product genuinely suits, e.g. ["Anniversaire", "Spécial enfant"],
        # or ["ALL"] for versatile products. From batch_classify_event_fit.
        "could_fit_event": raw.get("could_fit_event_llm") or ["ALL"],
        # OUR PREDICTION, read off the ingredient list by an LLM — deliberately NOT
        # merged into `dietary_tags` below, which holds Carrefour's own four
        # `type_envie` values. The name, the `source` field inside it and the wording
        # used downstream all have to keep the two apart: "Carrefour says this is
        # vegetarian" and "we read the ingredients and think it is" are different
        # promises. `profile: None` means unknown, never "contains meat".
        # From batch_classify_diets.
        "predicted_diet": raw.get("predicted_diet_llm"),
        # True for genuine build-your-own products — real structured Carrefour
        # data (composition_plateau below), not name-keyword guessing.
        "is_composable": is_composable,
        # False only for compose-it-yourself products with NO structured data
        # to back a real "Composer" flow (see derive_recommendable).
        "recommendable": recommendable,
        "persons": persons,
        # Diet restrictions only (Carrefour's own type_envie values) — read by the
        # composer and the dietary critic. Rewritten on every ingest so a product
        # Carrefour re-tags cannot keep a stale restriction.
        "dietary_tags": dietary_tags,
        # What the product IS (verrines, gougères, charcuterie) — lets the engine
        # tell a varied step from the same thing served three times.
        "family": derive_family(raw),
        "price_ref": price_ref,  # median across stores; None if no price data
        # ── Product details ──────────────────────────────────────
        "department": raw.get("carrefour_suppliers_department"),
        "bac_type": raw.get("bac_type"),
        "expression_pvc": raw.get("expression_pvc"),
        "delai_prepa": _safe_int(raw.get("delai_prepa")),
        "image_url": _image_url(raw.get("image"), PRODUCT_IMAGE_BASE),
        "categories": [
            {"id": c["category_id"], "name": c["category_name"]}
            for c in (raw.get("categories") or [])
        ],
        # ── Composition (plateaux/buffets) ───────────────────────
        "composition": composition,
        # "Build-your-own" grouped choices (e.g. "6 fromages parmi 22") — the
        # data the composer UI actually needs. None when not composable.
        "composition_plateau": composition_plateau,
        # ── Raw Carrefour data (source of truth) ─────────────────
        # Dietary info, allergens, ingredients etc. live here.
        # The LLM reads this directly — we don't pre-process it.
        "ingested_at": now,
        "raw": raw,
    }


def build_price_index(prices_file) -> dict[int, list[float]]:
    """Read ``products_prices.jsonl`` and return a price lookup by product.

    Flattens the nested store/price structure into a simple mapping
    ``{product_id: [price, price, ...]}``, keeping only rows that have
    a real numeric price (``prices: []`` rows are skipped).

    Args:
        prices_file: Path (or path-like) to ``products_prices.jsonl``.

    Returns:
        Dict mapping ``product_id`` to a list of all store prices for that product.
    """
    index: dict[int, list[float]] = {}
    _open = gzip.open(prices_file, "rt", encoding="utf-8") if Path(prices_file).suffix == ".gz" else open(prices_file, encoding="utf-8")
    with _open as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            pid = record["product_id"]
            flat: list[float] = []
            for store in record.get("stores", []):
                # Support both old structure (prices: [{price: x}])
                # and new structure (price: {price: x})
                prices_list = store.get("prices") or (
                    [store["price"]] if store.get("price") else []
                )
                for p in prices_list:
                    price = p.get("price")
                    if price is not None:
                        flat.append(float(price))
            if flat:
                index[pid] = flat
    return index


def transform_price_records(raw: dict) -> list[dict]:
    """Flatten one ``products_prices.jsonl`` record into individual price rows.

    Skips store entries that carry no price data.

    Args:
        raw: Raw record with ``product_id`` and nested ``stores`` list.

    Returns:
        List of ``{product_id, store_id, price}`` dicts, one per store with a price.
    """
    pid = raw["product_id"]
    docs = []
    for store in raw.get("stores", []):
        store_id = store.get("store_id")
        # Support both old structure (prices: [{price: x}]) and new
        # structure (price: {price: x}) — same as build_price_index.
        prices_list = store.get("prices") or (
            [store["price"]] if store.get("price") else []
        )
        for p in prices_list:
            price = p.get("price")
            if price is not None and store_id is not None:
                docs.append(
                    {
                        "product_id": pid,
                        "store_id": store_id,
                        "price": float(price),
                    }
                )
    return docs


def transform_store(raw: dict, store_concepts: dict[int, set[str]] | None = None) -> dict:
    """Transform one ``stores.jsonl`` record into a ``stores`` collection document.

    Builds a GeoJSON ``Point`` from ``longitude``/``latitude`` when available,
    enabling geospatial queries (e.g. find stores near a user).

    Args:
        raw:            Raw store dict from the JSONL export.
        store_concepts: ``{store_id: {concept_name, ...}}`` from
                        ``ingest.concepts.load_store_concepts`` — which of
                        Carrefour's curated concepts this store actually
                        carries (distinct from the messy raw ``concepts``
                        field below).

    Returns:
        A clean document ready for upsert (``_id`` = ``store_id``).
    """
    now = datetime.now(timezone.utc)

    geo = None
    try:
        lng = float(raw["longitude"])
        lat = float(raw["latitude"])
        if -180 <= lng <= 180 and -90 <= lat <= 90:
            geo = {"type": "Point", "coordinates": [lng, lat]}
        else:
            log.warning(
                "store_geo_out_of_bounds",
                store_id=raw.get("store_id"),
                name=raw.get("name"),
                lng=lng,
                lat=lat,
            )
    except (KeyError, TypeError, ValueError):
        pass

    return {
        "_id": raw["store_id"],
        "name": raw.get("name"),
        "code": raw.get("code"),
        "anabel_code": raw.get("anabel_code"),
        "type_label": raw.get("type_label"),
        "street_1": raw.get("street_1"),
        "street_2": raw.get("street_2"),
        "street_3": raw.get("street_3"),
        "city": raw.get("city"),
        "postcode": raw.get("postcode"),
        "is_active": raw.get("is_active", False),
        "withdrawal_store": raw.get("withdrawal_store", False),
        "drive": raw.get("drive", False),
        "geo": geo,
        "concepts": raw.get("concepts", []),
        # Carrefour's curated business taxonomy — which of the 18 concepts this
        # store actually carries (Statut=Activé rows only). Empty list if the
        # store isn't in the export. Drives the store-concept availability
        # filter in waib-api's engine.py, parallel to the existing price filter.
        "curated_concepts": sorted((store_concepts or {}).get(raw["store_id"], set())),
        "lad_postcodes": raw.get("lad_postcodes", []),
        "ingested_at": now,
    }
