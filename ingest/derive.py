"""
Derivation helpers — compute the minimal fields the AI pipeline needs
from raw Carrefour product records.

Philosophy: stay as close as possible to Carrefour's raw data.
We derive only what is strictly required for the app to function:
  - menu_step     → from LLM (see ingest/categorize.py) — NOT derived here anymore
  - persons       → from Carrefour's nb_portion field only
  - price_ref     → median across stores
  - recommendable → filter out "compose-it-yourself" products
  - dietary_tags  → the diet-related subset of Carrefour's own type_envie
  - family        → the head noun of the product name (verrines, gougères, charcuterie)

Everything else (dietary restrictions, allergens, occasion tags, etc.)
is kept as raw Carrefour data and left to the LLM to interpret.
"""

import statistics

from ingest.families import family_word
from ingest.log import get_logger

log = get_logger(__name__)


def derive_menu_step(product: dict) -> str | None:
    """Return the LLM-assigned menu_step injected upstream by batch_categorize().

    The categorization is done in bulk before transform_product() is called,
    and stored in the ``menu_step_llm`` key on the raw product dict.
    Returns None if not yet categorized (should not happen in normal flow).
    """
    return product.get("menu_step_llm")


# ── composable ────────────────────────────────────────────────────────────────
# "Build-your-own" products (e.g. "Plateau de 6 fromages" where the customer
# picks 6 cheeses from a real list of options) are NOT reliably identifiable
# from the name alone — e.g. "Plateau de 6 fromages" has no "à composer"/"au
# choix" wording at all, yet Carrefour supplies a full structured
# ``composition_plateau`` (groups of selectable pieces) for it. That structured
# field — not name-keyword matching — is the real, reliable signal: it's
# present (with actual choosable pieces) exactly on genuine build-your-own
# products, and absent everywhere else. ``left_picto_hyper``/``mots_cles`` are
# corroborating fallback signals for the rare product with the picto but no
# (or a malformed) composition_plateau block.


def _composition_plateau_groups(product: dict) -> list:
    comp = product.get("composition_plateau")
    if not isinstance(comp, dict):
        return []
    groups = comp.get("groups")
    return groups if isinstance(groups, list) else []


def derive_composable(product: dict) -> bool:
    """True for genuine build-your-own products — ONLY on real structured data.

    The ``left_picto_hyper`` / ``mots_cles`` fallbacks used to count too, for "the rare
    product with the picto but no composition_plateau block". Measured against the
    catalogue, that fallback caught nothing it was meant to: of the 32 products
    carrying the "A composer" picto, 21 have their structured groups and the other 11
    are all ``type_id: bundle`` opaque menu formulas ("Menu Classique", "Menu enfant",
    "Menu du Gastronome"…). It marked them composable, so the widget offered a
    "Composer" flow with nothing to choose from, and ``derive_recommendable`` let them
    through on the grounds that composable products are handled by that very flow.
    """
    return bool(_composition_plateau_groups(product))


# ── recommendable ──────────────────────────────────────────────────────────────
# "Compose-it-yourself" products with NO structured composition data (just
# vague name wording — "au choix", "à préciser" — and nothing Carrefour gives
# us to actually resolve the choice) are the ones the assistant genuinely can't
# handle; we flag those non-recommendable so they never surface in a menu
# suggestion the assistant can't back up with real choices.
# Products the customer builds via `derive_composable`'s REAL structured data
# ARE recommendable — that's exactly what the dedicated "Composer" flow (see
# is_composable on the stored document) is for, not a reason to hide them.

# NOT here: "à garnir". A product the customer garnishes themselves — "30 Navettes
# Natures (à garnir)" — is COMPLETE as sold: there is no choice for us to resolve, the
# rolls arrive plain and get filled at home. It belongs on a buffet, and this list was
# the only thing keeping it out. The keywords that remain describe a choice made at the
# counter from a card we never receive ("garniture au choix", "à préciser"), which the
# assistant genuinely cannot present.
_NON_RECOMMENDABLE_NAME_KEYWORDS = [
    "au choix",
    "à composer",
    "a composer",
    "composez",
    "à préciser",
    "a preciser",
]


def derive_recommendable(product: dict) -> bool:
    """False only for compose-it-yourself products with no structured data to
    back a real "Composer" flow. True for genuinely composable products
    (derive_composable) — the assistant CAN handle those via the dedicated
    composition UI, so excluding them entirely would be wrong."""
    if derive_composable(product):
        return True
    if _is_opaque_menu_bundle(product):
        return False
    name = (product.get("name") or "").lower()
    return not any(kw in name for kw in _NON_RECOMMENDABLE_NAME_KEYWORDS)


# Magento "bundle" products are formulas the customer assembles in store — a starter,
# a main and a dessert picked from a card we do not receive. All 12 active ones are
# "Menu X" ("Menu Classique", "Menu enfant", "Menu végétarien", "Menu du Gastronome",
# "Menu familial…"), all in Plats, and NONE carries a composition_plateau or even a
# named composition: there is nothing to tell the customer what they would eat.
#
# They are also a trap for the composer, which reads them as a cheap per-person main:
# on a 100-guest wedding it swapped a Bœuf Wellington for "Menu Classique" ×100 —
# 1290€, 43% of the budget, for a line the customer cannot inspect.
#
# The discriminator is `type_id`, not the name: the sushi platters are also called
# "Menu One", "Menu San", "Menu Love" and are perfectly explicit — they are
# `type_id: simple` with their piece count in the name.


def _is_opaque_menu_bundle(product: dict) -> bool:
    """A bundled formula with nothing to say about its contents."""
    if str(product.get("type_id") or "").strip().lower() != "bundle":
        return False
    if _composition_plateau_groups(product):
        return False
    pieces = (product.get("composition") or {}).get("pieces")
    return not pieces


# ── dietary_tags ──────────────────────────────────────────────────────────────
# Carrefour mixes two unrelated things in ``type_envie``: sensory/occasion tags
# (salé, froid, gastronomique…) and actual dietary restrictions (sans porc,
# végétarien…). waib-api needs the restrictions ISOLATED — it feeds them to the
# composer and to the dietary critic, which must not have to guess which of a
# dozen tags is a diet. So this is a strict projection of Carrefour's own
# vocabulary, never an inference: a product is only "végétarien" because
# Carrefour said so.
#
# This whitelist is the exhaustive set of diet values observed in the export
# (checked against all active products). A value Carrefour adds later is simply
# not surfaced until it is added here — deliberately fail-closed, because
# inventing a restriction is far worse than missing one.
_DIETARY_ENVIE_TAGS = frozenset({"sans porc", "sans viande", "sans poisson", "végétarien"})


def derive_dietary_tags(product: dict) -> list[str]:
    """Diet restrictions carried by Carrefour's ``type_envie``, in export order.

    Returns an empty list when the product has none — never None, so the API can
    treat "no tag" uniformly whether or not the field was ever written.
    """
    raw = product.get("type_envie") or []
    if isinstance(raw, str):
        raw = [raw]
    seen: list[str] = []
    for tag in raw:
        label = str(tag).strip().lower()
        if label in _DIETARY_ENVIE_TAGS and label not in seen:
            seen.append(label)
    return seen


# ── family ────────────────────────────────────────────────────────────────────


def derive_family(product: dict) -> str | None:
    """The kind of thing this product IS, from the head noun of its name.

    Same extraction that builds the per-store ``step_families`` hints, applied per
    product and stored on the document, because the API needs it for a question the
    hints cannot answer: is this STEP varied? Three products can be three distinct
    SKUs with no repeated ingredient and still be three verrines — "4 verrines
    saumon", "6 verrines tomates thon", "4 verrines pesto" was a real apéritif for a
    100-guest wedding, and nothing in the pipeline could see it as one experience
    served three times.

    Containers are skipped here (see ``family_word``): what a guest eats off a
    charcuterie platter and off a vegetable platter is not the same thing.
    """
    label = family_word(product.get("name") or "", skip_containers=True)
    return label or None


# ── persons ───────────────────────────────────────────────────────────────────


def derive_persons(product: dict) -> int | None:
    """Return how many people one unit serves, from Carrefour's nb_portion field.

    Returns None if the field is absent or not a positive integer — the LLM
    will infer coverage from the product name instead.
    No invented fallbacks (weight norms, piece counts, etc.).
    """
    val = product.get("nb_portion")
    if val is None:
        return None
    try:
        v = int(float(str(val).strip()))
        return v if v > 0 else None
    except (ValueError, TypeError):
        return None


# ── price_ref ─────────────────────────────────────────────────────────────────


def derive_price_ref(prices: list[float]) -> float | None:
    """Compute the median price across all stores.

    Used by the LLM when no store context is available.
    Returns None if the product has no price data at all.
    """
    if not prices:
        return None
    return round(statistics.median(prices), 2)
