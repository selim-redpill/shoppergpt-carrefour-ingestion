"""
LLM-based product categorization for Carrefour Traiteur.

Replaces the original static category→step mapping with a Gemini call.
Results are cached by product_id in MongoDB — each product is only categorized once.
Subsequent ingests use the cached value instantly.

Auth: same pattern as waib-api/gemini_http.py —
  - GOOGLE_GENAI_USE_VERTEXAI=true + ADC (prod / GCP)
  - GEMINI_API_KEY (dev / local AI Studio)

Usage (called from run.py before transform_product):
    step_cache = batch_categorize(db, raw_products)
    # step_cache: {product_id: menu_step}
"""

import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import google.auth
import google.auth.transport.requests
from pymongo import UpdateOne
from pymongo.database import Database

from ingest.log import get_logger

log = get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-3.5-flash-lite"
BATCH_SIZE = 50    # products per Gemini call
MAX_WORKERS = 10   # parallel Gemini calls
FALLBACK_STEP = "Plats"  # safe fallback for uncategorizable products — broadest, most generic step

VALID_STEPS = {
    "Apéritifs", "Entrées", "Plats", "Sauces", "Fromages",
    "Desserts", "Boissons", "Pains", "Petit Déj",
    "Table & Déco",
}

# Case-insensitive lookup (model sometimes returns UPPERCASE or mixed case)
_STEP_LOOKUP: dict[str, str] = {s.upper(): s for s in VALID_STEPS}

SYSTEM_PROMPT = """Tu es un expert en traiteur français. Catégorise chaque produit dans exactement un des steps suivants.
Lis attentivement les distinctions — elles sont importantes.

APÉRITIFS : finger food, amuse-bouches, canapés, mini-toasts, verrines, mousses en petits pots (format individuel), chips, crackers, dips, mini-brochettes, petits fours salés, mini-quiches, mini-burgers, pizzas découpées en toasts (ex: "pizza en 60 toasts"), œufs de poisson, plateau de charcuterie, plateau mixte apéro. Formats petits, à grignoter/partager debout.

ENTRÉES : plats froids ou chauds servis assis en début de repas. Carpaccio, foie gras entier/mi-cuit (en portion), terrine (en tranche), céviche, œufs mimosa, assiette de crudités, saumon fumé en tranche (format assiette, pas toast). Distinctions clés : foie gras en toast → Apéritifs. Saumon fumé en toast → Apéritifs. Saumon fumé en assiette/tranche → Entrées.

PLATS : plats principaux ET accompagnements. Viandes cuisinées (rôti, magret, souris d'agneau…), poissons cuisinés (pavé de saumon, filet…), volailles, lasagnes, gratins, pizzas entières, quiches entières, plats complets, sushis/makis, plateaux japonais, plateau du boucher, plateau BBQ. Également : légumes et accompagnements servis avec les plats (haricots verts, pommes de terre, salades vertes, légumes bruts, pois, carottes).

FROMAGES : fromages à la coupe ou en plateau (plateau de fromages), raclette, fondue.

DESSERTS : pâtisseries sucrées (gâteaux, tartes, macarons, entremets, éclairs, mille-feuilles, bûches, mignardises sucrées, fruits en dessert, coupes glacées), bonbons et confiseries. NE PAS inclure les viennoiseries du matin.

BOISSONS : toutes les boissons — eau, jus, sodas, champagne, vins, bières, cafés, thés, infusions.

PAINS : pain au sens strict — baguette, pain de campagne, pain de mie, pain aux céréales, pain surprise, pain de seigle, focaccia, pain burger (nature). PAS les viennoiseries.

PETIT DÉJ : viennoiseries (croissants, pains au chocolat, brioches, pains aux raisins, kouign-amann), assortiments petit-déjeuner, coffee break, paniers matinaux, chouquettes, madeleines, financiers. Produits consommés au petit-déjeuner ou à la pause café.

TABLE & DÉCO : vaisselle jetable, assiettes, gobelets, couverts plastique, serviettes en papier, nappes, chemins de table, bougies, ballons, décorations de fête, bouquets de fleurs, compositions florales, plantes. Également : produits ménagers (éponges, liquide vaisselle, sacs poubelle), ethylotests, et tout article non comestible.

SAUCES : sauces (mayonnaise, ketchup, nuoc-mâm, tapenade, anchoïade, béarnaise…), condiments (moutarde, cornichons…), assaisonnements (épices, sel, poivre, sucre), beurre. PAS les accompagnements alimentaires (légumes, salades, féculents) qui vont dans PLATS.

Réponds UNIQUEMENT avec du JSON valide où les clés sont les NUMÉROS des produits (pas les noms) :
{"1": "Desserts", "2": "Boissons", "3": "Apéritifs", ...}
Utilise exactement les noms de steps (avec accents et majuscules).
En cas de doute absolu, utilise "Plats"."""

# ── Dish role (main vs side) — second pass, Plats products only ────────────────
# The PLATS step deliberately lumps main dishes with their accompaniments (gratins,
# pommes de terre, légumes…). The composer/coverage then can't tell a protein main
# from a side, so it may pick two "plats" where one is just a garnish. This pass tags
# each Plats product main|side so the engine can require one main + optional sides.
# Conservative fallback = "main" (never demote a real main to a side).

VALID_ROLES = {"main", "side"}
ROLE_FALLBACK = "main"
_ROLE_LOOKUP: dict[str, str] = {r.upper(): r for r in VALID_ROLES}

ROLE_SYSTEM_PROMPT = """Tu es un expert en traiteur français. Pour chaque PLAT ci-dessous, indique s'il s'agit d'un plat principal (main) ou d'un accompagnement (side).

main (plat principal) : centré sur une protéine ou plat complet — viande/volaille/poisson/gibier cuisinés, rôtis, magrets, souris d'agneau, lasagnes, pizzas/quiches entières, sushis/makis, plats complets, gratins de viande/poisson.

side (accompagnement) : féculents et légumes servis EN ACCOMPAGNEMENT — gratin dauphinois, pommes de terre (purée, grenailles, sautées, dauphine), riz, pâtes nature, haricots verts, légumes poêlés/vapeur, salades vertes, crudités, ratatouille.

En cas de doute sur un plat végétarien substantiel et complet (ex. lasagnes de légumes) → main.

Réponds UNIQUEMENT en JSON valide où les clés sont les NUMÉROS des produits :
{"1": "main", "2": "side", ...}
En cas de doute absolu, utilise "main"."""


def _call_role_batch(batch: list[tuple[int, dict]]) -> dict[int, str]:
    """Call Gemini to tag a batch of (index, raw_product) as main|side. Returns {index: role}."""
    lines = [_format_product(i, raw) for i, raw in batch]
    prompt = f"{ROLE_SYSTEM_PROMPT}\n\nPlats :\n" + "\n".join(lines)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for attempt in range(3):
        try:
            response = _make_request(GEMINI_MODEL, payload)
            text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            result = {}
            for k, v in parsed.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                result[idx] = _ROLE_LOOKUP.get(str(v).upper().strip(), ROLE_FALLBACK)
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            log.warning("gemini_role_http_error", code=exc.code, batch_size=len(batch))
            return {i: ROLE_FALLBACK for i, _ in batch}
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
                continue
            log.warning("gemini_role_failed", error=str(exc), batch_size=len(batch))
            return {i: ROLE_FALLBACK for i, _ in batch}
    return {i: ROLE_FALLBACK for i, _ in batch}


def _load_role_cache(db: Database, product_ids: list[int]) -> dict[int, str]:
    cached = {}
    for doc in db.products.find(
        {"_id": {"$in": product_ids}, "dish_role": {"$ne": None}, "dish_role_source": "llm"},
        {"_id": 1, "dish_role": 1},
    ):
        cached[doc["_id"]] = doc["dish_role"]
    return cached


def _save_role_cache(db: Database, role_map: dict[int, str]) -> None:
    ops = [
        UpdateOne({"_id": pid}, {"$set": {"dish_role": role, "dish_role_source": "llm"}})
        for pid, role in role_map.items()
    ]
    if ops:
        db.products.bulk_write(ops, ordered=False)


def batch_classify_roles(
    db: Database,
    raw_products: list[dict],
    step_map: dict[int, str],
    force: bool = False,
) -> dict[int, str]:
    """Tag main|side for PLATS products only (cache-first, like batch_categorize).

    Args:
        db: MongoDB handle.
        raw_products: raw JSONL dicts (must have ``product_id``).
        step_map: ``{product_id: menu_step}`` from batch_categorize — selects Plats.
        force: ignore cache and re-classify.

    Returns:
        ``{product_id: "main"|"side"}`` for Plats products (empty for the rest).
    """
    plats_ids = [
        int(r["product_id"]) for r in raw_products if step_map.get(int(r["product_id"])) == "Plats"
    ]
    if not plats_ids:
        return {}
    id_to_raw = {int(r["product_id"]): r for r in raw_products}

    cached: dict[int, str] = {} if force else _load_role_cache(db, plats_ids)
    to_classify = [pid for pid in plats_ids if pid not in cached]
    log.info("dish_role_start", plats=len(plats_ids), from_cache=len(cached), via_llm=len(to_classify))
    if not to_classify:
        return cached

    indexed = [(i, id_to_raw[pid]) for i, pid in enumerate(to_classify, start=1)]
    batches = [indexed[i:i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]
    index_to_pid = {i: pid for i, pid in enumerate(to_classify, start=1)}
    llm_results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_call_role_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            llm_results.update(future.result())

    llm_by_pid = {index_to_pid[idx]: role for idx, role in llm_results.items() if idx in index_to_pid}
    _save_role_cache(db, llm_by_pid)
    final = {**cached, **llm_by_pid}
    dist: dict[str, int] = {}
    for role in final.values():
        dist[role] = dist.get(role, 0) + 1
    log.info("dish_role_complete", from_cache=len(cached), via_llm=len(llm_by_pid), distribution=dist)
    return final


# ── Drink role — which proportion rule sizes a drink (fourth pass) ────────────
# Carrefour supplies nb_portion (→ `persons`) on virtually every food product, but
# on only 1 of 166 drinks. Quantities for drinks were therefore left to the model,
# which sized EVERY drink to cover EVERY guest — four drinks meant four times the
# need (observed: 103€ of drinks on a 28€ step envelope).
#
# Carrefour publishes proportion rules per drink family ("Vins : 1 bouteille pour
# 4 personnes", "Champagnes : 1 pour 6"…). Those apply to the FAMILY as a whole,
# so the family is what we need to know per product. Tagged once here, at ingest,
# exactly like dish_role — the engine then does the arithmetic in code.

VALID_DRINK_ROLES = {
    "eau",
    "soft",
    "vin",
    "petillant",
    "champagne",
    "biere",
    "cidre",
    "spiritueux",
    "aperitif",
    "chaud",
    # Festive drinks that stand IN FOR alcohol: Champomy, alcohol-free beer,
    # alcohol-free cocktails. Filed as plain softs until now, which made a weak
    # kind of sense on the shelf and none at the table — they are what the guests
    # who do not drink alcohol raise at the toast, not what everyone drinks with
    # the meal. Sizing them as softs put 40 bottles of Champomy (120€) on a
    # 100-adult wedding.
    "sans_alcool",
}
# Cheapest family, drunk by everyone, never sized on adults alone: a
# misclassification lands on the safe side of both the budget and the guest count.
DRINK_ROLE_FALLBACK = "soft"
_DRINK_ROLE_LOOKUP: dict[str, str] = {r.upper(): r for r in VALID_DRINK_ROLES}

DRINK_ROLE_SYSTEM_PROMPT = """Tu es un expert en traiteur français. Pour chaque BOISSON ci-dessous, indique sa famille.

eau : eau plate ou gazeuse NON aromatisée (source, minérale).
soft : boissons non alcoolisées du quotidien — jus, nectars, sodas, limonades, thés glacés, boissons aux fruits, eaux aromatisées.
sans_alcool : boissons de FÊTE sans alcool, qui remplacent une boisson alcoolisée — jus de pomme pétillant type Champomy, cocktails sans alcool, bières sans alcool, vins/mousseux désalcoolisés. Le critère : le produit imite une boisson alcoolisée ou sert à trinquer.
vin : vin tranquille rouge, blanc ou rosé (y compris désigné par sa seule appellation : Riesling, Chablis, Coteaux-du-Layon, Sancerre…).
petillant : vin effervescent AUTRE que champagne — crémant, prosecco, mousseux, clairette.
champagne : champagne uniquement.
biere : bière AVEC alcool.
cidre : cidre.
spiritueux : alcools forts et liqueurs — whisky, vodka, rhum, gin, tequila, cognac, liqueurs.
aperitif : apéritifs à diluer — Aperol, Spritz, vermouth, porto, pastis.
chaud : boissons chaudes — café, thé en sachets/vrac, infusions, chocolat chaud.

ATTENTION aux pièges de nommage :
- "bière sans alcool" → sans_alcool (ni biere, ni soft)
- "cocktail sans alcool" → sans_alcool
- "jus de pomme pétillant" (Champomy) → sans_alcool (pas petillant)
- "eau aromatisée", "eau gazeuse aromatisée" → soft (pas eau)
- "thé glacé", "boisson au thé" → soft (pas chaud)
- un jus ou un soda ordinaire reste soft : "sans_alcool" est réservé à ce qui remplace un alcool.

Réponds UNIQUEMENT en JSON valide où les clés sont les NUMÉROS des produits :
{"1": "vin", "2": "soft", ...}
En cas de doute absolu, utilise "soft"."""

# Deterministic override, applied AFTER the model. These are naming traps where the
# family contradicts the words in the name, so a model reading that name is fooled
# by the same thing a keyword match would be — and the cost is asymmetric: sizing a
# flavoured water as table water also exempts it from budget arbitration, and
# sizing an alcohol-free beer on adults only under-serves everyone else.
_DRINK_ROLE_OVERRIDES: list[tuple[str, str]] = [
    # Alcohol-free versions of alcoholic drinks, and the sparkling apple juice sold
    # for toasting: they replace an alcohol, so they are sized for the guests who
    # don't drink one — not for everybody like a soda.
    (r"sans[\s-]?alcool", "sans_alcool"),
    (r"champomy", "sans_alcool"),
    (r"\beaux?\b.{0,20}aromatis", "soft"),
    (r"th[ée] glac|boisson au th[ée]|ice[\s-]?tea", "soft"),
    # Kept LAST: a box of tea bags / ground coffee is a hot drink, not a soft.
    # The iced-tea patterns above must win over this one.
    (r"\bth[ée]s?\b|caf[ée]|chicor[ée]e|infusion", "chaud"),
]


def _drink_role_override(name: str) -> str | None:
    """Family imposed by an unambiguous naming trap, or None."""
    for pattern, role in _DRINK_ROLE_OVERRIDES:
        if re.search(pattern, name, re.IGNORECASE):
            return role
    return None


def _call_drink_role_batch(batch: list[tuple[int, dict]]) -> dict[int, str]:
    """Ask Gemini for the family of a batch of drinks. Returns {index: role}."""
    lines = [_format_product(i, raw) for i, raw in batch]
    prompt = f"{DRINK_ROLE_SYSTEM_PROMPT}\n\nBoissons :\n" + "\n".join(lines)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for attempt in range(3):
        try:
            response = _make_request(GEMINI_MODEL, payload)
            text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            result = {}
            for k, v in parsed.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                result[idx] = _DRINK_ROLE_LOOKUP.get(
                    str(v).upper().strip(), DRINK_ROLE_FALLBACK
                )
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 2:
                time.sleep(2**attempt * 2)
                continue
            log.warning("gemini_drink_role_http_error", code=exc.code, batch_size=len(batch))
            return {i: DRINK_ROLE_FALLBACK for i, _ in batch}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if attempt < 2:
                time.sleep(1.5**attempt)
                continue
            log.warning("gemini_drink_role_failed", error=str(exc), batch_size=len(batch))
            return {i: DRINK_ROLE_FALLBACK for i, _ in batch}
    return {i: DRINK_ROLE_FALLBACK for i, _ in batch}


def _load_drink_role_cache(db: Database, product_ids: list[int]) -> dict[int, str]:
    cached = {}
    for doc in db.products.find(
        {"_id": {"$in": product_ids}, "drink_role": {"$ne": None}, "drink_role_source": "llm"},
        {"_id": 1, "drink_role": 1},
    ):
        cached[doc["_id"]] = doc["drink_role"]
    return cached


def _save_drink_role_cache(db: Database, role_map: dict[int, str]) -> None:
    ops = [
        UpdateOne({"_id": pid}, {"$set": {"drink_role": role, "drink_role_source": "llm"}})
        for pid, role in role_map.items()
    ]
    if ops:
        db.products.bulk_write(ops, ordered=False)


def batch_classify_drink_roles(
    db: Database,
    raw_products: list[dict],
    step_map: dict[int, str],
    force: bool = False,
) -> dict[int, str]:
    """Tag a drink family on BOISSONS products only (cache-first, like dish_role).

    Args:
        db: MongoDB handle.
        raw_products: raw JSONL dicts (must have ``product_id``).
        step_map: ``{product_id: menu_step}`` from batch_categorize — selects Boissons.
        force: ignore cache and re-classify.

    Returns:
        ``{product_id: role}`` for drinks (empty for the rest).
    """
    drink_ids = [
        int(r["product_id"])
        for r in raw_products
        if step_map.get(int(r["product_id"])) == "Boissons"
    ]
    if not drink_ids:
        return {}
    id_to_raw = {int(r["product_id"]): r for r in raw_products}

    cached: dict[int, str] = {} if force else _load_drink_role_cache(db, drink_ids)
    to_classify = [pid for pid in drink_ids if pid not in cached]
    log.info(
        "drink_role_start",
        drinks=len(drink_ids),
        from_cache=len(cached),
        via_llm=len(to_classify),
    )
    if not to_classify:
        return cached

    indexed = [(i, id_to_raw[pid]) for i, pid in enumerate(to_classify, start=1)]
    batches = [indexed[i : i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]
    index_to_pid = {i: pid for i, pid in enumerate(to_classify, start=1)}
    llm_results: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_call_drink_role_batch, batch): batch for batch in batches}
        for future in as_completed(futures):
            llm_results.update(future.result())

    llm_by_pid = {
        index_to_pid[idx]: role for idx, role in llm_results.items() if idx in index_to_pid
    }
    # Fallback for anything the model dropped, then the naming-trap overrides.
    overridden = 0
    for pid in to_classify:
        name = str(id_to_raw[pid].get("name") or "")
        forced = _drink_role_override(name)
        if forced:
            if llm_by_pid.get(pid) != forced:
                overridden += 1
            llm_by_pid[pid] = forced
        else:
            llm_by_pid.setdefault(pid, DRINK_ROLE_FALLBACK)

    _save_drink_role_cache(db, llm_by_pid)
    final = {**cached, **llm_by_pid}
    dist: dict[str, int] = {}
    for role in final.values():
        dist[role] = dist.get(role, 0) + 1
    log.info(
        "drink_role_complete",
        from_cache=len(cached),
        via_llm=len(llm_by_pid),
        overridden_by_name=overridden,
        distribution=dist,
    )
    return final


# ── Event fit — which occasions a product genuinely suits (third pass) ────────
# Complements menu_step: a product can be correctly categorized (e.g. Table & Déco)
# yet still be wrong for a given occasion (birthday candles for a business lunch).
# This tags each product with the events it plausibly fits, computed ONCE at ingest
# by a stronger model than the runtime per-request vetting call — so at compose time
# a sparse/risky pool can be widened with genuinely-relevant products instead of
# being padded with random ones. "ALL" is the universal tag for versatile products
# (a cheese platter, a plain baguette) that suit virtually any occasion — the model
# is told explicitly to use it rather than force an artificial partial list.

EVENT_CATEGORIES = [
    "Anniversaire",
    "Noël et Nouvel An",
    "Mariage",
    "Pâques",
    "Repas en famille",
    "Naissance et Baptême",
    "Brunch et Petit Déjeuner",
    "Apéro Dînatoire",
    "Spécial enfant",
    "Barbecue",
    "Pique-nique",
    "Dîner en amoureux",
    "Pot de départ",
]
_EVENT_LOOKUP: dict[str, str] = {e.upper(): e for e in EVENT_CATEGORIES}
_EVENT_LOOKUP["ALL"] = "ALL"
# Fail-safe default: never let a classification failure silently exclude a product
# from every occasion — "ALL" only ever widens the pool, so it's the safe fallback.
EVENT_FIT_FALLBACK = ["ALL"]

EVENT_FIT_SYSTEM_PROMPT = (
    "Tu es un expert en traiteur français. Pour chaque produit ci-dessous, indique à "
    "quel(s) type(s) d'événement il convient VRAIMENT, parmi cette liste :\n"
    + ", ".join(EVENT_CATEGORIES)
    + "\n\nRègles :\n"
    "- Si le produit est polyvalent et convient à pratiquement n'importe quelle occasion "
    "(ex. plateau de fromages, pain, eau minérale, salade composée), réponds UNIQUEMENT "
    "[\"ALL\"] — ne force pas une liste partielle artificielle pour un produit générique.\n"
    "- Si le produit est spécifique à un ou plusieurs événements précis (ex. bougies "
    "d'anniversaire, bûche de Noël, faire-part, décoration de baptême, panier pique-nique "
    "jetable), liste UNIQUEMENT ces événements précis — pas \"ALL\".\n"
    "- Sois strict sur les produits de niche/décoration/thème (Table & Déco, produits "
    "enfants, produits festifs) : c'est là que l'erreur coûte le plus cher (ex. des "
    "confettis d'anniversaire ne conviennent PAS à un pot de départ professionnel).\n"
    "- Un produit alimentaire neutre (viande, poisson, légume, dessert classique non "
    "thématique) est presque toujours \"ALL\".\n\n"
    "Réponds UNIQUEMENT en JSON valide où les clés sont les NUMÉROS des produits et les "
    "valeurs des LISTES de chaînes exactes de la liste ci-dessus (ou [\"ALL\"]) :\n"
    '{"1": ["ALL"], "2": ["Anniversaire", "Spécial enfant"], ...}\n'
    "En cas de doute absolu, réponds [\"ALL\"]."
)


def _call_event_fit_batch(batch: list[tuple[int, dict]]) -> dict[int, list[str]]:
    """Call Gemini to tag a batch of (index, raw_product) with event-fit labels."""
    lines = [_format_product(i, raw) for i, raw in batch]
    prompt = f"{EVENT_FIT_SYSTEM_PROMPT}\n\nProduits :\n" + "\n".join(lines)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    for attempt in range(3):
        try:
            response = _make_request(GEMINI_MODEL, payload)
            text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            result: dict[int, list[str]] = {}
            for k, v in parsed.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                if not isinstance(v, list):
                    result[idx] = list(EVENT_FIT_FALLBACK)
                    continue
                tags = [_EVENT_LOOKUP[t] for t in (str(x).upper().strip() for x in v) if t in _EVENT_LOOKUP]
                # "ALL" alongside specific tags is redundant/contradictory — keep just ALL.
                if "ALL" in tags or not tags:
                    result[idx] = ["ALL"]
                else:
                    result[idx] = sorted(set(tags))
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            log.warning("gemini_event_fit_http_error", code=exc.code, batch_size=len(batch))
            return {i: list(EVENT_FIT_FALLBACK) for i, _ in batch}
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
                continue
            log.warning("gemini_event_fit_failed", error=str(exc), batch_size=len(batch))
            return {i: list(EVENT_FIT_FALLBACK) for i, _ in batch}
    return {i: list(EVENT_FIT_FALLBACK) for i, _ in batch}


def _load_event_fit_cache(db: Database, product_ids: list[int]) -> dict[int, list[str]]:
    cached = {}
    for doc in db.products.find(
        {"_id": {"$in": product_ids}, "could_fit_event": {"$ne": None}, "could_fit_event_source": "llm"},
        {"_id": 1, "could_fit_event": 1},
    ):
        cached[doc["_id"]] = doc["could_fit_event"]
    return cached


def _save_event_fit_cache(db: Database, event_fit_map: dict[int, list[str]]) -> None:
    ops = [
        UpdateOne({"_id": pid}, {"$set": {"could_fit_event": tags, "could_fit_event_source": "llm"}})
        for pid, tags in event_fit_map.items()
    ]
    if ops:
        db.products.bulk_write(ops, ordered=False)


def batch_classify_event_fit(
    db: Database,
    raw_products: list[dict],
    force: bool = False,
) -> dict[int, list[str]]:
    """Tag every product with the event(s) it genuinely fits (cache-first, like
    batch_categorize). ``["ALL"]`` marks a versatile product suiting any occasion.

    Args:
        db: MongoDB handle.
        raw_products: raw JSONL dicts (must have ``product_id``).
        force: ignore cache and re-classify.

    Returns:
        ``{product_id: [event, ...]}`` for every input product.
    """
    all_ids = [int(r["product_id"]) for r in raw_products]
    id_to_raw = {int(r["product_id"]): r for r in raw_products}

    cached: dict[int, list[str]] = {} if force else _load_event_fit_cache(db, all_ids)
    to_classify = [pid for pid in all_ids if pid not in cached]
    log.info("event_fit_start", total=len(all_ids), from_cache=len(cached), via_llm=len(to_classify))
    if not to_classify:
        return cached

    indexed = [(i, id_to_raw[pid]) for i, pid in enumerate(to_classify, start=1)]
    batches = [indexed[i:i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]
    index_to_pid = {i: pid for i, pid in enumerate(to_classify, start=1)}
    llm_results: dict[int, list[str]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_call_event_fit_batch, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            llm_results.update(future.result())
            log.info("event_fit_progress", batches_done=completed, total_batches=len(batches))

    llm_by_pid = {index_to_pid[idx]: tags for idx, tags in llm_results.items() if idx in index_to_pid}
    _save_event_fit_cache(db, llm_by_pid)
    final = {**cached, **llm_by_pid}

    dist: dict[str, int] = {}
    for tags in final.values():
        for t in tags:
            dist[t] = dist.get(t, 0) + 1
    log.info("event_fit_complete", from_cache=len(cached), via_llm=len(llm_by_pid), distribution=dist)
    return final


# ── Diet prediction (OURS, not Carrefour's) ───────────────────────────────────
#
# WHAT THIS IS FOR: composing a VARIED menu. It answers "what kind of dish is this,
# broadly" so the assistant can put a meat main, a fish main and a meat-free main on the
# same wedding table instead of three trays of beef. It is a CONCEPT-level judgement.
#
# WHAT THIS IS NOT: a dietary guarantee. Carrefour Traiteur does not certify restrictions
# today, and neither do we. So this must never reach a customer as a promise — not
# "halal", not "casher", not "guaranteed vegetarian". The field name, the `source` marker
# and the wording sent to the model all keep it labelled as our own estimate.
#
# Deliberately NOT precise, and that is a product decision, not a shortcut. Judging at
# ingredient-trace level produced worse menus, not safer ones: "4 verrines pesto tomates
# et mozzarella" was ruled non-vegetarian by a "Traces éventuelles de POISSON" line, and
# "Cannellonis ricotta épinards" by beef gelatin used as a texture agent. Both are
# vegetarian dishes by any cook's reckoning, and excluding them shrinks the menu without
# protecting anyone — because the guarantee was never on offer in the first place.
#
# Why it exists at all: Carrefour's own diet field cannot carry a menu. Of the 265 main
# dishes it tags exactly one `végétarien`, "Tagliatelles au surimi", which is a fish
# dish, while 14 genuinely meat-free mains at a single store carry no tag. A 100-guest
# wedding was served two kinds of potato as its vegetarian option.

# Mutually exclusive, ordered from most to least restrictive. A dish gets exactly one.
DIET_PROFILES = ("vegan", "vegetarien", "poisson", "viande")
_DIET_PROFILE_LOOKUP: dict[str, str] = {p.upper(): p for p in DIET_PROFILES}

# What we believe the dish CONTAINS, read off the ingredient list. Religious
# compatibility is derived from these downstream rather than asserted here: pork and
# alcohol are observable in a list of ingredients, "halal" is a certification.
DIET_CONTAINS = (
    "porc",
    "alcool",
    "crustaces",
    "poisson",
    "viande",
    "lait",
    "oeuf",
    "gelatine",
)
_DIET_CONTAINS_LOOKUP: dict[str, str] = {c.upper(): c for c in DIET_CONTAINS}

# No fallback profile. A failed or impossible classification stays None — "unknown" is
# the honest answer and the guards downstream treat it as "no evidence", never as
# "contains meat" or "is vegetarian". Inventing "viande" here would quietly hide 77
# vegetarian dishes; inventing "vegetarien" would put fish on a vegetarian's plate.
DIET_PREDICTION_SOURCE = "waib_llm_ingredients"

DIET_SYSTEM_PROMPT = (
    "Tu es un chef traiteur qui trie une carte pour composer des menus variés. Pour "
    "chaque plat ci-dessous, tu disposes de son nom et de sa liste d'ingrédients. Dis à "
    "quelle CATÉGORIE DE PLAT il appartient, comme le ferait un cuisinier qui lit une "
    "carte — pas comme un service d'allergologie.\n\n"
    "`profil` — exactement UNE valeur :\n"
    "- \"vegan\" : plat entièrement végétal (légumes, céréales, légumineuses), sans "
    "fromage, sans œuf, sans crème.\n"
    "- \"vegetarien\" : plat sans viande ni poisson, à base de légumes, de fromage, "
    "d'œuf ou de pâtes — une pizza margherita, une quiche aux légumes, des cannellonis "
    "ricotta-épinards, des pâtes au pesto.\n"
    "- \"poisson\" : le plat est un plat de poisson ou de fruits de mer — c'est son "
    "ingrédient principal.\n"
    "- \"viande\" : le plat est un plat de viande, de volaille ou de charcuterie — "
    "c'est son ingrédient principal.\n\n"
    "RAISONNE AU CONCEPT DU PLAT, pas à la trace :\n"
    "- IGNORE totalement les mentions « traces éventuelles de… » : ce sont des "
    "avertissements d'usine, pas des ingrédients.\n"
    "- IGNORE les additifs techniques en quantité infime (gélatine, présure, arômes, "
    "bouillon) : des cannellonis ricotta-épinards restent un plat végétarien même si "
    "leur liste mentionne de la gélatine.\n"
    "- Fonde-toi sur les ingrédients PRINCIPAUX, ceux qui font le plat. Si le nom "
    "annonce un plat de légumes et que les ingrédients principaux sont des légumes, du "
    "fromage et des pâtes, c'est \"vegetarien\".\n"
    "- En revanche, un ingrédient animal qui FAIT le plat compte pleinement : jambon "
    "dans une pizza jambon-fromage, thon dans une verrine au thon, chorizo dans un "
    "soufflé au chorizo, lardons dans une quiche lorraine.\n\n"
    "`contient` — parmi "
    + ", ".join(DIET_CONTAINS)
    + " — ce que le plat contient DE FAÇON SIGNIFICATIVE, au même niveau de lecture "
    "(un plat au jambon → \"porc\" ; une sauce au vin → \"alcool\" ; une trace "
    "d'usine → rien).\n\n"
    "RÈGLE ABSOLUE : si la liste d'ingrédients est absente et que le nom ne permet pas "
    'de trancher, réponds `{"profil": null, "contient": []}`. Sinon, tranche : un plat '
    "sans profil est un plat que l'assistant ne pourra pas proposer pour varier un "
    "menu.\n\n"
    "Réponds UNIQUEMENT en JSON valide, les clés étant les NUMÉROS des plats :\n"
    '{"1": {"profil": "vegetarien", "contient": ["lait", "oeuf"]}, '
    '"2": {"profil": "viande", "contient": ["porc"]}, '
    '"3": {"profil": null, "contient": []}}'
)


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Carrefour's ingredient lists are HTML fragments with escaped entities
# ("P&acirc;te (eau, sel...)"). Long enough to carry the whole recipe, short enough to
# keep 50 of them in one prompt.
_MAX_INGREDIENTS_CHARS = 700


def _clean_ingredients(raw_html: str | None) -> str:
    """Flatten a Carrefour ingredient list into plain text. Empty when there is none."""
    if not raw_html:
        return ""
    text = html.unescape(str(raw_html))
    text = _HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)  # entities sometimes survive one pass ("&amp;eacute;")
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:_MAX_INGREDIENTS_CHARS]


def _format_diet_product(i: int, raw: dict) -> str:
    name = raw.get("name") or ""
    ingredients = _clean_ingredients(raw.get("ingredients"))
    return f'{i}. Nom: "{name}"\n   Ingrédients: {ingredients or "ABSENTS"}'


def _parse_diet_value(value: Any) -> dict[str, Any]:
    """One product's answer → a stored prediction. Anything unusable becomes unknown."""
    if not isinstance(value, dict):
        return {"profile": None, "contains": []}
    profile = _DIET_PROFILE_LOOKUP.get(str(value.get("profil") or "").upper().strip())
    raw_contains = value.get("contient")
    contains = (
        sorted(
            {
                _DIET_CONTAINS_LOOKUP[c]
                for c in (str(x).upper().strip() for x in raw_contains)
                if c in _DIET_CONTAINS_LOOKUP
            }
        )
        if isinstance(raw_contains, list)
        else []
    )
    # A profile that contradicts its own `contains` list is a model slip, and the two
    # halves come from the same call — so trust the ingredient-level detail and drop the
    # summary rather than keeping a "vegetarien" dish that admits to containing meat.
    if profile in ("vegan", "vegetarien") and ({"viande", "poisson", "crustaces"} & set(contains)):
        profile = None
    elif profile == "poisson" and "viande" in contains:
        profile = None
    return {"profile": profile, "contains": contains}


def _call_diet_batch(batch: list[tuple[int, dict]]) -> dict[int, dict[str, Any]]:
    """Call Gemini to predict the diet profile of a batch of (index, raw_product)."""
    lines = [_format_diet_product(i, raw) for i, raw in batch]
    prompt = f"{DIET_SYSTEM_PROMPT}\n\nPlats :\n" + "\n".join(lines)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }
    unknown = {"profile": None, "contains": []}
    for attempt in range(3):
        try:
            response = _make_request(GEMINI_MODEL, payload)
            text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            result: dict[int, dict[str, Any]] = {}
            for k, v in parsed.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue
                result[idx] = _parse_diet_value(v)
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            log.warning("gemini_diet_http_error", code=exc.code, batch_size=len(batch))
            return {i: dict(unknown) for i, _ in batch}
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
                continue
            log.warning("gemini_diet_failed", error=str(exc), batch_size=len(batch))
            return {i: dict(unknown) for i, _ in batch}
    return {i: dict(unknown) for i, _ in batch}


def _load_diet_cache(db: Database, product_ids: list[int]) -> dict[int, dict[str, Any]]:
    cached = {}
    for doc in db.products.find(
        {"_id": {"$in": product_ids}, "predicted_diet.source": DIET_PREDICTION_SOURCE},
        {"_id": 1, "predicted_diet": 1},
    ):
        cached[doc["_id"]] = doc["predicted_diet"]
    return cached


def _save_diet_cache(db: Database, diet_map: dict[int, dict[str, Any]]) -> None:
    ops = [
        UpdateOne({"_id": pid}, {"$set": {"predicted_diet": prediction}})
        for pid, prediction in diet_map.items()
    ]
    if ops:
        db.products.bulk_write(ops, ordered=False)


def batch_classify_diets(
    db: Database,
    raw_products: list[dict],
    force: bool = False,
) -> dict[int, dict[str, Any]]:
    """Predict each product's diet profile from its ingredient list (cache-first).

    Args:
        db: MongoDB handle.
        raw_products: raw JSONL dicts (must have ``product_id``).
        force: ignore cache and re-classify.

    Returns:
        ``{product_id: {"profile": str|None, "contains": [str], "source": ..., "model": ...}}``
        for every input product. ``profile: None`` means unknown — no ingredient list, or
        an answer we refused. It never means "contains meat".
    """
    all_ids = [int(r["product_id"]) for r in raw_products]
    id_to_raw = {int(r["product_id"]): r for r in raw_products}

    cached = {} if force else _load_diet_cache(db, all_ids)
    to_classify = [pid for pid in all_ids if pid not in cached]
    log.info("diet_start", total=len(all_ids), from_cache=len(cached), via_llm=len(to_classify))
    if not to_classify:
        return cached

    indexed = [(i, id_to_raw[pid]) for i, pid in enumerate(to_classify, start=1)]
    batches = [indexed[i:i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]
    index_to_pid = {i: pid for i, pid in enumerate(to_classify, start=1)}
    llm_results: dict[int, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_call_diet_batch, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            llm_results.update(future.result())
            log.info("diet_progress", batches_done=completed, total_batches=len(batches))

    llm_by_pid = {
        index_to_pid[idx]: {**pred, "source": DIET_PREDICTION_SOURCE, "model": GEMINI_MODEL}
        for idx, pred in llm_results.items()
        if idx in index_to_pid
    }
    _save_diet_cache(db, llm_by_pid)
    final = {**cached, **llm_by_pid}

    dist: dict[str, int] = {}
    for pred in final.values():
        dist[str(pred.get("profile"))] = dist.get(str(pred.get("profile")), 0) + 1
    log.info("diet_complete", from_cache=len(cached), via_llm=len(llm_by_pid), profiles=dist)
    return final


# ── Auth helpers (mirrors waib-api/gemini_http.py) ────────────────────────────

_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"
_AI_STUDIO_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


def _use_vertex() -> bool:
    return os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").strip().lower() in {"1", "true", "yes"}


def _vertex_url(model: str) -> str:
    project = (os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT") or "").strip()
    if not project:
        raise RuntimeError("GOOGLE_CLOUD_PROJECT must be set when GOOGLE_GENAI_USE_VERTEXAI=true")
    location = (os.getenv("GOOGLE_CLOUD_LOCATION") or "global").strip()
    host = "https://aiplatform.googleapis.com" if location == "global" else f"https://{location}-aiplatform.googleapis.com"
    return f"{host}/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent"


def _ai_studio_url(model: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GOOGLE_GENAI_USE_VERTEXAI=true with ADC, or set GEMINI_API_KEY")
    return f"{_AI_STUDIO_BASE}/{model}:generateContent?key={urllib.parse.quote(api_key)}"


def _make_request(model: str, payload: dict[str, Any]) -> dict[str, Any]:
    """POST to Gemini generateContent with Vertex ADC or AI Studio key."""
    url = _vertex_url(model) if _use_vertex() else _ai_studio_url(model)
    headers = {"Content-Type": "application/json"}

    if _use_vertex():
        creds, _ = google.auth.default(scopes=[_CLOUD_PLATFORM_SCOPE])
        creds.refresh(google.auth.transport.requests.Request())
        if not creds.token:
            raise RuntimeError("ADC did not return an access token")
        headers["Authorization"] = f"Bearer {creds.token}"

    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode())


# ── Gemini batch call ─────────────────────────────────────────────────────────

# How many pieces of an assortment to name. Enough to tell sweet from savoury, short
# enough to keep 50 products in one prompt.
_MAX_PIECES_SHOWN = 8


def _composition_summary(raw: dict) -> str:
    """What an assortment actually contains, from Carrefour's own piece list.

    Without it the classifier had to guess from the name and the shelf, and "24 Petits
    fours" (Département Boulangerie, rayon "Petits fours et mignardises") came out an
    APÉRITIF while holding mini Trianons and mini éclairs — a dessert. "48 Petits fours
    Réception" genuinely IS an apéritif, and the only thing telling the two apart is
    what is inside: quiches and saucisses on one side, chocolate on the other. Three
    products were misfiled this way, and the answer sat in the data the whole time.

    Feeds every classifier that goes through _format_product — menu_step, dish_role and
    event_fit all get it.
    """
    pieces = ((raw.get("composition") or {}).get("pieces")) or []
    names = []
    for piece in pieces:
        if isinstance(piece, str):
            label = piece.strip()
        elif isinstance(piece, dict):
            label = str(piece.get("name") or "").strip()
        else:
            continue
        if label:
            names.append(label)
    if not names:
        return ""
    shown = names[:_MAX_PIECES_SHOWN]
    more = f" (+{len(names) - len(shown)} autres)" if len(names) > len(shown) else ""
    return ", ".join(shown) + more


def _format_product(i: int, raw: dict) -> str:
    name = raw.get("name") or ""
    dept = raw.get("carrefour_suppliers_department") or ""
    cats = [c.get("category_name", "") for c in (raw.get("categories") or []) if c.get("category_name")]
    cats_str = ", ".join(cats[:4]) if cats else "aucune"
    line = f'{i}. Nom: "{name}" | Département: "{dept}" | Catégories: [{cats_str}]'
    contents = _composition_summary(raw)
    if contents:
        line += f" | Contient: [{contents}]"
    return line


def _call_batch(batch: list[tuple[int, dict]]) -> dict[int, str]:
    """Call Gemini for a batch of (index, raw_product) pairs. Returns {index: step}."""
    lines = [_format_product(i, raw) for i, raw in batch]
    prompt = f"{SYSTEM_PROMPT}\n\nProduits :\n" + "\n".join(lines)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0},
    }

    for attempt in range(3):
        try:
            response = _make_request(GEMINI_MODEL, payload)
            text = response["candidates"][0]["content"]["parts"][0]["text"].strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text)
            result = {}
            for k, v in parsed.items():
                try:
                    idx = int(k)
                except (ValueError, TypeError):
                    continue  # skip keys that are product names instead of indices
                # Normalize casing (model sometimes returns UPPERCASE)
                normalized = _STEP_LOOKUP.get(str(v).upper().strip(), FALLBACK_STEP)
                result[idx] = normalized
            return result
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < 2:
                time.sleep(2 ** attempt * 2)
                continue
            log.warning("gemini_batch_http_error", code=exc.code, batch_size=len(batch))
            return {i: FALLBACK_STEP for i, _ in batch}
        except Exception as exc:
            if attempt < 2:
                time.sleep(1.5 ** attempt)
                continue
            log.warning("gemini_batch_failed", error=str(exc), batch_size=len(batch))
            return {i: FALLBACK_STEP for i, _ in batch}

    return {i: FALLBACK_STEP for i, _ in batch}


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _load_cache(db: Database, product_ids: list[int]) -> dict[int, str]:
    cached = {}
    for doc in db.products.find(
        {"_id": {"$in": product_ids}, "menu_step": {"$ne": None}, "menu_step_source": "llm"},
        {"_id": 1, "menu_step": 1},
    ):
        cached[doc["_id"]] = doc["menu_step"]
    return cached


def _save_cache(db: Database, step_map: dict[int, str]) -> None:
    ops = [
        UpdateOne({"_id": pid}, {"$set": {"menu_step": step, "menu_step_source": "llm"}})
        for pid, step in step_map.items()
    ]
    if ops:
        db.products.bulk_write(ops, ordered=False)


# ── Public API ────────────────────────────────────────────────────────────────

def batch_categorize(
    db: Database,
    raw_products: list[dict],
    force: bool = False,
) -> dict[int, str]:
    """Categorize all products via Gemini, using MongoDB cache for already-seen SKUs.

    Auth: Vertex AI + ADC if GOOGLE_GENAI_USE_VERTEXAI=true, else GEMINI_API_KEY.

    Args:
        db: MongoDB database handle.
        raw_products: List of raw JSONL dicts (must have ``product_id``).
        force: If True, ignore cache and re-categorize everything.

    Returns:
        Dict mapping product_id → menu_step for all input products.
    """
    all_ids = [int(r["product_id"]) for r in raw_products]
    id_to_raw = {int(r["product_id"]): r for r in raw_products}

    cached: dict[int, str] = {} if force else _load_cache(db, all_ids)
    to_classify = [pid for pid in all_ids if pid not in cached]

    auth_mode = "vertex+ADC" if _use_vertex() else "AI Studio key"
    log.info(
        "categorize_start",
        total=len(all_ids),
        from_cache=len(cached),
        via_llm=len(to_classify),
        auth=auth_mode,
    )

    if not to_classify:
        return cached

    indexed = [(i, id_to_raw[pid]) for i, pid in enumerate(to_classify, start=1)]
    batches = [indexed[i:i + BATCH_SIZE] for i in range(0, len(indexed), BATCH_SIZE)]
    log.info("categorize_batches", batches=len(batches), workers=min(MAX_WORKERS, len(batches)))

    index_to_pid = {i: pid for i, pid in enumerate(to_classify, start=1)}
    llm_results: dict[int, str] = {}

    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(batches))) as executor:
        futures = {executor.submit(_call_batch, batch): batch for batch in batches}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            llm_results.update(future.result())
            log.info("categorize_progress", batches_done=completed, total_batches=len(batches))

    llm_by_pid = {index_to_pid[idx]: step for idx, step in llm_results.items() if idx in index_to_pid}
    _save_cache(db, llm_by_pid)

    final = {**cached, **llm_by_pid}

    dist: dict[str, int] = {}
    for step in final.values():
        dist[step] = dist.get(step, 0) + 1
    log.info("categorize_complete", from_cache=len(cached), via_llm=len(llm_by_pid), distribution=dist)

    return final
