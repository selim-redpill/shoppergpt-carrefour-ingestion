"""
Product sub-families per store × menu step — grounding for the API's vector search.

The API's query planner (``waib-api``'s ``engine.plan_step_queries``) turns an event
brief into product-name-like search queries, because the embedding space is product
NAMES. With nothing to ground it, it guesses sub-families from general knowledge and
regularly searches for things this catalogue simply does not carry — the vector search
then returns the nearest (often irrelevant) neighbours rather than nothing, so the miss
is silent. Observed on Table & Déco: generic party supplies ("assiettes jetables",
"ballons") queried for a romantic dinner at a store whose entire décor assortment is
9 flower bouquets, 2 candle references and chopsticks.

This module condenses each (store, step) catalogue into the list of sub-families it
ACTUALLY contains — e.g. for Boissons: vin, jus, soda, champagne, eau, nectar, liqueur,
café, bière, cidre, thé, limonade, whisky, crémant, rhum, vodka, sangria. Short enough
to inject into a prompt (a few hundred tokens for a whole menu), specific enough that
the planner picks real sub-families instead of inventing them.

Deliberately NOT an LLM call: the head noun of a French retail product name is a
reliable family label, so plain word-frequency does the job — at ingestion time, once,
rather than on every chat turn (a full per-store catalogue listing would be ~9k tokens
of product names per call, re-read by each of the 8-15 LLM calls a single compose makes).

Extraction, per product name:
  1. Strip packaging noise — parenthesised text, a trailing "- 24 toasts"/"- 75cl"
     size suffix, inline units, punctuation, the leading count of "48 Petits fours".
  2. Walk the remaining words to the first one that names a FAMILY, skipping
     format/size/portion words (mini, maxi, assortiment, menu, trio…) — so
     "Menu familial gratin pour 4 personnes" yields ``gratin``, not ``menu``.
  3. Group case/accent/plural variants together (crêpes ≡ crèpes, éclairs ≡ eclairs,
     verrine ≡ verrines) and label each group with its most frequent surface form.

Families are returned most-frequent-first and capped, but the long tail is the point:
a single-product family (chevreuil, tajine, paëlla, sangria) is exactly the kind of
option the planner would never have guessed on its own.
"""

import re
import unicodedata
from collections import Counter
from typing import Dict, List

# Safety bound per step, NOT an intended trimmer: the distinct-family count saturates
# well before this (store 474's widest step tops out under 80, and the full uncapped
# hint for its whole 10-step menu costs ~610 tokens vs ~500 when capped at 40 — the rare
# tail is almost free). Set high on purpose: a one-product family (chevreuil, sangria,
# paëlla) is precisely what the planner would never have guessed, so trimming by
# frequency would cut the most valuable entries first. Here only to stop an unexpectedly
# sprawling catalogue from bloating the prompt without bound.
MAX_FAMILIES_PER_STEP = 80

# Grammatical glue — never a family on its own.
_STOPWORDS = frozenset(
    {
        "de",
        "du",
        "des",
        "la",
        "le",
        "les",
        "un",
        "une",
        "et",
        "au",
        "aux",
        "à",
        "a",
        "en",
        "pour",
        "avec",
        "sans",
        "sur",
        "dans",
        "d",
        "l",
        "the",
    }
)

# Words that describe the FORMAT, SIZE or PORTIONING of a product rather than what it
# is. Skipped over (not dropped) so extraction falls through to the real head noun:
# "Petite salade" → salade, "Trio de choux jambon comté" → choux, "Menu familial
# gratin" → gratin. Kept deliberately narrow and generic (standard French retail
# vocabulary, not store-specific): over-listing here would silently swallow real
# families. Note "plateau"/"planche"/"assiette" are NOT here — a "Plateau de fromages"
# is a genuine Carrefour concept and a real family, not a packaging detail.
_FORMAT_WORDS = frozenset(
    {
        "mini",
        "minis",
        "maxi",
        "maxis",
        "demi",
        "demie",
        "petit",
        "petite",
        "petits",
        "petites",
        "grand",
        "grande",
        "grands",
        "grandes",
        "gros",
        "grosse",
        "assortiment",
        "assortiments",
        "coffret",
        "lot",
        "pack",
        "sachet",
        "boite",
        "boîte",
        "barquette",
        "plaque",
        "seau",
        "menu",
        "menus",
        "formule",
        "formules",
        "familial",
        "familiale",
        "duo",
        "duos",
        "trio",
        "trios",
        "quatuor",
        # "Douzaine d'huîtres" is a portion count, not a family — it topped Entrées
        # before this, hiding the actual huîtres/coquilles families beneath it.
        "douzaine",
        "douzaines",
    }
)

# Containers and occasion words: they say how a product is PACKAGED or WHEN it is
# eaten, never what it is. Skipped only for the per-product family (see
# ``family_word``'s ``skip_containers``), never for the store hints, where "plateau" is
# a genuine Carrefour concept the query planner should know the store carries.
#
# The distinction matters for menu variety: a charcuterie platter, a vegetable platter
# and a cheese platter are three different experiences, and collapsing them onto
# "plateau" would make a varied apéritif look repetitive.
_CONTAINER_WORDS = frozenset(
    {
        "plateau", "plateaux", "assiette", "assiettes", "planche", "planches", "box",
        "coffret", "panier", "paniers", "apéro", "apero", "apéritif", "aperitif",
        "apéritifs", "aperitifs",
    }
)

_PAREN_RE = re.compile(r"\([^)]*\)")
# Carrefour appends size/portion info after a dash: "Pain surprise polaire - 24 toasts".
_TRAILING_DASH_RE = re.compile(r"[-–].*$")
_UNIT_RE = re.compile(
    r"\b\d+([.,]\d+)?\s*"
    r"(g|kg|ml|cl|l|cm|mm|pièces?|pieces?|parts?|personnes?|pers|toasts?|x)\b",
    re.IGNORECASE,
)
_PUNCT_RE = re.compile(r"[\"'’,.:;!?/&]")
_WS_RE = re.compile(r"\s+")


def _normalize(name: str) -> str:
    """Strip packaging/size noise from a raw product name."""
    text = _PAREN_RE.sub(" ", name)
    text = _TRAILING_DASH_RE.sub(" ", text)
    text = _UNIT_RE.sub(" ", text)
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def _fold(word: str) -> str:
    """Accent- and plural-insensitive grouping key.

    Merges surface variants of the same family that the catalogue spells
    inconsistently: crêpes/crèpes, éclairs/eclairs, cœur/coeur, verrine/verrines.
    Trailing s/x is dropped only on words long enough that it is plural rather than
    part of the stem (keeps "jus" and "vin"/"vins" from colliding wrongly).
    """
    folded = unicodedata.normalize("NFKD", word.lower().replace("œ", "oe"))
    folded = "".join(c for c in folded if not unicodedata.combining(c))
    if len(folded) > 4 and folded[-1] in ("s", "x"):
        folded = folded[:-1]
    return folded


def family_word(name: str, skip_containers: bool = False) -> str:
    """The family-naming head word of one product name, or "" if none stands out.

    ``skip_containers`` also walks past packaging and occasion words, so a "Plateau
    apéro charcuterie" yields ``charcuterie`` instead of ``plateau``. Used for the
    per-product ``family`` field, where the question is what the guest EATS; left off
    for the store hints, where "plateau" is itself a family the store carries.
    """
    skipped = _FORMAT_WORDS | _CONTAINER_WORDS if skip_containers else _FORMAT_WORDS
    for word in _normalize(name).split(" "):
        if not word or word.isdigit():
            continue
        lowered = word.lower()
        if lowered in _STOPWORDS or lowered in skipped:
            continue
        # A lone letter/digit fragment ("l", "4") names nothing.
        if len(lowered) < 3:
            continue
        return lowered
    return ""


# Kept as the private alias the store-hint code already uses.
_family_word = family_word


def extract_families(names: List[str], max_families: int = MAX_FAMILIES_PER_STEP) -> List[str]:
    """Condense product names into their sub-families, most frequent first.

    Args:
        names: Raw product names of one (store, step) catalogue.
        max_families: Hard cap on the returned list — bounds the API prompt.

    Returns:
        Lowercased family labels, frequency-descending then alphabetical (so a
        re-ingestion of unchanged data produces a byte-identical list).
    """
    # Count by folded key, but remember how the catalogue actually spells it so the
    # label reads naturally ("crêpes", not the accent-stripped "crepes").
    counts: Counter = Counter()
    surfaces: Dict[str, Counter] = {}
    for name in names:
        word = _family_word(name or "")
        if not word:
            continue
        key = _fold(word)
        counts[key] += 1
        surfaces.setdefault(key, Counter())[word] += 1

    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [surfaces[key].most_common(1)[0][0] for key, _ in ranked[:max_families]]
