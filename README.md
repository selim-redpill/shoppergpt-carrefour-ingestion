# shopper-gpt-carrefour-ingest

ETL pipeline that ingests the Carrefour Traiteur exports into MongoDB and Pinecone,
for the ShopperGPT assistant (`waib-api`).

## Principle

Stay as close as possible to Carrefour's raw data. The full source record is kept
under `raw` on every product, and the assistant reads it directly. We derive only
what the engine cannot infer at runtime — and we never invent a value: a missing
portion count stays missing rather than becoming an estimate, because a number the
LLM can multiply is a number it will trust.

## Run

```bash
cp .env.example .env      # MONGO_URI, MONGO_DB, GEMINI_API_KEY, OPENAI_API_KEY, PINECONE_API_KEY
poetry install
poetry run python run.py                        # everything, in order
```

| Command | What it does |
|---|---|
| `run.py --stores` | `stores` collection (+ geo, delivery modes, curated concepts) |
| `run.py --prices` | `prices` collection — one row per (product, store) that has a price |
| `run.py --products` | `products` collection, including the LLM classifications |
| `run.py --catalogue` | per-store availability summaries; **needs `--products` + `--prices` first** |
| `run.py --pinecone` | embeds product names and upserts the vectors |
| `--force-categorize` | ignores the classification cache and re-asks the model |
| `--reset-pinecone` | wipes all vectors before re-ingesting (drops stale ones) |

Put the exports in `data/` (gitignored). Both `products.jsonl` and the dated GCS
names (`catalogue_products_YYYY-MM-DD.jsonl.gz`) are picked up — the most recent
wins, see `ingest/config.py`.

## What gets derived

Everything else on the document is raw Carrefour data, unmodified.

| Field | Source | Notes |
|---|---|---|
| `menu_step` | Gemini | Apéritifs, Entrées, Plats, Sauces, Fromages, Desserts, Boissons, Pains, Petit Déj, Table & Déco. Must stay in sync with `MENU_STEPS_ORDERED` in the API's `engine.py`. |
| `dish_role` | Gemini | `main` / `side`, on Plats only — the step mixes a roast with its gratin, and the engine needs one protein main. |
| `drink_role` | Gemini | `eau, soft, vin, petillant, champagne, biere, cidre, spiritueux, aperitif, chaud` — selects which per-guest proportion rule applies. Boissons only. |
| `could_fit_event` | Gemini | Occasions the product genuinely suits, or `["ALL"]`. Mirrors `EVENT_CATEGORIES` in the API. |
| `volume_ml` | `raw.weight` | Bottle volume, Boissons only. The raw field is a decimal *string*, and on food it is a mass in grams — hence the guard, plus a 200 ml plausibility floor (a 34 g box of tea bags lives in the same field). |
| `dietary_tags` | `raw.type_envie` | The diet subset of Carrefour's own tags (`sans porc`, `sans viande`, `sans poisson`, `végétarien`) isolated from the sensory ones, for the composer and the dietary critic. A strict whitelist — never inferred. |
| `persons` | `raw.nb_portion` | `None` when Carrefour gives nothing. No fallback. |
| `price_ref` | `prices` | Median across stores, used only when no store is selected — the store's own price always wins. |
| `is_composable` / `composition_plateau` | `raw.composition_plateau` | Genuine build-your-own products, from Carrefour's structured groups (never name keywords). Piece `code`s are kept: the cart API needs them verbatim. |
| `recommendable` | product name | `False` only for "au choix / à composer" products with no structured data behind them. Excluded from Mongo and Pinecone unless `INGEST_NON_RECOMMENDABLE=true`. |
| `delai_prepa` | raw | Global lead time in days; per-store overrides stay in `raw.carrefour_delay`. |

On stores, `--catalogue` adds `step_catalogue` (product count per step),
`step_families` (which sub-families the store actually carries — grounds the API's
query planner) and `step_typical_cost` (median €/guest per step).

### The classification cache

Each Gemini classification is written with a `<field>_source: "llm"` marker and
reused on every later ingest, so a product is classified once. The cache is keyed
on `product_id` alone: if Carrefour ever renames a product in place, its old
classifications persist. Measured at zero drift over the June→August exports
(1343 products, no name change); `--force-categorize` is the remedy if it happens.

## Pinecone

The embedded text is the **product name only** — ingredients and keywords dilute
the vector. Metadata is `menu_step` + `status`, the only two things the API filters
on. Index: `waib-carrefour-dev-large` / `waib-carrefour-prod-large` (1536 dims).

## Tests

```bash
poetry run pytest
```

Pure-function tests only — no MongoDB, no network. They cover what decides the
data: the derivations, the document shape, and the drink-family overrides.

## Maintenance scripts

```bash
poetry run python scripts/missing_portions.py       # products with no nb_portion, for Carrefour
poetry run python scripts/drop_fossil_fields.py     # dry-run; --apply to unset dead fields
poetry run python scripts/reclassify_stale_steps.py # dry-run; --apply to re-ask on removed steps
```
