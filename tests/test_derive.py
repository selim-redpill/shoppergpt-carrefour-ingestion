"""Unit tests for the derivation helpers — the functions that decide what the
recommendation engine believes about a product.

No MongoDB, no network: every function here is a pure transform of one raw
Carrefour record. These replace three earlier scripts that sampled the live
database and called ``derive_menu_step`` on documents lacking the
``menu_step_llm`` key it reads, so they compared every product against ``None``
and reported a 100% reclassification rate that meant nothing.
"""

from ingest.derive import (
    derive_composable,
    derive_dietary_tags,
    derive_menu_step,
    derive_persons,
    derive_price_ref,
    derive_recommendable,
)


class TestDietaryTags:
    """Diet restrictions must be a strict projection of Carrefour's type_envie."""

    def test_keeps_only_diet_values(self):
        raw = {"type_envie": ["salé", "froid", "gastronomique", "sans porc", "végétarien"]}
        assert derive_dietary_tags(raw) == ["sans porc", "végétarien"]

    def test_accepts_a_bare_string(self):
        assert derive_dietary_tags({"type_envie": "sans poisson"}) == ["sans poisson"]

    def test_normalises_case_and_spacing(self):
        assert derive_dietary_tags({"type_envie": ["Sans Porc", " végétarien "]}) == [
            "sans porc",
            "végétarien",
        ]

    def test_deduplicates(self):
        assert derive_dietary_tags({"type_envie": ["sans porc", "sans porc"]}) == ["sans porc"]

    def test_never_invents_a_restriction(self):
        # "sans gluten" is NOT in Carrefour's type_envie vocabulary. Surfacing it
        # from a product name or an ingredient list would be a safety claim we
        # have no basis for.
        assert derive_dietary_tags({"type_envie": ["bio", "sans gluten"]}) == []

    def test_missing_field_is_empty_list_not_none(self):
        assert derive_dietary_tags({}) == []
        assert derive_dietary_tags({"type_envie": None}) == []


class TestPersons:
    """Guest coverage comes from nb_portion or nowhere — no estimates."""

    def test_reads_nb_portion(self):
        assert derive_persons({"nb_portion": 6}) == 6

    def test_accepts_a_decimal_string(self):
        assert derive_persons({"nb_portion": "8.0000"}) == 8

    def test_absent_stays_unknown(self):
        assert derive_persons({}) is None

    def test_zero_and_negative_are_not_coverage(self):
        assert derive_persons({"nb_portion": 0}) is None
        assert derive_persons({"nb_portion": -2}) is None

    def test_garbage_stays_unknown(self):
        assert derive_persons({"nb_portion": "n/a"}) is None

    def test_never_falls_back_to_weight(self):
        # A 750 g product is not "a portion for 750 people", nor for any guessed
        # number: an invented coverage silently multiplies into the quantities.
        assert derive_persons({"weight": "750.0000", "nb_pieces_dans_boite": 24}) is None


class TestComposableAndRecommendable:
    def test_structured_plateau_is_composable(self):
        raw = {"composition_plateau": {"groups": [{"name": "Fromages", "pieces": [{"code": "0-0"}]}]}}
        assert derive_composable(raw) is True

    def test_picto_alone_is_enough(self):
        assert derive_composable({"left_picto_hyper": "A composer"}) is True

    def test_name_wording_alone_is_not(self):
        # "Plateau à composer" with nothing structured behind it is precisely the
        # case the assistant cannot honour.
        assert derive_composable({"name": "Plateau à composer"}) is False

    def test_composable_products_stay_recommendable(self):
        raw = {"name": "Plateau de 6 fromages au choix",
               "composition_plateau": {"groups": [{"pieces": [{"code": "0-0"}]}]}}
        assert derive_recommendable(raw) is True

    def test_unbacked_choice_wording_is_not_recommendable(self):
        assert derive_recommendable({"name": "Sandwich garniture au choix"}) is False

    def test_ordinary_product_is_recommendable(self):
        assert derive_recommendable({"name": "Plateau du charcutier"}) is True


class TestPriceRefAndMenuStep:
    def test_median_not_mean(self):
        # A single outlier store must not move the reference price.
        assert derive_price_ref([10.0, 10.0, 10.0, 100.0]) == 10.0

    def test_no_price_is_none(self):
        assert derive_price_ref([]) is None

    def test_menu_step_is_the_injected_llm_value(self):
        assert derive_menu_step({"menu_step_llm": "Boissons"}) == "Boissons"

    def test_menu_step_absent_when_not_classified(self):
        assert derive_menu_step({"name": "Plateau"}) is None
