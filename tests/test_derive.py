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

    def test_le_picto_seul_ne_suffit_plus(self):
        # Mesuré sur le catalogue : des 32 produits portant le picto « A composer »,
        # 21 ont leurs groupes structurés et les 11 autres sont des formules « Menu X »
        # opaques. Le picto seul offrait donc un parcours « Composer » sans rien à
        # choisir, et faisait passer ces formules pour recommandables.
        assert derive_composable({"left_picto_hyper": "A composer"}) is False

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

    def test_un_produit_a_garnir_soi_meme_reste_recommandable(self):
        # « 30 Navettes Natures (à garnir) » est COMPLET tel que vendu : aucun choix à
        # résoudre, les navettes arrivent nature et se garnissent à la maison. Le
        # mot-clé « à garnir » était la seule chose qui les tenait hors du catalogue,
        # alors qu'elles ont leur place sur un buffet.
        for name in ("30 Navettes Natures (à garnir)", "30 Navettes Sésames (à garnir)"):
            assert derive_recommendable({"name": name, "type_id": "simple"}) is True

    def test_ordinary_product_is_recommendable(self):
        assert derive_recommendable({"name": "Plateau du charcutier"}) is True


class TestFormulesOpaques:
    """Les « Menu X » bundles n'ont rien à montrer au client.

    Ce sont des formules assemblées en magasin — une entrée, un plat, un dessert
    choisis sur une carte qu'on ne reçoit pas. Les 12 bundles actifs sont tous des
    « Menu X », tous dans Plats, et aucun ne porte de composition : impossible de dire
    au client ce qu'il mangerait. Le compositeur, lui, y voyait un plat principal pas
    cher — sur un mariage de 100 convives il a remplacé un Bœuf Wellington par
    « Menu Classique » ×100, soit 1290 €, 43 % du budget.
    """

    def test_un_bundle_sans_contenu_est_ecarte(self):
        assert derive_recommendable({"name": "Menu Classique", "type_id": "bundle"}) is False

    def test_le_discriminant_est_le_type_pas_le_nom(self):
        # Les plateaux de sushis s'appellent aussi « Menu One », « Menu San »,
        # « Menu Love » — et sont parfaitement explicites : type_id simple, nombre de
        # pièces dans le nom. Filtrer sur le mot « menu » les supprimerait à tort.
        for name in ("Menu One - 9 pièces", "Menu San - 14 pièces", "Menu Love - 40 pièces"):
            assert derive_recommendable({"name": name, "type_id": "simple"}) is True

    def test_un_bundle_qui_dit_son_contenu_reste_recommandable(self):
        # La règle porte sur l'opacité, pas sur le conditionnement : un bundle dont on
        # peut lister le contenu est présentable au client.
        product = {
            "name": "Menu de Noël",
            "type_id": "bundle",
            "composition": {"pieces": [{"name": "Foie gras"}, {"name": "Chapon"}]},
        }
        assert derive_recommendable(product) is True

    def test_les_vrais_produits_a_composer_sont_intacts(self):
        # C'est la limite de la règle : elle ne doit toucher QUE les formules « Menu X ».
        # Les plateaux de fromages, les assortiments de pâtisseries au choix et les
        # pizzas à saveurs multiples portent tous une composition structurée, donc ils
        # restent composables ET recommandables.
        for name in (
            "Plateau de 6 fromages",
            "Assortiment de 10 pâtisseries classiques au choix",
            "Assiette du charcutier à composer",
            "Pizza - 4 saveurs au choix - 8 parts",
        ):
            product = {
                "name": name,
                "type_id": "plateau",
                "composition_plateau": {"groups": [{"pieces": [{"code": "0-0"}]}]},
            }
            assert derive_composable(product) is True, name
            assert derive_recommendable(product) is True, name

    def test_un_bundle_avec_de_vrais_choix_structures_reste_composable(self):
        product = {
            "name": "Menu à composer",
            "type_id": "bundle",
            "composition_plateau": {"groups": [{"pieces": [{"code": "0-0"}]}]},
        }
        assert derive_composable(product) is True
        assert derive_recommendable(product) is True


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
