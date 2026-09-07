"""Unit tests for transform_product's own derivations — the document shape the
API consumes.

Focus on the two fields that feed quantity arithmetic (``volume_ml``) and
allergy/diet reasoning (``dietary_tags``), plus the price flattening, since a
wrong value there propagates all the way into the customer's real cart.
"""

from ingest.transform import _volume_ml, transform_price_records, transform_product


class TestVolumeMl:
    """raw.weight is a decimal STRING, and on food it is a mass — not a volume."""

    def test_parses_the_decimal_string(self):
        assert _volume_ml({"weight": "750.0000"}, "Boissons") == 750

    def test_only_on_drinks(self):
        assert _volume_ml({"weight": "750.0000"}, "Plats") is None

    def test_half_bottle_is_a_real_volume(self):
        assert _volume_ml({"weight": "375.0000"}, "Boissons") == 375

    def test_rejects_a_mass_masquerading_as_a_volume(self):
        # A 34 g box of tea bags and 180 g of instant coffee both live in this
        # field. Treated as millilitres they would make the per-guest ratio
        # demand dozens of units.
        assert _volume_ml({"weight": "34.0000"}, "Boissons") is None
        assert _volume_ml({"weight": "180.0000"}, "Boissons") is None

    def test_missing_or_garbage_is_unknown(self):
        assert _volume_ml({}, "Boissons") is None
        assert _volume_ml({"weight": None}, "Boissons") is None
        assert _volume_ml({"weight": "abc"}, "Boissons") is None
        assert _volume_ml({"weight": "0.0000"}, "Boissons") is None


class TestTransformProduct:
    @staticmethod
    def _raw(**over):
        raw = {
            "product_id": 42,
            "sku": "000042",
            "name": "Champagne brut - 75cl",
            "status": "Activé",
            "menu_step_llm": "Boissons",
            "drink_role_llm": "champagne",
            "weight": "750.0000",
            "type_envie": ["froid", "sans porc"],
            "nb_portion": None,
        }
        raw.update(over)
        return raw

    def test_drink_document_carries_family_and_volume(self):
        doc = transform_product(self._raw(), {42: [21.9, 23.5, 22.0]})
        assert doc["_id"] == 42
        assert doc["status"] == "active"
        assert doc["menu_step"] == "Boissons"
        assert doc["drink_role"] == "champagne"
        assert doc["volume_ml"] == 750
        assert doc["persons"] is None
        assert doc["dietary_tags"] == ["sans porc"]
        assert doc["price_ref"] == 22.0

    def test_food_gets_no_drink_fields(self):
        doc = transform_product(
            self._raw(name="Rôti de veau", menu_step_llm="Plats", drink_role_llm=None,
                      nb_portion="6", weight="1200.0000"),
            {},
        )
        assert doc["drink_role"] is None
        assert doc["volume_ml"] is None
        assert doc["persons"] == 6
        assert doc["price_ref"] is None

    def test_inactive_status_is_normalised(self):
        assert transform_product(self._raw(status="Désactivé"), {})["status"] == "inactive"

    def test_raw_record_is_kept_whole(self):
        # The API reads type_allergene, carrefour_delay, nb_pieces_dans_boite…
        # straight out of raw — dropping anything here silently blinds it.
        raw = self._raw(type_allergene=["Lait"], carrefour_delay={"delay": {"338": 2}})
        doc = transform_product(raw, {})
        assert doc["raw"]["type_allergene"] == ["Lait"]
        assert doc["raw"]["carrefour_delay"] == {"delay": {"338": 2}}


class TestPriceRecords:
    def test_flattens_both_export_shapes(self):
        old_shape = {"product_id": 1, "stores": [{"store_id": 338, "prices": [{"price": "4.99"}]}]}
        new_shape = {"product_id": 1, "stores": [{"store_id": 338, "price": {"price": "4.99"}}]}
        assert transform_price_records(old_shape) == transform_price_records(new_shape)
        assert transform_price_records(new_shape) == [
            {"product_id": 1, "store_id": 338, "price": 4.99}
        ]

    def test_store_without_a_price_is_dropped(self):
        # "The store has a price" IS the availability signal in waib-api — a row
        # with no price must never become a zero-priced one.
        raw = {"product_id": 1, "stores": [{"store_id": 338, "prices": []}, {"store_id": 339}]}
        assert transform_price_records(raw) == []
