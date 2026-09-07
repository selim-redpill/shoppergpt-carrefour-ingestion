"""Unit tests for the deterministic drink-family overrides.

The model classifies drinks by name, so the dangerous cases are the ones where
the name contradicts the family. These overrides run AFTER the model and are
order-sensitive: the iced-tea rules must win over the generic tea rule, or every
Lipton pêche becomes a hot drink.
"""

import pytest

from ingest.categorize import VALID_DRINK_ROLES, _drink_role_override


@pytest.mark.parametrize(
    "name,expected",
    [
        # Alcohol-free versions of alcoholic drinks: sizing these on adults only
        # would under-serve everyone else at the table.
        ("6 Bières au citron sans alcool Tourtel", "soft"),
        ("6 bières 1664 sans alcool", "soft"),
        ("Cocktail sans alcool passion Mister Cocktail", "soft"),
        # Flavoured water is not table water — treating it as such also exempts
        # it from budget arbitration.
        ("Eau aromatisée au citron Volvic", "soft"),
        ("Eau gazeuse aromatisée orange et grenade San Pellegrino", "soft"),
        # Iced tea is a soft drink; these must beat the generic tea rule below.
        ("Thé glacé pêche Lipton", "soft"),
        ("Boisson au thé vert glacé saveur citron menthe Lipton", "soft"),
        ("ICE TEA LIPTON 2L", "soft"),
        # Sparkling apple juice sold as a celebration drink.
        ("Jus de pomme pétillant Champomy", "soft"),
        # Tea to brew and ground coffee are hot drinks — they are sold by mass,
        # so they must stay out of the per-litre arithmetic.
        ("Thé 5 fruits rouges Lipton", "chaud"),
        ("Thé vert menthe Carrefour", "chaud"),
        ("Café moulu Tradition", "chaud"),
        ("Café Soluble Chicorée Original Ricore", "chaud"),
        # Nothing to override: the model's own answer stands.
        ("Champagne brut Tsarine - 75cl", None),
        ("Vin rouge Bordeaux Mouton Cadet", None),
        ("Eau de source état naturel Cristaline", None),
        ("Bière blonde IPA Castelain", None),
    ],
)
def test_override(name, expected):
    assert _drink_role_override(name) == expected


def test_every_override_target_is_a_valid_family():
    for name in ("6 bières sans alcool", "Thé vert menthe", "Champomy"):
        forced = _drink_role_override(name)
        assert forced is None or forced in VALID_DRINK_ROLES
