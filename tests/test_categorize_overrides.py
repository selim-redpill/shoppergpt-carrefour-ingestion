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
        # Alcohol-free versions of alcoholic drinks belong to their own family: they
        # stand IN FOR an alcohol, so they are sized for the guests who skip it, not
        # for the whole table like a soda. Filed as softs, they were ordered by the
        # dozen — 40 bottles of Champomy on a wedding of 100 adults.
        ("6 Bières au citron sans alcool Tourtel", "sans_alcool"),
        ("6 bières 1664 sans alcool", "sans_alcool"),
        ("Cocktail sans alcool passion Mister Cocktail", "sans_alcool"),
        # Flavoured water is not table water — treating it as such also exempts
        # it from budget arbitration.
        ("Eau aromatisée au citron Volvic", "soft"),
        ("Eau gazeuse aromatisée orange et grenade San Pellegrino", "soft"),
        # Iced tea is a soft drink; these must beat the generic tea rule below.
        ("Thé glacé pêche Lipton", "soft"),
        ("Boisson au thé vert glacé saveur citron menthe Lipton", "soft"),
        ("ICE TEA LIPTON 2L", "soft"),
        # Sparkling apple juice sold for toasting — same family, same reason.
        ("Jus de pomme pétillant Champomy", "sans_alcool"),
        # Tea to brew and ground coffee are hot drinks — they are sold by mass,
        # so they must stay out of the per-litre arithmetic.
        ("Thé 5 fruits rouges Lipton", "chaud"),
        ("Thé vert menthe Carrefour", "chaud"),
        ("Café moulu Tradition", "chaud"),
        ("Café Soluble Chicorée Original Ricore", "chaud"),
        # A plain juice or soda stays a soft: `sans_alcool` is reserved for what
        # replaces an alcohol, and widening it would put the everyday drinks on the
        # children's ratio.
        ("Jus d'orange 100% pur fruit pressé Carrefour", None),
        ("Soda à l'orange Orangina", None),
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
