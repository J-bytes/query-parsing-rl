from __future__ import annotations

import pytest

from qprl.data.rules import (
    RuleLabeller,
    extract_color,
    extract_gender_age,
    extract_material,
    extract_price,
    extract_size,
)


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("bluetooth speaker under $50", {"price_max": 50.0}),
        ("dress less than 30 dollars", {"price_max": 30.0}),
        ("dresses under 50", {"price_max": 50.0}),
        ("gaming chair over $150", {"price_min": 150.0}),
        ("lamp $20-$40", {"price_min": 20.0, "price_max": 40.0}),
        ("rug between 100 and 200 dollars", {"price_min": 100.0, "price_max": 200.0}),
        ("tv under $1,200", {"price_max": 1200.0}),
        ("toys for kids under 5", {}),  # age, not price
        ("dog food under 5 lbs", {}),
        ("dog food under 50 lbs", {}),  # regression: '5' must not match inside '50'
        ("$25 gift card", {}),  # amount is the product, not a bound
        ("over the door hooks", {}),
    ],
)
def test_price(query: str, expected: dict[str, float]) -> None:
    assert extract_price(query) == expected


def test_price_require_currency() -> None:
    assert extract_price("vitamins for men over 50", require_currency=True) == {}
    assert extract_price("vitamins for men over 50") == {"price_min": 50.0}
    assert extract_price("shoes under $60", require_currency=True) == {"price_max": 60.0}
    assert extract_price("rug between 100 and 200", require_currency=True) == {}


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("mens running shoes", {"gender": "men"}),
        ("women's jacket", {"gender": "women"}),
        ("toddler girl dress", {"gender": "women", "age_group": "baby"}),
        ("boys soccer cleats", {"gender": "men", "age_group": "kids"}),
        ("men women matching shirts", {"gender": "unisex"}),
        ("teen backpack", {"age_group": "teen"}),
        ("boyfriend jeans", {}),
    ],
)
def test_gender_age(query: str, expected: dict[str, str]) -> None:
    assert extract_gender_age(query) == expected


def test_color_material_size() -> None:
    assert extract_color("rose gold watch") == "rose gold"
    assert extract_color("orange juice") is None
    assert extract_color("white noise machine") is None
    assert extract_material("stainless steel bottle") == "stainless steel"
    assert extract_size("queen size sheets") == "queen"
    assert extract_size("8x10 rug") == "8x10"
    assert extract_size("samsung 55 inch tv") == "55 inch"
    assert extract_size("hoodie xxl") == "xxl"
    assert extract_size("shoes size 10.5") == "10.5"
    assert extract_size("king cobra figurine") is None


def test_brand_lexicon_longest_match() -> None:
    rl = RuleLabeller(["under armour", "armour", "nike"])
    assert rl.extract_brand("under armour hoodie") == "under armour"
    assert rl.extract_brand("nike shoes") == "nike"
    assert rl.extract_brand("generic shoes") is None
    lab = rl.label("nike mens shoes under $80 black")
    assert lab == {"brand": "nike", "gender": "men", "price_max": 80.0, "color": "black"}


def test_from_missing_file() -> None:
    assert RuleLabeller.from_file("does/not/exist.txt").brands == []
    assert RuleLabeller.from_file(None).brands == []
