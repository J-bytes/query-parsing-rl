from __future__ import annotations

import json

from hypothesis import given
from hypothesis import strategies as st

from qprl.schema import (
    FIELDS,
    normalize_parse,
    normalize_text,
    normalize_value,
    parse_completion,
    to_target_json,
)


def test_normalize_text_rules() -> None:
    assert normalize_text("  Men's  Running-Shoes ") == "mens running shoes"
    assert normalize_text("Levi’s") == "levis"
    assert normalize_text("size 10.5") == "size 10.5"
    assert normalize_text("...") == ""


def test_normalize_value_types() -> None:
    assert normalize_value("price_max", "$1,200") == 1200.0
    assert normalize_value("price_max", 19.999) == 20.0
    assert normalize_value("price_max", True) is None
    assert normalize_value("price_max", "cheap") is None
    assert normalize_value("gender", "Women") == "women"
    assert normalize_value("gender", "female") is None  # not in enum
    assert normalize_value("color", ["Red", "Blue"]) == "red blue"
    assert normalize_value("color", "") is None
    assert normalize_value("unknown", "x") is None
    assert normalize_value("brand", None) is None


def test_parse_strict() -> None:
    r = parse_completion('{"product_type": "case", "price_max": 20}')
    assert r.strict_json and r.lenient_json and r.schema_ok and r.format_ok
    assert r.parse == {"product_type": "case", "price_max": 20.0}


def test_parse_lenient_fenced() -> None:
    r = parse_completion('Sure!\n```json\n{"brand": "Nike"}\n```')
    assert not r.strict_json and r.lenient_json and r.schema_ok and not r.format_ok
    assert r.parse == {"brand": "nike"}


def test_parse_schema_violations() -> None:
    assert not parse_completion('{"foo": "bar"}').schema_ok
    assert not parse_completion('{"price_max": "20"}').schema_ok
    assert not parse_completion('{"gender": "female"}').schema_ok
    assert not parse_completion('{"brand": ""}').schema_ok
    assert parse_completion('{"price_max": "20"}').parse == {"price_max": 20.0}


def test_parse_garbage() -> None:
    for text in ["", "no json here", "[1, 2]", "{broken", '"just a string"']:
        r = parse_completion(text)
        assert not r.strict_json and not r.format_ok and r.parse == {}


def test_parse_skips_non_object_braces() -> None:
    r = parse_completion('{ not json } then {"brand": "sony"}')
    assert r.lenient_json and r.parse == {"brand": "sony"}


def test_target_json_canonical_order() -> None:
    s = to_target_json({"price_max": 50.0, "product_type": "lamp", "brand": "ikea"})
    assert s == '{"product_type": "lamp", "brand": "ikea", "price_max": 50}'
    assert to_target_json({}) == "{}"


label_values = st.one_of(
    st.text(alphabet="abcdefghij klm0123456789", min_size=1, max_size=12),
    st.floats(min_value=0, max_value=1e5, allow_nan=False),
)


@st.composite
def parses(draw: st.DrawFn) -> dict[str, object]:
    keys = draw(st.sets(st.sampled_from(list(FIELDS))))
    out: dict[str, object] = {}
    for k in keys:
        if k == "gender":
            out[k] = draw(st.sampled_from(["men", "women", "unisex"]))
        elif k == "age_group":
            out[k] = draw(st.sampled_from(["baby", "kids", "teen", "adult"]))
        elif FIELDS[k][0] is float:
            out[k] = draw(st.floats(min_value=0, max_value=1e5, allow_nan=False))
        else:
            out[k] = draw(st.text(alphabet="abcdefghij klm0123456789", min_size=1, max_size=12))
    return out


@given(parses())
def test_roundtrip_target_json(p: dict[str, object]) -> None:
    """Serialising a normalised parse and parsing it back is the identity."""
    norm = normalize_parse(p)
    r = parse_completion(to_target_json(norm))
    assert r.strict_json and r.format_ok
    assert r.parse == norm


@given(parses())
def test_normalize_idempotent(p: dict[str, object]) -> None:
    once = normalize_parse(p)
    assert normalize_parse(once) == once


@given(st.text(max_size=80))
def test_parse_never_raises(text: str) -> None:
    r = parse_completion(text)
    assert set(r.parse) <= set(FIELDS)
    if r.format_ok:
        assert json.loads(text.strip()) is not None


# --- tests added after mutation testing ---


def test_normalisation_details() -> None:
    assert normalize_text("air_max") == "air max"
    assert normalize_text("size 10.") == "size 10"
    assert normalize_value("price_max", 9.991) == 9.99
    assert normalize_value("price_max", "$9.991") == 9.99
    assert normalize_value("brand", {"a": 1}) is None
    assert normalize_value("brand", 3) == "3"


def test_recovery_picks_first_object_and_nested_brace() -> None:
    assert parse_completion('x {"brand": "a"} {"brand": "b"}').parse == {"brand": "a"}
    assert parse_completion('{{"brand": "a"}').parse == {"brand": "a"}


def test_target_json_keeps_unicode() -> None:
    assert to_target_json({"brand": "café"}) == '{"brand": "café"}'
