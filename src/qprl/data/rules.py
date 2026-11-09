"""Deterministic rule-based labeller.

Two jobs:
  1. Hints for the LLM labeller (the prompt shows what the rules found).
  2. A baseline row in the results table: "how far do regexes get?".

Rules are written for precision, not recall. Where a pattern is ambiguous it is
skipped rather than guessed, and the reason is written next to it. Price is the
one field where rules beat the LLM, so the merge step lets rule prices override.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from qprl.schema import normalize_text

COLORS = [
    "black", "white", "red", "blue", "green", "yellow", "orange", "purple", "pink",
    "brown", "gray", "grey", "silver", "gold", "beige", "navy", "teal", "turquoise",
    "maroon", "burgundy", "tan", "ivory", "cream", "khaki", "olive", "lavender",
    "coral", "mint", "charcoal", "magenta", "aqua", "bronze", "copper", "rose gold",
    "navy blue", "light blue", "dark blue", "sky blue", "royal blue", "baby blue",
    "light pink", "hot pink", "dark green", "lime green", "forest green", "off white",
    "multicolor", "clear", "transparent",
]  # fmt: skip
# Ambiguous colour words we deliberately do not tag: "orange juice", "white
# noise", "black friday" etc. are handled by these phrase blockers.
COLOR_BLOCKERS = {
    "orange": ["orange juice", "orange oil", "orange extract"],
    "white": ["white noise", "white board", "whiteboard", "white out"],
    "black": ["black friday", "blackout", "black light"],
    "gold": ["gold bond", "goldfish"],
    "clear": ["clear coat", "clearance"],
}

MATERIALS = [
    "leather", "faux leather", "cotton", "organic cotton", "wool", "merino wool",
    "silk", "linen", "polyester", "nylon", "denim", "suede", "velvet", "fleece",
    "cashmere", "bamboo", "wood", "wooden", "oak", "walnut", "pine", "metal", "steel",
    "stainless steel", "aluminum", "aluminium", "iron", "cast iron", "copper",
    "brass", "glass", "ceramic", "porcelain", "plastic", "silicone", "rubber",
    "acrylic", "marble", "granite", "concrete", "canvas", "vinyl", "foam",
    "memory foam", "latex", "titanium", "carbon fiber", "resin", "stone", "paper",
]  # fmt: skip

GENDER_TOKENS = {
    "men": "men", "mens": "men", "man": "men", "male": "men", "guys": "men",
    "boys": "men", "boy": "men",
    "women": "women", "womens": "women", "woman": "women", "ladies": "women",
    "lady": "women", "female": "women", "girls": "women", "girl": "women",
    "unisex": "unisex",
}  # fmt: skip
AGE_TOKENS = {
    "baby": "baby", "babies": "baby", "infant": "baby", "infants": "baby",
    "newborn": "baby", "toddler": "baby", "toddlers": "baby",
    "kids": "kids", "kid": "kids", "child": "kids", "children": "kids",
    "childrens": "kids", "boys": "kids", "boy": "kids", "girls": "kids",
    "girl": "kids", "youth": "kids",
    "teen": "teen", "teens": "teen", "teenager": "teen", "teenage": "teen",
    "adult": "adult", "adults": "adult",
}  # fmt: skip

_NUM = r"(\d+(?:\.\d+)?)"
_CUR_WORD = r"(?:dollars?|bucks|usd)"
# Unit words after a number mean it is not a price ("under 5 lbs", "kids under 5 years").
_NOT_PRICE_UNIT = (
    r"(?:years?|yrs?|months?|mos?|lbs?|pounds?|oz|ounces?|inch(?:es)?|in|ft|feet|foot|"
    r"cm|mm|gb|tb|mah|w|watts?|v|volts?|pcs?|pieces?|pack|count|ct|calories|cal|mph|hz|qt|l)\b"
)

_MAX_TRIGGER = r"(?:under|below|less than|cheaper than|up to|max|maximum|no more than|<)"
_MIN_TRIGGER = r"(?:over|above|more than|at least|min|minimum|>)"

_RANGE_RE = re.compile(
    rf"\$\s?{_NUM}\s?(?:-|to)\s?\$?\s?{_NUM}"  # $10-$20, $10 to 20
    rf"|between\s+\$?{_NUM}\s+(?:and|-)\s+\$?{_NUM}\s*{_CUR_WORD}?"  # between 10 and 20 (dollars)
    rf"|{_NUM}\s?(?:-|to)\s?{_NUM}\s?{_CUR_WORD}"  # 10-20 dollars
)
_MAX_RE = re.compile(
    rf"{_MAX_TRIGGER}\s+(\$)?\s?{_NUM}(?![\d.])\s*({_CUR_WORD})?(?!\s*{_NOT_PRICE_UNIT})"
)
_MIN_RE = re.compile(
    rf"{_MIN_TRIGGER}\s+(\$)?\s?{_NUM}(?![\d.])\s*({_CUR_WORD})?(?!\s*{_NOT_PRICE_UNIT})"
)

# Bare trigger + number without "$" or a currency word is a price only if the
# number is >= this. "toys for kids under 5" is an age, "dresses under 50" a price.
MIN_BARE_PRICE = 10.0

_SIZE_PATTERNS = [
    re.compile(r"\bsize\s+(\d+(?:\.\d+)?|x{0,3}s|x{0,3}l|m|small|medium|large)\b"),
    re.compile(r"\b(\d?x{1,3}l|x{1,3}s|\dx)\b"),  # xl, xxl, 2xl, 3x, xs
    re.compile(r"\b(california king|cal king|queen|king|twin xl|twin)\b(?= size| bed| sheet| mattress| comforter| frame|$)"),
    re.compile(r"\b(\d+(?:\.\d+)?\s?x\s?\d+(?:\.\d+)?(?:\s?x\s?\d+(?:\.\d+)?)?)\b"),  # 8x10, 10 x 10
    re.compile(
        r"\b(\d+(?:\.\d+)?\s?(?:inch(?:es)?|in|\"|ft|feet|foot|cm|mm|oz|ounces?|lbs?|"
        r"gallons?|gal|qt|quarts?|ml|liters?|litres?|gb|tb|mah|quart))(?=\s|$)"
    ),
]  # fmt: skip


def _has_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![\w]){re.escape(phrase)}(?![\w])", text) is not None


def _longest_lexicon_match(text: str, lexicon: list[str]) -> str | None:
    best: str | None = None
    for item in lexicon:
        if _has_phrase(text, item) and (best is None or len(item) > len(best)):
            best = item
    return best


def extract_price(query: str, require_currency: bool = False) -> dict[str, float]:
    """Price bounds. With require_currency=True only matches carrying "$" or a
    currency word count; that high-precision subset is what the merge step lets
    override the LLM. "vitamins for men over 50" is why bare numbers are weaker.
    """
    q = query.lower().replace(",", "")
    m = _RANGE_RE.search(q)
    if m:
        explicit = "$" in m.group(0) or re.search(_CUR_WORD, m.group(0)) is not None
        if explicit or not require_currency:
            nums = [float(g) for g in m.groups() if g is not None]
            return {"price_min": min(nums[:2]), "price_max": max(nums[:2])}
    out: dict[str, float] = {}
    for pattern, key in ((_MAX_RE, "price_max"), (_MIN_RE, "price_min")):
        m = pattern.search(q)
        if not m:
            continue
        dollar, num, cur = m.group(1), float(m.group(2)), m.group(3)
        explicit = bool(dollar or cur)
        if explicit or (not require_currency and num >= MIN_BARE_PRICE):
            out[key] = num
    return out


def extract_gender_age(query: str) -> dict[str, str]:
    tokens = normalize_text(query.replace("'", "")).split()
    out: dict[str, str] = {}
    genders = {GENDER_TOKENS[t] for t in tokens if t in GENDER_TOKENS}
    ages = {AGE_TOKENS[t] for t in tokens if t in AGE_TOKENS}
    if len(genders) == 1:
        out["gender"] = genders.pop()
    elif {"men", "women"} <= genders:  # "his and hers", "men women" -> unisex
        out["gender"] = "unisex"
    # "toddler girl" gives {baby, kids}: the more specific age wins. Kids-words
    # like "girl"/"boy" say "child", toddler/baby/teen say which child.
    for specific in ("baby", "teen"):
        if specific in ages and ages <= {specific, "kids"}:
            ages = {specific}
    if len(ages) == 1:
        out["age_group"] = ages.pop()
    return out


def extract_color(query: str) -> str | None:
    q = normalize_text(query)
    color = _longest_lexicon_match(q, COLORS)
    if color is None:
        return None
    for blocker in COLOR_BLOCKERS.get(color, []):
        if blocker in q:
            return None
    return color


def extract_material(query: str) -> str | None:
    return _longest_lexicon_match(normalize_text(query), MATERIALS)


def extract_size(query: str) -> str | None:
    q = query.lower()
    for pat in _SIZE_PATTERNS:
        m = pat.search(q)
        if m:
            return normalize_text(m.group(1))
    return None


class RuleLabeller:
    def __init__(self, brands: list[str] | None = None) -> None:
        # Sorted longest-first so multi-word brands win ("under armour" over "armour").
        self.brands = sorted(brands or [], key=len, reverse=True)
        self._brand_set = set(self.brands)

    @classmethod
    def from_file(cls, path: str | Path | None) -> RuleLabeller:
        if path is None or not Path(path).exists():
            return cls([])
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        return cls([b for b in lines if b.strip()])

    def extract_brand(self, query: str) -> str | None:
        tokens = normalize_text(query).split()
        best: str | None = None
        for n in (4, 3, 2, 1):
            for i in range(len(tokens) - n + 1):
                cand = " ".join(tokens[i : i + n])
                if cand in self._brand_set and (best is None or len(cand) > len(best)):
                    best = cand
            if best:
                return best
        return None

    def label(self, query: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out.update(extract_price(query))
        out.update(extract_gender_age(query))
        for key, fn in (
            ("color", extract_color),
            ("material", extract_material),
            ("size", extract_size),
            ("brand", self.extract_brand),
        ):
            val = fn(query)
            if val:
                out[key] = val
        return out
