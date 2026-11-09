"""Output schema for the query-parsing task, plus parsing and normalisation.

A parse is a flat JSON object. Every field is optional: a field that the query
does not mention is *omitted* (never set to null or ""). Keeping the object flat
makes field-level precision/recall well defined and keeps the reward simple.

This module is the single source of truth for:
  * which fields exist and what type each one has,
  * how a model completion is turned into a dict (strict and lenient),
  * how values are normalised before two parses are compared.
Every stage (labelling, eval, DPO pair scoring, GRPO reward) imports from here,
so "correct" means exactly the same thing everywhere.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

# Field name -> (python type, short description used in the prompt).
FIELDS: dict[str, tuple[type, str]] = {
    "product_type": (str, "what the shopper wants, as a short noun phrase (e.g. 'running shoes')"),
    "brand": (str, "brand or manufacturer named in the query"),
    "color": (str, "colour(s) requested"),
    "size": (str, "size, dimension or capacity (e.g. 'xl', '10.5', 'queen', '55 inch', '64gb')"),
    "material": (str, "material (e.g. 'leather', 'stainless steel', 'cotton')"),
    "gender": (str, "one of: men, women, unisex"),
    "age_group": (str, "one of: baby, kids, teen, adult (only if stated or clearly implied)"),
    "price_min": (float, "lower price bound in the query's currency, as a number"),
    "price_max": (float, "upper price bound in the query's currency, as a number"),
    "compatible_with": (
        str,
        "device or product this item must fit or work with (e.g. 'iphone 12')",
    ),
}

ENUMS: dict[str, set[str]] = {
    "gender": {"men", "women", "unisex"},
    "age_group": {"baby", "kids", "teen", "adult"},
}

STRING_FIELDS = [k for k, (t, _) in FIELDS.items() if t is str]

_WS = re.compile(r"\s+")
# Keep letters, digits, '.', and spaces. Dots matter for sizes like "10.5".
_PUNCT = re.compile(r"[^\w\s.]")


def normalize_text(value: str) -> str:
    """Lowercase, drop punctuation except '.', collapse whitespace.

    Deliberately conservative: no stemming or plural folding. "shoe" vs "shoes"
    counts as a mismatch. This keeps the metric strict and easy to explain; the
    labelling prompt asks for the surface form used in the query, which makes the
    strictness fair to the model.
    """
    # Apostrophes are deleted, not spaced: "men's" -> "mens", "levi's" -> "levis".
    value = value.lower().replace("_", " ").replace("'", "").replace("’", "")
    value = _PUNCT.sub(" ", value)
    value = _WS.sub(" ", value).strip()
    return value.strip(".").strip()


def normalize_value(name: str, value: Any) -> Any | None:  # noqa: PLR0911
    """Normalise one field value. Returns None if the value is unusable."""
    if name not in FIELDS or value is None:
        return None
    typ = FIELDS[name][0]
    if typ is float:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return round(float(value), 2)
        if isinstance(value, str):
            m = re.search(r"\d+(?:\.\d+)?", value.replace(",", ""))
            return round(float(m.group()), 2) if m else None
        return None
    if isinstance(value, list):  # e.g. ["red", "blue"] -> "red blue"
        value = " ".join(str(v) for v in value)
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return None
    text = normalize_text(str(value))
    if not text:
        return None
    if name in ENUMS and text not in ENUMS[name]:
        return None
    return text


def normalize_parse(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalise a parse dict: drop unknown keys, empty values and bad types."""
    out: dict[str, Any] = {}
    for k, v in obj.items():
        nv = normalize_value(k, v)
        if nv is not None:
            out[k] = nv
    return out


@dataclass
class ParseResult:
    """Outcome of turning a raw completion into a parse.

    strict_json: the whole completion (whitespace-stripped) is one JSON object.
    lenient_json: a JSON object could be recovered from somewhere in the text.
    schema_ok: the object has only known keys, each with the right type / enum value.
    parse: normalised dict (empty if nothing could be recovered).
    """

    strict_json: bool = False
    lenient_json: bool = False
    schema_ok: bool = False
    parse: dict[str, Any] = field(default_factory=dict)

    @property
    def format_ok(self) -> bool:
        """Format bonus condition: strict JSON *and* schema-valid."""
        return self.strict_json and self.schema_ok


def _schema_ok(obj: dict[str, Any]) -> bool:
    for k, v in obj.items():
        if k not in FIELDS:
            return False
        typ = FIELDS[k][0]
        if typ is float:
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                return False
        else:
            if not isinstance(v, str) or not v.strip():
                return False
            if k in ENUMS and normalize_text(v) not in ENUMS[k]:
                return False
    return True


def parse_completion(text: str) -> ParseResult:
    """Turn a model completion into a ParseResult.

    Strict mode accepts only a bare JSON object (what we train for). Lenient mode
    also recovers the first JSON object embedded in surrounding text or a
    ```json fence, so that field-level metrics still credit a model that got the
    content right but the wrapper wrong. Both are reported separately.
    """
    res = ParseResult()
    stripped = text.strip()
    obj: Any = None
    try:
        obj = json.loads(stripped)
        if isinstance(obj, dict):
            res.strict_json = True
            res.lenient_json = True
        else:
            obj = None
    except (json.JSONDecodeError, ValueError):
        obj = None
    if obj is None:
        start = stripped.find("{")
        while start != -1 and obj is None:
            try:
                cand, _ = json.JSONDecoder().raw_decode(stripped[start:])
                if isinstance(cand, dict):
                    obj = cand
                    res.lenient_json = True
            except (json.JSONDecodeError, ValueError):
                pass
            start = stripped.find("{", start + 1)
    if obj is None:
        return res
    res.schema_ok = _schema_ok(obj)
    res.parse = normalize_parse(obj)
    return res


def to_target_json(parse: dict[str, Any]) -> str:
    """Canonical serialisation used as the SFT target.

    Keys follow FIELDS order, compact separators, numbers without trailing .0 when
    integral. A fixed canonical form gives the model one right answer per parse.
    """
    ordered: dict[str, Any] = {}
    for k in FIELDS:
        if k in parse:
            v = parse[k]
            if isinstance(v, float) and v.is_integer():
                v = int(v)
            ordered[k] = v
    return json.dumps(ordered, ensure_ascii=False, separators=(", ", ": "))
