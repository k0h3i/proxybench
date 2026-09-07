"""Apply the finite formatting rules of benchmark-v1."""

from datetime import date
import re
import unicodedata


MONTHS = {name: i for i, name in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
ENUMS = {
    "scope_type": {"INDIVIDUAL_FUND", "FUND_GROUP"},
    "participation": {"VOTED", "DID_NOT_VOTE", "OTHER"},
    "direction": {"FOR", "AGAINST", "ABSTAIN", "WITHHOLD", "OTHER"},
    "management_recommendation": {"FOR", "AGAINST", "ABSTAIN", "WITHHOLD", "NONE", "OTHER"},
    "disclosed_management_alignment": {
        "WITH_MANAGEMENT", "AGAINST_MANAGEMENT", "NOT_APPLICABLE", "OTHER"},
}
ALIASES = {
    "participation": {"did not vote": "DID_NOT_VOTE"},
    "disclosed_management_alignment": {
        "for": "WITH_MANAGEMENT", "against": "AGAINST_MANAGEMENT", "na": "NOT_APPLICABLE"},
}


def text(value):
    return " ".join(unicodedata.normalize("NFC", value).split())


LEGACY_GUIDE = "benchmark-v1"
ACTIVE_GUIDE = "benchmark-v1-date-order-v2"
POLICY_VERSIONS = {LEGACY_GUIDE: "finite-date-v1", ACTIVE_GUIDE: "finite-date-v2"}


def policy_version(guide_version):
    if guide_version not in POLICY_VERSIONS:
        raise ValueError("Unsupported guide version: " + str(guide_version))
    return POLICY_VERSIONS[guide_version]


def calendar_date(value, *, guide_version=LEGACY_GUIDE):
    policy_version(guide_version)
    value = value.strip()
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return date.fromisoformat(value).isoformat()
    match = re.fullmatch(r"([0-9]{1,2})/([0-9]{1,2})/([0-9]{4})", value)
    if match:
        a, b, year = map(int, match.groups())
        if guide_version == ACTIVE_GUIDE:
            return date(year, a, b).isoformat()
        candidates = set()
        for month, day in ((a, b), (b, a)):
            try:
                candidates.add(date(year, month, day).isoformat())
            except ValueError:
                pass
        if len(candidates) == 1:
            return candidates.pop()
        raise ValueError("Invalid or ambiguous numeric date.")
    match = re.fullmatch(r"([A-Za-z]{3}) +([0-9]{1,2}), +([0-9]{4})", value)
    if match:
        month, day, year = match.groups()
    else:
        match = re.fullmatch(r"([0-9]{1,2})-([A-Za-z]{3})-([0-9]{4})", value)
        if not match:
            raise ValueError("Date is outside the finite wire grammar.")
        day, month, year = match.groups()
    if month.upper() not in MONTHS:
        raise ValueError("Unknown month.")
    return date(int(year), MONTHS[month.upper()], int(day)).isoformat()


def quantity(value):
    if not re.fullmatch(r"(?:0|[1-9][0-9]*|[0-9]{1,3}(?:,[0-9]{3})+)(?:\.[0-9]+)?", value):
        raise ValueError("Invalid decimal quantity string.")
    whole, _, fraction = value.replace(",", "").partition(".")
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return whole + ("." + fraction if fraction else "")


def scalar(value, kind, *, guide_version=LEGACY_GUIDE):
    policy_version(guide_version)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A resolved scalar must be a nonempty string.")
    if kind in ENUMS:
        normalized = text(value).casefold()
        aliases = {v.casefold(): v for v in ENUMS[kind]}
        aliases.update(ALIASES.get(kind, {}))
        if normalized not in aliases:
            raise ValueError(f"Unknown {kind}; use OTHER with source wording when appropriate.")
        return aliases[normalized]
    if kind == "date":
        return calendar_date(value, guide_version=guide_version)
    if kind == "amount":
        return quantity(value)
    if kind == "identifier":
        return value.strip()
    if kind == "name":
        return text(value).casefold()
    return text(value)
