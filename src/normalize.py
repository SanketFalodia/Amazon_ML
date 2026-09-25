"""Multi-view normalization for names and addresses (language/country agnostic)."""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

LEGAL = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "llc", "l.l.c", "ltd",
    "limited", "pvt", "private", "llp", "plc", "gmbh", "sa", "sas", "sarl", "bv", "nv", "ag",
    "pte", "opc", "huf", "and", "&", "the", "dba", "trading", "enterprises", "enterprise",
}
NAME_ABBR = {
    "intl": "international", "tech": "technology", "svcs": "services", "svc": "service",
    "mgmt": "management", "assoc": "associates", "bros": "brothers", "mfg": "manufacturing",
    "univ": "university", "natl": "national", "grp": "group", "hldgs": "holdings",
}
ADDR_ABBR = {
    "rd": "road", "st": "street", "ave": "avenue", "blvd": "boulevard", "dr": "drive",
    "ln": "lane", "hwy": "highway", "n": "north", "s": "south", "e": "east", "w": "west",
    "nr": "near", "opp": "opposite", "apt": "apartment", "fl": "floor", "bldg": "building",
    "sec": "sector", "extn": "extension", "indl": "industrial",
}
LANDMARK = {"near", "nr", "opposite", "opp", "behind", "next", "beside", "adjacent"}
US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia",
    "ks", "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj",
    "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt",
    "va", "wa", "wv", "wi", "wy", "dc",
}


def _base(x: str) -> str:
    x = unicodedata.normalize("NFKC", str(x)).lower().replace("&", " and ")
    x = re.sub(r"[^\w\s]", " ", x)
    return re.sub(r"\s+", " ", x).strip()


def _tokens(x: str, abbr: dict, drop_legal: bool = False) -> list[str]:
    toks = [abbr.get(t, t) for t in _base(x).split()]
    if drop_legal:
        toks = [t for t in toks if t not in LEGAL]
    return toks


def views_name(x: str) -> dict:
    full = _tokens(x, NAME_ABBR)
    clean = _tokens(x, NAME_ABBR, drop_legal=True)
    return {
        "name_raw": _base(x),
        "name_full": " ".join(full),
        "name_no_legal": " ".join(clean),
        "name_sorted": " ".join(sorted(clean)),
        "name_compact": "".join(clean),
        "name_init": "".join(w[0] for w in clean),
    }


def views_addr(x: str) -> dict:
    toks = _tokens(x, ADDR_ABBR)
    digits = re.findall(r"\d+", x)
    pincode = next((d for d in digits if len(d) == 6), "")
    zip5 = next((d for d in digits if len(d) == 5), "")
    state = next((t for t in toks if t in US_STATES), "")
    # crude city = last non-digit token before the state, if any
    nond = [t for t in toks if not t.isdigit()]
    city = ""
    if state and state in nond:
        idx = nond.index(state)
        city = nond[idx - 1] if idx - 1 >= 0 else ""
    return {
        "addr_raw": _base(x),
        "addr_full": " ".join(toks),
        "addr_no_digits": " ".join(t for t in toks if not t.isdigit()),
        "addr_sorted": " ".join(sorted(t for t in toks if not t.isdigit())),
        "pincode": pincode,
        "zip5": zip5,
        "house_no": digits[0] if digits else "",
        "is_landmark": int(bool(LANDMARK & set(toks))),
        "city": city,
        "state": state,
    }


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [views_name(r["business_name"]) | views_addr(r["business_address"])
            for _, r in df.iterrows()]
    norm = pd.DataFrame(cols, index=df.index)
    out = pd.concat([df.reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
    # soundex-lite key
    try:
        import jellyfish
        out["name_meta"] = out["name_no_legal"].map(
            lambda s: jellyfish.metaphone(s) if s else "")
    except Exception:
        out["name_meta"] = out["name_init"]
    return out
