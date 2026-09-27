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


def _meta_series(names: pd.Series) -> pd.Series:
    try:
        import jellyfish
        return names.map(lambda s: jellyfish.metaphone(s) if s else "")
    except Exception:
        return None  # caller falls back to name_init


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Unchanged, in-memory version -- fine for small frames. For anything
    multi-million rows, use normalize_frame_to_parquet instead (memory-bounded)."""
    cols = [views_name(r["business_name"]) | views_addr(r["business_address"])
            for _, r in df.iterrows()]
    norm = pd.DataFrame(cols, index=df.index)
    out = pd.concat([df.reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
    meta = _meta_series(out["name_no_legal"])
    out["name_meta"] = meta if meta is not None else out["name_init"]
    return out


def normalize_frame_to_parquet(df: pd.DataFrame, path: str, chunk_size: int = 200_000) -> None:
    """Memory-bounded version of normalize_frame: processes `chunk_size` rows at a
    time and streams each chunk straight to a parquet file, so peak memory is
    O(chunk_size) instead of O(len(df)). Prevents the OOM you hit at ~5M-row files
    (holding the full row-dict list + a duplicate concatenated DataFrame at once).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    n = len(df)
    try:
        for start in range(0, n, chunk_size):
            chunk = df.iloc[start:start + chunk_size]
            cols = [views_name(r["business_name"]) | views_addr(r["business_address"])
                    for _, r in chunk.iterrows()]
            norm = pd.DataFrame(cols, index=chunk.index)
            out = pd.concat([chunk.reset_index(drop=True), norm.reset_index(drop=True)], axis=1)
            meta = _meta_series(out["name_no_legal"])
            out["name_meta"] = meta if meta is not None else out["name_init"]

            table = pa.Table.from_pandas(out, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, table.schema)
            writer.write_table(table)

            print(f"  [normalize] {min(start + chunk_size, n)}/{n} rows -> {path}")
            del chunk, cols, norm, out, table
    finally:
        if writer is not None:
            writer.close()