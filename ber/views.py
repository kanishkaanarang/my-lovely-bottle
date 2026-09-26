"""Additional lossy text views; originals always remain available."""
import re
import unicodedata
from functools import lru_cache
from urllib.parse import urlsplit

from .common import clean_nulls, core_name, digits_ascii, fold_latin, normalize

ADDRESS = {
    "US": {"st": "street", "rd": "road", "ave": "avenue", "blvd": "boulevard", "ste": "suite", "hwy": "highway", "ln": "lane", "dr": "drive"},
    "India": {"rd": "road", "nr": "near", "opp": "opposite", "nagar": "nagar", "flr": "floor", "st": "street", "marg": "road"},
    "France": {"st": "saint", "ste": "sainte", "r": "rue", "bd": "boulevard", "av": "avenue", "pl": "place", "rte": "route"},
}


def script(text):
    found = set()
    for c in text:
        if not c.isalpha():
            continue
        name = unicodedata.name(c, "")
        found.add(next((s for s in ("LATIN", "DEVANAGARI", "BENGALI", "TAMIL") if s in name), "OTHER"))
    return next(iter(found)) if len(found) == 1 else "MIXED" if found else "EMPTY"


def indic_fold(text):
    # Secondary view only. Nasal equivalences are deliberately lossy.
    text = unicodedata.normalize("NFD", normalize(text))
    text = text.replace("़", "").replace("ঁ", "ং").replace("ँ", "ं")
    for pattern in ("ङ्", "ञ्", "ण्", "न्", "म्"):
        text = text.replace(pattern, "ं")
    return unicodedata.normalize("NFC", text)


def chargrams(text):
    text = "".join(text.split())
    return sorted({"g" + text[i:i+3].encode().hex() for i in range(max(0, len(text)-2))})


def phonetic(text):
    # Local Soundex per Latin token; low-recall optional channel, never identity proof.
    groups = {c: str(i) for i, group in enumerate(("bfpv", "cgjkqsxz", "dt", "l", "mn", "r"), 1) for c in group}
    keys = []
    for word in fold_latin(text).split():
        if not word.isascii() or not word.isalpha():
            continue
        code, previous = word[0], groups.get(word[0], "0")
        for c in word[1:]:
            value = groups.get(c, "0")
            if value != "0" and value != previous:
                code += value
            previous = value
        keys.append((code + "000")[:4])
    return " ".join(sorted(keys))


def name_aliases(text):
    parts = re.split(r"\b(?:dba|doing business as|t/a|formerly)\b", text, flags=re.I)
    aliases = {normalize(p) for p in parts if normalize(p)}
    if re.search(r"(?:https?://|www\.|\.[a-z]{2,}(?:/|$))", text, re.I):
        match = re.search(r"(?:https?://|www\.)[^\s\[\]()<>,]+", text, re.I)
        value = match[0] if match else text.strip("[]() ")
        try:
            host = urlsplit(value if "://" in value else "//"+value).hostname
        except ValueError:
            host = None
        if host:
            aliases.add(normalize(host.removeprefix("www.").split(".")[0]))
    return tuple(sorted(aliases))


def address_parts(text, country):
    raw = digits_ascii(clean_nulls(text)).casefold()
    postcode_pattern = r"\b\d{6}\b" if country == "India" else r"\b\d{5}(?:-\d{4})?\b"
    codes = re.findall(postcode_pattern, raw)
    number = re.search(r"(?<!\w)(\d+)(?:\s*[-–]\s*(\d+))?(?:\s*(bis|ter)\b)?(?:\s+(1/2|1/4|3/4))?", raw)
    lo = int(number[1]) if number else -1
    hi = int(number[2]) if number and number[2] else lo
    if hi < lo:
        hi = lo
    unit = re.search(r"(?:suite|ste|unit|apt|apartment|flat|#)\s*([\w-]+)", raw)
    expanded = " ".join(ADDRESS.get(country, {}).get(t, t) for t in fold_latin(normalize(text)).split())
    if country == "France":
        expanded = re.sub(r"\bcedex(?:\s+\d{1,2})?\b", "", expanded)
        expanded = re.sub(r"\bparis\s+\d{1,2}(?:e|eme|er)\b", "paris", expanded)
    expanded = " ".join(expanded.split())
    street = next((t for t in expanded.split() if t.isalpha() and t not in {"bis", "ter", "no", "number"}), "")
    # A weak locality proxy, never asserted to be an authoritative geographic parse.
    locality = normalize(text.split(",")[-1]) if "," in text else ""
    return dict(expanded=expanded, postcode=(codes[-1][:5] if country != "India" else codes[-1]) if codes else "",
                house_lo=lo, house_hi=hi, house_suffix=number[3] or "" if number else "",
                fraction=number[4] or "" if number else "", unit=unit[1] if unit else "",
                street_key=f"{lo}:{street}" if lo >= 0 and street else "", locality=locality)


@lru_cache(maxsize=50_000)
def views(name, address, country, disabled=()):
    n, a = normalize(name), normalize(address)
    fn, fa = (n, a) if "accents" in disabled else (fold_latin(n), fold_latin(a))
    core = fn if "suffix" in disabled else core_name(name, country, "accents" not in disabled)
    result = dict(n=n, a=a, folded=fn, core=core, compact="".join(fn.split()),
                  sorted=core if "reordering" in disabled else " ".join(sorted(core.split())), indic=indic_fold(n), phonetic=phonetic(core),
                  script=script(name), aliases=name_aliases(name), folded_address=fa)
    result.update(address_parts(address, country))
    if "abbreviations" in disabled: result["expanded"] = fa
    if "accents" in disabled:
        result["expanded"] = " ".join(ADDRESS.get(country, {}).get(t, t) for t in a.split()) if "abbreviations" not in disabled else a
    return result
