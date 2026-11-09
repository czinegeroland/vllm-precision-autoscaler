"""Version names and parsing (pure Python, no torch: used by settings.py and
unit tests). The kinds themselves live in versions.py."""

from __future__ import annotations

# name -> (bits, method); must match versions.KINDS
KNOWN = {
    "high": (16, "unquantized"),
    "fp8": (8, "fp8_per_channel_w8a8"),
}
ALIASES = {"bf16": "high", "fp16": "high", "16": "high", "8": "fp8"}


def canonical(name: str) -> str:
    name = name.strip().lower()
    return ALIASES.get(name, name)


def parse_versions(raw: str) -> tuple[str, ...]:
    """'high,fp8' -> ('high', 'fp8'). 'high' (the loaded weights) is always included."""
    names = [canonical(x) for x in raw.split(",") if x.strip()]
    unknown = [x for x in names if x not in KNOWN]
    if unknown:
        raise ValueError(f"unknown precision version(s) {unknown}; available: {sorted(KNOWN)}")
    if "high" not in names:
        names.insert(0, "high")
    seen: list[str] = []
    for x in names:
        if x not in seen:
            seen.append(x)
    return tuple(seen)
