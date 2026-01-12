# src/swissdox/text.py
from __future__ import annotations
import html
import re


_WS_RE = re.compile(r"\s+")


def clean_text(s: str) -> str:
    """Light text cleanup: unescape, normalize whitespace, trim quotes."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s).replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = _WS_RE.sub(" ", s).strip()
    return s.strip(' "“”„\'')


def clean_xml_swissdox(s: str) -> str:
    """Swissdox content is often HTML/XML-ish. Remove tags and normalize."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = re.sub(r"</p>", "\n", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = _WS_RE.sub(" ", s).strip()
    return s