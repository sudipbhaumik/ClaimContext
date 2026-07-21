"""Text normaliser: light cleanup without destroying document structure.

Rules applied (in order):
1. Always preserve ``<!-- PAGE n -->`` markers and tab-separated table rows.
2. Strip page-number boilerplate lines matching ``^Page N of M$``.
3. Collapse runs of blank lines to at most ``_MAX_BLANK`` consecutive blanks.
4. Strip leading and trailing blank lines from the result.

What is NOT done (deliberately):
- No repeated-header/footer block detection: fragile heuristic, low value
  for this corpus. The only repeating boilerplate is page numbers (covered
  by rule 2).
- No whitespace normalisation within lines: fixed-width table column
  alignment depends on preserved inter-column spaces.
- No lowercasing or stemming: that belongs in the retriever, not the corpus.
"""

from __future__ import annotations

import re

_PAGE_NUMBER_RE = re.compile(r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE)
_MONEY_RE = re.compile(r"\$[\s\d,]+\.\d{2}")
_TABLE_SEP_RE = re.compile(r"[─═=]{4,}")
_COL_GAP_RE = re.compile(r"\s{2,}")
_MAX_BLANK = 2


def _is_table_line(line: str) -> bool:
    """Heuristic: is this line part of a fixed-width table?

    Lines are exempt from boilerplate stripping if they:
    - contain a formatted money value (``$  450.00``), or
    - consist mostly of box-drawing or equals-sign separators, or
    - have 3+ runs of 2+ spaces (column gap pattern in fixed-width tables).

    Failure mode: a centred prose heading with multiple spaces may be
    classified as a table line. The consequence is benign — the line is
    preserved rather than stripped, which is the safer outcome.
    """
    if _MONEY_RE.search(line):
        return True
    if _TABLE_SEP_RE.search(line):
        return True
    return len(_COL_GAP_RE.findall(line)) >= 3


def normalise(text: str) -> str:
    lines = text.splitlines()
    result: list[str] = []
    blank_run = 0

    for line in lines:
        # Always keep page markers intact
        if line.startswith("<!-- PAGE"):
            blank_run = 0
            result.append(line)
            continue

        # Strip page-number boilerplate (e.g. "Page 1 of 3") unless it looks
        # like a table line (belt-and-suspenders guard)
        if _PAGE_NUMBER_RE.match(line) and not _is_table_line(line):
            continue

        stripped = line.strip()
        if not stripped:
            blank_run += 1
            if blank_run <= _MAX_BLANK:
                result.append("")
            continue

        blank_run = 0
        result.append(line)

    # Trim leading/trailing blank lines
    while result and not result[0].strip():
        result.pop(0)
    while result and not result[-1].strip():
        result.pop()

    return "\n".join(result)
