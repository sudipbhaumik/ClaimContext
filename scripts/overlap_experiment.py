"""Overlap experiment: see chunk boundaries on WR-001 endorsement at overlap 0 / 64 / 256."""

from pathlib import Path

from claimcontext.config import Settings
from claimcontext.ingestion import chunk_document, run_discover_extract

results = run_discover_extract(Path("data/documents"), Path("/tmp/exp.json"))
r = next(x for x in results if x.doc_id == "POL-5504-endorsement-WR001")

print(f"\nDocument: {r.doc_id}  ({len(r.document.text.splitlines())} lines)\n")

for overlap in [0, 64, 256]:
    s = Settings(chunk_overlap=overlap)
    chunks = chunk_document(r.document, s)
    print(f"{'=' * 70}")
    print(f"overlap={overlap}: {len(chunks)} chunk(s)")
    for i, c in enumerate(chunks):
        print(f"\n  chunk {i}  page={c.page}  section={c.section!r}")
        # Show first and last 120 chars so you can see what's at each boundary
        text = c.text
        if len(text) > 240:
            print(f"  START: {text[:120]!r}")
            print(f"  END:   {text[-120:]!r}")
        else:
            print(f"  TEXT:  {text!r}")
