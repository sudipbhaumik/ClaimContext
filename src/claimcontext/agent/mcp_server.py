"""MCP server exposing the citation-checker tool (spec-5b).

Why citation-checker and not the full agent: exposing run_agent() would require
solving external-caller identity (who is the MCP client, and what Principal do they
resolve to?) — a real design question this project's auth model doesn't yet answer,
and forcing it into this spec would be scope creep on a POC deliverable. The
citation-checker is stateless, takes already-produced data as input, needs no
entitlement context of its own, and is genuinely useful standalone — an external
client can audit any grounded-answer system's citations, not just this one.
(See specs/spec-5b-agent-tools-hardening.md, "Decisions made".)

API verified against the installed `mcp==2.0.0` package before writing this file
(per CLAUDE.md §2A.5 / spec-5b's own warning about MCP API drift). This version's
server class is `mcp.server.mcpserver.MCPServer` — NOT `mcp.server.fastmcp.FastMCP`,
which is what most current online examples show (that module doesn't exist in 2.0).
Confirmed via inspect.signature() against the installed package, not assumed from
docs or memory.

Run: `uv run python -m claimcontext.agent.mcp_server` (stdio transport).
"""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from claimcontext.agent.tools.citation_checker import check_citations_raw

mcp_server = MCPServer(
    name="claimcontext-citation-checker",
    version="0.1.0",
    instructions=(
        "Detects FABRICATED citations in a grounded-answer system's response: a "
        "cited doc_id that was never actually retrieved. Does NOT verify that "
        "retrieved citations are the CORRECT evidence for the question — a "
        "citation can pass this check while still being the wrong source's "
        "content (see ClaimContext's KNOWN_ISSUES.md KI-1)."
    ),
)


@mcp_server.tool()
def check_citations(answer: str, retrieved_doc_ids: list[str]) -> dict:
    """Check an answer's self-reported citations against what was actually retrieved.

    Args:
        answer: The full answer text, including any bracketed source references
            (e.g. "[doc_id | p.1 | Section]" or similar — the parser tolerates
            format variation, see citation_checker.py).
        retrieved_doc_ids: The doc_ids that were actually retrieved and made
            available to the system that produced `answer`.

    Returns a dict with: claimed_doc_ids, retrieved_doc_ids, fabricated,
    has_fabricated_citations. has_fabricated_citations=False means no invented
    citation was found — it does NOT mean the answer is correctly grounded.
    """
    result = check_citations_raw(answer, retrieved_doc_ids)
    return result.model_dump()


if __name__ == "__main__":
    mcp_server.run(transport="stdio")
