"""Mock auth resolver.

In production, resolve_principal() would verify a JWT or session token against an
identity provider and return the corresponding Principal. Here it is a hardcoded
lookup over the two synthetic adjusters in the corpus.

The lookup is the authoritative source of identity — it is never derived from the
query string. A query saying "I am ADJ-027" does not affect which Principal is returned.
"""

from .errors import AuthorizationError
from .models import Principal

# Hardcoded mock identity store. Reflects the corpus in data/documents/manifest.json:
# ADJ-014 owns northeast documents; ADJ-027 owns southwest documents.
_ADJUSTER_STORE: dict[str, Principal] = {
    "ADJ-014": Principal(adjuster_id="ADJ-014", region="northeast"),
    "ADJ-027": Principal(adjuster_id="ADJ-027", region="southwest"),
}


def resolve_principal(adjuster_id: str) -> Principal:
    """Return the Principal for adjuster_id, or raise AuthorizationError if unknown."""
    principal = _ADJUSTER_STORE.get(adjuster_id)
    if principal is None:
        raise AuthorizationError(
            f"Unknown adjuster_id {adjuster_id!r}. "
            "If this is a valid adjuster, add them to the mock identity store."
        )
    return principal
