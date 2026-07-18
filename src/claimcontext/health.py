"""Health check — loads config and prints key settings to prove the package starts."""

from claimcontext.config import get_settings


def health() -> None:
    s = get_settings()
    print(f"app_name:    {s.app_name}")
    print(f"environment: {s.environment}")
    print(f"qdrant_url:  {s.qdrant_url}")


if __name__ == "__main__":
    health()
