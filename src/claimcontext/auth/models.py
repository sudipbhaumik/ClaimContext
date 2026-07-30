from pydantic import BaseModel


class Principal(BaseModel):
    adjuster_id: str  # e.g. "ADJ-014"
    region: str  # e.g. "northeast"
