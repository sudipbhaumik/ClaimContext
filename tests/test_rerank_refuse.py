"""Tests for spec-2c: cross-encoder reranker + refuse gate.

Proof 1 and 2 are live proofs requiring Qdrant + bge-reranker model.
Proof 3 and 4 are live proofs requiring Qdrant + bge-reranker + Ollama.
Unit tests run without any external dependencies.

Run unit tests only:
    pytest tests/test_rerank_refuse.py -m "not retrieval and not llm" -v

Run rerank proofs (no LLM required):
    pytest tests/test_rerank_refuse.py -m "retrieval" -v -s

Run refuse proofs (requires Ollama):
    pytest tests/test_rerank_refuse.py -m "llm" -v -s
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from claimcontext.retrieval.ask import _REFUSE_MESSAGE
from claimcontext.retrieval.models import RetrievalResult

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_result(chunk_id: str, doc_id: str, score: float, text: str = "") -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        doc_id=doc_id,
        doc_type="policy",
        policy_number=None,
        claim_number=None,
        page=1,
        section="",
        score=score,
        text=text or f"text for {chunk_id}",
        embedding_model="BAAI/bge-large-en-v1.5",
        chunker_version="v1",
    )


# ── Unit: Reranker ────────────────────────────────────────────────────────────


class TestReranker:
    def _mock_reranker(self, scores: list[float]):
        """Return a Reranker with the CrossEncoder model pre-injected as a mock."""
        from claimcontext.config import Settings
        from claimcontext.retrieval.reranker import Reranker

        settings = Settings()
        reranker = Reranker(settings)
        mock_model = MagicMock()
        mock_model.predict.return_value = scores
        reranker._model = mock_model  # bypass lazy load
        return reranker

    def test_reorders_by_score(self) -> None:
        """Reranker must sort candidates descending by cross-encoder score."""
        candidates = [
            _make_result("A", "doc-A", score=0.9),
            _make_result("B", "doc-B", score=0.7),
            _make_result("C", "doc-C", score=0.5),
        ]
        # A low, B mid, C high — opposite of original order
        reranker = self._mock_reranker([0.1, 0.3, 0.9])
        result = reranker.rerank("test query", candidates)

        assert result[0].chunk_id == "C"
        assert result[1].chunk_id == "B"
        assert result[2].chunk_id == "A"

    def test_score_replaced(self) -> None:
        """Output .score must be the cross-encoder score, not the original."""
        candidates = [_make_result("X", "doc-X", score=0.99)]
        reranker = self._mock_reranker([0.42])
        result = reranker.rerank("query", candidates)

        assert abs(result[0].score - 0.42) < 1e-6
        assert result[0].score != 0.99  # original score overwritten

    def test_empty_candidates(self) -> None:
        from claimcontext.config import Settings
        from claimcontext.retrieval.reranker import Reranker

        settings = Settings()
        reranker = Reranker(settings)
        assert reranker.rerank("query", []) == []


# ── Unit: refuse gate (via ask()) ─────────────────────────────────────────────


class TestRefuseGate:
    def _make_settings(self, threshold: float = 0.55):
        from claimcontext.config import Settings

        return Settings(refuse_threshold=threshold)

    def _make_reranker_with_score(self, score: float):
        """Return a Reranker whose rerank() sets every result's score to `score`."""
        from claimcontext.retrieval.reranker import Reranker

        reranker = MagicMock(spec=Reranker)
        reranker.rerank.side_effect = lambda query, candidates: [
            c.model_copy(update={"score": score}) for c in candidates
        ]
        return reranker

    def test_refuse_when_below_threshold(self) -> None:
        from claimcontext.retrieval.ask import ask

        settings = self._make_settings(threshold=0.55)
        reranker = self._make_reranker_with_score(0.30)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("A", "doc-A", 0.8)]
        mock_llm = MagicMock()

        result = ask(
            query="anything",
            retriever=mock_retriever,
            llm=mock_llm,
            settings=settings,
            reranker=reranker,
        )

        assert result.refused is True
        assert result.answer == _REFUSE_MESSAGE
        assert result.citations == []
        mock_llm.complete.assert_not_called()

    def test_no_refuse_when_above_threshold(self) -> None:
        from claimcontext.retrieval.ask import ask

        settings = self._make_settings(threshold=0.55)
        reranker = self._make_reranker_with_score(0.92)

        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("A", "doc-A", 0.8)]
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "The answer."

        with patch("claimcontext.retrieval.ask._load_prompt", return_value="system"):
            result = ask(
                query="covered peril?",
                retriever=mock_retriever,
                llm=mock_llm,
                settings=settings,
                reranker=reranker,
            )

        assert result.refused is False
        assert result.answer == "The answer."
        mock_llm.complete.assert_called_once()

    def test_refuse_message_does_not_disclose_existence(self) -> None:
        """§6B: the refusal message must not say 'nothing found' or 'no documents'."""
        assert "nothing" not in _REFUSE_MESSAGE.lower()
        assert "no document" not in _REFUSE_MESSAGE.lower()
        assert "not found" not in _REFUSE_MESSAGE.lower()
        assert "no record" not in _REFUSE_MESSAGE.lower()

    def test_no_reranker_skips_refuse_gate(self) -> None:
        """When reranker=None, the refuse gate is not applied."""
        from claimcontext.retrieval.ask import ask

        settings = self._make_settings(threshold=0.99)  # would always refuse
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("A", "doc-A", 0.1)]
        mock_llm = MagicMock()
        mock_llm.complete.return_value = "answer"

        with patch("claimcontext.retrieval.ask._load_prompt", return_value="system"):
            result = ask(
                query="q",
                retriever=mock_retriever,
                llm=mock_llm,
                settings=settings,
                reranker=None,
            )

        assert result.refused is False
        mock_llm.complete.assert_called_once()

    def test_refused_askresult_has_no_citations(self) -> None:
        from claimcontext.retrieval.ask import ask

        settings = self._make_settings(threshold=0.99)
        reranker = self._make_reranker_with_score(0.01)
        mock_retriever = MagicMock()
        mock_retriever.search.return_value = [_make_result("A", "doc-A", 0.5)]
        mock_llm = MagicMock()

        result = ask("q", mock_retriever, mock_llm, settings, reranker)

        assert result.citations == []
        assert result.refused is True


# ── Proof 1: domain-mismatch diagnostic — "deductible Section 4.2" ──────────


@pytest.mark.retrieval
def test_proof1_deductible_domain_mismatch_diagnostic() -> None:
    """Documents the domain mismatch for clause-reference queries with bge-reranker-base.

    spec-2b degraded 'deductible Section 4.2' from dense rank 1 to fused rank 6.
    The spec-2c reranker cannot fully restore this regression because:
      - The corpus chunk text does not contain "Section 4.2" literally (section numbers
        appear only in the query, not in the chunk headings or body text).
      - bge-reranker-base scores all policy passages very low (< 0.1) for this query —
        the model has no signal to distinguish which policy's deductible section matches.
      - With refuse_threshold=0.55, the gate fires and prevents a potentially wrong answer.

    This is a finding for spec-4 golden-set calibration: either lower the threshold
    (if clause-reference queries should be answered with the best available match) or
    rewrite the query as a natural language question ("what is the deductible for
    policy POL-3301") which the reranker handles confidently.

    Assertions:
      1. POL-3301-policy is present in the reranked results (not dropped entirely).
      2. All top-5 scores are below 0.1 (confirming no confident match — domain limit).
      3. The refuse gate would fire at the current threshold (score < refuse_threshold).
    """
    from claimcontext.config import Settings
    from claimcontext.retrieval.hybrid_retriever import HybridRetriever
    from claimcontext.retrieval.reranker import Reranker

    settings = Settings()
    retriever = HybridRetriever(settings)
    retriever.check_index_staleness()
    reranker = Reranker(settings)

    query = "deductible Section 4.2"
    candidates = retriever.search(query, top_k=settings.top_k * 3)
    reranked = reranker.rerank(query, candidates)

    top_score = reranked[0].score if reranked else 0.0

    print(f"\nProof 1 — '{query}' reranked top-5 (domain-mismatch diagnostic):")
    for i, r in enumerate(reranked[:5]):
        print(f"  [{i + 1}] score={r.score:.5f}  doc={r.doc_id}  sec={r.section[:40]!r}")
    print(f"  top_score={top_score:.5f}  threshold={settings.refuse_threshold:.3f}")
    print(f"  refuse gate fires: {top_score < settings.refuse_threshold}")

    policy_rank = next((i for i, r in enumerate(reranked) if "POL-3301-policy" in r.doc_id), None)
    rank_display = policy_rank + 1 if policy_rank is not None else "not found"
    print(f"  POL-3301-policy at reranked rank: {rank_display}")

    # Assertion 1: target document not dropped
    assert policy_rank is not None, "POL-3301-policy dropped from reranked results entirely"
    # Assertion 2: all top-5 scores are low — confirming domain mismatch, not a retrieval bug
    assert top_score < 0.1, (
        f"Unexpected high confidence for clause-reference query: top_score={top_score:.5f}. "
        "If this passes, the model or corpus changed — re-examine the proof."
    )
    # Assertion 3: refuse gate fires (conservative behavior is correct when no confident match)
    assert top_score < settings.refuse_threshold, (
        f"Refuse gate should fire for low-confidence clause-reference query. "
        f"top_score={top_score:.5f} >= threshold={settings.refuse_threshold:.3f}. "
        "Consider whether this query should be refused or threshold should be lowered."
    )


# ── Proof 2: flagship ordering — "wind-driven rain coverage" ─────────────────


@pytest.mark.retrieval
def test_proof2_flagship_ordering() -> None:
    """spec-2b had endorsement WR-001 vs Exclusion 2.3 at ranks 1-2, margin 0.00153.
    Reranker must keep both in top-3 and produce a wider, more meaningful margin.

    Print actual scores — these numbers tell whether the reranker is genuinely
    discriminating or just reordering noise.
    """
    from claimcontext.config import Settings
    from claimcontext.retrieval.hybrid_retriever import HybridRetriever
    from claimcontext.retrieval.reranker import Reranker

    settings = Settings()
    retriever = HybridRetriever(settings)
    retriever.check_index_staleness()
    reranker = Reranker(settings)

    query = "wind-driven rain coverage"
    candidates = retriever.search(query, top_k=settings.top_k * 3)
    reranked = reranker.rerank(query, candidates)

    print(f"\nProof 2 — '{query}' reranked top-5:")
    for i, r in enumerate(reranked[:5]):
        print(f"  [{i + 1}] score={r.score:.5f}  doc={r.doc_id}  sec={r.section[:40]!r}")

    rrf_margin = 0.00153  # from spec-2b
    margin = reranked[0].score - reranked[1].score if len(reranked) > 1 else 0.0
    print(f"  reranked margin rank1 vs rank2: {margin:.5f}  (RRF baseline: {rrf_margin:.5f})")

    endorsement_rank = next(
        (i for i, r in enumerate(reranked) if "endorsement-WR001" in r.doc_id), None
    )
    exclusion_rank = next(
        (i for i, r in enumerate(reranked) if "POL-5504-policy" in r.doc_id), None
    )

    assert endorsement_rank is not None and endorsement_rank <= 2, (
        f"Endorsement WR-001 at rank {endorsement_rank} — expected top-3. "
        f"Top-3: {[r.doc_id for r in reranked[:3]]}"
    )
    assert exclusion_rank is not None and exclusion_rank <= 3, (
        f"POL-5504-policy (Exclusion 2.3 chunk) at rank {exclusion_rank} — expected top-4. "
        f"Top-4: {[r.doc_id for r in reranked[:4]]}"
    )
    assert margin > rrf_margin, (
        f"Reranked margin {margin:.5f} is not wider than RRF margin {rrf_margin:.5f} — "
        "reranker is not adding signal over RRF"
    )


# ── Proof 3 + 4: refuse gate fires / does not false-refuse ───────────────────


@pytest.mark.retrieval
def test_proof3_refuse_gate_scores() -> None:
    """Print raw reranker scores for an off-corpus query so the threshold can be calibrated.
    Does not require Ollama — only the reranker model.

    An off-corpus query ("CEO salary") should produce top reranked score < refuse_threshold.
    """
    from claimcontext.config import Settings
    from claimcontext.retrieval.hybrid_retriever import HybridRetriever
    from claimcontext.retrieval.reranker import Reranker

    settings = Settings()
    retriever = HybridRetriever(settings)
    retriever.check_index_staleness()
    reranker = Reranker(settings)

    off_corpus_query = "what is the CEO annual salary and bonus structure"
    candidates = retriever.search(off_corpus_query, top_k=settings.top_k * 3)
    reranked = reranker.rerank(off_corpus_query, candidates)

    top_score = reranked[0].score if reranked else 0.0
    print("\nProof 3 — off-corpus query reranked scores (top-5):")
    for i, r in enumerate(reranked[:5]):
        print(f"  [{i + 1}] score={r.score:.5f}  doc={r.doc_id}")
    print(f"  threshold={settings.refuse_threshold:.3f}  top_score={top_score:.5f}")
    print(f"  would refuse: {top_score < settings.refuse_threshold}")

    assert top_score < settings.refuse_threshold, (
        f"Off-corpus query top score {top_score:.5f} >= threshold {settings.refuse_threshold}. "
        "Refuse gate would NOT fire. Consider raising refuse_threshold."
    )


@pytest.mark.retrieval
def test_proof4_answerable_query_not_refused() -> None:
    """A clearly answerable query must have top reranked score above refuse_threshold."""
    from claimcontext.config import Settings
    from claimcontext.retrieval.hybrid_retriever import HybridRetriever
    from claimcontext.retrieval.reranker import Reranker

    settings = Settings()
    retriever = HybridRetriever(settings)
    retriever.check_index_staleness()
    reranker = Reranker(settings)

    query = "what does the wind-driven rain endorsement cover"
    candidates = retriever.search(query, top_k=settings.top_k * 3)
    reranked = reranker.rerank(query, candidates)

    top_score = reranked[0].score if reranked else 0.0
    print(f"\nProof 4 — answerable query top score: {top_score:.5f}")
    would_refuse = top_score < settings.refuse_threshold
    print(f"  threshold={settings.refuse_threshold:.3f}  would refuse: {would_refuse}")

    assert top_score >= settings.refuse_threshold, (
        f"Answerable query top score {top_score:.5f} < threshold {settings.refuse_threshold}. "
        "False refuse — consider lowering refuse_threshold."
    )
