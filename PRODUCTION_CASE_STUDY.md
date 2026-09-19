# ClaimContext — Production Judgment & Case Study

### 0. Scope Declaration

**Production-style project.**

ClaimContext uses production patterns — an entitlement-filtered retrieval pipeline, a refuse gate, a CI-style eval harness with a documented-exceptions list, request-level tracing, and health/readiness endpoints. But it has never carried real user or system traffic.

It runs against a synthetic 30-document corpus. It's exercised through `pytest` and manual CLI/API/UI runs on a single developer machine. There is no deployed instance, no external caller, and no `.github/` CI pipeline. Every claim below is verified against local test runs, not production telemetry.

---

### 0.5. Highlights — the moments worth reading first

If you only have a few minutes, these are the four moments in the project that best show the engineering judgment behind it:

- **The MPS non-determinism that flipped a decision gate.** A reranker score drifted 0.076 between runs on the same query — enough to flip the refuse gate. The fix was to pin the reranker to CPU. Slower, but reproducible. A decision gate has to give the same answer for the same input, or it isn't a gate.
- **The section-embedding fix that was measured to be a no-op and reverted.** A fix for terse claim notes was implemented, re-indexed, and then measured to have zero effect on the notes it was meant to help — while actively regressing an unrelated policy query. Fully reverted. See KI-2.
- **The same-claim filter that closes cross-claim contamination — but explicitly not the harder subset.** A filter drops chunks from a different claim than the one being asked about. It fixes one real failure and is honest that a related failure (same claim, wrong subtopic) remains open as KI-10, not buried inside the fixed issue.
- **The 9.5-hour hang from a missing timeout.** An external call without an explicit `timeout=` blocked for 9.5 hours against an unreachable address. A `tenacity` retry wrapper was documented as "hardened" but never engaged, because a hang never raises. Fixed with explicit timeouts. The original trigger was not cleanly reproduced afterward, and that's disclosed.

Each of these is expanded below.

---

### 1. Problem & Constraint

**What it does, and what it must not do.** This system retrieves and cites source text from policy documents, claim notes, and claim documents. The goal is to help an adjuster answer a coverage question faster. It must not compute payouts, deductibles, or coverage determinations. It must not answer from volatile system-of-record fields such as claim status, reserves, or payments. And it must not present an ungrounded or cross-claim-contaminated answer as if it were grounded.

This boundary is enforced in code, not just in intent.

`_is_tier3_query()` in [`src/claimcontext/retrieval/ask.py`](src/claimcontext/retrieval/ask.py) refuses claim-status, reserve, and payment questions before retrieval runs. These fields are explicitly excluded from the embedded metadata schema. A claim's status can change the moment a document is written, and an embedded value is a snapshot with no way to signal it went stale.

`refuse_threshold` (default `0.55`, in [`src/claimcontext/config.py`](src/claimcontext/config.py)) makes `ask()` refuse when the top reranked score is weak. The system is built to say "I don't have enough in the documents to answer this" instead of guessing.

**Why a confident hallucination is worse than a refusal here.** An adjuster acting on a wrong "no investigation has occurred" answer, or a cross-claim-contaminated citation, causes a real downstream error — a wrong file referenced, a wrong coverage clause cited. A refusal costs the adjuster one follow-up lookup. A wrong confident answer can go undetected and end up in a claim record. That asymmetry is why the refuse gate exists as a first-class code path.

On every refusal branch, `ask()` returns `refused=True` with `citations=[]` and `retrieved_chunks=[]`. Commit `eb4c586` fixed a real leak where scored-but-refused chunks were being returned in the response.

**Non-goals**, stated in `specs/CLAUDE.md` §2 and verified as absent from the code. No module in `src/claimcontext/` computes a monetary amount. There is no payout, deductible, or coverage math anywhere. There is no claim-decision automation — the system informs, the adjuster decides. There is no live connector to the claims system of record for Tier-3 fields.

**Budgets that are actually configured, not aspirational.** `llm_timeout_seconds=30`, `agent_llm_timeout_seconds=15`, `agent_tool_timeout_seconds=10`, `qdrant_timeout_seconds=10`, `agent_max_tool_calls=10`, `agent_max_sub_queries=4`. All in `config.py`, all overridable via `.env`, none hardcoded in business logic. The cost budget is informal, not code-enforced. Real API spend is capped to roughly $10–$20 for eval and demo runs by defaulting to local Ollama. There is no token-budget guard inside `ask()` itself.

---

### 2. Architecture Decisions & Tradeoffs

**Decision 1 — Hand-implemented RRF fusion, with a separate cross-encoder rerank stage.**

Reciprocal Rank Fusion is implemented by hand in [`src/claimcontext/retrieval/rrf.py`](src/claimcontext/retrieval/rrf.py) rather than hidden inside a library call. The cross-encoder rerank (`bge-reranker-base`, [`src/claimcontext/retrieval/reranker.py`](src/claimcontext/retrieval/reranker.py)) runs as a distinct stage after fusion.

The reason for hybrid retrieval is that dense embeddings catch semantic similarity while BM25 catches exact-term matches — policy numbers, clause names — that embeddings often miss. The reason for hand-implementing RRF was a stated project requirement: it needed to be explainable. `specs/CLAUDE.md` says: "RRF — hand-implemented, never hidden in a library. I must be able to explain it."

**Verified in code.** `HybridRetriever.search()` calls dense (`Retriever.search()`) and sparse (`BM25Index.search()`) independently. Then `rrf()` fuses. Then `Reranker.rerank()` re-scores the fused top-N. Three distinct stages, never collapsed. Matches the ordering rule in `CLAUDE.md`: "fuse first, then rerank... never rerank before RRF."

**Cost of this choice, partly measured and partly inferred.** The reranker is likely the single most expensive stage per query — this is an inference from how cross-encoders work, not a live-load measurement. `bge-reranker-base` runs a full cross-encoder pass over `rerank_top_n` candidates. It's pinned to `device="cpu"` because commit `891ef50` found MPS gave non-deterministic scores. A 0.076 delta between runs was enough to flip the refuse gate's decision for a borderline query (q06). CPU pinning trades GPU speed for determinism — a real cost accepted deliberately.

What remains unmeasured without live traffic: the actual wall-clock latency distribution under concurrent load. `KI-4` documents that local Ollama's single-worker queue degrades under sustained concurrent calls (reproduced with roughly 72 concurrent RAGAS eval requests). This is evidence the current local-model path has a real concurrency ceiling. It is not evidence about how the system would behave under realistic multi-user load, because it has never been exposed to any.

**Decision 2 — Structure-aware chunking, with a separate path for claim notes.**

Structure-aware chunking ([`src/claimcontext/ingestion/chunker.py`](src/claimcontext/ingestion/chunker.py): `_split_into_sections()`, `_split_section()`) was chosen over naive fixed-size chunking. Policy documents have meaningful section boundaries that fixed-size windows would cut through. The cost is that claim-note chunks (`_chunk_notes()`) never get a `section` label — `chunker.py` sets `section=""` for `claim_note` doc types, because notes don't have real headings.

**Verified in code, and verified as a real cost, not a hypothetical one.** This design choice is the direct root cause of `KI-2`. Terse claim-note shorthand ("Coverage applies under Section I. No applicable exclusions identified.") scores near-zero against natural-language questions on the general-domain cross-encoder. There's no section label to anchor retrieval.

A fix that tried to add one (embed `section + text`, mirroring the reranker's own format) was implemented, re-indexed, and measured. It had zero effect on its own target — the `if section else text` branch always fell through to plain text for exactly the chunks that needed help. It also regressed an unrelated query: `POL-3301-policy`'s best rank for a coverage question moved from 7th to 12th place. This was confirmed by re-embedding the full 97-chunk corpus both ways and comparing cosine scores directly.

The fix was fully reverted. `chunker_version` went back to `v1`. The corpus was re-indexed. The real fix — redesigning `_chunk_notes()` to synthesize an anchor from note-internal markers like `[NOTE-1003-01]` — is scoped but not built.

---

### 3. Evaluation Evidence

**What was tested, and what wasn't.** Retrieval, generation, and agent-routing correctness are tested against an 11-entry RAG golden set and a 4-entry agent-trajectory golden set. The real pipeline runs end-to-end (not mocked). This evidence does not establish behavior at corpus scale, under concurrent multi-user load, across multi-turn conversations, or against adversarial input beyond one synthetic prompt-injection case.

**RAG evaluation** (`tests/test_eval_harness.py`, `src/claimcontext/eval/`):

- Golden set: `data/eval/golden_set_v1.jsonl`, 11 hand-authored entries. Includes a temporal case (policy expiry vs. loss date), a Tier-3 refusal case, an entitlement-refusal case, and the sparse-context informative-absence case (q06).
- Scored with RAGAS (context precision, context recall, faithfulness, answer relevance) using a different-model-family judge — local Ollama `mistral`, distinct from the `llama3.2` answer model. This is a design choice to reduce self-preference bias. See `eval_ragas_llm_provider` in `config.py`.
- `KNOWN_EVAL_EXCEPTIONS = frozenset({"q02", "q05", "q08"})` in `tests/test_eval_harness.py`. The harness fails the gate on any unexpected failure, but three specific, root-caused, documented gaps are allowed to fail without blocking the build. This is deliberate. A strict "must pass 100%" gate would either force hiding a known gap behind a loosened assertion, or block all progress on an unrelated change until every known issue is fixed. The disclosed tradeoff: an allowlist can rot if entries are never revisited. Mitigated by requiring (per the file's own note) that an entry's removal happens in the same change as its fix.
- Threshold calibration is evidence-based, not guessed. `src/claimcontext/eval/calibration.py` measures actual score bands (off-corpus max 0.0036, clause-reference max 0.0773, weak-partial around 0.563, answerable minimum 0.5948) and reports a "safe band" the current `refuse_threshold=0.55` sits inside. This is real, printed output from a real run.

**Agent evaluation** (`tests/test_agent_eval.py`, `src/claimcontext/agent_eval/`):

- Golden set: `data/agent_eval/trajectory_golden_set_v1.jsonl`, 4 entries. Trajectory-based, not answer-based. Checks whether the agent routed correctly (single vs. multi-part decompose) and whether multi-part composition preserved per-claim citation provenance (`_cross_step_provenance_holds()`).
- Failure-mode classification is real code (`FailureMode` enum, `src/claimcontext/agent_eval/schema.py`), not just design intent. A wrong trajectory is tagged with why it's wrong.

**Honest limits of this evidence.**

- **Corpus size.** 30 documents total in `data/documents/`. A synthetic corpus built to exercise specific cases (cross-claim contamination, sparse notes, temporal edge case), not a representative sample of real claim volume or diversity.
- **Golden-set size.** 11 RAG entries and 4 agent-trajectory entries. Enough to catch specific, named regressions. Not enough for statistically meaningful score distributions. The scorecard output itself carries the caveat: "Scores are directional... treat movements <0.05 as noise."
- **No multi-turn evaluation.** `specs/CLAUDE.md` schedules session memory as `spec-9a` (LangGraph checkpointer, thread_id). Not built. Every eval entry is a single-turn query.
- **Adversarial coverage is narrow.** The one adversarial case tested is prompt-injection-via-retrieved-content awareness — the prompt explicitly instructs the model to treat source blocks as reference-only. There is no red-team suite, no fuzzing, no jailbreak corpus.
- **Judge bias is only partly addressed.** `KI-5` states that the same-family-vs-different-family judge bias delta was never empirically measured. The different-model-family choice is a design mitigation, not a validated one.

---

### 4. Failure & Recovery

**Case A — Generation hedged instead of reasoning about informative absence (KI-3).**

The system failed when a claim had only an FNOL on file, with no investigation notes (by deliberate corpus design). The correct answer was "no investigation has been conducted yet." Instead the model produced a generic hedge: "The available documents do not contain enough information to answer this question." Retrieval and the refuse gate both worked correctly. The FNOL scored 0.563, above threshold, and was the only — genuinely correct — evidence.

It was detected through the golden-set entry q06's `expected_behavior` mismatch during eval harness runs.

The fix was a change to the generation prompt (`prompts/rag_v2.txt`). The prompt is versioned per `CLAUDE.md` §2A.3 — `rag_v1.txt` is kept, not overwritten, so prior eval history stays attributable to the prompt version that produced it. The new rule instructs the model to state absence as fact when the context is sparse but genuinely complete.

Verified live: `ask()` now returns the correct, cited answer for q06. But the underlying judge-vs-behavior gap remains unresolved. RAGAS's Answer Relevance metric scored the corrected answer 0.00 even though the behavior is now correct. This is disclosed in `KNOWN_ISSUES.md` as an accepted evaluation-layer limitation, not silently ignored.

A second path is part of the same story: a two-step classify-then-generate architecture was considered as a deeper fix. A live reproduction test was run specifically to check whether it was still warranted after the prompt fix landed. It wasn't — no reproducible generation-side failure remained. Declining to build unneeded architecture is recorded in `KNOWN_ISSUES.md` alongside the fix.

**Case B — Cross-claim citation contamination (KI-1), and the still-open half of it (KI-10).**

The system failed when a query naming one specific claim (`CLM-1004`) returned a final answer citing a chunk from a different claim (`CLM-1003-estimate`). The refuse gate's threshold check had no concept of "does this evidence belong to the claim being asked about" — only "did it score high enough."

It was detected by re-tracing golden-set entry q08's actual evidence against `KNOWN_ISSUES.md`'s original (initially overclaiming) description. The re-trace surfaced that q08 was actually two distinct failures in one run, not one.

The fix was `_filter_same_claim()` in `retrieval/ask.py`. Every reranked chunk surviving to the threshold check must carry the same `claim_number` as the query names (or `None`, for claim-agnostic reference material like policies). Mismatches are dropped before `top_score` is computed.

Verified live: q08 now cites only `CLM-1004-*` chunks. A deliberate negative test also proves the filter has no effect on the other failure mode.

That other failure mode — KI-10 — is still open. A same-claim, wrong-subtopic chunk (`CLM-1004-fnol` at 0.86) can outscore the actually-correct chunk (`CLM-1004-notes` at 0.0001). It's filed as its own tracked issue rather than buried inside KI-1's write-up, because a same-claim filter structurally cannot catch a chunk that is from the same claim.

**Case C — An external call with no timeout hung for 9.5 hours (KI-9).**

The system failed during spec-7b's failure-injection testing. A `QdrantClient` construction with no explicit `timeout=` argument (`_claim_owner()` in `agent/routing.py`, and `collect_allowed_ids()` in `auth/entitlement.py`) blocked for 9.5 hours against an unreachable address instead of failing fast. A `tenacity` retry wrapper around the call was documented as "hardened," but a hang never raises an exception for `tenacity` to retry on. The retry machinery silently never engaged.

It was detected by the injection test itself (`test_observability.py`), not a live incident.

The fix was to add explicit `timeout=settings.qdrant_timeout_seconds` at both call sites. Verified live: the same address now returns in about 0.1s and degrades gracefully to `None`, its documented advisory-not-authoritative contract.

An honest residual gap is disclosed rather than hidden. Re-testing the exact original address after the fix returned a fast `ConnectionRefusedError`. So the original 9.5-hour hang's precise trigger was never cleanly reproduced. The fix closes two real, unconditional gaps — any unreachable address now has a time bound — regardless of whether the original symptom's exact mechanism is fully understood.

---

### 5. Operational Readiness & Reflection

**What would be needed before a real workload.** A live claims-system connector for Tier-3 fields. Load-tested concurrency limits. A CI pipeline. Recovery from a bad deploy would require re-running the pinned ingestion pipeline against a snapshotted corpus and rolling back a config value, not a data migration — because the vector index is fully derived, not primary state.

**Telemetry and auditability — what exists, verified in code.**

- **Access audit log.** `_audit()` in `retrieval/ask.py` logs `adjuster_id`, `region`, a SHA-256-truncated query hash, `chunks_retrieved`, and `decision` on every request. Raw query text is deliberately excluded from the general log. `_query_hash()`'s own docstring notes that a separate, access-controlled sink would be needed for investigations that require the raw query.
- **Distributed tracing.** Every model and vector-store call is wrapped in a Langfuse span (`observability/tracing.py`, `Tracer.span()`). It's designed fail-safe: a tracing exporter failure cannot break the request path. `Tracer.shutdown()` is only called at process shutdown (lifespan), never on the request path, because it's a blocking network call.
- **Health/readiness split.** `/health` in `api/app.py` is a pure liveness check. `/ready` actually calls `retriever.check_index_staleness()` and returns 503 with a reason string on failure. Models are warmed up eagerly at startup (`lifespan()`) specifically so "ready" and "actually fast" are the same claim, not two different ones.

**What's missing, honestly.**

- No metrics or dashboard layer. Langfuse gives trace-level detail, not aggregated latency or error-rate dashboards or alerting.
- No token-usage or cost tracking per request. Cost discipline today is "default to free local Ollama," not a measured per-request budget.
- No structured request-ID correlation across the audit log and the trace store beyond what Langfuse's own trace ID provides.

**Rollback and reproducibility — what's pinned, verified in code and config.**

- **Index versioning.** Every chunk carries `embedding_model` and `chunker_version` in its Tier-2 metadata (`config.py`: `embedding_model="BAAI/bge-large-en-v1.5"`, `chunker_version="v1"`). Per `CLAUDE.md` §2A.4, a mismatch at query time requires a full re-index. Stale vectors are treated as incompatible, not merely old. This was exercised for real during the KI-2 revert — `chunker_version` was bumped to `v2`, then reverted to `v1`, and the corpus was re-indexed both times.
- **Prompt versioning.** Prompts are files, not inline strings (`prompts/rag_v1.txt`, `prompts/rag_v2.txt`). They're selected via `_PROMPT_VERSION` in `ask.py`. Old versions are kept, not overwritten, so a prior eval run's score stays attributable to the exact prompt that produced it. Rolling back generation behavior is a one-line constant change, not a re-derivation from memory of what the old prompt said.
- **Dependency pinning.** `uv.lock` is authoritative. `DEPENDENCIES-LOCKED.md` documents the rationale and is machine-verified against it (verification date stamped 2026-07-29). A rollback of the dependency graph is a lockfile revert, not a re-research effort.
- **What is NOT pinned together as a single deployable unit.** There is no single manifest tying a given LLM model version, prompt version, `chunker_version`, and `refuse_threshold` to one deployable "release." These are independently configured values in `.env` and `config.py`. A rollback today means manually reverting several independent config values and confirming their combination was the one previously validated, not restoring one version tag. `KI-6` already documents one instance of this risk materializing: `.env` had `REFUSE_THRESHOLD=0.4` while calibration recommended `0.55`. An unsynced drift that happened to not matter (all discussed scores were far from both values) but could have.

**Operational next steps, in order of what would actually block a live workload first.**

1. **Live Tier-3 connector.** Tier-3 questions are refused outright today. A real deployment needs the documented-but-unbuilt claims-system-of-record connector before Tier-3 questions can be answered instead of only safely refused.
2. **Concurrency and load testing.** `KI-4`'s local-Ollama single-worker degradation is the only concurrency evidence that exists. There is no number for how the full pipeline behaves under realistic concurrent multi-adjuster load, and no autoscaling or queueing story beyond FastAPI's own async handling.
3. **CI pipeline.** There is no `.github/` workflow. `mypy`, `ruff`, and tests are run manually. `CLAUDE.md` names GitHub Actions → ECR → Fargate with an eval gate in the pipeline as the intended shape. None of it is built.
4. **Rate limiting and access-control hardening.** Entitlement (region and adjuster-scoped Qdrant filter, `auth/entitlement.py`) is real and enforced once identity is known. But `resolve_principal()` in `auth/resolver.py` is explicitly documented as a mock — a hardcoded lookup over two synthetic adjusters, keyed only by the `adjuster_id` string in the request. There is no JWT or session verification that the caller is that adjuster. The docstring itself states that production would verify a JWT or session token against an identity provider. That layer is not built. There is also no API-level rate limiting.
5. **PII redaction.** Explicitly scoped as "theatrical against synthetic data" in `CLAUDE.md` §6B. Real PII redaction tooling (e.g. Presidio) is named as the production path but not implemented.
