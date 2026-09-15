# Known Issues

Consolidated defect log for issues found during development but **deliberately
deferred** until the end-to-end app runs (post spec-9b). These are not silent gaps:
each one is enforced by `KNOWN_EVAL_EXCEPTIONS` in `tests/test_eval_harness.py`
(Proof 5b), which fails the eval gate on any *unexpected* regression while letting
these specific, understood entries fail without blocking. Full analysis and evidence:
`specs/spec-4-handoff.md`, "The refuse-gate architectural finding."

Deferred by decision, not oversight — fixing #1 in particular is an architecture
decision (retrieval vs. gate vs. output-faithfulness) that deserves its own
options-and-tradeoffs pass, not a reflexive patch mid-spec.

---

## Defects

| ID | Severity | Summary | Affects | Status |
|---|---|---|---|---|
| KI-1 | **High** | Refuse gate can open on the wrong evidence, producing a confident, cross-contaminated answer | q08 | Deferred |
| KI-2 | **High** | Terse adjuster-note shorthand is unfindable by the cross-encoder against natural-language questions | q02, q05, q06, q08 | Deferred |
| KI-3 | Medium | Generation hedges instead of reasoning about informative absence | q06 | Deferred |
| KI-4 | Low | RAGAS scoring concurrency breaks against local Ollama at full scale | — (infra) | Deferred |
| KI-5 | Low | Judge-bias delta (same-family vs. different-family score comparison) not measured | — (infra) | Deferred |
| KI-6 | Low | `refuse_threshold` config drift: `.env`=0.4, calibration recommends 0.55 | — (config) | Deferred |
| KI-7 | Low | Reasoning-model judge (`gpt-5-mini`) has recurring token-budget friction | — (infra) | Deferred |
| KI-8 | Medium | Cross-cutting: literal/substring matching is brittle against paraphrase — recurs at every layer that uses it | q02, q05, q06, q08 (KI-2 root cause) + spec-5a Tier-3 routing | Deferred |
| KI-9 | **High** | The network cannot be trusted to fail fast — every external client needs its own explicit timeout | `_claim_owner()` (routing.py), `EntitlementScope.collect_allowed_ids()` (entitlement.py) | **Fixed** (spec-7b) |

---

### KI-1 — Grounding gate gap: threshold-on-top-score misses wrong-subtopic context

**Root cause:** The refuse gate asks "does the top candidate clear
`refuse_threshold`" — not "does the retrieved set contain what's needed to answer
*this* question." Those are different axes (relevance-to-claim vs.
relevance-to-question) that a single similarity score collapses into one number. When
a same-claim, wrong-sub-topic chunk scores high enough to open the gate while the
actual answer-bearing chunk (buried by KI-2) never reaches the LLM, the system doesn't
refuse — it answers, ungrounded, sometimes blending in a **different claim's**
content entirely.

**Evidence (q08):** `CLM-1004-fnol` scored 0.86 (wrong sub-topic, opens gate);
`CLM-1004-notes` (contains the literal adjuster conclusion) scored 0.0001, ranked
12th/30, never reached `rerank_top_n=5`. Generated answer cited `CLM-1003-estimate` —
a different claim. RAGAS confirmed: `context_precision=0.0, context_recall=0.0,
answer_relevance=0.0`. Non-deterministic: a later run of the same question answered
correctly (LLM sampling variance on a borderline retrieval set) — the underlying gap
is unchanged, it just didn't trigger that run.

**Affects:** q08. (Structurally could affect any claim-scoped question where a
wrong-subtopic same-claim chunk outscores the correct one.)

**Candidate fix — open, none chosen:**
- Query-aware context-sufficiency check — not straightforward: the gate has no way to
  know at query time which chunk is "the" answer; that's golden-set knowledge.
- Same-claim/same-entity consistency check — would catch q08's cross-claim
  contamination specifically, but is a narrow heuristic (claim-scoped questions only).
- Output-faithfulness gating (verify citations support the answer, post-generation) —
  catches it downstream, costs a second LLM call.
- Fix retrieval at the root (see KI-2) — may resolve q08 without any gate change.

**Status:** Deferred to a dedicated grounding-robustness / retrieval-quality spec.

---

### KI-2 — Terse claim-note shorthand is unfindable by the cross-encoder

**Root cause:** `bge-reranker-base` (general-domain) scores terse, shorthand claim
notes (e.g. `"Coverage applies under Section I. No applicable exclusions
identified."`) near-zero against natural-language questions, even when the note is
the literal answer. This is the retrieval-side root cause behind three of the four
known exceptions.

**Evidence:**
- q02: top score 0.059 (`CLM-1003-estimate`); `CLM-1003-notes` (has the denial
  reasoning) doesn't make top-5.
- q05: top score 0.143 (`CLM-1001-letter`); `CLM-1001-notes` scores 0.035/0.008.
- q08: see KI-1 — `CLM-1004-notes` scores 0.0001, ranked 12th/30.
- q06: related but distinct manifestation — see KI-3; the *only* evidence that exists
  (an FNOL) scores 0.563 and correctly passes the gate, so KI-2 isn't the cause of
  q06's failure, generation is (KI-3).

**Failure modes from this one cause:**
- *Safe* (q02, q05): no chunk clears threshold → gate correctly refuses. Uninformative
  but not wrong.
- *Dangerous* (q08): a different, wrong-subtopic chunk clears threshold instead →
  see KI-1.

**Candidate fix:** Index `section + text` for dense embeddings (BM25 already does
this per spec-2b) to close the embedding-side asymmetry, and/or investigate whether a
different reranker or reranking strategy handles terse/shorthand text better. Same
cross-cutting seam flagged in spec-2c/spec-3 ("Section-Identifier Indexing Gap" in
`docs/BUILD-JOURNAL.md`).

**Status:** Deferred to the same grounding-robustness / retrieval-quality spec as KI-1.

---

### KI-3 — Generation doesn't reason about informative absence

**Root cause:** Given sparse-but-real context (an FNOL with no investigation notes,
by deliberate corpus design — CLM-1005 has none), the system hedges ("not enough
information") instead of stating the true, informative fact ("no investigation has
been conducted; only the initial FNOL is on file"). Retrieval and the gate both do
their job correctly here (0.563 > threshold, only the FNOL is passed through, and it's
genuinely the only evidence) — this is purely a generation/prompt limitation, not
related to KI-1/KI-2.

**Affects:** q06.

**Candidate fix:** Prompt/generation change to reason about absence-as-fact when
context is sparse but present. Relabel q06 from REFUSE to ANSWER only once both (a)
generation produces the informative-absence answer, and (b) `ground_truth_answer` is
written for q06.

**Do not** raise `refuse_threshold` above 0.563 to make q06 "pass" — that silently
relabels it via the gate instead of fixing generation (see `recommend_threshold()` in
`src/claimcontext/eval/schema.py`).

**Status:** Deferred to whichever future spec touches the answer-generation prompt.

---

### KI-8 — Cross-cutting: literal/substring matching is brittle against paraphrase

**Pattern, not a single bug.** The same underlying weakness — matching *literal
words* instead of *meaning* — has now surfaced independently at three different
layers of this system. Naming it once, here, so the next occurrence is recognized
as the same pattern rather than diagnosed from scratch again.

- **KI-2 (retrieval):** the cross-encoder scores terse claim-note shorthand near-zero
  against a natural-language question asking for the same fact in different words.
  "Coverage applies under Section I" vs. "what did the adjuster conclude" — same
  meaning, no shared vocabulary the model can lean on.
- **`ask.py`'s Tier-3 guard (`_is_tier3_query`):** pattern-matches literal phrases
  ("reserve amount", "claim status") against the raw query text. Documented in the
  code itself as "leaky... may slip through" for paraphrase variation — a known,
  accepted weakness at the time it was written (spec-2c/spec-4).
- **spec-5a (routing → generation composition):** the agent's decompose step
  paraphrases a query into sub-questions before they reach `ask()`. When a Tier-3
  query gets paraphrased in decomposition, the literal words `_is_tier3_query()`
  depends on can vanish in the rewrite — the guard doesn't fire, and the query falls
  through to a generic refusal instead of the correct, informative Tier-3 refusal.
  Reproduced deterministically (6/6) during spec-5a proof-writing before the router's
  multi-part classifier was fixed to stop mis-routing that query in the first place.

**Why this matters as a named pattern, not three isolated bugs:** every new layer
added to this system that does its own literal/substring matching against
user-phrased text inherits this same fragility by default — and, per the spec-5a
finding, layers can now *compound* it (a paraphrasing layer sitting in front of a
literal-matching layer breaks the literal-matching layer in a new way neither one's
own tests would catch in isolation). Any future guard, classifier, or router built on
substring/keyword matching should be evaluated against paraphrase variation
specifically, not just against the exact phrasings its author happened to type while
building it.

**Candidate fix — open, none chosen (same posture as KI-1/KI-2):** semantic/embedding-
based classification instead of substring matching for guards like `_is_tier3_query`;
keeping literal-sensitive guards upstream of any paraphrasing step (i.e., classify
Tier-3 on the *original* query before decomposition ever touches it, not after);
or accepting the current mitigations (spec-5a's few-shot-tuned router now avoids
mis-routing the specific Tier-3 case that triggered this) as good enough pending a
real incident.

**Status:** Deferred — noted as a cross-cutting design consideration for whichever
future spec next builds a new classifier/guard, not owned by a specific fix spec.

---

### KI-9 — The network cannot be trusted to fail fast; every external client needs its own explicit timeout

**Root cause:** Two `QdrantClient` constructions had no `timeout=` argument at all —
`_claim_owner()` in `agent/routing.py` (the router's entitlement pre-filter lookup)
and `EntitlementScope.collect_allowed_ids()` in `auth/entitlement.py` (the sparse-side
entitlement filter, called from every principal-scoped `ask()`). Every other
`QdrantClient` in the codebase (`retriever.py`, `sparse.py`, `metadata_filter.py`,
`qdrant_writer.py`) already sets `timeout=settings.qdrant_timeout_seconds` —
these two were the exceptions, not the rule.

**Evidence:** Discovered live during spec-7b's failure-injection testing — a test
pointed at an unreachable Qdrant address (`http://localhost:1`, expecting a fast
refuse, the same assumption spec-6's environmental proof and spec-7a's staleness
proofs made elsewhere in this project) and the connection blocked for **9.5 hours**
instead of failing within seconds. `_claim_owner()`'s own docstring explicitly
claimed hardening ("the already-hardened `_ask_with_retry` path... retry on
specific transient failure types") — but a `tenacity` retry loop can only retry
an exception that's actually raised. A hung connection with no timeout never
raises anything for `tenacity` to catch; the first attempt just blocks forever,
and the retry machinery around it never fires. The hardening claim was real for
*retryable failures*, but silently didn't cover *hangs*.

**Honest caveat:** re-testing the exact fix against the exact same address
(`http://localhost:1`) after adding explicit timeouts returned a fast
`ConnectionRefusedError` in ~0.1s — meaning the original 9.5-hour hang's precise
trigger was not cleanly reproduced, and the exact mechanism remains undiagnosed
(environment/OS-state-dependent, possibly transient). This does **not** weaken the
finding: the audit that followed from taking the hang seriously found two real,
unconditional gaps (no timeout means no bound, regardless of whether this specific
address hangs or refuses on any given day) that are fixed now as defense-in-depth,
not contingent on reproducing the original symptom.

**Fix (spec-7b):** both call sites now take an explicit `timeout` — `_claim_owner()`
passes `settings.qdrant_timeout_seconds`; `collect_allowed_ids()` gained a `timeout`
parameter, threaded from `ask.py`'s call site as `settings.qdrant_timeout_seconds`.
Verified live: `_claim_owner()` against the address that originally hung now
returns in ~0.1s and degrades to `None` (its existing "advisory, not authoritative"
contract), never hanging.

**The generalizable lesson, worth carrying into every future spec that adds a new
external call:** an explicit client-side timeout is not optional hardening to add
"if there's time" — it is the *only* thing standing between an unreachable
dependency and an indefinitely hung request. The OS/network stack is not a
reliable fast-fail backstop. Any new external client (a future spec-9b cache
backend, spec-8b's Fargate health checks, a real Tier-3 connector) needs this
audited at construction time, not discovered via a 9.5-hour surprise.

**Status:** Fixed (the two identified gaps). Listed here rather than only in a
spec handoff because the lesson is cross-cutting, matching KI-8's convention.

---

### Minor / tracked

| ID | Issue | Detail |
|---|---|---|
| KI-4 | RAGAS scoring concurrency breaks against local Ollama at scale | `asyncio.gather` fires all metrics × entries with no cap. Fine against a hosted API judge; against local Ollama (single-worker), later-queued requests time out before being served. Reproduced with 8 ANSWER entries (~72 concurrent requests). Fix: cap concurrency (semaphore or sequential) for `provider == "ollama"`. **Cross-cutting pattern, not isolated to eval:** the same root cause (local Ollama's single-worker request queue degrading under sustained back-to-back load, while working fine in isolation) has now recurred in spec-7b's `test_observability.py` — the combined 6-test file intermittently stalls on the 4th test after three prior tests' worth of LLM calls, though every test passes reliably alone. Naming it once, here: any future spec adding a sustained-load test/eval against local Ollama should assume this constraint applies and cap concurrency or run serially with cooldown, not rediscover it per-spec. Spec-8's move to a hosted LLM for cloud deploy removes this constraint entirely for production traffic — it only affects local dev/test. |
| KI-5 | Judge-bias delta not measured | Design decision (different model family than the answer LLM) made, not empirically validated with a same-family-vs-different-family score comparison. Attempted once, blocked by KI-4, not completed. |
| KI-6 | `refuse_threshold` config drift | `.env` has `REFUSE_THRESHOLD=0.4`; calibration (Proof 4) recommends `0.55`. Doesn't change any current conclusion (all discussed scores are far below or comfortably above both values) but should be synced. |
| KI-7 | Reasoning-model judge token-budget friction | `gpt-5-mini` needed `max_tokens=4096` (default 1024 caused `IncompleteOutputException` on every call — reasoning tokens draw from the same budget as visible output). Even at 4096, one call still hit the limit. Accumulating evidence a non-reasoning judge model would be cleaner long-term. |

---

## How this file is enforced

`tests/test_eval_harness.py::KNOWN_EVAL_EXCEPTIONS = {"q02", "q05", "q06", "q08"}` —
Proof 5b runs the live pipeline and RAGAS scoring, then asserts *no unexpected*
entry failures beyond this set. A new regression (anything not listed here) still
fails the proof. When an issue above is fixed, remove its entry ID from
`KNOWN_EVAL_EXCEPTIONS` and this file in the same change — an entry passing while
still listed here as deferred is a stale doc, not a passing test.
