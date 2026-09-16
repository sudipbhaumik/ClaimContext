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
| KI-1 | **High** | Cross-claim citation contamination — a different claim's chunk reaching the final answer | q08 (contamination half) | **Fixed** (`spec-grounding-robustness`) |
| KI-2 | **High** | Terse adjuster-note shorthand is unfindable by the cross-encoder against natural-language questions | q02, q05, q06, q08, `test_proof4_composition_preserves_claim_provenance` (xfail) | **Parked** — fix attempted, ineffective AND regressed policy retrieval, fully reverted; real fix needs ingestion changes |
| KI-3 | Medium | Generation hedges instead of reasoning about informative absence | q06 | Deferred |
| KI-4 | Low | RAGAS scoring concurrency breaks against local Ollama at full scale | — (infra) | Deferred |
| KI-5 | Low | Judge-bias delta (same-family vs. different-family score comparison) not measured | — (infra) | Deferred |
| KI-6 | Low | `refuse_threshold` config drift: `.env`=0.4, calibration recommends 0.55 | — (config) | Deferred |
| KI-7 | Low | Reasoning-model judge (`gpt-5-mini`) has recurring token-budget friction | — (infra) | Deferred |
| KI-8 | Medium | Cross-cutting: literal/substring matching is brittle against paraphrase — recurs at every layer that uses it | q02, q05, q06, q08 (KI-2 root cause) + spec-5a Tier-3 routing | Deferred |
| KI-9 | **High** | The network cannot be trusted to fail fast — every external client needs its own explicit timeout | `_claim_owner()` (routing.py), `EntitlementScope.collect_allowed_ids()` (entitlement.py) | **Fixed** (spec-7b) |
| KI-10 | **High** | Refuse gate can open on a same-claim, wrong-subtopic chunk — the gate-opening half of KI-1, not fixed by KI-1's same-claim filter | q08 | Deferred |

---

### KI-1 — Grounding gate gap: cross-claim citation contamination

**Scope corrected during `spec-grounding-robustness`.** q08 was originally
described here as one failure ("threshold-on-top-score misses wrong-subtopic
context... sometimes blending in a different claim's content"). Re-tracing
the actual evidence against the golden-set entry showed q08 exhibits **two
distinct failures** in the same run, not one — a same-claim, wrong-subtopic
gate-opening failure, and a separate cross-claim citation failure. They need
different fixes. The gate-opening failure is now tracked separately as
**KI-10** (structurally cannot be caught by a same-claim filter). This entry
is narrowed to the cross-claim citation failure only, which is what the
same-claim/same-entity consistency check built in `spec-grounding-robustness`
actually closes.

**Root cause:** a chunk from a *different* claim than the one the query names
can still reach `rerank_top_n` and get cited in the final answer — the refuse
gate's threshold check has no concept of "does this evidence belong to the
claim being asked about," only "did it score high enough."

**Evidence (q08):** the generated answer cited `CLM-1003-estimate` — a
different claim than `CLM-1004`, the claim q08's question names — despite
`CLM-1003-estimate` never being part of the correct evidence set. (The
*other* half of q08's original evidence — `CLM-1004-fnol` wrongly outscoring
`CLM-1004-notes` — is KI-10, not this issue.)

**Affects:** q08 (citation-contamination half).

**Fix (`spec-grounding-robustness`):** `_filter_same_claim()` in
`retrieval/ask.py` — a same-claim/same-entity consistency check. When a
query names a specific claim, every chunk surviving rerank must carry that
same `claim_number` (or `None`, for claim-agnostic reference material like
policies/endorsements) before the threshold check; a mismatch is dropped,
and if that empties the candidate set, the query refuses. `top_score` for
the refuse-threshold check is computed from the filtered list, never the
original reranked list, so a since-dropped chunk's score can't wrongly gate
the decision. **This fix does not address KI-10** — verified directly: run
against a synthetic same-claim-wrong-subtopic case, the filter passes both
chunks through unchanged, exactly as expected.

**Verified live:** the flagship q08 query (`ask()`, `ADJ-027`,
"What did the adjuster conclude about coverage for CLM-1004...") returned
citations `['CLM-1004-fnol', 'CLM-1004-estimate', 'CLM-1004-letter',
'POL-5504-policy']` — no `CLM-1003` chunk, the exact contamination this
fix targets. Unit-level proof (deterministic, not dependent on LLM/retrieval
variance): cross-claim chunk dropped, same-claim and `claim_number=None`
chunks kept, no-claim-named queries pass through as a no-op.

**Status:** Fixed. See KI-10 for the separate, still-open gate-opening case
this fix does not address.

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

**Candidate fix #1 — tried, empirically confirmed ineffective:** index
`section + text` for dense embeddings, mirroring `reranker.py`'s own
`f"{section} {text}".strip() if section else text` concatenation exactly
(implemented in `spec-grounding-robustness`, `ingestion/pipeline.py`).
**Live re-measurement after a full reindex showed zero score change** —
q02/q05/q08's top-candidate scores were byte-identical to the pre-fix
numbers above. Root cause: `chunker.py` sets `section=""` unconditionally
for every `claim_note` chunk (`chunker.py` line ~303 — notes are chunked via
`_chunk_notes()`, which never runs heading detection at all). Since every
piece of KI-2's evidence (`CLM-1003-notes`, `CLM-1001-notes`,
`CLM-1004-notes`) *is* a claim-note chunk, the fix's `if section else text`
branch always fell through to plain text — for exactly the chunks it was
meant to fix. The reranker has this identical blind spot (it was the format
being mirrored), and BM25's earlier "already does this" claim doesn't hold
up either — BM25 finds notes (when it does) via literal term overlap in the
note text itself, not because of a section-label anchor.

**The code change was initially kept ("does no harm") — that claim was
wrong, and the fix was fully reverted once measured.** Re-embedding the
full corpus both ways and comparing cosine scores directly showed the
section-prefix change actively regressed an unrelated query:
`POL-3301-policy`'s best rank for "what perils are covered under policy
POL-3301?" moved from **7th** (inside `top_k=10`, `test_proof1_coverage_
question_returns_relevant_chunks` passes) to **12th** (outside `top_k`,
test fails) — confirmed via the same live re-embedding methodology, not
theorized. The section-label prefix pulled *other* policy chunks' embeddings
closer to unrelated queries, displacing this one. No benefit to KI-2's own
evidence (confirmed above) plus a measured cost elsewhere is not a
change worth keeping under any framing. `ingestion/pipeline.py` reverted to
plain `c.text` embedding; `chunker_version` reverted to `v1`; corpus
re-reindexed and the regression confirmed gone (`POL-3301-policy` back to
7th, both previously-failing tests pass again).

**Candidate fix #2 — not attempted, requires ingestion pipeline changes,
parked:** claim-note chunks need *some* anchor other than a section heading
that doesn't exist for them — e.g. synthesizing a label from `doc_type` +
the note's own opening context, or restructuring `_chunk_notes()` to derive
a pseudo-section from note structure (note entries often have their own
internal markers, e.g. `[NOTE-1003-01] 2026-03-05 — ADJ-027`, which could
seed a label). This is real ingestion-pipeline design work, not a small
patch — parked rather than attempted under this spec's original scope.

**Status:** Parked, fully reverted (nothing from fix #1 remains in the
codebase). Fix #2 is scoped but not started, deferred to a future
ingestion-focused spec (not yet assigned a name/number).

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

### KI-10 — Refuse gate can open on a same-claim, wrong-subtopic chunk

**Root cause:** identical mechanism to KI-1's original description, but this
is the half of it that a same-claim consistency check structurally cannot
catch. `CLM-1004-fnol` (wrong subtopic, but genuinely part of claim
`CLM-1004` — the same claim the query names) scored 0.86 and cleared
`refuse_threshold`, while `CLM-1004-notes` (the correct evidence, also
`CLM-1004`) scored 0.0001 and never reached `rerank_top_n`. A filter that
only checks "does this chunk belong to the claim named in the query" passes
`CLM-1004-fnol` straight through — it *is* that claim, just the wrong section
of it.

**Split from KI-1 during `spec-grounding-robustness`:** KI-1 originally
described q08 as citing "a different claim's content," which is true, but
q08 actually exhibits two distinct failures in the same run — a same-claim
gate-opening failure (this issue) and a separate cross-claim citation
failure (KI-1, as scoped after the split). KI-1's same-claim filter, built in
`spec-grounding-robustness`, closes the cross-claim case. It does not, and
structurally cannot, close this one. Filed as its own ID rather than left as
a note inside a spec's handoff specifically so it stays visible in
`KNOWN_ISSUES.md` after that spec closes and its handoff is no longer the
active planning surface.

**Affects:** q08 (gate-opening half).

**Candidate fix — open, none chosen, carried forward from KI-1's original
list:**
- Query-aware context-sufficiency check — the gate has no way to know at
  query time which chunk is "the" answer; that's golden-set knowledge.
- Output-faithfulness gating (verify citations support the answer, post-
  generation) — catches it downstream, costs a second LLM call.
- `spec-grounding-robustness`'s KI-2 fix (embedding `section + text` for
  dense retrieval) may reduce how often this triggers by making
  `CLM-1004-notes` easier to find in the first place, but does not
  structurally prevent a wrong-subtopic same-claim chunk from ever
  outscoring the correct one again — measured, not assumed, in that spec's
  proofs.

**Status:** Deferred. Not owned by a specific future spec yet — surface at
the next planning pass.

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
