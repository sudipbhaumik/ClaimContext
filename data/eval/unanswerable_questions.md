# Unanswerable Questions

Questions the synthetic corpus genuinely cannot answer — used to prove the
refuse-on-weak-context gate (spec-2c). The system must refuse, not hallucinate.

---

1. **What is the total amount paid out on claim CLM-1001?**
   *Cannot answer: payment disbursement records are not in any document.
   The estimate exists, but actual payment confirmation is absent.*

2. **Has policyholder Margaret Chen filed any prior claims in the past five years?**
   *Cannot answer: claims history is not present in the corpus.
   Only the current claims (CLM-1001, CLM-1002) are represented.*

3. **Who was the at-fault driver in the CLM-1002 accident, and what is their insurer?**
   *Cannot answer: the other party fled the scene and their details are not documented
   in any FNOL, notes, or correspondence in the corpus.*

4. **What is the adjuster ADJ-027's caseload and average resolution time?**
   *Cannot answer: adjuster workload metrics are not in any document.
   Only claim-specific notes for ADJ-027's assigned claims are present.*

5. **Was a public adjuster or attorney retained by the insured for CLM-1003?**
   *Cannot answer: the denial letter and claim notes make no mention of
   legal representation. The corpus contains no such record.*
