"""
Clause library — committed lists of natural prose fragments.

The generator samples from these with a seeded RNG. No LLM is called
at generation time; variety comes from recombination, not inference.
"""

# ── Auto coverage clauses ─────────────────────────────────────────────────────

AUTO_COVERAGE_CLAUSES: list[str] = [
    (
        "COVERAGE A — BODILY INJURY LIABILITY\n"
        "We will pay damages for bodily injury for which any insured becomes legally responsible "
        "because of an auto accident. We will settle or defend, as we consider appropriate, any "
        "claim or suit asking for these damages. Our duty to settle or defend ends when our limit "
        "of liability for this coverage has been exhausted by payment of judgments or settlements."
    ),
    (
        "COVERAGE B — PROPERTY DAMAGE LIABILITY\n"
        "We will pay damages for property damage for which any insured becomes legally responsible "
        "because of an auto accident. We will settle or defend any claim or suit asking for these "
        "damages. Defense costs are paid in addition to and do not reduce the limits of liability."
    ),
    (
        "COVERAGE C — COLLISION\n"
        "We will pay for direct and accidental loss to your covered auto or a non-owned auto, "
        "including its equipment, minus any applicable deductible, caused by collision. "
        "'Collision' means the upset of your covered auto or a non-owned auto, or their impact "
        "with another vehicle or object."
    ),
    (
        "COVERAGE D — COMPREHENSIVE (OTHER THAN COLLISION)\n"
        "We will pay for direct and accidental loss to your covered auto or a non-owned auto, "
        "including its equipment, minus any applicable deductible, caused by other than collision. "
        "This includes but is not limited to: fire, theft or larceny, explosion or earthquake, "
        "windstorm, hail, water or flood, malicious mischief or vandalism, riot or civil commotion, "
        "or contact with a bird or animal."
    ),
    (
        "COVERAGE E — UNINSURED MOTORISTS\n"
        "We will pay compensatory damages which an insured is legally entitled to recover from the "
        "owner or operator of an uninsured motor vehicle because of bodily injury sustained by an "
        "insured and caused by an accident. The owner's or operator's liability for these damages "
        "must arise out of the ownership, maintenance, or use of the uninsured motor vehicle."
    ),
    (
        "COVERAGE F — MEDICAL PAYMENTS\n"
        "We will pay reasonable expenses incurred for necessary medical and funeral services because "
        "of bodily injury caused by accident and sustained by an insured. Expenses must be incurred "
        "within three years from the date of the accident."
    ),
]

AUTO_EXCLUSION_CLAUSES: list[tuple[str, str]] = [
    (
        "EXCLUSION 2.1 — INTENTIONAL ACTS",
        "We do not provide coverage for any insured who intentionally causes bodily injury or "
        "property damage. This exclusion applies even if the insured is unable to appreciate the "
        "nature or consequences of such acts.",
    ),
    (
        "EXCLUSION 2.2 — RACING",
        "We do not provide coverage for any insured for bodily injury or property damage arising "
        "from the use of a vehicle as a conveyance in, or in practice or preparation for, any "
        "prearranged or organized racing, speed, or demolition contest.",
    ),
    (
        "EXCLUSION 2.3 — COMMERCIAL USE",
        "We do not provide Liability Coverage for any vehicle while being used as a public or "
        "livery conveyance. This exclusion does not apply to a share-the-expense car pool.",
    ),
    (
        "EXCLUSION 2.4 — NUCLEAR",
        "We do not provide coverage for loss caused directly or indirectly by nuclear reaction, "
        "nuclear radiation, or radioactive contamination, all whether controlled or uncontrolled "
        "or however caused, or any consequence of any of these.",
    ),
    (
        "EXCLUSION 2.5 — WAR",
        "We do not provide coverage for loss caused by or resulting from war, invasion, acts of "
        "foreign enemies, hostilities, civil war, rebellion, insurrection, military power, "
        "confiscation, nationalization, destruction by government or public authority.",
    ),
]

# ── Property coverage clauses ─────────────────────────────────────────────────

PROPERTY_COVERAGE_CLAUSES: list[str] = [
    (
        "SECTION I — PROPERTY COVERAGES\n"
        "COVERAGE A — DWELLING\n"
        "We insure for direct physical loss to the property described in the Declarations under "
        "Coverage A unless the loss is excluded under Section I — Perils Insured Against."
    ),
    (
        "COVERAGE B — OTHER STRUCTURES\n"
        "We cover other structures on the residence premises set apart from the dwelling by clear "
        "space. This includes structures connected to the dwelling by only a fence, utility line, "
        "or similar connection. The limit of liability for this coverage will not be more than "
        "10% of the limit of liability that applies to Coverage A."
    ),
    (
        "COVERAGE C — PERSONAL PROPERTY\n"
        "We cover personal property owned or used by an insured while it is anywhere in the world. "
        "At your request, we will cover personal property owned by others while the property is "
        "on the part of the residence premises occupied by an insured. Our limit of liability for "
        "personal property usually located at an insured's residence, other than the residence "
        "premises, is 10% of the limit of liability for Coverage C."
    ),
    (
        "COVERAGE D — LOSS OF USE\n"
        "The limit of liability for Coverage D is the total limit for the coverages in a. and b. "
        "below. a. Additional Living Expense — If a loss covered under Section I makes that part "
        "of the residence premises where you reside not fit to live in, we cover any necessary "
        "increase in living expenses incurred by you so that your household can maintain its normal "
        "standard of living. Payment is for the shortest time required to repair or replace the "
        "damage or, if you permanently relocate, the shortest time required for your household to "
        "settle elsewhere."
    ),
]

PROPERTY_EXCLUSION_CLAUSES: list[tuple[str, str]] = [
    (
        "EXCLUSION 2.1 — FLOOD",
        "We do not insure for loss caused directly or indirectly by flood, surface water, waves, "
        "tidal water, overflow of a body of water, or spray from any of these, whether or not "
        "driven by wind. This exclusion applies regardless of any other cause or event that "
        "contributes concurrently or in any sequence to the loss. Flood damage is available "
        "through the National Flood Insurance Program (NFIP).",
    ),
    (
        "EXCLUSION 2.2 — EARTH MOVEMENT",
        "We do not insure for loss caused by earth movement, including but not limited to: "
        "earthquake, landslide, mine subsidence, mudflow, earth sinking, rising, or shifting. "
        "Direct loss by fire, explosion, theft, or breakage of glass resulting from earth movement "
        "is covered.",
    ),
    (
        "EXCLUSION 2.3 — WIND-DRIVEN RAIN",
        "We do not insure for loss caused by or resulting from wind-driven rain, meaning rain "
        "that enters the dwelling through openings caused by or resulting from wind or hail. "
        "This exclusion applies unless the dwelling structure has first sustained direct physical "
        "loss caused by a covered windstorm peril that creates the opening through which the "
        "rain enters. See Endorsement WR-001 if attached — that endorsement modifies this exclusion.",
    ),
    (
        "EXCLUSION 2.4 — NEGLECT",
        "We do not insure for loss caused by neglect of an insured to use all reasonable means "
        "to save and preserve property at and after the time of a loss.",
    ),
    (
        "EXCLUSION 2.5 — INTENTIONAL LOSS",
        "We do not insure for loss arising out of any act committed by or at the direction of an "
        "insured with the intent to cause a loss. This exclusion applies to all insureds even if "
        "only one insured commits the act.",
    ),
    (
        "EXCLUSION 2.6 — ORDINANCE OR LAW",
        "We do not insure for loss caused by enforcement of any ordinance or law regulating the "
        "construction, repair, or demolition of a building or other structure, unless specifically "
        "provided under this policy.",
    ),
]

# ── Endorsement language ──────────────────────────────────────────────────────

ENDORSEMENT_WR001_BODY = """\
ENDORSEMENT WR-001 — WIND-DRIVEN RAIN COVERAGE EXTENSION

This endorsement modifies insurance provided under:
  HOMEOWNERS PROPERTY INSURANCE POLICY

Effective Date of Endorsement: {endorsement_date}
Policy Number: {policy_number}
Policyholder: {policyholder_name}

MODIFICATION TO SECTION I — EXCLUSIONS
EXCLUSION 2.3 — WIND-DRIVEN RAIN

The exclusion for wind-driven rain contained in Exclusion 2.3 of the base policy is hereby
DELETED and REPLACED in its entirety with the following:

EXCLUSION 2.3 (AS AMENDED) — WIND-DRIVEN RAIN
We do not insure for loss caused by or resulting from wind-driven rain EXCEPT where all of
the following conditions are met:
  (a) A windstorm of sufficient force to constitute a covered peril under this policy
      strikes the insured premises;
  (b) The windstorm creates an opening in the roof, walls, or other structural element
      of the dwelling through which rain subsequently enters; and
  (c) The resulting interior damage is a direct consequence of rain entry through that
      opening.

Where these conditions are satisfied, interior damage caused by wind-driven rain entering
through a storm-created opening IS COVERED under the terms of this policy, subject to
the applicable dwelling limit and deductible set forth in the Declarations.

This endorsement does not extend coverage to rain entry through pre-existing openings,
gaps, cracks, or openings resulting from wear, deterioration, or inadequate maintenance.

All other terms and conditions of the policy remain unchanged.

Authorized Representative: _________________________
Date: {endorsement_date}
"""

# ── Condition clauses ─────────────────────────────────────────────────────────

CONDITION_CLAUSES: list[str] = [
    (
        "DUTIES AFTER AN OCCURRENCE\n"
        "In the event of an occurrence, you must: (a) give prompt notice to us or our agent; "
        "(b) promptly forward to us every notice, demand, summons, or other process relating to "
        "the occurrence; (c) cooperate with us in the investigation, settlement, or defense of "
        "any claim or suit; (d) submit to examination under oath by any person named by us, as "
        "often as we reasonably require; (e) submit a signed, sworn proof of loss within 60 days "
        "after our request."
    ),
    (
        "APPRAISAL\n"
        "If you and we do not agree on the amount of loss, either may demand an appraisal of the "
        "loss. In this event, each party will select a competent and impartial appraiser within "
        "20 days after receiving a written request from the other. The two appraisers will select "
        "an umpire. If they cannot agree upon an umpire within 15 days, you or we may request "
        "that the choice be made by a judge of a court of record in the state in which the "
        "residence premises is located."
    ),
    (
        "SUIT AGAINST US\n"
        "No action can be brought against us unless there has been full compliance with all of "
        "the terms under this Section and the action is started within one year after the date "
        "of loss."
    ),
    (
        "LOSS PAYMENT\n"
        "We will adjust all losses with you. We will pay you unless some other person is named "
        "in the policy or is legally entitled to receive payment. Loss will be payable 60 days "
        "after we receive your proof of loss and: (a) reach an agreement with you; (b) there is "
        "an entry of a final judgment; or (c) there is a filing of an appraisal award with us."
    ),
    (
        "SUBROGATION\n"
        "An insured may waive in writing before a loss all rights of recovery against any person. "
        "If not waived, we may require an assignment of rights of recovery for a loss to the "
        "extent that payment is made by us. If an assignment is sought, an insured must sign and "
        "deliver all related papers and cooperate with us."
    ),
]

# ── FNOL text fragments ───────────────────────────────────────────────────────

FNOL_LOSS_DESCRIPTIONS_AUTO: list[str] = [
    "The insured reports that the covered vehicle was struck from behind while stopped at a red light. "
    "The at-fault driver was identified and exchanged insurance information at the scene. "
    "No injuries were reported. The insured drove the vehicle home and states it is operable but has "
    "visible rear-end damage.",
    "The insured reports a sideswipe collision while merging onto the freeway. The other vehicle did not "
    "stop. The insured was unable to obtain the other party's information. A police report was filed. "
    "The vehicle sustained damage to the driver-side doors and rear quarter panel.",
    "The insured reports a single-vehicle collision with a highway guardrail during adverse weather "
    "conditions. No other vehicles were involved. Emergency services were dispatched but the insured "
    "declined medical transport. Vehicle is not driveable and was towed from the scene.",
]

FNOL_LOSS_DESCRIPTIONS_PROPERTY: list[str] = [
    "The insured reports discovery of standing water in the basement following heavy rainfall. "
    "The source of entry is believed to be through window wells and foundation cracks. "
    "Personal property and flooring are affected. Insured has engaged a water mitigation service "
    "and is awaiting an inspection appointment.",
    "The insured reports wind and rain damage following a severe storm event. The roof sustained "
    "visible damage and rain entered the dwelling through the compromised area, affecting the "
    "master bedroom ceiling and upper hallway. Insured has placed temporary tarping over the "
    "affected roof section.",
    "The insured reports a fire originating in the kitchen that spread to the adjacent dining area. "
    "The fire department responded and the fire was contained. The insured and family are "
    "temporarily displaced. Damage assessment is pending.",
]

# ── Adjuster note phrasings ───────────────────────────────────────────────────

NOTE_PHRASINGS_FNOL: list[str] = [
    "FNOL received {date}. {loss_desc} Insured reached by phone; loss reported within policy period.",
    "Initial contact made with insured. DOL {loss_date}; reported {report_date}. "
    "Assigned to my desk for investigation. Reserve set pending inspection.",
    "FNOL processed. Confirmed coverage active at time of loss. Referred to field team for inspection.",
]

NOTE_PHRASINGS_INSPECTION: list[str] = [
    "Field inspection completed {date}. Damage consistent with reported cause. "
    "Photos taken; estimate to follow.",
    "Inspected {date} with contracted estimator. Confirmed mechanism of loss. "
    "Preliminary damage estimate in process.",
    "Contacted insured to schedule insp. Insp completed {date}. "
    "Damage documented; no discrepancies noted with FNOL description.",
    "Insp. completed. Damage to {area} confirmed. Estimator on site. Report expected {eta}.",
]

NOTE_PHRASINGS_COVERAGE_REVIEW: list[str] = [
    "Coverage review completed. Policy active at DOL. No exclusions implicated. Proceeding to estimate review.",
    "Policy reviewed. Confirmed {policy_number} was in force on {loss_date}. "
    "Coverage applies under Section I. No applicable exclusions identified.",
    "Coverage analysis complete. Exclusion 2.3 reviewed — {exclusion_finding}.",
    "Reviewed policy {policy_number}. Coverage determination: {outcome}. Basis: {basis}.",
]

NOTE_PHRASINGS_ESTIMATE: list[str] = [
    "Estimate received from {estimator}. Total: ${amount:,.2f}. "
    "Reviewed for reasonableness — within expected range for this type of loss.",
    "Estimate reviewed and approved. Authorized ${amount:,.2f} for repairs. "
    "Insured notified of approval.",
    "Repair estimate reviewed. Line items cross-checked against labor/parts schedules. "
    "Minor adjustment applied. Final authorized amount: ${amount:,.2f}.",
]

NOTE_PHRASINGS_DECISION: list[str] = [
    "Claim resolved. Payment issued to insured for authorized repairs. File closed pending receipt.",
    "Coverage determination issued. {outcome}. Insured notified by letter dated {date}. "
    "File remains open pending appeal period.",
    "Denial letter sent {date}. Reason: {reason}. Insured advised of rights and appeal process.",
    "Settlement reached. Insured accepted ${amount:,.2f}. Release signed. File closed.",
]

# ── Damage line items ─────────────────────────────────────────────────────────

LINE_ITEMS_AUTO: list[tuple[str, float]] = [
    ("Rear bumper assembly, OEM, painted to match", 485.00),
    ("Rear bumper reinforcement bar", 215.00),
    ("Trunk lid assembly, OEM", 890.00),
    ("Trunk lid latch and striker", 78.50),
    ("Rear tail light assembly — left", 245.00),
    ("Rear tail light assembly — right", 245.00),
    ("Rear quarter panel — right, R&R", 1_250.00),
    ("Driver-side door shell, OEM", 760.00),
    ("Driver-side door glass, tempered", 195.00),
    ("Door trim panel — driver, replacement", 165.00),
    ("Paint and refinishing — 2 panels, blending", 680.00),
    ("Sensor — rear parking assist", 140.00),
    ("Labor — structural alignment (frame check)", 325.00),
    ("Labor — mechanical R&R, miscellaneous", 210.00),
    ("Rental vehicle — 7 days @ $45/day", 315.00),
]

LINE_ITEMS_PROPERTY: list[tuple[str, float]] = [
    ("Roof shingles — tear off and replace, 24 sq.", 4_200.00),
    ("Roof decking — replace damaged sections, 8 sheets", 640.00),
    ("Ice and water barrier, 200 sq. ft.", 310.00),
    ("Ridge cap shingles", 185.00),
    ("Gutter — aluminum, 60 LF, replace", 420.00),
    ("Ceiling drywall — master bedroom, 120 sq. ft.", 510.00),
    ("Ceiling drywall — upper hallway, 80 sq. ft.", 340.00),
    ("Insulation — blown-in replacement, 200 sq. ft.", 290.00),
    ("Interior paint — 2 rooms, walls and ceiling", 780.00),
    ("Hardwood floor — water-stained section, 40 sq. ft.", 680.00),
    ("Temporary tarping — emergency weatherproofing", 350.00),
    ("Debris removal and cleanup", 275.00),
    ("Labor — carpentry, miscellaneous", 460.00),
    ("Permit — roofing, county fee", 125.00),
]

LINE_ITEMS_PROPERTY_FLOOD: list[tuple[str, float]] = [
    ("Water extraction — basement, 1,200 sq. ft.", 1_800.00),
    ("Moisture mapping and air quality assessment", 450.00),
    ("Dehumidifier — commercial, 5-day rental", 600.00),
    ("Drywall removal — water-damaged sections, 320 sq. ft.", 960.00),
    ("Insulation removal and disposal — basement walls", 540.00),
    ("Flooring — vinyl plank, remove and replace, 480 sq. ft.", 2_880.00),
    ("Mold remediation — treated area 450 sq. ft.", 3_200.00),
    ("Carpet — replacement, 180 sq. ft.", 810.00),
    ("Personal property — documented loss (see schedule)", 4_750.00),
    ("Sump pump — replacement, incl. installation", 1_100.00),
    ("Window well covers — install, 3 units", 390.00),
    ("Labor — miscellaneous carpentry and trim", 620.00),
]

# ── Correspondence letter bodies ──────────────────────────────────────────────

LETTER_ACKNOWLEDGEMENT_BODIES: list[str] = [
    (
        "We have received your claim and have assigned it to Claim Number {claim_number} for "
        "processing. We appreciate you bringing this matter to our attention promptly.\n\n"
        "An adjuster has been assigned to your claim and will contact you within two business "
        "days to discuss the next steps, including scheduling an inspection of the damaged "
        "property. Please retain all receipts related to any emergency repairs or temporary "
        "measures you have taken to prevent further loss.\n\n"
        "In the meantime, please do not authorize permanent repairs until our adjuster has had "
        "the opportunity to inspect the damage. Emergency temporary repairs to prevent further "
        "damage are permitted and should be documented with photographs.\n\n"
        "If you have any questions, please contact our Claims Department at the number listed "
        "on the front of your policy."
    ),
    (
        "Thank you for reporting your claim. We understand that a loss of this nature can be "
        "stressful, and we are committed to processing your claim fairly and efficiently.\n\n"
        "We have reviewed the information submitted and are currently investigating the "
        "circumstances of the reported loss. We may contact you for additional documentation, "
        "including photographs, estimates, or other supporting materials as our review progresses.\n\n"
        "Your cooperation during this process is important and appreciated. Please ensure that "
        "any damaged property is preserved in its current state to the extent possible, pending "
        "our inspection."
    ),
]

LETTER_DENIAL_BODY_FLOOD = (
    "After careful review of your claim and the applicable policy provisions, we have determined "
    "that the reported loss is not covered under your current policy.\n\n"
    "REASON FOR DENIAL:\n"
    "Your policy, {policy_number}, contains Exclusion 2.1 — Flood, which states:\n\n"
    '    "We do not insure for loss caused directly or indirectly by flood, surface water, '
    "waves, tidal water, overflow of a body of water, or spray from any of these, whether "
    'or not driven by wind."\n\n'
    "Based on our investigation, the damage to your property was caused by the entry of "
    "surface water and the overflow of subsurface water following the heavy rainfall event "
    "of {loss_date}. This constitutes flood or surface water intrusion as defined under "
    "the policy exclusion cited above.\n\n"
    "We recognize this outcome may be disappointing. Flood coverage is available through "
    "the National Flood Insurance Program (NFIP) and may be purchased for future protection. "
    "We encourage you to contact your agent to discuss your coverage options.\n\n"
    "YOUR RIGHTS:\n"
    "You have the right to request a full review of this decision. If you believe this "
    "determination is incorrect, please submit a written appeal within 30 days of the date "
    "of this letter, along with any additional documentation you wish us to consider. "
    "You may also contact your state's Department of Insurance if you believe this denial "
    "is improper."
)

LETTER_COVERAGE_APPROVAL_BODY = (
    "We have completed our review of your claim and are pleased to confirm that the reported "
    "loss is covered under your policy.\n\n"
    "BASIS FOR COVERAGE:\n"
    "Your policy, {policy_number}, provides coverage for the type of loss described in your "
    "claim. Our adjuster has reviewed the damage and confirmed that the cause of loss falls "
    "within the covered perils under Section I of your policy.\n\n"
    "AUTHORIZED AMOUNT:\n"
    "Based on our review of the inspection findings and the submitted repair estimate, we "
    "have authorized payment in the amount of ${authorized_amount:,.2f}, which reflects the "
    "cost of reasonable and necessary repairs minus your applicable deductible.\n\n"
    "Payment will be processed and mailed to the address on file within 10 business days. "
    "Please retain all receipts for completed repairs, as we may request them for our records.\n\n"
    "If the final repair costs differ materially from the authorized estimate, please contact "
    "us before authorizing additional work so we can review and adjust as appropriate."
)

# ── Adjusters, inspectors, insurers ──────────────────────────────────────────

ADJUSTER_NAMES: dict[str, str] = {
    "ADJ-014": "James Kowalski",
    "ADJ-027": "Patricia Reyes",
}

INSPECTOR_NAMES: list[str] = [
    "T. Nakamura",
    "B. Okonkwo",
    "C. Vandermeer",
    "S. Patel",
]

INSURER_NAME = "Meridian General Insurance Company"
INSURER_ADDRESS = "One Commerce Plaza, Suite 800, Albany, NY 12260"

# ── General boilerplate ───────────────────────────────────────────────────────

POLICY_HEADER_BOILERPLATE = """\
{insurer_name}
{insurer_address}
────────────────────────────────────────────────────────────────────────────────
THIS POLICY IS A LEGAL CONTRACT BETWEEN THE POLICYHOLDER NAMED IN THE
DECLARATIONS AND {insurer_name_upper}. PLEASE READ IT CAREFULLY.
────────────────────────────────────────────────────────────────────────────────
"""

POLICY_FOOTER_BOILERPLATE = """\

────────────────────────────────────────────────────────────────────────────────
NOTICE: This is a summary representation. The complete policy terms and
conditions govern in the event of any conflict. This document does not alter
or waive any provisions of the policy.

{insurer_name} is licensed to do business in all 50 states and the District
of Columbia. Complaints may be directed to the appropriate state Department
of Insurance.
────────────────────────────────────────────────────────────────────────────────
Page {page} of {total_pages}   Policy No. {policy_number}   CONFIDENTIAL
"""

GENERAL_CONDITIONS_TEXT = """\
SECTION IV — CONDITIONS

{conditions}

SECTION V — DEFINITIONS

"Bodily injury" means bodily harm, sickness, or disease, including required
care, loss of services, and death that results.

"Property damage" means physical injury to, destruction of, or loss of use of
tangible property.

"Occurrence" means an accident, including continuous or repeated exposure to
substantially the same general harmful conditions.

"Insured" means you and, if residents of your household, your relatives and any
other person under the age of 21 in your care or the care of a resident relative.
"""
