# Switchboard — Phase 1 dataset

Spoken-register phone-agent traces for SFT. Each row is one full customer-service call:
a system prompt in the agent's voice, a caller, tool calls against a per-scenario catalog,
and tool results. The training signal is **how identifiers survive a phone call**: the
caller says a DNI/NIE/passport/IBAN/phone/date the way people actually say it (in groups,
with dots, "be de Barcelona", fifteen/fifty slips, dropped groups), and the agent must
chunk-read it back, catch checksum failures, re-ask only the group that is wrong, and
never write to a system with an unconfirmed value. Secondary axes: say/do discipline
(no past tense before a tool returns), spoken style (no markdown, no ISO dates, no
snake_case read aloud), language fidelity (es/en), and two narration modes
(`narrate` = one short line before a slow tool, `silent` = empty content on tool turns).

## Files

| file | rows | what |
|---|---|---|
| `scenarios_p1.jsonl` | 240 | seeded scenarios: customer, in-universe db, identifier plan (truth / what the caller says first / twist / corruption / spoken style), tool catalog, expected tool sequence |
| `phase1_accepted.jsonl` | 202 | all `traces/batch_*.judged.jsonl` concatenated, deduped by `scenario_id` |
| `phase1_final.jsonl` | 202 | the same rows after a full re-run of the validator (0 dropped) |
| `phase1_sft.jsonl` | 202 | Axolotl chat rows: `{messages, tools, meta}` — `meta` carries scenario_id, vertical, language, style, intent, target_patterns |
| `phase1_final_report.json` | 202 | per-trace validator errors/warnings |
| `phase1_stats.json` | — | everything summarised below |

## How it was made

1. **Seed** — `pipeline/switchboard/seed_scenarios.py` generates 240 scenarios across
   3 verticals x 2 languages x 2 narration styles x 13 intents. Identifiers are generated
   *valid* (real DNI/NIE control letters, IBAN mod-97, E.164) by `validators.py`, then a
   twist (`wrong_control_letter`, `one_digit_wrong`) and/or a speech corruption
   (`b_v_confusion`, `fifteen_fifty`, `zero_o`, `digit_swap`, `dropped_group`,
   `merged_groups`, `m_n_confusion`, `letter_dropped`) is planned on top.
2. **Author** — traces written in batches of 10 against `AUTHORING_BRIEF.md`, which fixes
   the in-universe rules (e.g. `verify_identity` returns ok whenever both fields match the
   db row) so tool results cannot be invented.
3. **Validate** — `validate_trace.py` programmatically rejects: unanswered tool calls,
   non-JSON args, unknown tools, missing required params, identifier arguments that are not
   the canonical truth, a write tool before a chunked read-back, a twist used without a
   re-ask, completion claimed in past tense before a write tool returned, question + write
   in one turn, markdown/ISO dates/snake_case in spoken content, wrong language, and
   style violations. Batches were re-authored until 240/240 passed.
4. **Judge** — an LLM judge scores four axes 1-5 (naturalness, say_do, identifier_handling,
   language) and passes/rejects each trace. Only passing traces reach `*.judged.jsonl`.
5. **Assemble** — this dataset: concatenate, dedupe, re-validate the pooled file, render SFT.

## Acceptance funnel

```
scenarios seeded              240
traces written                240   (1 per scenario, no scenario unwritten)
validator-accepted            240   (100% — batches were re-authored until clean)
judge-passed                  202   (84.2% of written; 38 rejected)
pooled + deduped              202   (0 duplicate scenario_ids, 0 missing input files)
final validator re-run        202   (100% accepted, 0 dropped)
rendered to SFT               202
```

The judge rejected 38 traces. Dominant reasons (a trace can have several): **31** invent a
fact the tools never returned, **23** stage the `verification_mismatch_then_alt_id` twist by
having `verify_identity` return a mismatch for arguments that exactly match the db row
(forbidden by the brief), **21** contain a read-back or digit-count error the caller then
"confirms", **1** passes a plan code outside the tool's enum. Note that the programmatic
validator cannot see any of these — they are all semantic — which is why the judge stage
exists.

## Shape of the final 202

- **Vertical**: retail 74, banking 66, telecom 62. **Language**: es 120, en 82.
  **Style**: narrate 101, silent 101. **Intent**: 13 intents, 7-20 rows each.
- **Turns**: 10.1 assistant turns/call (median 10, range 7-15), 7.7 user turns,
  3.4 tool calls. Spoken assistant turns average 15 words (max 42) — phone-length, as intended.
- **Size**: ~589 est. tokens per row of messages (chars/4); ~1424 with the tool schemas
  included. ~119k message tokens / ~288k total.
- **Identifier work**: 449 identifiers over 202 calls. 44 carry a checksum/digit twist
  (42 traces) and 68 carry a speech corruption (65 traces); 26 distinct spoken styles.
- **Search twists**: none 174, near_miss_name 11, too_many_results 9,
  not_found_then_widen 7, verification_mismatch_then_alt_id 1.

## Judge means

| axis | final 202 | all 240 graded | the 38 rejected |
|---|---|---|---|
| naturalness | 4.94 | 4.92 | 4.86 |
| say_do | 4.55 | 4.32 | 3.03 |
| identifier_handling | 4.83 | 4.71 | 4.06 |
| language | 4.99 | 4.99 | 5.00 |
| mean of axes | 4.83 | 4.74 | 4.24 |

Rejection is driven almost entirely by **say_do** (3.03 vs 4.55): the calls that failed were
fluent and correctly-spoken, they just asserted things the tools had not returned.

## Near-duplicate check

Verbatim overlap of assistant sentences across traces (no MinHash; exact normalised match):

| sentence length | occurrences | unique | fraction repeated in >1 trace |
|---|---|---|---|
| all | 3844 | 2653 | **38.2%** |
| >= 4 words | 2770 | 2360 | **21.0%** |
| >= 8 words | 1427 | 1370 | **6.2%** |

No two traces share an identical assistant script. The repetition is concentrated in the
short formulaic phone furniture you *want* a phone agent to have — "un momento, lo compruebo."
(28 traces), "que tenga buen día." (20), "and your date of birth?" (19) — and it thins out
fast with length, so the substantive content is not templated.

## Known gaps

- **`verification_mismatch_then_alt_id` is effectively absent**: 24 seeded, 1 survived.
  Nearly every author staged it by faking a mismatch on correct arguments, which the judge
  correctly rejected. The scenario needs a redesign (the caller must give a genuinely
  different-from-db value first) before it can be trained on. Note the validator also
  *skips* its write-sequence check for this twist, so it is doubly under-covered.
- **Twist/corruption density is low**: 405 of 449 identifiers have `twist: none` and 381
  have `corruption: none`. Only ~21% of traces exercise a checksum failure and ~32% a
  speech corruption; the "catch it and re-ask one group" behaviour is the point of the set
  and could carry more weight in phase 2.
- **Attrition is not uniform**: banking and telecom lost more traces than retail
  (80->66, 80->62 vs 80->74), and `invoice_question` (12->7) and `callback_request` (15->11)
  are thin. es/en drifted from 60/40 seeded to 59/41.
- **Judge coverage is uneven in form, not in fact**: the 24 `*.scores.json` files use three
  different per-trace schemas (`traces` / `per_trace` / `scores`), and 2 rejected traces
  (`sw_7_0112`, `sw_7_0136`) carry a rejection reason but no per-axis row — so axis means
  are over 238 of 240, while pass/fail is known for all 240. Every one of the 202 final rows
  has full axis scores.
- **Single-judge, no human audit**: no inter-rater or human spot-check of the judge's
  accept decisions, so the 202 are "one judge said yes", not gold.
- **No held-out split**: this file is the whole pool. Carve out an eval slice before
  training, ideally stratified by vertical x language x style.
- **All identifiers are Spanish-shaped** (DNI/NIE/CIF, ES postal codes, +34), even in the
  82 English calls. Phone/IBAN normalizers cover a few other countries but no scenario uses them.
