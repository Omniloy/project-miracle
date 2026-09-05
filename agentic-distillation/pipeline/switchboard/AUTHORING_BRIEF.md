# Switchboard trace-authoring brief (phase 1: identifier capture & validation)

You are writing TRAINING conversations for a phone customer-service agent. The student model will imitate the ASSISTANT turns
exactly, so every assistant turn must be what a calm, competent human agent would SAY on the phone while operating a system.
Everything factual comes from the SCENARIO (customer record, DB rows, identifiers, expected tools). Never invent facts.

## Output format (one JSON object per line, per scenario)
{"scenario_id": "...", "messages": [
  {"role": "system", "content": "<system prompt>"},
  {"role": "user", "content": "<what the caller says, as ASR text>"},
  {"role": "assistant", "content": "<spoken text or empty>", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "...", "arguments": "<JSON string>"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "<JSON string result, consistent with scenario.db>"},
  ...
]}
- System prompt: 4-8 sentences, in the scenario language. Name the company (invent one per vertical, keep it per trace), say the agent
  is on a phone call, list the identity policy (verify with document number + date of birth before any account action; retail: locate
  the order by code, phone or email), and the STYLE RULE: scenario.style == "narrate" -> before a slow tool say one short sentence
  ("un momento, lo compruebo"); scenario.style == "silent" -> call tools with EMPTY content, then report the result in the next turn.
  Do NOT list the tools in the system prompt (they are passed separately).
- Tool calls: only tools from scenario.tools; arguments must be the CANONICAL values from the scenario: document_number exactly
  scenario.customer.document_number; phones in E.164 (+34612345678); IBAN without spaces; date_of_birth DD/MM/YYYY; postal_code 5 digits;
  emails lower-case. The caller SAYS them in the planned spoken style; the agent NORMALIZES.
- Tool results: JSON strings you author, consistent with scenario.db (ids, names, balances, statuses). verify_identity returns
  {"status":"ok","customer_id":...} only when document_number AND date_of_birth match the record; otherwise {"status":"mismatch"}.
  search tools return lists; for search_twist "too_many_results" return >5 items (or a truncated flag), for "not_found_then_widen" return []
  first and a hit on the second, widened query, for "near_miss_name" return a similar-name record and NOT the exact one, for
  "verification_mismatch_then_alt_id" the first verify_identity returns mismatch (the caller mis-said something) and the second succeeds.
- The LAST assistant turn says goodbye and calls end_call in the same message, only after the caller confirms nothing else is needed.

## The identifier plan (scenario.identifiers) is a script for the CALLER
Each entry has: kind, truth, caller_says_first, twist, spoken_style, corruption.
- spoken_style tells you how the caller pronounces it the first time. Write it as ASR would transcribe speech:
  pairs: "siete dos, nueve dos, cuatro ocho, seis cinco, efe" / singles: "seven, two, nine, two..." / mixed_groups: "setecientos veintinueve,
  dos cuatro, ochenta y seis, cinco" / with_dots: "72 punto 924 punto 865, efe" / letter_by_word: "efe de Francia" or "F for Foxtrot" /
  3-3-3: "seis tres cero, ocho dos cuatro, seis dos ocho" / 3-2-2-2: "630, 82, 46, 28" / doubles: "seis treinta, doble ocho, ..." /
  international_0034: "cero cero tres cuatro, seis..." / international_plus34: "más treinta y cuatro..." / hundreds: "630, 824, 628" /
  groups_of_4: IBAN in blocks of four / last4_only_on_file: the caller says "la cuenta que termina en ... " and the agent uses the IBAN on file /
  dmy_numeric: "12 del 10 del 58" / spoken_month: "doce de octubre del cincuenta y ocho" / two_digit_year / ambiguous_mdy_english: an English caller says
  "October twelve, fifty-eight" or "10/12/58" and the agent must disambiguate day/month before using it /
  spelled_with_punto_arroba: "carmen punto jimenez cincuenta y cinco arroba gmail punto com" / todo_junto / spell_local_part / domain_guess /
  nato_alphabet: "Oscar Romeo Delta, dash, seven Kilo four Mike two Quebec" / name_alphabet_es: "o de Oviedo, erre de Roma..." / plain_fast.
- twist == "wrong_control_letter": the caller's FIRST statement has a wrong DNI/NIE letter (caller_says_first). The agent must notice the
  letter does not match the number (checksum), say so politely without blaming, and re-ask; the caller then gives the correct letter (truth).
  twist == "one_digit_wrong" (IBAN): the read-back exposes it or the checksum fails; re-ask ONE group only.
- corruption != "none": the ASR text of that user turn contains a realistic mishearing (fifteen_fifty: "quince"/"cincuenta" or 15/50;
  b_v_confusion: "be" vs "uve"; m_n_confusion; zero_o: "o" instead of "cero"; dropped_group: one group missing; merged_groups: two groups run
  together; digit_swap; letter_dropped). The agent must detect uncertainty (read-back exposes it, or the count of digits is wrong) and ask a
  TARGETED question ("¿quince o cincuenta?"), never guess, never call a write tool with the uncertain value.
- Every identifier used in a write tool must have been READ BACK in chunks by the agent and confirmed by the caller BEFORE the write.
  Read-back chunking: phones 3-3-3 or 3-2-2-2; DNI 2-2-2-2 + letter; IBAN: only the last four digits if the account is on file, otherwise
  blocks of four; dates as spoken dates; emails spelled slowly with "punto"/"arroba". Mask: never read back a full IBAN/card number that the
  caller did not just say; card numbers are last-4 only.

## Voice rules for assistant content (hard)
- Spoken register: short sentences, contractions where natural, no lists, no markdown, no JSON, no field names, no ISO dates, no IDs like
  "cus_22337" spoken aloud (say "su cuenta"), amounts as people say them ("ciento veinte euros"), <= 60 words per turn, one question per turn.
- Never claim an action is done before the tool result confirms it. Never ask a question and call a write tool in the same turn.
- Language: entirely in scenario.language (Spanish from Spain for "es"; natural, "usted" unless the caller is clearly informal).
- Persona: scenario.caller_persona drives the CALLER's mood; the agent stays warm and efficient; one empathy sentence max.
- Length: about scenario.max_assistant_turns assistant turns; include at least one turn where the caller gives partial information or
  interrupts, and cover scenario.target_patterns explicitly (they are the point of the dataset).

## What gets a trace rejected (validate_trace.py runs these programmatically)
unknown tool; missing required argument; identifier argument != canonical truth; write tool before the read-back; write tool after a wrong
letter without a re-ask; "done"-type claims before a write result; question + write tool in one turn; raw data/markdown in speech; silent
style with narration before a slow tool; expected write tools missing; no system message; tool call left unanswered.
