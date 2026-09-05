#!/usr/bin/env python3
"""Seed deterministic scenarios for the identifier-capture family. Each scenario fixes the ground truth (customer record,
identifiers, DB rows, the write tool calls a correct agent must make) and the CALLER's behaviour (how they will say each
identifier, which utterance is ASR-corrupted, whether they misstate a check letter). The trace author never invents facts.
Usage: seed_scenarios.py OUT.jsonl --n 240 --seed 7"""
import argparse, json, random, string
from validators import dni_letter, iban_fix_check, normalize_phone
from tools_catalog import CATALOG

FIRST = {"es": ["Lucía", "Mateo", "Carmen", "Álvaro", "Noa", "Javier", "Aitana", "Iker", "Marta", "Hugo", "Ainhoa", "Pau"],
         "en": ["Grace", "Oliver", "Priya", "Marcus", "Fiona", "Daniel", "Amara", "Theo", "Sofia", "Liam", "Nadia", "Ethan"]}
LAST = {"es": ["García", "Fernández", "Smyth-Ruiz", "López", "Etxeberria", "Martín", "Jiménez", "Rovira", "Domínguez", "Iglesias"],
        "en": ["Whitmore", "Okafor", "Smyth", "Patel", "Lindqvist", "Brennan", "Duarte", "Chen", "Kowalski", "Hargreaves"]}
CITIES = {"es": [("Madrid", "28013"), ("Valencia", "46001"), ("Bilbao", "48001"), ("Sevilla", "41004"), ("Zaragoza", "50001")],
          "en": [("Madrid", "28013"), ("Barcelona", "08002"), ("Málaga", "29001"), ("Alicante", "03001")]}
ID_STYLES = {  # how the caller says identifiers (spoken-form plans the author must follow)
    "dni": ["pairs", "singles", "mixed_groups", "with_dots", "letter_by_word"],
    "phone": ["3-3-3", "3-2-2-2", "singles", "with_dots", "doubles", "international_0034", "international_plus34", "hundreds"],
    "iban": ["groups_of_4", "last4_only_on_file", "singles_slow"],
    "dob": ["dmy_numeric", "spoken_month", "two_digit_year", "ambiguous_mdy_english"],
    "email": ["spelled_with_punto_arroba", "todo_junto", "spell_local_part", "domain_guess"],
    "code": ["nato_alphabet", "name_alphabet_es", "plain_fast"],
    "postal": ["singles", "pairs", "hundreds"],
}
CORRUPTIONS = ["fifteen_fifty", "b_v_confusion", "m_n_confusion", "zero_o", "dropped_group", "merged_groups", "digit_swap", "letter_dropped", "none"]
PATTERNS_ID = ["Elicit identifier with format hint", "Chunked read-back before use", "Checksum failure -> polite re-ask of one group", "Normalize spoken separators before tool call",
               "Mask sensitive identifier on read-back", "Partial capture across turns", "Letter disambiguation by word", "Date-format disambiguation", "Caller lacks the document -> alternative identifier",
               "Phone number international prefix normalization", "Email spelled aloud -> canonical", "Correction of one group only", "Never write with unvalidated identifier", "ASR mishearing repair"]
INTENTS = {
    "banking": [("block_lost_card", ["verify_identity", "block_card"]), ("transfer_to_iban", ["verify_identity", "get_accounts", "transfer_funds"]), ("update_phone", ["verify_identity", "update_contact_phone"]), ("check_transactions", ["verify_identity", "get_accounts", "get_transactions"]), ("callback_request", ["create_callback"])],
    "telecom": [("change_plan", ["verify_identity", "get_lines", "change_plan"]), ("report_outage", ["verify_identity", "report_outage"]), ("port_in", ["verify_identity", "port_in_request"]), ("invoice_question", ["verify_identity", "get_invoice"])],
    "retail": [("order_status", ["search_order", "get_order_status"]), ("change_address", ["search_order", "update_delivery_address"]), ("cancel_order", ["search_order", "cancel_order"]), ("book_table", ["create_reservation"])],
}

def rand_dni(r):
    n = r.randint(1000000, 99999999); return f"{n:08d}{dni_letter(n)}"
def rand_nie(r):
    p = r.choice("XYZ"); n7 = r.randint(0, 9999999); num = int(str("XYZ".index(p)) + f"{n7:07d}"); return f"{p}{n7:07d}{dni_letter(num)}"
def rand_passport(r):
    return "".join(r.choice(string.ascii_uppercase) for _ in range(3)) + f"{r.randint(0, 999999):06d}"
def rand_phone(r):
    return "+34" + r.choice("67") + "".join(r.choice(string.digits) for _ in range(8))
def rand_iban(r):
    bban = "2100" + f"{r.randint(0, 9999):04d}" + f"{r.randint(0, 99):02d}" + f"{r.randint(0, 9999999999):010d}"
    return iban_fix_check("ES", bban)
def rand_code(r):
    return "ORD-" + "".join(r.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6))
def rand_dob(r):
    return f"{r.randint(1, 28):02d}/{r.randint(1, 12):02d}/{r.randint(1955, 2004)}"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=240); ap.add_argument("--seed", type=int, default=7); a = ap.parse_args()
    r = random.Random(a.seed); rows = []
    for i in range(a.n):
        lang = "es" if i % 5 < 3 else "en"  # 60/40
        vertical = ["banking", "telecom", "retail"][i % 3]
        style = "narrate" if (i // 3) % 2 == 0 else "silent"
        intent, tool_plan = r.choice(INTENTS[vertical])
        first, last = r.choice(FIRST[lang]), r.choice(LAST[lang]); city, postal = r.choice(CITIES[lang])
        doc_type = r.choices(["dni", "nie", "passport"], [0.6, 0.25, 0.15])[0]
        doc = {"dni": rand_dni, "nie": rand_nie, "passport": rand_passport}[doc_type](r)
        cust = {"customer_id": f"cus_{r.randint(10000, 99999)}", "name": f"{first} {last}", "document_type": doc_type, "document_number": doc,
                "date_of_birth": rand_dob(r), "phone": rand_phone(r), "email": f"{first.lower().replace('á','a').replace('í','i').replace('ú','u')}.{last.lower().split('-')[0].replace('í','i').replace('é','e').replace('á','a')}{r.randint(1,99)}@{r.choice(['gmail.com','hotmail.es','outlook.com','proton.me'])}",
                "postal_code": postal, "city": city}
        db = {"customers": [cust]}
        if vertical == "banking":
            db["accounts"] = [{"account_id": f"acc_{r.randint(100000, 999999)}", "customer_id": cust["customer_id"], "type": "checking", "balance_eur": round(r.uniform(120, 6400), 2), "card_last4": f"{r.randint(0, 9999):04d}"}]
            if intent == "check_transactions":
                db["transactions"] = [{"account_id": db["accounts"][0]["account_id"], "date": f"{r.randint(1,28):02d}/08/2026", "amount_eur": round(-r.uniform(3, 220), 2), "merchant": m} for m in r.sample(["Mercadona", "Repsol", "Amazon", "Renfe", "Netflix", "Farmacia Central", "Zara", "Uber"], 6)]
        if vertical == "telecom":
            db["lines"] = [{"line_phone": cust["phone"], "plan_code": r.choice(["BASIC_10", "PLUS_25"]), "status": "active"}]
        if vertical == "retail":
            db["orders"] = [{"order_id": f"ord_{r.randint(100000, 999999)}", "order_code": rand_code(r), "customer_phone": cust["phone"], "customer_email": cust["email"], "status": r.choice(["processing", "shipped"]), "items": r.choice([["auriculares"], ["zapatillas talla 42"], ["cafetera"]]), "delivery": {"street": f"Calle {r.choice(['Mayor', 'Alcalá', 'Colón', 'Real'])} {r.randint(1, 120)}", "postal_code": postal, "city": city}}]
        # identifiers the caller must give, with spoken-form plan and truth-vs-said (checksum failure cases)
        ids = []
        if "verify_identity" in tool_plan:
            said = doc; twist = "none"
            if doc_type in ("dni", "nie") and r.random() < 0.3:  # caller misstates the letter -> checksum failure pattern
                said = doc[:-1] + r.choice([c for c in "TRWAGMYFPDXBNJZSQVHLCKE" if c != doc[-1]]); twist = "wrong_control_letter"
            ids.append({"kind": doc_type, "truth": doc, "caller_says_first": said, "twist": twist, "spoken_style": r.choice(ID_STYLES["dni"]), "corruption": r.choice(CORRUPTIONS) if r.random() < 0.3 else "none"})
            ids.append({"kind": "date_of_birth", "truth": cust["date_of_birth"], "caller_says_first": cust["date_of_birth"], "twist": "none", "spoken_style": r.choice(ID_STYLES["dob"]), "corruption": "none"})
        if intent in ("update_phone", "callback_request", "port_in", "book_table", "report_outage"):
            ph = rand_phone(r) if intent != "report_outage" else cust["phone"]
            ids.append({"kind": "phone", "truth": ph, "caller_says_first": ph, "twist": "none", "spoken_style": r.choice(ID_STYLES["phone"]), "corruption": r.choice(CORRUPTIONS) if r.random() < 0.35 else "none"})
        if intent == "transfer_to_iban":
            ib = rand_iban(r); tw = "none"; said = ib
            if r.random() < 0.25: said = ib[:-3] + str((int(ib[-3]) + 1) % 10) + ib[-2:]; tw = "one_digit_wrong"
            ids.append({"kind": "iban", "truth": ib, "caller_says_first": said, "twist": tw, "spoken_style": r.choice(ID_STYLES["iban"]), "corruption": "none"})
            ids.append({"kind": "amount_eur", "truth": round(r.choice([50, 120, 250, 380.5, 1000]), 2), "caller_says_first": None, "twist": "none", "spoken_style": "spoken_amount", "corruption": "none"})
        if intent in ("order_status", "change_address", "cancel_order"):
            key = r.choice(["order_code", "phone", "email"])
            truth = db["orders"][0]["order_code"] if key == "order_code" else (cust["phone"] if key == "phone" else cust["email"])
            ids.append({"kind": key if key != "order_code" else "code", "truth": truth, "caller_says_first": truth, "twist": "none", "spoken_style": r.choice(ID_STYLES["code" if key == "order_code" else ("phone" if key == "phone" else "email")]), "corruption": r.choice(CORRUPTIONS) if r.random() < 0.3 else "none"})
        if intent in ("change_address", "report_outage"):
            ids.append({"kind": "postal_code", "truth": postal if intent == "report_outage" else r.choice(CITIES[lang])[1], "caller_says_first": None, "twist": "none", "spoken_style": r.choice(ID_STYLES["postal"]), "corruption": "none"})
        if intent == "block_lost_card":
            ids.append({"kind": "card_last4", "truth": db["accounts"][0]["card_last4"], "caller_says_first": db["accounts"][0]["card_last4"], "twist": "none", "spoken_style": r.choice(["pairs", "singles"]), "corruption": r.choice(["fifteen_fifty", "none", "none"])})
        # result-shape twists for search tools
        search_twist = "none"
        if vertical == "retail" and r.random() < 0.35: search_twist = r.choice(["too_many_results", "not_found_then_widen", "near_miss_name"])
        if vertical != "retail" and "verify_identity" in tool_plan and r.random() < 0.15: search_twist = "verification_mismatch_then_alt_id"
        patterns = r.sample(PATTERNS_ID, 3) + ["Chunked read-back before use", "Never write with unvalidated identifier"]
        rows.append({"scenario_id": f"sw_{a.seed}_{i:04d}", "vertical": vertical, "language": lang, "style": style, "intent": intent, "expected_tool_sequence": tool_plan,
                     "customer": cust, "db": db, "identifiers": ids, "search_twist": search_twist, "target_patterns": sorted(set(patterns)),
                     "caller_persona": {"age": r.randint(22, 78), "mood": r.choice(["calm", "hurried", "confused", "irritated", "chatty"]), "has_document_at_hand": r.random() > 0.15},
                     "tools": CATALOG[vertical], "max_assistant_turns": r.choice([10, 12, 14, 18])})
    with open(a.out, "w") as f:
        for s in rows: f.write(json.dumps(s, ensure_ascii=False) + "\n")
    import collections
    print("scenarios", len(rows), collections.Counter(s["vertical"] for s in rows), collections.Counter(s["language"] for s in rows), collections.Counter(s["style"] for s in rows), "twists", collections.Counter(i["twist"] for s in rows for i in s["identifiers"]), "corruptions", sum(1 for s in rows for i in s["identifiers"] if i["corruption"] != "none"))

if __name__ == "__main__":
    main()
