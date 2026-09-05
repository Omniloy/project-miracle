"""Tool catalogs (OpenAI function schemas) for the three first-phase verticals. Identifier arguments are always the
CANONICAL normalized string: DNI 8 digits+letter, NIE X/Y/Z+7+letter, phone E.164 (+34...), IBAN no spaces, DOB DD/MM/YYYY."""
def T(name, desc, props, req, slow=False):
    return {"type": "function", "function": {"name": name, "description": desc + (" (slow: 2-4 s)" if slow else ""),
            "parameters": {"type": "object", "properties": props, "required": req}}}
ID = {"type": "string", "description": "Customer identifier: DNI (8 digits + letter), NIE (X/Y/Z + 7 digits + letter) or passport number, normalized without spaces or dashes"}
PHONE = {"type": "string", "description": "Phone number in E.164, e.g. +34612345678"}
COMMON = [
    T("verify_identity", "Verify the caller. Returns customer_id on success, or 'mismatch'.", {"document_number": ID, "date_of_birth": {"type": "string", "description": "DD/MM/YYYY"}}, ["document_number", "date_of_birth"]),
    T("search_customer", "Find customers by name, phone or email. May return 0, 1 or many results.", {"name": {"type": "string"}, "phone": PHONE, "email": {"type": "string"}}, [], slow=True),
    T("transfer_to_human", "Warm-transfer the call to a human agent.", {"reason": {"type": "string", "enum": ["policy_requires_human", "caller_insists", "fraud_suspected", "out_of_scope", "technical_failure"]}, "summary": {"type": "string"}}, ["reason", "summary"]),
    T("create_callback", "Schedule a callback.", {"phone": PHONE, "preferred_time": {"type": "string"}, "topic": {"type": "string"}}, ["phone", "topic"]),
    T("end_call", "End the call after the caller confirms nothing else is needed.", {}, []),
]
BANKING = COMMON + [
    T("get_accounts", "List accounts for a verified customer.", {"customer_id": {"type": "string"}}, ["customer_id"]),
    T("get_transactions", "List transactions of an account in a date range.", {"account_id": {"type": "string"}, "from_date": {"type": "string", "description": "DD/MM/YYYY"}, "to_date": {"type": "string", "description": "DD/MM/YYYY"}}, ["account_id"], slow=True),
    T("block_card", "Block a debit/credit card by its last 4 digits.", {"customer_id": {"type": "string"}, "card_last4": {"type": "string", "pattern": "^\\d{4}$"}, "reason": {"type": "string", "enum": ["lost", "stolen", "fraud_suspected", "damaged"]}}, ["customer_id", "card_last4", "reason"]),
    T("transfer_funds", "Transfer money to an IBAN.", {"from_account_id": {"type": "string"}, "to_iban": {"type": "string", "description": "IBAN without spaces"}, "amount_eur": {"type": "number"}}, ["from_account_id", "to_iban", "amount_eur"], slow=True),
    T("update_contact_phone", "Update the customer's contact phone.", {"customer_id": {"type": "string"}, "phone": PHONE}, ["customer_id", "phone"]),
]
TELECOM = COMMON + [
    T("get_lines", "List phone lines of a verified customer.", {"customer_id": {"type": "string"}}, ["customer_id"]),
    T("get_invoice", "Get an invoice by month.", {"customer_id": {"type": "string"}, "month": {"type": "string", "description": "MM/YYYY"}}, ["customer_id", "month"], slow=True),
    T("change_plan", "Change the plan of a line.", {"line_phone": PHONE, "plan_code": {"type": "string", "enum": ["BASIC_10", "PLUS_25", "UNLIMITED_40"]}}, ["line_phone", "plan_code"]),
    T("report_outage", "Report a service outage for a postal code.", {"postal_code": {"type": "string", "pattern": "^\\d{5}$"}, "line_phone": PHONE}, ["postal_code", "line_phone"]),
    T("port_in_request", "Start a number port-in from another operator.", {"customer_id": {"type": "string"}, "phone_to_port": PHONE, "current_operator": {"type": "string"}}, ["customer_id", "phone_to_port", "current_operator"], slow=True),
]
RETAIL = [t for t in COMMON if t["function"]["name"] != "verify_identity"] + [
    T("search_order", "Find orders by order code, phone or email. May return 0, 1 or many.", {"order_code": {"type": "string", "description": "e.g. ORD-7K4M2Q"}, "phone": PHONE, "email": {"type": "string"}}, [], slow=True),
    T("get_order_status", "Status of one order.", {"order_id": {"type": "string"}}, ["order_id"]),
    T("update_delivery_address", "Change the delivery address of an unshipped order.", {"order_id": {"type": "string"}, "street": {"type": "string"}, "postal_code": {"type": "string", "pattern": "^\\d{5}$"}, "city": {"type": "string"}}, ["order_id", "street", "postal_code", "city"]),
    T("cancel_order", "Cancel an unshipped order.", {"order_id": {"type": "string"}, "reason": {"type": "string"}}, ["order_id", "reason"]),
    T("create_reservation", "Book a table.", {"name": {"type": "string"}, "phone": PHONE, "date": {"type": "string", "description": "DD/MM/YYYY"}, "time": {"type": "string", "description": "HH:MM 24h"}, "party_size": {"type": "integer"}}, ["name", "phone", "date", "time", "party_size"]),
]
CATALOG = {"banking": BANKING, "telecom": TELECOM, "retail": RETAIL}
