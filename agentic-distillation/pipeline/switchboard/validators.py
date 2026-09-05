"""Identifier validators/normalizers used by the Switchboard scenario seeder and trace validator.
Pure functions; no I/O. Spanish DNI/NIE control letter, CIF control, IBAN mod-97, Luhn, E.164 (ES/US/UK basic),
Spanish postal codes, dates (DD/MM/YYYY canonical), emails. All normalizers return None when invalid."""
import re, datetime

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"

def dni_letter(number: int) -> str:
    return DNI_LETTERS[number % 23]

def normalize_dni(s: str):
    """'12.345.678-Z', '12345678z', '12 345 678 Z' -> '12345678Z' if the control letter is right, else None."""
    t = re.sub(r"[\s.\-]", "", s or "").upper()
    m = re.fullmatch(r"(\d{7,8})([A-Z])", t)
    if not m: return None
    num = m.group(1).zfill(8)
    return num + m.group(2) if dni_letter(int(num)) == m.group(2) else None

def normalize_nie(s: str):
    """NIE: X/Y/Z + 7 digits + letter; X->0, Y->1, Z->2 before computing the DNI letter."""
    t = re.sub(r"[\s.\-]", "", s or "").upper()
    m = re.fullmatch(r"([XYZ])(\d{7})([A-Z])", t)
    if not m: return None
    num = int("XYZ".index(m.group(1)).__str__() + m.group(2))
    return t if dni_letter(num) == m.group(3) else None

def cif_control(s: str):
    """Spanish CIF (company tax id): letter + 7 digits + control (digit or letter). Returns the expected control char."""
    t = re.sub(r"[\s.\-]", "", s or "").upper()
    if not re.fullmatch(r"[ABCDEFGHJKLMNPQRSUVW]\d{7}[0-9A-J]", t): return None
    digits = t[1:8]
    total = 0
    for i, d in enumerate(digits):
        n = int(d)
        if i % 2 == 0:  # positions 1,3,5,7 (odd positions in the spec) are doubled
            n *= 2; n = n // 10 + n % 10
        total += n
    c = (10 - total % 10) % 10
    letter_control = "JABCDEFGHI"[c]
    if t[0] in "PQRSNW" or t[1:3] == "00":  # entities whose control must be a letter
        return letter_control
    if t[0] in "ABEH":  # entities whose control must be a digit
        return str(c)
    return str(c) if t[-1].isdigit() else letter_control

def normalize_cif(s: str):
    t = re.sub(r"[\s.\-]", "", s or "").upper()
    ctrl = cif_control(t)
    return t if ctrl is not None and t[-1] == ctrl else None

def normalize_iban(s: str):
    """Remove spaces/dots; upper-case; verify mod-97 == 1 and the country length (ES=24, DE=22, FR=27, GB=22, PT=25, IT=27)."""
    t = re.sub(r"[\s.\-]", "", s or "").upper()
    lengths = {"ES": 24, "DE": 22, "FR": 27, "GB": 22, "PT": 25, "IT": 27, "NL": 18, "BE": 16}
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{11,30}", t): return None
    if t[:2] in lengths and len(t) != lengths[t[:2]]: return None
    rearranged = t[4:] + t[:4]
    digits = "".join(str(int(ch, 36)) for ch in rearranged)
    return t if int(digits) % 97 == 1 else None

def iban_fix_check(country: str, bban: str) -> str:
    """Build a valid IBAN for a BBAN by computing its check digits."""
    digits = "".join(str(int(ch, 36)) for ch in bban + country + "00")
    check = 98 - int(digits) % 97
    return f"{country}{check:02d}{bban}"

def luhn_ok(number: str) -> bool:
    d = [int(c) for c in re.sub(r"\D", "", number or "")]
    if len(d) < 12: return False
    s = 0
    for i, n in enumerate(reversed(d)):
        if i % 2 == 1:
            n *= 2; n = n - 9 if n > 9 else n
        s += n
    return s % 10 == 0

def normalize_phone(s: str, default_country="ES"):
    """Basic E.164 normalizer for ES (9 digits starting 6/7/8/9, +34 / 0034 prefixes), US (10 digits, +1), UK (+44)."""
    t = re.sub(r"[\s.\-()]", "", s or "")
    t = re.sub(r"^00", "+", t)
    if t.startswith("+34") and re.fullmatch(r"\+34[6789]\d{8}", t): return t
    if t.startswith("+1") and re.fullmatch(r"\+1[2-9]\d{9}", t): return t
    if t.startswith("+44") and re.fullmatch(r"\+44[1-9]\d{8,9}", t): return t
    if t.startswith("+"): return None
    if default_country == "ES" and re.fullmatch(r"[6789]\d{8}", t): return "+34" + t
    if default_country == "US" and re.fullmatch(r"[2-9]\d{9}", t): return "+1" + t
    return None

def normalize_postal_es(s: str):
    t = re.sub(r"\s", "", s or "")
    return t if re.fullmatch(r"(0[1-9]|[1-4]\d|5[0-2])\d{3}", t) else None

def normalize_date(s: str, order="DMY"):
    """'5/8/1990', '05-08-90', '1990-08-05' -> '05/08/1990' (canonical DD/MM/YYYY). order tells how to read x/y/z."""
    t = (s or "").strip()
    m = re.fullmatch(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})", t)
    if m: y, a, b = int(m.group(1)), int(m.group(2)), int(m.group(3)); d, mo = (b, a)
    else:
        m = re.fullmatch(r"(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})", t)
        if not m: return None
        a, b, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if y < 100: y += 1900 if y > 30 else 2000
        d, mo = (a, b) if order == "DMY" else (b, a)
    try: return datetime.date(y, mo, d).strftime("%d/%m/%Y")
    except ValueError: return None

def normalize_email(s: str):
    t = (s or "").strip().lower().replace(" arroba ", "@").replace(" at ", "@").replace(" punto ", ".").replace(" dot ", ".").replace(" ", "")
    return t if re.fullmatch(r"[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}", t) else None

NORMALIZERS = {"dni": normalize_dni, "nie": normalize_nie, "cif": normalize_cif, "iban": normalize_iban, "phone": normalize_phone,
               "postal_code": normalize_postal_es, "date_of_birth": normalize_date, "email": normalize_email}

if __name__ == "__main__":
    assert normalize_dni("12345678Z") == "12345678Z" and normalize_dni("12345678A") is None
    assert normalize_nie("X1234567L") == "X1234567L"
    assert normalize_iban("GB82 WEST 1234 5698 7654 32") == "GB82WEST12345698765432"
    assert normalize_iban(iban_fix_check("ES", "21000418450200051332")) is not None
    assert luhn_ok("4539 1488 0343 6467")
    assert normalize_phone("612 34 56 78") == "+34612345678" and normalize_phone("0034 612345678") == "+34612345678"
    assert normalize_date("5/8/1990") == "05/08/1990" and normalize_email("juan punto perez arroba gmail punto com") == "juan.perez@gmail.com"
    print("validators ok")
