"""Turn spoken ES/EN number and letter words into a compact digit/letter string so read-backs can be compared with canonical values.
'siete dos, nueve dos, cuatro ocho, seis cinco, efe de Francia' -> '72924865F'; 'six thirty, double eight, forty-six' -> '6308846'."""
import re
ES_UNITS = {"cero": 0, "uno": 1, "una": 1, "un": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9,
            "diez": 10, "once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15, "dieciséis": 16, "dieciseis": 16, "diecisiete": 17, "dieciocho": 18, "diecinueve": 19,
            "veinte": 20, "veintiuno": 21, "veintiuna": 21, "veintidós": 22, "veintidos": 22, "veintitrés": 23, "veintitres": 23, "veinticuatro": 24, "veinticinco": 25, "veintiséis": 26, "veintiseis": 26, "veintisiete": 27, "veintiocho": 28, "veintinueve": 29,
            "treinta": 30, "cuarenta": 40, "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90}
ES_HUNDREDS = {"cien": 100, "ciento": 100, "doscientos": 200, "trescientos": 300, "cuatrocientos": 400, "quinientos": 500, "seiscientos": 600, "setecientos": 700, "ochocientos": 800, "novecientos": 900}
EN_UNITS = {"zero": 0, "oh": 0, "o": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
            "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
ES_LETTERS = {"a": "A", "be": "B", "ce": "C", "de": "D", "e": "E", "efe": "F", "ge": "G", "hache": "H", "i": "I", "jota": "J", "ka": "K", "ele": "L", "eme": "M", "ene": "N", "eñe": "Ñ", "o": "O", "pe": "P", "cu": "Q", "erre": "R", "ere": "R", "ese": "S", "te": "T", "u": "U", "uve": "V", "equis": "X", "zeta": "Z", "ceta": "Z"}
NATO = {"alfa": "A", "alpha": "A", "bravo": "B", "charlie": "C", "delta": "D", "echo": "E", "foxtrot": "F", "golf": "G", "hotel": "H", "india": "I", "juliet": "J", "juliett": "J", "kilo": "K", "lima": "L", "mike": "M", "november": "N", "oscar": "O", "papa": "P", "quebec": "Q", "romeo": "R", "sierra": "S", "tango": "T", "uniform": "U", "victor": "V", "whiskey": "W", "xray": "X", "yankee": "Y", "zulu": "Z"}

def spoken_to_compact(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[,.;:¿?¡!\-–/()]", " ", t)
    toks = t.split()
    out = []; i = 0; last_num_end = -1
    while i < len(toks):
        w = toks[i]
        if re.fullmatch(r"\d+", w): out.append(w); i += 1; last_num_end = i; continue
        # 'doble ocho' / 'double eight' / 'triple dos'
        if w in ("doble", "double", "triple") and i + 1 < len(toks):
            nxt = toks[i + 1]; d = ES_UNITS.get(nxt, EN_UNITS.get(nxt))
            if d is not None and d < 10: out.append(str(d) * (2 if w != "triple" else 3)); i += 2; continue
        if w in ES_HUNDREDS:
            val = ES_HUNDREDS[w]; j = i + 1
            if j < len(toks) and toks[j] in ES_UNITS:
                val += ES_UNITS[toks[j]]; j += 1
                if j + 1 < len(toks) and toks[j] == "y" and toks[j + 1] in ES_UNITS and ES_UNITS[toks[j + 1]] < 10: val += ES_UNITS[toks[j + 1]]; j += 2
            out.append(str(val)); i = j; last_num_end = i; continue
        if w in ES_UNITS:
            val = ES_UNITS[w]; j = i + 1
            if val >= 30 and j + 1 < len(toks) and toks[j] == "y" and toks[j + 1] in ES_UNITS and ES_UNITS[toks[j + 1]] < 10: val += ES_UNITS[toks[j + 1]]; j += 2
            out.append(str(val)); i = j; last_num_end = i; continue
        if w in EN_UNITS:
            val = EN_UNITS[w]; j = i + 1
            if val >= 20 and val % 10 == 0 and j < len(toks) and toks[j] in EN_UNITS and EN_UNITS[toks[j]] < 10: val += EN_UNITS[toks[j]]; j += 1
            out.append(str(val)); i = j; last_num_end = i; continue
        if w == "hundred" and out and out[-1].isdigit(): out[-1] = str(int(out[-1]) * 100); i += 1; continue
        if w in ("mil", "thousand"):
            base = int(out.pop()) if (out and out[-1].isdigit() and last_num_end == i) else 1
            val = base * 1000; j = i + 1
            if j < len(toks) and toks[j] in ES_HUNDREDS: val += ES_HUNDREDS[toks[j]]; j += 1
            if j < len(toks) and toks[j] in ES_UNITS:
                val += ES_UNITS[toks[j]]; j += 1
                if j + 1 < len(toks) and toks[j] == "y" and toks[j + 1] in ES_UNITS and ES_UNITS[toks[j + 1]] < 10: val += ES_UNITS[toks[j + 1]]; j += 2
            out.append(str(val)); i = j; last_num_end = i; continue
        # letters: 'efe de francia' -> F ; NATO ; 'f for foxtrot' ; single letters
        if w in ES_LETTERS and not (w in ("a", "e", "o", "u", "i", "de", "te", "ka") and i + 1 < len(toks) and toks[i + 1] in ES_UNITS):  # avoid articles before numbers
            if w in ("de", "y", "a", "o", "e", "u", "i", "te", "ka") and w not in ("efe", "uve", "zeta"):  # short function words: only accept as letter if followed by 'de <word>' or at the end of a code
                if i + 1 < len(toks) and toks[i + 1] == "de": out.append(ES_LETTERS[w]); i += 3; continue
                i += 1; continue
            out.append(ES_LETTERS[w])
            if i + 2 < len(toks) and toks[i + 1] == "de": i += 3
            else: i += 1
            continue
        if w in NATO: out.append(NATO[w]); i += 1; continue
        if re.fullmatch(r"[a-z]", w) and i + 2 < len(toks) and toks[i + 1] in ("for", "as", "de", "como"): out.append(w.upper()); i += 3; continue
        if re.fullmatch(r"[a-z]", w) and (i + 1 == len(toks) or toks[i + 1] in ES_UNITS or toks[i + 1] in EN_UNITS or re.fullmatch(r"\d+|[a-z]", toks[i + 1])): out.append(w.upper()); i += 1; continue
        i += 1
    return "".join(out).upper()

if __name__ == "__main__":
    assert spoken_to_compact("siete dos, nueve dos, cuatro ocho, seis cinco, efe de Francia") == "72924865F", spoken_to_compact("siete dos, nueve dos, cuatro ocho, seis cinco, efe de Francia")
    assert "6308246" in spoken_to_compact("seis tres cero, ochenta y dos, cuarenta y seis, veintiocho")
    assert spoken_to_compact("six thirty, double eight, forty-six") == "6308846"
    assert spoken_to_compact("Oscar Romeo Delta seven Kilo four Mike two Quebec") == "ORD7K4M2Q"
    assert "12101958" in spoken_to_compact("doce del diez de mil novecientos cincuenta y ocho")
    print("spoken_digits ok")
