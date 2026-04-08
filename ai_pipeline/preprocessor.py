import re

# ── Normalisasi Singkatan ─────────────────────────────────────────────────────

SLANG_MAP = {
    "yg": "yang", "dgn": "dengan", "utk": "untuk", "krn": "karena",
    "tdk": "tidak", "tdk": "tidak", "sdh": "sudah", "blm": "belum",
    "bgt": "banget", "jg": "juga", "sy": "saya", "hrs": "harus",
    "tsb": "tersebut", "svp": "tolong", "mhn": "mohon", "thx": "terima kasih",
    "info": "informasi", "tlg": "tolong", "kpd": "kepada", "bpk": "bapak",
    "ibu": "ibu", "jl": "jalan", "jln": "jalan", "gg": "gang", "rt": "rt",
    "rw": "rw", "kel": "kelurahan", "kec": "kecamatan", "kab": "kabupaten",
    "pls": "tolong", "asap": "segera", "udh": "sudah", "gak": "tidak",
    "ga": "tidak", "nggak": "tidak", "emg": "memang", "klo": "kalau",
    "kalo": "kalau", "gimana": "bagaimana", "aja": "saja", "doang": "saja",
}

def normalize_slang(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        clean = word.strip(".,!?;:()")
        if clean.lower() in SLANG_MAP:
            replacement = SLANG_MAP[clean.lower()]
            # Pertahankan tanda baca yang menempel
            word = word.replace(clean, replacement)
        result.append(word)
    return " ".join(result)


# ── Preprocessor Utama ────────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    # 1. Lowercase — kecuali tag PII, simpan dulu
    pii_tags = re.findall(r'\[NIK\]|\[PHONE\]|\[EMAIL\]|\[REKENING\]', text)
    placeholder = {}
    for i, tag in enumerate(pii_tags):
        key = f"__PII{i}__"
        text = text.replace(tag, key, 1)
        placeholder[key] = tag

    # 2. Lowercase
    text = text.lower()

    # 3. Normalisasi singkatan/slang
    text = normalize_slang(text)

    # 4. Hapus karakter non-alfanumerik berlebihan — tambah underscore ke whitelist
    text = re.sub(r'[^\w\s.,!?_]', ' ', text)

    # 5. Normalisasi tanda seru/tanya berlebihan
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    # 6. Hapus angka standalone KECUALI yang kecil (RT/RW/nomor pendek)
    text = re.sub(r'\b\d{5,}\b', '', text)

    # 7. Normalisasi spasi ganda
    text = re.sub(r'\s+', ' ', text).strip()

    # 8. Kembalikan tag PII
    for key, tag in placeholder.items():
        text = text.replace(key.lower(), tag)

    return text


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Jln Rungkut Asri berlubang bgt, sdh 3 bulan tdk diperbaiki!!",
        "Mhn segera ditindaklanjuti, banjir di kel. Gubeng udh parah bgt!!!",
        "Tlg perbaiki lampu jalan yg mati di Wonokromo, udh gelap bgt",
        "Sampah menumpuk di gg. Mawar RT 05 RW 02, baunya tdk tertahankan",
        "NIK saya [NIK], hubungi di [PHONE] atau [EMAIL]",  # output PII filter
    ]

    for t in test_cases:
        result = preprocess(t)
        print(f"INPUT : {t}")
        print(f"OUTPUT: {result}")
        print()