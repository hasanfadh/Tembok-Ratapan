"""
convert_jsonl_format.py
Konversi sp4n_train.jsonl dari format anotasi ke format inference_baseline.

Perubahan:
  - 'kategori'         → 'category'
  - intent 4 kelas     → 2 kelas (complaint | non-complaint)
  - sentiment ID       → EN (positif→positive, netral→neutral, negatif→negative)
  - system prompt      → diseragamkan dengan inference_baseline.py

Cara pakai:
    python convert_jsonl_format.py data/processed/sp4n_train.jsonl
    python convert_jsonl_format.py data/processed/sp4n_train.jsonl --output data/processed/sp4n_train_final.jsonl
"""

import json
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# MAPPING
# ─────────────────────────────────────────────

# Intent 4 kelas → 2 kelas
INTENT_MAP = {
    "complaint":   "complaint",
    "report":      "complaint",    # laporan = complaint
    "question":    "non-complaint",
    "suggestion":  "non-complaint",
}

# Kategori ID → EN (sesuai inference_baseline)
KATEGORI_MAP = {
    "Infrastruktur": "infrastruktur",
    "Sosial":        "lainnya",
    "Pendidikan":    "lainnya",
    "Kesehatan":     "lainnya",
    "Lingkungan":    "kebersihan",
    "Administrasi":  "kependudukan",
    "Lainnya":       "lainnya",
    # fallback lowercase
    "infrastruktur": "infrastruktur",
    "sosial":        "lainnya",
    "pendidikan":    "lainnya",
    "kesehatan":     "lainnya",
    "lingkungan":    "kebersihan",
    "administrasi":  "kependudukan",
    "lainnya":       "lainnya",
}

# Sentiment ID → EN
SENTIMENT_MAP = {
    "positif": "positive",
    "netral":  "neutral",
    "negatif": "negative",
    "positive": "positive",
    "neutral":  "neutral",
    "negative": "negative",
}

SYSTEM_PROMPT = (
    "Kamu adalah sistem klasifikasi pengaduan masyarakat Kota Surabaya. "
    "Analisis teks pengaduan dan kembalikan hasil dalam format JSON. "
    "Jawab HANYA dengan JSON tanpa penjelasan tambahan."
)

OUTPUT_FORMAT_DESC = (
    "Kembalikan HANYA JSON berikut tanpa penjelasan tambahan:\n"
    "{\n"
    "  \"intent\": \"<complaint | non-complaint>\",\n"
    "  \"category\": \"<infrastruktur | kependudukan | perizinan | kebersihan | keamanan | lainnya>\",\n"
    "  \"location\": \"<lokasi yang disebutkan, atau null>\",\n"
    "  \"object\": \"<objek masalah utama>\",\n"
    "  \"urgency\": \"<low | medium | high>\",\n"
    "  \"sentiment\": \"<positive | neutral | negative>\",\n"
    "  \"summary\": \"<ringkasan 1 kalimat dalam Bahasa Indonesia>\"\n"
    "}"
)


# ─────────────────────────────────────────────
# KONVERSI
# ─────────────────────────────────────────────

def convert_output(old_output: str) -> str:
    """Parse output lama, remap field, return JSON string baru."""
    try:
        data = json.loads(old_output)
    except Exception:
        # Coba ekstrak JSON dari dalam teks
        import re
        match = re.search(r"\{[\s\S]*\}", old_output)
        if match:
            try:
                data = json.loads(match.group())
            except Exception:
                return old_output  # gagal parse, kembalikan apa adanya
        else:
            return old_output

    new_data = {
        "intent":    INTENT_MAP.get(str(data.get("intent", "complaint")).lower(), "complaint"),
        "category":  KATEGORI_MAP.get(str(data.get("kategori", data.get("category", "lainnya"))), "lainnya"),
        "location":  data.get("location") or data.get("lokasi") or None,
        "object":    data.get("object") or data.get("objek") or "",
        "urgency":   str(data.get("urgency", "medium")).lower(),
        "sentiment": SENTIMENT_MAP.get(str(data.get("sentiment", "neutral")).lower(), "neutral"),
        "summary":   data.get("summary") or data.get("ringkasan") or "",
    }

    # Validasi urgency
    if new_data["urgency"] not in {"low", "medium", "high"}:
        new_data["urgency"] = "medium"

    return json.dumps(new_data, ensure_ascii=False)


def build_new_instruction(old_input: str) -> str:
    """Buat instruction + input dalam format inference_baseline."""
    return (
        f"Kamu adalah sistem klasifikasi pengaduan masyarakat Kota Surabaya.\n"
        f"Analisis teks pengaduan berikut dan kembalikan hasil dalam format JSON.\n\n"
        f"Teks pengaduan:\n\"\"\"{old_input}\"\"\"\n\n"
        f"{OUTPUT_FORMAT_DESC}"
    )


def convert_record(record: dict) -> dict:
    """Konversi satu record JSONL."""
    old_input  = record.get("input", "")
    old_output = record.get("output", "{}")

    # Bersihkan prefix "Judul: ... \nLaporan: " jika ada, ambil isi laporan saja
    # Tapi tetap pertahankan konteks judul
    new_instruction = build_new_instruction(old_input)
    new_output      = convert_output(old_output)

    return {
        "instruction": new_instruction,
        "input":       "",   # sudah masuk ke instruction
        "output":      new_output,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",           help="Path JSONL input (sp4n_train.jsonl)")
    parser.add_argument("--output", "-o",  default=None, help="Path output (default: sp4n_train_final.jsonl)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.parent / "sp4n_train_final.jsonl"

    records   = []
    errors    = 0
    skipped   = 0

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                converted = convert_record(record)
                records.append(converted)
            except Exception as e:
                errors += 1

    # Tulis output
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[done] {len(records)} record dikonversi → {output_path}")
    if errors:
        print(f"[warn] {errors} record gagal dikonversi")

    # Preview distribusi
    print("\n── Preview 3 record pertama ──")
    for r in records[:3]:
        out = json.loads(r["output"])
        print(f"  intent={out['intent']} | category={out['category']} | "
              f"urgency={out['urgency']} | sentiment={out['sentiment']}")
        print(f"  summary: {out.get('summary', '')[:80]}")
        print()

    # Distribusi label
    from collections import Counter
    intents    = Counter(json.loads(r["output"])["intent"]    for r in records)
    categories = Counter(json.loads(r["output"])["category"]  for r in records)
    urgencies  = Counter(json.loads(r["output"])["urgency"]   for r in records)
    sentiments = Counter(json.loads(r["output"])["sentiment"] for r in records)

    print("── Distribusi Label ──")
    print(f"intent    : {dict(intents)}")
    print(f"category  : {dict(categories)}")
    print(f"urgency   : {dict(urgencies)}")
    print(f"sentiment : {dict(sentiments)}")


if __name__ == "__main__":
    main()