"""
preprocessing.py
Preprocessing pipeline untuk dataset SP4N-LAPOR
Input  : CSV hasil scraping (kolom: id, judul, isi, kategori, url)
Output : CSV bersih siap inference + JSONL untuk fine-tuning
"""

import pandas as pd
import re
import json
import os
from pathlib import Path


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv("lapor_dataset.csv")
    print(f"[load] {len(df)} baris dimuat dari {path}")
    return df


# ─────────────────────────────────────────────
# 2. DROP KOLOM TIDAK PERLU
# ─────────────────────────────────────────────

def drop_columns(df: pd.DataFrame, cols: list = ["url"]) -> pd.DataFrame:
    df = df.drop(columns=[c for c in cols if c in df.columns])
    print(f"[drop] Kolom dihapus: {cols}")
    return df


# ─────────────────────────────────────────────
# 3. CLEANING TEKS
# ─────────────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # Hapus URL
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Hapus karakter HTML entity
    text = re.sub(r"&[a-z]+;", " ", text)

    # Hapus karakter non-alfanumerik kecuali tanda baca umum
    text = re.sub(r"[^\w\s.,!?;:()\-']", " ", text)

    # Normalisasi whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["judul"] = df["judul"].apply(clean_text)
    df["isi"]   = df["isi"].apply(clean_text)
    print("[clean] Kolom judul & isi sudah dibersihkan")
    return df


# ─────────────────────────────────────────────
# 4. NORMALISASI KATEGORI
# ─────────────────────────────────────────────

KATEGORI_MAP = {
    # Infrastruktur
    "infrastruktur": "Infrastruktur",
    "jalan": "Infrastruktur",
    "jembatan": "Infrastruktur",
    # Sosial
    "sosial": "Sosial",
    "bansos": "Sosial",
    "kemiskinan": "Sosial",
    # Pendidikan
    "pendidikan": "Pendidikan",
    "sekolah": "Pendidikan",
    # Kesehatan
    "kesehatan": "Kesehatan",
    "rs": "Kesehatan",
    "puskesmas": "Kesehatan",
    # Lingkungan
    "lingkungan": "Lingkungan",
    "sampah": "Lingkungan",
    "banjir": "Lingkungan",
    # Administrasi
    "administrasi": "Administrasi",
    "perizinan": "Administrasi",
    "ktp": "Administrasi",
    # Lainnya
    "lainnya": "Lainnya",
}

def normalize_kategori(df: pd.DataFrame) -> pd.DataFrame:
    def _map(k):
        if not isinstance(k, str):
            return "Lainnya"
        k_lower = k.lower().strip()
        for key, val in KATEGORI_MAP.items():
            if key in k_lower:
                return val
        return k.title()  # fallback: title case asli

    df["kategori"] = df["kategori"].apply(_map)
    print("[normalize] Kategori dinormalisasi")
    print(df["kategori"].value_counts().to_string())
    return df


# ─────────────────────────────────────────────
# 5. FILTER KUALITAS DATA
# ─────────────────────────────────────────────

def filter_quality(df: pd.DataFrame, min_isi_len: int = 20) -> pd.DataFrame:
    before = len(df)
    df = df.dropna(subset=["isi", "judul"])
    df = df[df["isi"].str.len() >= min_isi_len]
    df = df.drop_duplicates(subset=["isi"])
    after = len(df)
    print(f"[filter] {before - after} baris dihapus → sisa {after} baris")
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# 6. BUAT INPUT TEKS UNTUK INFERENCE
# ─────────────────────────────────────────────

def build_inference_input(row: pd.Series) -> str:
    """
    Gabungkan judul + isi sebagai satu teks input untuk model.
    Format: 'Judul: <judul>\nLaporan: <isi>'
    """
    return f"Judul: {row['judul']}\nLaporan: {row['isi']}"


def add_inference_column(df: pd.DataFrame) -> pd.DataFrame:
    df["input_text"] = df.apply(build_inference_input, axis=1)
    print("[build] Kolom input_text ditambahkan")
    return df


# ─────────────────────────────────────────────
# 7. EXPORT
# ─────────────────────────────────────────────

def export_csv(df: pd.DataFrame, output_path: str):
    df.to_csv(output_path, index=False)
    print(f"[export] CSV disimpan → {output_path}")


def export_jsonl_for_finetuning(df: pd.DataFrame, output_path: str):
    """
    Format JSONL untuk fine-tuning / prompt-response pair.
    Setiap baris: {"prompt": ..., "label": {"kategori": ...}}
    Label lain (intent, location, dll) akan diisi setelah anotasi.
    """
    SYSTEM_PROMPT = (
        "Kamu adalah sistem klasifikasi laporan pengaduan masyarakat Indonesia. "
        "Ekstrak informasi berikut dari laporan:\n"
        "- intent: jenis pengaduan (complaint / question / suggestion / report)\n"
        "- kategori: bidang laporan\n"
        "- location: lokasi spesifik yang disebutkan (atau null)\n"
        "- object: objek/masalah utama yang dilaporkan\n"
        "- urgency: tingkat urgensi (low / medium / high)\n"
        "- sentiment: sentimen pelapor (positif / netral / negatif)\n"
        "- summary: ringkasan singkat 1 kalimat\n\n"
        "Jawab dalam format JSON saja."
    )

    records = []
    for _, row in df.iterrows():
        record = {
            "id": int(row["id"]),
            "system": SYSTEM_PROMPT,
            "user": row["input_text"],
            "label": {
                "kategori": row["kategori"],
                # Kolom ini diisi manual / semi-auto saat anotasi
                "intent": None,
                "location": None,
                "object": None,
                "urgency": None,
                "sentiment": None,
                "summary": None
            }
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[export] JSONL disimpan → {output_path} ({len(records)} baris)")


# ─────────────────────────────────────────────
# 8. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_preprocessing(
    input_path: str,
    output_dir: str = "data/processed"
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = load_data(input_path)
    df = drop_columns(df, cols=["url"])
    df = clean_columns(df)
    df = normalize_kategori(df)
    df = filter_quality(df, min_isi_len=20)
    df = add_inference_column(df)

    # Export
    export_csv(df, os.path.join(output_dir, "sp4n_clean.csv"))
    export_jsonl_for_finetuning(df, os.path.join(output_dir, "sp4n_for_finetuning.jsonl"))

    # Preview
    print("\n── Preview 3 baris pertama ──")
    for _, row in df.head(3).iterrows():
        print(f"\n[ID {row['id']}] {row['judul'][:60]}...")
        print(f"  kategori : {row['kategori']}")
        print(f"  input    : {row['input_text'][:120]}...")

    return df


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/sp4n_raw.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/processed"

    df = run_preprocessing(input_file, output_dir)
    print(f"\n[done] Preprocessing selesai. {len(df)} data siap digunakan.")