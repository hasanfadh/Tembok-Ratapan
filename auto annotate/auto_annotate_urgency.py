"""
auto_annotate_urgency.py
Anotasi otomatis kolom 'urgency' untuk dataset SP4N menggunakan Ollama lokal.
Output: CSV dengan kolom urgency terisi (low / medium / high)

Cara pakai:
    python auto_annotate_urgency.py data/processed/sp4n_clean.csv
    python auto_annotate_urgency.py data/processed/sp4n_clean.csv --model qwen2.5:7b --batch 10
"""

import argparse
import json
import re
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen2.5:7b"   # ganti sesuai model yang kamu pull
TIMEOUT       = 30             # detik per request
RETRY_MAX     = 2              # retry jika gagal

URGENCY_PROMPT_TEMPLATE = """\
Kamu adalah sistem klasifikasi laporan pengaduan masyarakat Indonesia.
Tugasmu: tentukan tingkat urgensi laporan berikut.

Definisi:
- high   : mengancam keselamatan jiwa, darurat, atau sudah berlangsung lama dan berdampak luas
- medium : mengganggu aktivitas warga, perlu segera tapi tidak darurat
- low    : pertanyaan umum, saran, atau masalah ringan

Laporan:
Judul: {judul}
Isi: {isi}

Jawab HANYA dengan satu kata: high, medium, atau low.
Jangan tambahkan penjelasan apapun."""


# ─────────────────────────────────────────────
# INFERENCE SATU BARIS
# ─────────────────────────────────────────────

def infer_urgency(judul: str, isi: str, model: str) -> str:
    prompt = URGENCY_PROMPT_TEMPLATE.format(
        judul=judul[:300],
        isi=isi[:600]   # trim panjang agar cepat
    )

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 10,    # cukup untuk 1 kata
        }
    }

    for attempt in range(RETRY_MAX + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            raw = resp.json().get("response", "").strip().lower()

            # Ekstrak label dari respons
            if "high" in raw:
                return "high"
            elif "medium" in raw:
                return "medium"
            elif "low" in raw:
                return "low"
            else:
                return "unknown"

        except requests.exceptions.Timeout:
            if attempt < RETRY_MAX:
                time.sleep(2)
            else:
                return "error_timeout"
        except Exception as e:
            if attempt < RETRY_MAX:
                time.sleep(2)
            else:
                return f"error"

    return "error"


# ─────────────────────────────────────────────
# BATCH ANNOTATION
# ─────────────────────────────────────────────

def annotate(df: pd.DataFrame, model: str, resume: bool = True) -> pd.DataFrame:
    if "urgency" not in df.columns:
        df["urgency"] = None

    # Resume: skip baris yang sudah punya label valid
    valid_labels = {"high", "medium", "low"}
    todo_idx = df[~df["urgency"].isin(valid_labels)].index.tolist()

    print(f"[annotate] Total baris : {len(df)}")
    print(f"[annotate] Perlu label : {len(todo_idx)}")
    print(f"[annotate] Model       : {model}")
    print(f"[annotate] Ollama URL  : {OLLAMA_URL}\n")

    if not todo_idx:
        print("[annotate] Semua baris sudah teranotasi!")
        return df

    errors = 0
    start  = time.time()

    for i, idx in enumerate(tqdm(todo_idx, desc="Annotating", unit="row")):
        row = df.loc[idx]
        label = infer_urgency(
            judul=str(row.get("judul", "")),
            isi=str(row.get("isi", "")),
            model=model
        )
        df.at[idx, "urgency"] = label

        if label.startswith("error"):
            errors += 1

        # Auto-save setiap 100 baris
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate    = (i + 1) / elapsed
            eta     = (len(todo_idx) - i - 1) / rate if rate > 0 else 0
            tqdm.write(f"  >> {i+1}/{len(todo_idx)} | {rate:.1f} row/s | ETA {eta/60:.1f} menit")

    elapsed = time.time() - start
    print(f"\n[done] Selesai dalam {elapsed/60:.1f} menit")
    print(f"[done] Error: {errors} baris")

    return df


# ─────────────────────────────────────────────
# QUALITY CHECK REPORT
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    print("\n── Distribusi Urgency ──")
    print(df["urgency"].value_counts().to_string())
    pct = df["urgency"].value_counts(normalize=True) * 100
    print("\n── Persentase ──")
    print(pct.round(1).to_string())

    # Contoh tiap label
    for label in ["high", "medium", "low"]:
        sample = df[df["urgency"] == label].head(2)
        print(f"\n── Contoh [{label.upper()}] ──")
        for _, row in sample.iterrows():
            print(f"  [{row['id']}] {str(row['judul'])[:80]}")
            print(f"       {str(row['isi'])[:100]}...")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Auto-annotate urgency via Ollama")
    parser.add_argument("input",              help="Path ke CSV input (sp4n_clean.csv)")
    parser.add_argument("--model", "-m",      default=DEFAULT_MODEL, help="Nama model Ollama")
    parser.add_argument("--output", "-o",     default=None,          help="Path output CSV (default: overwrite input)")
    parser.add_argument("--no-resume",        action="store_true",   help="Mulai dari awal, abaikan label lama")
    args = parser.parse_args()

    input_path  = args.input
    output_path = args.output or input_path  # default overwrite
    model       = args.model
    resume      = not args.no_resume

    # Cek Ollama aktif
    try:
        r = requests.get("http://localhost:11434", timeout=5)
        print("[check] Ollama aktif ✓")
    except Exception:
        print("[ERROR] Ollama tidak berjalan! Jalankan dulu: ollama serve")
        return

    # Load
    df = pd.read_csv(input_path)
    print(f"[load] {len(df)} baris dari {input_path}")

    # Annotate
    df = annotate(df, model=model, resume=resume)

    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[save] Hasil disimpan → {output_path}")

    # Report
    print_report(df)


if __name__ == "__main__":
    main()