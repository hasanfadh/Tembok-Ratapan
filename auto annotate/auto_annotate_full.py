"""
auto_annotate_full.py
Anotasi otomatis SEMUA label (intent, kategori, location, object, urgency, sentiment, summary)
dari dataset SP4N menggunakan Ollama lokal.
Output: 
  - sp4n_annotated.csv       (semua kolom + label)
  - sp4n_train.jsonl         (siap fine-tuning, format instruction tuning)

Cara pakai:
    python auto_annotate_full.py data/processed/sp4n_clean.csv
    python auto_annotate_full.py data/processed/sp4n_clean.csv --model qwen2.5:7b --output data/processed
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
DEFAULT_MODEL = "qwen2.5:7b"
TIMEOUT       = 45
RETRY_MAX     = 2

SYSTEM_INSTRUCTION = (
    "Kamu adalah sistem ekstraksi informasi laporan pengaduan masyarakat Indonesia. "
    "Jawab HANYA dengan JSON valid, tanpa penjelasan, tanpa markdown, tanpa backtick."
)

ANNOTATION_PROMPT = """\
Ekstrak informasi dari laporan pengaduan berikut. Jawab HANYA JSON, tidak ada teks lain.

Laporan:
Judul: {judul}
Isi: {isi}

Format JSON yang harus dikembalikan:
{{
  "intent": "complaint|question|suggestion|report",
  "kategori": "Infrastruktur|Sosial|Pendidikan|Kesehatan|Lingkungan|Administrasi|Lainnya",
  "location": "lokasi spesifik atau null",
  "object": "objek atau masalah utama yang dilaporkan",
  "urgency": "low|medium|high",
  "sentiment": "positif|netral|negatif",
  "summary": "ringkasan 1 kalimat maksimal 20 kata"
}}

Definisi urgency:
- high   : darurat, mengancam keselamatan, atau sudah lama tidak ditangani
- medium : mengganggu aktivitas, perlu ditangani tapi tidak darurat  
- low    : pertanyaan, saran, atau masalah ringan

Jawab hanya JSON:"""

TRAINING_INSTRUCTION = (
    "Kamu adalah sistem klasifikasi laporan pengaduan masyarakat Indonesia. "
    "Ekstrak informasi berikut dari laporan: intent, kategori, location, object, urgency, sentiment, dan summary. "
    "Jawab dalam format JSON."
)

VALID_INTENTS    = {"complaint", "question", "suggestion", "report"}
VALID_KATEGORI   = {"Infrastruktur", "Sosial", "Pendidikan", "Kesehatan", "Lingkungan", "Administrasi", "Lainnya"}
VALID_URGENCY    = {"low", "medium", "high"}
VALID_SENTIMENT  = {"positif", "netral", "negatif"}


# ─────────────────────────────────────────────
# INFERENCE + PARSE
# ─────────────────────────────────────────────

def call_ollama(prompt: str, model: str) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": SYSTEM_INSTRUCTION,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 200,
        }
    }
    for attempt in range(RETRY_MAX + 1):
        try:
            resp = requests.post(OLLAMA_URL, json=payload, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            if attempt < RETRY_MAX:
                time.sleep(3)
        except Exception:
            if attempt < RETRY_MAX:
                time.sleep(3)
    return ""


def extract_json(raw: str) -> dict:
    """Ekstrak JSON dari respons model, toleran terhadap noise."""
    # Coba langsung parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Coba cari blok JSON dengan regex
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    return {}


def validate_and_fix(data: dict, original_kategori: str = None) -> dict:
    """Validasi label, fallback ke nilai default jika tidak valid."""
    result = {}

    result["intent"]    = data.get("intent", "complaint").lower()
    if result["intent"] not in VALID_INTENTS:
        result["intent"] = "complaint"

    raw_kat = data.get("kategori", original_kategori or "Lainnya")
    result["kategori"]  = raw_kat if raw_kat in VALID_KATEGORI else (original_kategori or "Lainnya")

    result["location"]  = data.get("location") or None
    if result["location"] and str(result["location"]).lower() in ("null", "none", "-", ""):
        result["location"] = None

    result["object"]    = data.get("object") or data.get("objek") or ""

    result["urgency"]   = data.get("urgency", "medium").lower()
    if result["urgency"] not in VALID_URGENCY:
        result["urgency"] = "medium"

    result["sentiment"] = data.get("sentiment", "netral").lower()
    if result["sentiment"] not in VALID_SENTIMENT:
        result["sentiment"] = "netral"

    result["summary"]   = data.get("summary") or data.get("ringkasan") or ""

    return result


def annotate_row(judul: str, isi: str, model: str, original_kategori: str = None) -> dict:
    prompt = ANNOTATION_PROMPT.format(
        judul=str(judul)[:300],
        isi=str(isi)[:700]
    )
    raw    = call_ollama(prompt, model)
    data   = extract_json(raw)
    result = validate_and_fix(data, original_kategori)
    return result


# ─────────────────────────────────────────────
# BATCH ANNOTATION
# ─────────────────────────────────────────────

LABEL_COLS = ["intent", "kategori", "location", "object", "urgency", "sentiment", "summary"]

def is_complete(row: pd.Series) -> bool:
    """Cek apakah baris sudah punya semua label valid."""
    for col in LABEL_COLS:
        val = row.get(col)
        if val is None or (isinstance(val, float)) or str(val).strip() == "":
            return False
    return True


def run_annotation(df: pd.DataFrame, model: str) -> pd.DataFrame:
    # Inisialisasi kolom kalau belum ada
    for col in LABEL_COLS:
        if col not in df.columns:
            df[col] = None

    todo_idx = [i for i, row in df.iterrows() if not is_complete(row)]

    print(f"[annotate] Total baris : {len(df)}")
    print(f"[annotate] Perlu label : {len(todo_idx)}")
    print(f"[annotate] Skip (done) : {len(df) - len(todo_idx)}")
    print(f"[annotate] Model       : {model}\n")

    if not todo_idx:
        print("[annotate] Semua baris sudah teranotasi!")
        return df

    errors = 0
    start  = time.time()

    for i, idx in enumerate(tqdm(todo_idx, desc="Annotating", unit="row")):
        row    = df.loc[idx]
        labels = annotate_row(
            judul=row.get("judul", ""),
            isi=row.get("isi", ""),
            model=model,
            original_kategori=row.get("kategori")
        )

        for col, val in labels.items():
            df.at[idx, col] = val

        if not labels.get("summary"):
            errors += 1

        # Progress log tiap 100 baris
        if (i + 1) % 100 == 0:
            elapsed = time.time() - start
            rate    = (i + 1) / elapsed
            eta     = (len(todo_idx) - i - 1) / rate if rate > 0 else 0
            tqdm.write(f"  >> {i+1}/{len(todo_idx)} | {rate:.1f} row/s | ETA {eta/60:.1f} menit")

    elapsed = time.time() - start
    print(f"\n[done] Selesai dalam {elapsed/60:.1f} menit")
    print(f"[done] Baris bermasalah: {errors}")

    return df


# ─────────────────────────────────────────────
# CONVERT KE FORMAT TRAINING (JSONL)
# ─────────────────────────────────────────────

def build_output_json(row: pd.Series) -> str:
    """Format output JSON untuk training."""
    output = {
        "intent":    row.get("intent", ""),
        "kategori":  row.get("kategori", ""),
        "location":  row.get("location") if pd.notna(row.get("location")) else None,
        "object":    row.get("object", ""),
        "urgency":   row.get("urgency", ""),
        "sentiment": row.get("sentiment", ""),
        "summary":   row.get("summary", "")
    }
    return json.dumps(output, ensure_ascii=False)


def export_training_jsonl(df: pd.DataFrame, output_path: str):
    records = []
    skipped = 0

    for _, row in df.iterrows():
        # Skip baris yang labelnya tidak lengkap
        if not is_complete(row):
            skipped += 1
            continue

        record = {
            "instruction": TRAINING_INSTRUCTION,
            "input": f"Judul: {row['judul']}\nLaporan: {row['isi']}",
            "output": build_output_json(row)
        }
        records.append(record)

    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[export] JSONL training → {output_path}")
    print(f"[export] {len(records)} baris ditulis, {skipped} baris diskip")


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_report(df: pd.DataFrame):
    print("\n" + "="*50)
    print("DISTRIBUSI LABEL")
    print("="*50)
    for col in ["intent", "kategori", "urgency", "sentiment"]:
        if col in df.columns:
            print(f"\n── {col.upper()} ──")
            print(df[col].value_counts().to_string())

    print("\n── Contoh hasil anotasi ──")
    sample = df.dropna(subset=["summary"]).head(3)
    for _, row in sample.iterrows():
        print(f"\n[ID {row['id']}] {str(row['judul'])[:60]}")
        print(f"  intent   : {row.get('intent')}")
        print(f"  urgency  : {row.get('urgency')}")
        print(f"  sentiment: {row.get('sentiment')}")
        print(f"  summary  : {row.get('summary')}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input",           help="Path CSV input (sp4n_clean.csv)")
    parser.add_argument("--model", "-m",   default=DEFAULT_MODEL)
    parser.add_argument("--output", "-o",  default=None, help="Folder output (default: folder input)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output) if args.output else input_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_out  = output_dir / "sp4n_annotated.csv"
    jsonl_out = output_dir / "sp4n_train.jsonl"

    # Cek Ollama
    try:
        requests.get("http://localhost:11434", timeout=5)
        print("[check] Ollama aktif ✓\n")
    except Exception:
        print("[ERROR] Ollama tidak berjalan!")
        return

    # Load — resume dari annotated kalau sudah ada
    if csv_out.exists():
        df = pd.read_csv(csv_out)
        print(f"[load] Resume dari {csv_out} ({len(df)} baris)")
    else:
        df = pd.read_csv(input_path)
        print(f"[load] {len(df)} baris dari {input_path}")

    # Annotate
    df = run_annotation(df, model=args.model)

    # Save CSV
    df.to_csv(csv_out, index=False)
    print(f"[save] CSV → {csv_out}")

    # Export JSONL training
    export_training_jsonl(df, str(jsonl_out))

    # Report
    print_report(df)
    print(f"\n[DONE] File siap fine-tuning: {jsonl_out}")


if __name__ == "__main__":
    main()
