"""
inference_baseline.py
TembokRatapan — AI Pipeline
Anggota A (Hasan) | Fase 1

Tujuan:
- Test koneksi ke Ollama lokal
- Test structured prompt ke Qwen2.5:3b
- Parse output JSON dari model
- Baseline sebelum PII filter & fine-tuning diintegrasikan

Jalankan: python inference_baseline.py
Pastikan Ollama sudah aktif: ollama run qwen2.5:3b
"""

import requests
import json
import time

# ── Config ————————————————————————————————————————————————————————————————————

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

# ── Structured Prompt ─────────────────────────────────────────────────────────

def build_prompt(text: str) -> str:
    return f"""Kamu adalah sistem klasifikasi pengaduan masyarakat Kota Surabaya.
Analisis teks pengaduan berikut dan kembalikan hasil dalam format JSON.

Teks pengaduan:
\"\"\"{text}\"\"\"

Kembalikan HANYA JSON berikut tanpa penjelasan tambahan:
{{
  "intent": "<complaint | non-complaint>",
  "category": "<infrastruktur | kependudukan | perizinan | kebersihan | keamanan | lainnya>",
  "location": "<lokasi yang disebutkan, atau null>",
  "object": "<objek masalah utama>",
  "urgency": "<low | medium | high>",
  "sentiment": "<positive | neutral | negative>",
  "summary": "<ringkasan 1 kalimat dalam Bahasa Indonesia>"
}}"""


# ── Ollama Client ─────────────────────────────────────────────────────────────

def call_ollama(prompt: str, timeout: int = 60) -> str | None:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,   # rendah agar output konsisten
            "top_p": 0.9,
            "num_predict": 512,
        }
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.exceptions.ConnectionError:
        print("[ERROR] Ollama tidak bisa diakses. Pastikan Ollama sudah aktif.")
        return None
    except requests.exceptions.Timeout:
        print("[ERROR] Request timeout. Model terlalu lambat merespons.")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# ── JSON Parser ───────────────────────────────────────────────────────────────

def parse_json_output(raw: str) -> dict | None:
    """
    Ekstrak JSON dari response model.
    Model kadang menambahkan teks sebelum/sesudah JSON — ini handle itu.
    """
    try:
        # Coba parse langsung
        return json.loads(raw.strip())
    except json.JSONDecodeError:
        pass

    # Coba ekstrak blok JSON dari dalam teks
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

    print("[WARN] Gagal parse JSON dari output model.")
    print(f"[RAW OUTPUT]\n{raw}")
    return None


# ── Main Inference ────────────────────────────────────────────────────────────

def run_inference(text: str) -> dict | None:
    print(f"\n{'='*60}")
    print(f"[INPUT] {text}")
    print(f"{'='*60}")

    prompt = build_prompt(text)

    print("[INFO] Mengirim ke Ollama...")
    start = time.time()
    raw_output = call_ollama(prompt)
    elapsed = time.time() - start

    if raw_output is None:
        return None

    print(f"[INFO] Response diterima ({elapsed:.1f}s)")
    result = parse_json_output(raw_output)

    if result:
        print("[OUTPUT JSON]")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    return result


# ── Test Cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    # Kasus normal
    "Jalan di depan rumah saya di Rungkut Asri berlubang besar dan sudah "
    "berbahaya buat pengendara motor. Sudah lebih dari 3 bulan tidak diperbaiki.",

    # Kasus dengan urgensi tinggi
    "Banjir parah di kawasan Gubeng sejak kemarin malam, rumah warga sudah "
    "terendam. Mohon segera ditindaklanjuti!",

    # Kasus bukan pengaduan
    "Terima kasih atas pelayanan Dinas Kebersihan yang sudah cepat merespons "
    "laporan sampah minggu lalu.",

    # Kasus ambigu / singkat
    "Lampu jalan mati di Wonokromo.",
]


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("TembokRatapan — Inference Baseline Test")
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama URL: {OLLAMA_URL}\n")

    # Cek koneksi Ollama
    print("[INFO] Mengecek koneksi Ollama...")
    try:
        ping = requests.get("http://localhost:11434", timeout=5)
        print("[INFO] Ollama aktif ✅\n")
    except Exception:
        print("[ERROR] Ollama tidak aktif. Jalankan dulu: ollama run qwen2.5:3b")
        exit(1)

    results = []
    success = 0

    for i, text in enumerate(TEST_CASES):
        print(f"\n[TEST {i+1}/{len(TEST_CASES)}]")
        result = run_inference(text)
        if result:
            results.append(result)
            success += 1
        time.sleep(1)  # jeda antar request

    # Ringkasan
    print(f"\n{'='*60}")
    print(f"HASIL: {success}/{len(TEST_CASES)} test berhasil di-parse sebagai JSON")
    print(f"{'='*60}")

    if success == len(TEST_CASES):
        print("✅ Baseline inference berjalan normal. Siap lanjut ke PII filter.")
    else:
        print("⚠️  Ada test yang gagal. Cek raw output di atas untuk debug prompt.")