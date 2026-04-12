"""
generate_modelfile.py
Generate Modelfile untuk mendaftarkan model fine-tuned ke Ollama.

Cara pakai:
    python generate_modelfile.py path/ke/file.gguf
    python generate_modelfile.py C:/Users/hp/Downloads/sp4n-qwen2.5-7b.Q4_K_M.gguf
"""

import argparse
import subprocess
import sys
from pathlib import Path

SYSTEM_PROMPT = """Kamu adalah sistem klasifikasi laporan pengaduan masyarakat Indonesia (SP4N-LAPOR).
Tugasmu adalah mengekstrak informasi terstruktur dari laporan pengaduan dan mengembalikannya dalam format JSON.

Field yang harus diekstrak:
- intent: complaint | question | suggestion | report
- kategori: Infrastruktur | Sosial | Pendidikan | Kesehatan | Lingkungan | Administrasi | Lainnya
- location: lokasi spesifik yang disebutkan, atau null
- object: objek atau masalah utama yang dilaporkan
- urgency: low | medium | high
- sentiment: positif | netral | negatif
- summary: ringkasan 1 kalimat maksimal 20 kata

Selalu jawab dalam format JSON valid tanpa penjelasan tambahan."""

MODELFILE_TEMPLATE = """FROM {gguf_path}

SYSTEM \"\"\"{system_prompt}\"\"\"

PARAMETER temperature 0.1
PARAMETER num_predict 256
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
"""


def generate(gguf_path: str, model_name: str, output_dir: str):
    gguf = Path(gguf_path).resolve()
    if not gguf.exists():
        print(f"[ERROR] File tidak ditemukan: {gguf}")
        sys.exit(1)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modelfile_path = out_dir / "Modelfile"
    content = MODELFILE_TEMPLATE.format(
        gguf_path=str(gguf).replace("\\", "/"),
        system_prompt=SYSTEM_PROMPT
    )

    with open(modelfile_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Modelfile dibuat → {modelfile_path}")
    print(f"\nLangkah selanjutnya — jalankan di PowerShell:\n")
    print(f"  ollama create {model_name} -f \"{modelfile_path}\"")
    print(f"  ollama run {model_name}")
    print(f"\nAtau jalankan otomatis sekarang? (y/n): ", end="")

    ans = input().strip().lower()
    if ans == "y":
        print(f"\n[run] ollama create {model_name} -f {modelfile_path}")
        result = subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path)],
            capture_output=False
        )
        if result.returncode == 0:
            print(f"\n[OK] Model '{model_name}' berhasil didaftarkan ke Ollama!")
            print(f"Jalankan dengan: ollama run {model_name}")
        else:
            print(f"\n[ERROR] Gagal. Coba jalankan manual perintah di atas.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gguf",              help="Path ke file .gguf hasil export Colab")
    parser.add_argument("--name", "-n",      default="sp4n-lapor", help="Nama model di Ollama")
    parser.add_argument("--output", "-o",    default=".",           help="Folder output Modelfile")
    args = parser.parse_args()

    generate(args.gguf, args.name, args.output)


if __name__ == "__main__":
    main()