"""
SP4N LAPOR - Web Scraper untuk Fine-tuning Dataset
===================================================
Scrapes halaman Kisah Sukses lalu masuk ke detail setiap aduan.
Data yang diambil: ID, Judul, Isi, Kategori Laporan

Fitur:
  - Loop otomatis sampai halaman terakhir (tidak perlu tahu total halaman)
  - Resume otomatis kalau scraping terputus di tengah jalan
  - Simpan progress ke checkpoint setiap selesai 1 laporan

Install dulu:
    pip install requests beautifulsoup4 pandas

Jalankan:
    python lapor_scraper.py

Untuk mulai dari awal (hapus checkpoint):
    python lapor_scraper.py --reset
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import json
import logging
import os
import sys

# ─── Konfigurasi ────────────────────────────────────────────────────────────
BASE_URL        = "https://www.lapor.go.id"
LIST_URL        = BASE_URL + "/kisah-sukses?page={page}"
OUTPUT_CSV      = "lapor_dataset.csv"
OUTPUT_JSON     = "lapor_dataset.json"
CHECKPOINT_FILE = "checkpoint.json"   # menyimpan progress scraping

DELAY_MIN = 2.0   # detik — jangan dikurangi agar tidak diblokir
DELAY_MAX = 4.0

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8",
}

# ─── Setup logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("scraper.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─── Session ─────────────────────────────────────────────────────────────────
session = requests.Session()
session.headers.update(HEADERS)

# Cookie sesi login LAPOR
# Kalau expired (redirect ke landing page), ambil ulang dari:
#   browser -> F12 -> Console -> ketik: document.cookie
COOKIES = {
    "lapor": "eyJpdiI6InZ1QXVmY2s5XC9nMU96Wk1KSHRLMENnPT0iLCJ2YWx1ZSI6ImVFdms1SW5KVUJWbmliXC80cXFLRXlhdUJLdjJUblJPQlNROThEeUlZU29obm5TVGp1NHVPK3BYK1hlWHdHamRxZFNCUm9FbkFhbDJEblRtZEJZQWc0dz09IiwibWFjIjoiOWFiNTg3MTNiNzEzY2JiZjcwYmEzMWYxNmFhYWYwNTUzMjI1MTRkZmVlYWU2ZTY1YTNiMDExYjdjYTIzYjUxYSJ9",
    "_gid": "GA1.3.394002479.1774970210",
}
for name, value in COOKIES.items():
    session.cookies.set(name, value, domain="lapor.go.id", path="/")

# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT  —  simpan & load progress
# ═════════════════════════════════════════════════════════════════════════════

def load_checkpoint() -> tuple[set, list]:
    """
    Load progress dari file checkpoint.
    Return: (seen_ids, all_data)
    """
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
            cp = json.load(f)
        seen_ids = set(cp.get("seen_ids", []))
        all_data = cp.get("all_data", [])
        log.info(f"Resume dari checkpoint: {len(all_data)} laporan sudah tersimpan.")
        return seen_ids, all_data
    return set(), []


def save_checkpoint(seen_ids: set, all_data: list):
    """Simpan progress ke file checkpoint."""
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {"seen_ids": list(seen_ids), "all_data": all_data},
            f, ensure_ascii=False
        )


def reset_checkpoint():
    """Hapus checkpoint untuk mulai dari awal."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        log.info("Checkpoint dihapus. Mulai dari awal.")


# ═════════════════════════════════════════════════════════════════════════════
# FETCH & PARSE
# ═════════════════════════════════════════════════════════════════════════════

def get_page(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch URL dan kembalikan BeautifulSoup. Retry otomatis jika gagal."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=30)  # naikkan ke 30 untuk server LAPOR yang lambat
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.Timeout:
            # Timeout: retry cepat, tidak perlu nunggu lama
            log.warning(f"Timeout attempt {attempt}/{retries}: {url}")
            time.sleep(2)
        except requests.RequestException as e:
            # Error lain (koneksi putus, dll): nunggu lebih lama
            log.warning(f"Attempt {attempt}/{retries} gagal ({url}): {e}")
            time.sleep(DELAY_MAX * attempt)
    log.error(f"Semua percobaan gagal: {url}")
    return None


def get_report_links(page_num: int) -> list[str] | None:
    """
    Ambil semua link detail aduan dari satu halaman list.
    Return None  -> halaman gagal di-fetch (koneksi error)
    Return []    -> halaman berhasil di-fetch tapi kosong (sudah mentok)
    Return [...] -> daftar URL detail
    """
    url = LIST_URL.format(page=page_num)
    log.info(f"── Halaman {page_num}: {url}")
    soup = get_page(url)

    if soup is None:
        return None   # gagal fetch, bukan berarti mentok

    links = []
    for card in soup.select("div.post-title.h4.mg-0"):
        a_tag = card.find("a", href=True)
        if a_tag:
            href = a_tag["href"]
            if not href.startswith("http"):
                href = BASE_URL + href
            links.append(href)

    log.info(f"   -> {len(links)} laporan ditemukan")
    return links


def parse_detail(url: str) -> dict | None:
    """
    Ambil data detail satu aduan.
    Elemen HTML:
      ID       : <p>Tracking ID : <b>#XXXXX</b></p>
      Judul    : <h1 class="complaint-title">
      Isi      : <div class="complaint-excerpt"> <p>
      Kategori : <a href=".../laporan/kategori/...">
    """
    soup = get_page(url)
    if not soup:
        return None

    # ID
    tracking_id = None
    for p in soup.find_all("p"):
        if "Tracking ID" in p.get_text():
            b_tag = p.find("b")
            if b_tag:
                tracking_id = b_tag.get_text(strip=True).replace("#", "")
            break

    # Judul
    title_tag = soup.find("h1", class_="complaint-title")
    judul = title_tag.get_text(strip=True) if title_tag else None

    # Isi
    excerpt_div = soup.find("div", class_="complaint-excerpt")
    isi = None
    if excerpt_div:
        p_tag = excerpt_div.find("p")
        isi = p_tag.get_text(strip=True) if p_tag else excerpt_div.get_text(strip=True)

    # Kategori
    kategori = None
    kat_a = soup.select_one('a[href*="/laporan/kategori/"]')
    if kat_a:
        kategori = kat_a.get_text(strip=True)

    if not all([tracking_id, judul, isi]):
        log.warning(f"Data tidak lengkap: {url}")

    return {
        "id"      : tracking_id,
        "judul"   : judul,
        "isi"     : isi,
        "kategori": kategori,
        "url"     : url,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LOOP  —  auto-stop + resume
# ═════════════════════════════════════════════════════════════════════════════

def scrape():
    seen_ids, all_data = load_checkpoint()

    page = 1
    empty_streak = 0       # hitung halaman kosong berturut-turut
    MAX_EMPTY_STREAK = 3   # stop kalau 3 halaman kosong berturut-turut

    while True:
        links = get_report_links(page)

        # ── Gagal fetch (koneksi error, bukan mentok) ──
        if links is None:
            log.warning("Gagal fetch halaman list. Coba lagi dalam 10 detik...")
            time.sleep(10)
            continue   # coba halaman yang sama lagi

        # ── Halaman kosong → kemungkinan sudah mentok ──
        if not links:
            empty_streak += 1
            log.info(f"Halaman {page} kosong ({empty_streak}/{MAX_EMPTY_STREAK}).")
            if empty_streak >= MAX_EMPTY_STREAK:
                log.info("Sudah mentok. Scraping selesai!")
                break
            page += 1
            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))
            continue

        empty_streak = 0   # reset kalau halaman ada isinya

        # ── Loop tiap link detail di halaman ini ──
        for link in links:
            log.info(f"   Detail: {link}")
            data = parse_detail(link)

            if data is None:
                log.warning("      x Gagal parse, skip.")
            elif data["id"] in seen_ids:
                log.info(f"      x Duplikat ID={data['id']}, skip.")
            else:
                all_data.append(data)
                seen_ids.add(data["id"])
                log.info(
                    f"      v ID={data['id']} | "
                    f"{(data['judul'] or '')[:45]} | {data['kategori']}"
                )
                # Simpan checkpoint setiap 1 laporan berhasil
                save_checkpoint(seen_ids, all_data)

            time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

        log.info(f"   Total terkumpul sejauh ini: {len(all_data)} laporan\n")

        page += 1
        time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))

    return all_data


# ═════════════════════════════════════════════════════════════════════════════
# SAVE FINAL
# ═════════════════════════════════════════════════════════════════════════════

def save_final(data: list[dict]):
    """Simpan hasil akhir ke CSV dan JSON."""
    if not data:
        log.warning("Tidak ada data.")
        return

    df = pd.DataFrame(data, columns=["id", "judul", "isi", "kategori", "url"])
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    log.info(f"CSV tersimpan -> {OUTPUT_CSV} ({len(df)} baris)")

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"JSON tersimpan -> {OUTPUT_JSON}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if "--reset" in sys.argv:
        reset_checkpoint()

    log.info("=== SP4N LAPOR Scraper dimulai ===")
    results = scrape()
    save_final(results)
    log.info(f"=== Selesai. Total {len(results)} laporan berhasil diambil. ===")