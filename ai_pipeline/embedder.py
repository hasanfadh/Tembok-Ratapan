"""
embedder.py

Fungsi:
- Generate embedding dari teks aduan (384 dimensi)
- Simpan embedding ke ChromaDB
- Cek duplikat via cosine similarity (threshold 0.85)

Jalankan: python embedder.py
"""

from sentence_transformers import SentenceTransformer  # type: ignore
import chromadb # type: ignore
from chromadb.config import Settings # type: ignore
import numpy as np
import os

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
VECTOR_STORE_PATH = "./vector_store"
COLLECTION_NAME = "complaints_vectors"
DUPLICATE_THRESHOLD = 0.6

# ── Init Model & ChromaDB ─────────────────────────────────────────────────────

print("[INFO] Loading embedding model...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("[INFO] Model loaded ✅")

chroma_client = chromadb.PersistentClient(
    path=VECTOR_STORE_PATH,
    settings=Settings(anonymized_telemetry=False)
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# ── Core Functions ────────────────────────────────────────────────────────────

def generate_embedding(text: str) -> list[float]:
    """Ubah teks jadi vektor 384 dimensi."""
    embedding = embedding_model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def check_duplicate(text: str, complaint_id: str = None) -> dict:
    """
    Cek apakah aduan merupakan duplikat.
    Return: {is_duplicate, similarity_score, duplicate_of}
    """
    if collection.count() == 0:
        return {"is_duplicate": False, "similarity_score": 0.0, "duplicate_of": None}

    embedding = generate_embedding(text)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=1,
        include=["distances", "metadatas"]
    )

    if not results["distances"][0]:
        return {"is_duplicate": False, "similarity_score": 0.0, "duplicate_of": None}

    # ChromaDB cosine distance: 0 = identik, 1 = sangat berbeda
    # Konversi ke similarity: similarity = 1 - distance
    distance = results["distances"][0][0]
    similarity = round(1 - distance, 4)
    nearest_id = results["metadatas"][0][0].get("complaint_id")

    # Jangan flag duplikat dengan dirinya sendiri
    if complaint_id and nearest_id == complaint_id:
        return {"is_duplicate": False, "similarity_score": similarity, "duplicate_of": None}

    is_duplicate = similarity >= DUPLICATE_THRESHOLD

    return {
        "is_duplicate": is_duplicate,
        "similarity_score": similarity,
        "duplicate_of": nearest_id if is_duplicate else None
    }


def add_to_vector_store(complaint_id: str, cleaned_text: str, category: str) -> None:
    """Simpan embedding aduan ke ChromaDB."""
    embedding = generate_embedding(cleaned_text)

    collection.add(
        ids=[complaint_id],
        embeddings=[embedding],
        documents=[cleaned_text],
        metadatas=[{
            "complaint_id": complaint_id,
            "category": category,
        }]
    )
    print(f"[INFO] Embedding disimpan: {complaint_id}")


# ── Test ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\nTembokRatapan — Embedder Test")
    print(f"Model   : {MODEL_NAME}")
    print(f"Store   : {VECTOR_STORE_PATH}")
    print(f"Threshold: {DUPLICATE_THRESHOLD}\n")

    # Simulasi aduan yang masuk
    aduan_list = [
        {
            "id": "aduan-001",
            "text": "jalan rungkut asri berlubang besar sudah 3 bulan tidak diperbaiki",
            "category": "infrastruktur"
        },
        {
            "id": "aduan-002",
            "text": "banjir parah di kawasan gubeng rumah warga sudah terendam",
            "category": "infrastruktur"
        },
        {
            "id": "aduan-003",
            "text": "jalan di rungkut asri rusak parah berlubang berbahaya buat motor",  # duplikat aduan-001
            "category": "infrastruktur"
        },
        {
            "id": "aduan-004",
            "text": "lampu jalan mati di wonokromo sudah gelap banget",
            "category": "infrastruktur"
        },
    ]

    print("=" * 60)
    for aduan in aduan_list:
        print(f"\n[ADUAN] {aduan['id']}: {aduan['text']}")

        # Cek duplikat sebelum simpan
        dup_result = check_duplicate(aduan["text"], aduan["id"])

        if dup_result["is_duplicate"]:
            print(f"[DUPLIKAT] similarity={dup_result['similarity_score']} "
                  f"→ duplikat dari {dup_result['duplicate_of']}")
        else:
            print(f"[BARU] similarity={dup_result['similarity_score']} → disimpan ke ChromaDB")
            add_to_vector_store(aduan["id"], aduan["text"], aduan["category"])

    print(f"\n{'='*60}")
    print(f"Total embedding tersimpan: {collection.count()}")
    print("✅ Embedder siap digunakan.")