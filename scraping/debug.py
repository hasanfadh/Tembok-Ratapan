import requests
from bs4 import BeautifulSoup

session = requests.Session()
session.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"})
session.cookies.set("lapor", "eyJpdiI6InZ1QXVmY2s5XC9nMU96Wk1KSHRLMENnPT0iLCJ2YWx1ZSI6ImVFdms1SW5KVUJWbmliXC80cXFLRXlhdUJLdjJUblJPQlNROThEeUlZU29obm5TVGp1NHVPK3BYK1hlWHdHamRxZFNCUm9FbkFhbDJEblRtZEJZQWc0dz09IiwibWFjIjoiOWFiNTg3MTNiNzEzY2JiZjcwYmEzMWYxNmFhYWYwNTUzMjI1MTRkZmVlYWU2ZTY1YTNiMDExYjdjYTIzYjUxYSJ9", domain="lapor.go.id", path="/")

url = "https://www.lapor.go.id/laporan/detil/proyek-jalan-tol-pejagan-pemalang-seksi-3"
resp = session.get(url, timeout=30)
soup = BeautifulSoup(resp.text, "html.parser")

print("=== TITLE ===")
print(soup.title.string)

print("\n=== TRACKING ID ===")
for p in soup.find_all("p"):
    if "Tracking" in p.get_text():
        print(p)

print("\n=== JUDUL ===")
print(soup.find("h1", class_="complaint-title"))