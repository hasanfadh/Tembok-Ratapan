import re

def is_valid_nik(nik):
    # Bersihkan dari spasi atau karakter pemisah sebelum validasi
    nik = re.sub(r'[\s.-]', '', nik)
    if len(nik) != 16 or not nik.isdigit():
        return False
    
    day = int(nik[6:8])
    month = int(nik[8:10])
    
    # NIK Wanita: tanggal + 40
    if not (1 <= day <= 31 or 41 <= day <= 71):
        return False
    
    if not (1 <= month <= 12):
        return False
    
    return True

def mask_pii(text):
    # 1. EMAIL (Sudah oke, tapi ditambahkan agar lebih robust)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

    # 2. PHONE (Perbaikan: Menangani spasi dan tanda hubung di tengah nomor)
    # Pattern ini menangkap 08xx atau +62xx yang mungkin dipisah spasi/strip
    phone_pattern = r'(\+62|08)(\s?[\d-]{2,4}){2,4}\b'
    
    # Kita gunakan fungsi substitusi agar tidak sembarang angka panjang kena hantam
    def phone_fixer(match):
        clean_num = re.sub(r'[\s-]', '', match.group(0))
        if 9 <= len(clean_num) <= 15:
            return '[PHONE]'
        return match.group(0)

    text = re.sub(phone_pattern, phone_fixer, text)

    # 3. FIX: Kode Area Telepon Rumah (Landline)
    landline_pattern = r'\b(021|022|0231|0251|0265|024|0271|0274|0281|031|0341|0361|0370|0380|061|0651|0711|0721|0751|0761|0778|0511|0542|0561|0411|0431|0911|0967)[\s-]?\d{3,4}[\s-]?\d{3,4}\b'
    text = re.sub(landline_pattern, '[PHONE]', text)

    # 4. NIK vs REKENING (Perbaikan: Deteksi angka yang mungkin dipisah spasi/strip)
    # Pattern ini mencari deretan angka 10-16 digit yang mungkin terpisah spasi
    potential_num_pattern = r'\b(?:\d[\s.-]?){10,16}\b'

    def replace_number(match):
        raw_val = match.group(0)
        # Bersihkan untuk cek panjang asli
        clean_num = re.sub(r'[\s.-]', '', raw_val)
        
        if len(clean_num) == 16 and is_valid_nik(clean_num):
            return '[NIK]'
        elif 10 <= len(clean_num) <= 16:
            return '[REKENING]'
        return raw_val
    
    text = re.sub(potential_num_pattern, replace_number, text)

    return text
