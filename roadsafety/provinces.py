"""Reference table of Thailand's 77 provinces and helpers to normalise messy names.

Region follows the six-region geographic grouping used by the Royal Society of
Thailand (North, Northeast, Central, East, West, South).
"""

from __future__ import annotations

import re

import pandas as pd

# (Thai name, English name, region)
PROVINCES: list[tuple[str, str, str]] = [
    # North
    ("เชียงใหม่", "Chiang Mai", "North"),
    ("เชียงราย", "Chiang Rai", "North"),
    ("ลำปาง", "Lampang", "North"),
    ("ลำพูน", "Lamphun", "North"),
    ("แม่ฮ่องสอน", "Mae Hong Son", "North"),
    ("น่าน", "Nan", "North"),
    ("พะเยา", "Phayao", "North"),
    ("แพร่", "Phrae", "North"),
    ("อุตรดิตถ์", "Uttaradit", "North"),
    # Central
    ("กรุงเทพมหานคร", "Bangkok", "Central"),
    ("อ่างทอง", "Ang Thong", "Central"),
    ("พระนครศรีอยุธยา", "Phra Nakhon Si Ayutthaya", "Central"),
    ("ชัยนาท", "Chai Nat", "Central"),
    ("กำแพงเพชร", "Kamphaeng Phet", "Central"),
    ("ลพบุรี", "Lop Buri", "Central"),
    ("นครนายก", "Nakhon Nayok", "Central"),
    ("นครปฐม", "Nakhon Pathom", "Central"),
    ("นครสวรรค์", "Nakhon Sawan", "Central"),
    ("นนทบุรี", "Nonthaburi", "Central"),
    ("ปทุมธานี", "Pathum Thani", "Central"),
    ("เพชรบูรณ์", "Phetchabun", "Central"),
    ("พิจิตร", "Phichit", "Central"),
    ("พิษณุโลก", "Phitsanulok", "Central"),
    ("สมุทรปราการ", "Samut Prakan", "Central"),
    ("สมุทรสาคร", "Samut Sakhon", "Central"),
    ("สมุทรสงคราม", "Samut Songkhram", "Central"),
    ("สระบุรี", "Saraburi", "Central"),
    ("สิงห์บุรี", "Sing Buri", "Central"),
    ("สุโขทัย", "Sukhothai", "Central"),
    ("สุพรรณบุรี", "Suphan Buri", "Central"),
    ("อุทัยธานี", "Uthai Thani", "Central"),
    # Northeast
    ("อำนาจเจริญ", "Amnat Charoen", "Northeast"),
    ("บึงกาฬ", "Bueng Kan", "Northeast"),
    ("บุรีรัมย์", "Buri Ram", "Northeast"),
    ("ชัยภูมิ", "Chaiyaphum", "Northeast"),
    ("กาฬสินธุ์", "Kalasin", "Northeast"),
    ("ขอนแก่น", "Khon Kaen", "Northeast"),
    ("เลย", "Loei", "Northeast"),
    ("มหาสารคาม", "Maha Sarakham", "Northeast"),
    ("มุกดาหาร", "Mukdahan", "Northeast"),
    ("นครพนม", "Nakhon Phanom", "Northeast"),
    ("นครราชสีมา", "Nakhon Ratchasima", "Northeast"),
    ("หนองบัวลำภู", "Nong Bua Lam Phu", "Northeast"),
    ("หนองคาย", "Nong Khai", "Northeast"),
    ("ร้อยเอ็ด", "Roi Et", "Northeast"),
    ("สกลนคร", "Sakon Nakhon", "Northeast"),
    ("ศรีสะเกษ", "Si Sa Ket", "Northeast"),
    ("สุรินทร์", "Surin", "Northeast"),
    ("อุบลราชธานี", "Ubon Ratchathani", "Northeast"),
    ("อุดรธานี", "Udon Thani", "Northeast"),
    ("ยโสธร", "Yasothon", "Northeast"),
    # East
    ("ฉะเชิงเทรา", "Chachoengsao", "East"),
    ("จันทบุรี", "Chanthaburi", "East"),
    ("ชลบุรี", "Chon Buri", "East"),
    ("ปราจีนบุรี", "Prachin Buri", "East"),
    ("ระยอง", "Rayong", "East"),
    ("สระแก้ว", "Sa Kaeo", "East"),
    ("ตราด", "Trat", "East"),
    # West
    ("กาญจนบุรี", "Kanchanaburi", "West"),
    ("เพชรบุรี", "Phetchaburi", "West"),
    ("ประจวบคีรีขันธ์", "Prachuap Khiri Khan", "West"),
    ("ราชบุรี", "Ratchaburi", "West"),
    ("ตาก", "Tak", "West"),
    # South
    ("ชุมพร", "Chumphon", "South"),
    ("กระบี่", "Krabi", "South"),
    ("นครศรีธรรมราช", "Nakhon Si Thammarat", "South"),
    ("นราธิวาส", "Narathiwat", "South"),
    ("ปัตตานี", "Pattani", "South"),
    ("พังงา", "Phang Nga", "South"),
    ("พัทลุง", "Phatthalung", "South"),
    ("ภูเก็ต", "Phuket", "South"),
    ("ระนอง", "Ranong", "South"),
    ("สตูล", "Satun", "South"),
    ("สงขลา", "Songkhla", "South"),
    ("สุราษฎร์ธานี", "Surat Thani", "South"),
    ("ตรัง", "Trang", "South"),
    ("ยะลา", "Yala", "South"),
]

PROVINCE_TABLE = pd.DataFrame(PROVINCES, columns=["province_th", "province_en", "region"])

# Nicknames, abbreviations and spelling variants seen in Thai government files.
_ALIASES: dict[str, str] = {
    "กทม": "กรุงเทพมหานคร",
    "กทม.": "กรุงเทพมหานคร",
    "กรุงเทพ": "กรุงเทพมหานคร",
    "กรุงเทพฯ": "กรุงเทพมหานคร",
    "bangkokmetropolis": "กรุงเทพมหานคร",
    "อยุธยา": "พระนครศรีอยุธยา",
    "ayutthaya": "พระนครศรีอยุธยา",
    "โคราช": "นครราชสีมา",
    "korat": "นครราชสีมา",
    "ประจวบ": "ประจวบคีรีขันธ์",
    "สุราษฎร์": "สุราษฎร์ธานี",
    "สุราษฏร์ธานี": "สุราษฎร์ธานี",  # ฏ typed instead of ฎ
    "นครศรีฯ": "นครศรีธรรมราช",
    "หนองบัวลําภู": "หนองบัวลำภู",  # sara am typed as nikhahit + sara aa
    "ลําปาง": "ลำปาง",
    "ลําพูน": "ลำพูน",
    "อํานาจเจริญ": "อำนาจเจริญ",
    "buriram": "บุรีรัมย์",
    "chonburi": "ชลบุรี",
    "lopburi": "ลพบุรี",
    "prachinburi": "ปราจีนบุรี",
    "singburi": "สิงห์บุรี",
    "suphanburi": "สุพรรณบุรี",
    "sisaket": "ศรีสะเกษ",
    "nongbualamphu": "หนองบัวลำภู",
    "phangnga": "พังงา",
}


def _key(name: str) -> str:
    """Lower-case, drop whitespace/punctuation and the 'province' prefix."""
    s = str(name).strip()
    s = re.sub(r"^(จังหวัด|จ\.)\s*", "", s)
    s = re.sub(r"\s*(province|changwat)$", "", s, flags=re.IGNORECASE)
    s = s.replace("ํา", "ำ")  # nikhahit + sara aa -> sara am
    return re.sub(r"[\s\-_.']", "", s.lower())


_LOOKUP: dict[str, str] = {}
for th, en, _ in PROVINCES:
    _LOOKUP[_key(th)] = th
    _LOOKUP[_key(en)] = th
for alias, th in _ALIASES.items():
    _LOOKUP[_key(alias)] = th


def normalize_province(series: pd.Series) -> pd.Series:
    """Map raw province strings to the canonical Thai name (NaN when unknown)."""
    uniques = series.dropna().astype(str).unique()
    mapping = {raw: _LOOKUP.get(_key(raw)) for raw in uniques}
    return series.map(mapping)
