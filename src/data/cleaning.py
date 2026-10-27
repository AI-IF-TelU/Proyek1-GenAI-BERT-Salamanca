"""
Pembersihan teks CLICK-ID dan pemecahan artikel menjadi kalimat.

Artefak scraping yang ditangani (dilihat langsung dari data raw):
    - paragraf disambung dengan koma:            "...lalu.,Mantan kades ..."
    - entitas ber-hyperlink hilang jadi koma:    "Presiden ketiga Indonesia , di Jakarta"
    - kalimat menempel tanpa spasi:              "ujarnya.Bobby mengaku ..."
    - dateline / nama portal di awal artikel:    "REPUBLIKA.CO.ID, JAKARTA -- ", "Medan - ", " , - "
    - ekor detik (kode editor + tag):            "(iwd/iwd) korupsi dana desa"
    - boilerplate portal:                        "loading...", "#GrowFearless with FIMELA", "Baca juga: ..."
    - HTML, URL, \\xa0, karakter rusak (\\ufffd), emoji, aksara non-latin
"""

import html
import re
import unicodedata

HTML_TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"(https?://\S+|www\.\S+|pic\.twitter\.com/\S+)", re.IGNORECASE)
EMAIL = re.compile(r"\S+@\S+\.\S+")
MENTION_HASHTAG = re.compile(r"[@#]\w+")

QUOTE_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'", "`": "'",
    "–": "-", "—": "-", "−": "-",
    "…": "...",
    "�": '"',   # di data ini karakter rusak hampir selalu bekas tanda kutip
    "\xa0": " ", "​": "", "﻿": "",
})

# Boilerplate yang dibuang dari teks artikel sebelum dipecah per kalimat
BOILERPLATE = [
    re.compile(r"loading\.\.\.", re.IGNORECASE),
    re.compile(r"#GrowFearless with \w+", re.IGNORECASE),
    re.compile(r"The post\s*,?\s*\.?\s*$"),
    re.compile(r"Disadur dari .*$"),
    # ekor detik: "(iwd/iwd) tag tag tag" di akhir artikel
    re.compile(r"\(\s*[a-z]{2,5}\s*/\s*[a-z]{2,5}\s*\)[^()]*$"),
]

# Dateline di awal artikel
AGENCY_DATELINE = re.compile(
    r"^(?:REPUBLIKA\.CO\.ID|KOMPAS\.com|TEMPO\.CO|Liputan6\.com|Okezone|SINDOnews|TRIBUNNEWS\.COM"
    r"|Bola\.com|Merdeka\.com|Fimela\.com|KapanLagi\.com|WowKeren\.com)"
    r"\s*,\s*(?:[A-Za-z][A-Za-z .]{0,40}?)?\s*-{1,2}\s*",
    re.IGNORECASE,
)
CITY_DATELINE = re.compile(r"^[A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+){0,2}\s+-{1,2}\s+")
LEADING_JUNK = re.compile(r"^[\s,.\-:;]+")

# Kalimat yang mengandung frasa ini dibuang utuh (navigasi portal, bukan isi berita)
DROP_SENTENCE = re.compile(
    r"\b(baca juga|baca selengkapnya|lihat juga|simak juga|simak video|tonton juga|"
    r"saksikan video|editor\s*:|penulis\s*:|reporter\s*:|advertisement|scroll to)\b"
)

# Gelar/singkatan yang titiknya bukan akhir kalimat
ABBREV_END = re.compile(
    r"(?:\b[A-Z]|\b(?:Dr|Drs|Dra|Prof|Ir|H|Hj|KH|St|No|Jl|Jln|Tbk|Sdr|Bpk|Ny|Tn|Kol|Brigjen|"
    r"Mayjen|Letjen|Jend|Kapt|Kompol|AKBP|AKP|Ipda|Iptu|Aiptu|Bripka|dkk|dll|dsb|Rp|Pdt|Sel|Kab|Kec|Kel))\.$"
)

SENTENCE_SPLIT = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"']))\s+(?=[\"'(]?[A-Z0-9])")

# Setelah lowercase, hanya karakter ini yang dipertahankan
DISALLOWED = re.compile(r"[^a-z0-9\s.,!?:;'\"()%/&+\-]")
MULTI_SPACE = re.compile(r"\s+")


def normalize(text) -> str:
    """Normalisasi dasar yang sama untuk judul dan isi artikel (masih case-sensitive)."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = HTML_TAG.sub(" ", text)
    text = URL.sub(" ", text)
    text = EMAIL.sub(" ", text)
    text = unicodedata.normalize("NFKC", text.translate(QUOTE_MAP))
    return text


def finalize(text: str) -> str:
    """Lowercase, buang karakter aneh, rapikan spasi dan tanda baca."""
    text = MENTION_HASHTAG.sub(" ", text)
    text = text.lower()
    text = DISALLOWED.sub(" ", text)
    text = re.sub(r"\s+([.,!?:;%])", r"\1", text)   # "indonesia , di" -> "indonesia, di"
    text = re.sub(r"([,.!?])\1{1,}", r"\1", text)    # ",," -> ","  (tapi "..." jadi ".")
    text = re.sub(r",\s*([.!?])", r"\1", text)       # "dan,." -> "dan."
    text = re.sub(r"\(\s*\)", " ", text)
    text = MULTI_SPACE.sub(" ", text).strip()
    return text.strip(" ,;:-")


def clean_title(title) -> str:
    """Bersihkan satu judul berita. Judul tidak dipecah per kalimat."""
    return finalize(normalize(title))


def _strip_dateline(text: str) -> str:
    text = LEADING_JUNK.sub("", text)
    text = AGENCY_DATELINE.sub("", text)
    text = CITY_DATELINE.sub("", text)
    return LEADING_JUNK.sub("", text)


def _repair_structure(text: str) -> str:
    text = text.replace("\r", " ").replace("\n", " ")
    # pemisah paragraf: akhir kalimat lalu koma -> spasi
    text = re.sub(r"([.!?][\"']?)\s*,+\s*", r"\1 ", text)
    # deretan koma bekas entitas yang hilang -> satu koma
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    # "termasuk,dan" -> "termasuk, dan" (koma desimal "7,3" tidak kena)
    text = re.sub(r",(?=[A-Za-z\"])", ", ", text)
    # kalimat menempel: "ujarnya.Bobby" -> "ujarnya. Bobby"
    text = re.sub(r"(?<=[a-z][a-z\")])([.!?]\"?)(?=[A-Z\"])", r"\1 ", text)
    return MULTI_SPACE.sub(" ", text).strip()


def split_sentences(text: str) -> list:
    pieces = SENTENCE_SPLIT.split(text)
    sentences = []
    for piece in pieces:
        if sentences and ABBREV_END.search(sentences[-1]):
            sentences[-1] = sentences[-1] + " " + piece
        else:
            sentences.append(piece)
    return sentences


def is_good_sentence(sentence: str, min_words: int, min_alpha_ratio: float) -> bool:
    if len(sentence.split()) < min_words:
        return False
    letters = sum(c.isalpha() for c in sentence)
    if letters / max(len(sentence.replace(" ", "")), 1) < min_alpha_ratio:
        return False
    return DROP_SENTENCE.search(sentence) is None


def clean_article(content, min_words: int = 3, min_alpha_ratio: float = 0.5) -> list:
    """Bersihkan isi satu artikel dan kembalikan daftar kalimat yang sudah lowercase."""
    text = normalize(content)
    if not text.strip():
        return []
    for pattern in BOILERPLATE:
        text = pattern.sub(" ", text)
    text = _strip_dateline(text)
    text = _repair_structure(text)

    sentences = []
    for raw in split_sentences(text):
        s = finalize(raw)
        if is_good_sentence(s, min_words, min_alpha_ratio):
            sentences.append(s)
    return sentences
