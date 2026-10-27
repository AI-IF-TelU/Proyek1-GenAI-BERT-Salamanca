"""
Step 1 - Persiapan data CLICK-ID.

Jalankan dari root repo:
    python -m src.data.prepare            (pakai ulang split di data/splits/ kalau sudah ada)
    python -m src.data.prepare --resplit  (buat ulang split dari awal)

Output:
    data/splits/titles_{train,val,test}.csv   judul berlabel, split stratified 80/10/10
    data/processed/pretrain_{train,val}.txt   korpus MLM, satu kalimat per baris,
                                              baris kosong sebagai pemisah artikel
    reports/data_stats.json                   statistik untuk laporan
"""

import argparse
import json
import random
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SEED, data_cfg, paths
from src.data.cleaning import clean_article, clean_title


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_raw_articles() -> tuple:
    """
    Gabungkan raw XLSX dan raw CSV lalu dedup berdasarkan URL.

    XLSX dipakai sebagai sumber utama karena isinya lengkap (46.119 artikel, sesuai paper),
    sedangkan CSV versi Kaggle terpotong (misalnya tribunnews hanya 1.500 dari 9.662).
    CSV tetap digabung karena ada beberapa ratus artikel yang hanya ada di CSV.
    """
    frames = []
    for path in sorted(paths.raw_xlsx_dir.glob("*.xlsx")):
        print(f"  baca {path.name}")
        frames.append(pd.read_excel(path).assign(portal=path.stem, origin="xlsx"))
    for path in sorted(paths.raw_csv_dir.glob("*.csv")):
        frames.append(pd.read_csv(path).assign(portal=path.stem, origin="csv"))

    df = pd.concat(frames, ignore_index=True)
    n_rows = len(df)
    n_xlsx = int((df.origin == "xlsx").sum())
    df["url"] = df["url"].astype(str).str.strip()
    df = df.drop_duplicates(subset="url", keep="first")
    df = df[df["content"].notna() | df["title"].notna()].reset_index(drop=True)

    stats = {
        "rows_xlsx": n_xlsx,
        "rows_csv": n_rows - n_xlsx,
        "articles_after_url_dedup": len(df),
        "articles_without_content": int(df["content"].isna().sum()),
        "articles_per_portal": df["portal"].value_counts().sort_index().to_dict(),
    }
    return df, stats


def load_annotated() -> tuple:
    """Load 15.000 judul berlabel (suara mayoritas 3 annotator) dan dedup judul yang sama."""
    df = pd.read_csv(paths.annotated_main)
    n_raw = len(df)
    df = df.rename(columns={"label_score": "label", "label": "label_name"})
    df["text"] = df["title"].map(clean_title)
    df = df[df["text"].str.len() > 0]

    # judul yang sama persis setelah dibersihkan: buang semua kalau labelnya bertentangan,
    # simpan satu kalau labelnya sama (supaya judul yang sama tidak muncul di train dan test)
    label_per_text = df.groupby("text")["label"].nunique()
    conflicting = set(label_per_text[label_per_text > 1].index)
    n_dup_rows = int(df.duplicated("text", keep="first").sum())
    df = df[~df["text"].isin(conflicting)].drop_duplicates("text", keep="first").reset_index(drop=True)

    stats = {
        "rows_raw": n_raw,
        "duplicate_title_rows": n_dup_rows,
        "titles_with_conflicting_labels": len(conflicting),
        "rows_after_dedup": len(df),
        "label_counts": {"clickbait": int((df.label == 1).sum()), "non-clickbait": int((df.label == 0).sum())},
    }
    return df[["title", "text", "label", "label_name"]], stats


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------
def split_titles(df: pd.DataFrame) -> dict:
    holdout = data_cfg.val_ratio + data_cfg.test_ratio
    train, rest = train_test_split(df, test_size=holdout, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(
        rest, test_size=data_cfg.test_ratio / holdout, stratify=rest["label"], random_state=SEED
    )
    return {"train": train.reset_index(drop=True), "val": val.reset_index(drop=True), "test": test.reset_index(drop=True)}


# ---------------------------------------------------------------------------
# Leakage
# ---------------------------------------------------------------------------
class TitleMatcher:
    """
    Cek cepat apakah sebuah kalimat sama dengan atau memuat salah satu judul test.

    Brute force (1.500 judul x ~1 juta kalimat) terlalu lambat, jadi judul diindeks
    berdasarkan n kata pertamanya dan hanya kandidat yang n-gram-nya cocok yang dicek.
    """

    def __init__(self, titles, n: int = 4):
        self.n = n
        self.exact = set(titles)
        self.index = defaultdict(list)
        self.short = []
        for t in self.exact:
            words = t.split()
            if len(words) >= n:
                self.index[tuple(words[:n])].append(t)
            else:
                self.short.append(t)

    def leaks(self, sentence: str) -> bool:
        if sentence in self.exact:
            return True
        words = sentence.split()
        for i in range(len(words) - self.n + 1):
            for t in self.index.get(tuple(words[i:i + self.n]), ()):
                if t in sentence:
                    return True
        return any(f" {t} " in f" {sentence} " for t in self.short)


# ---------------------------------------------------------------------------
# Korpus pretraining
# ---------------------------------------------------------------------------
def build_pretrain_corpus(articles: pd.DataFrame, matcher: TitleMatcher) -> tuple:
    docs = []
    removed_title = removed_content = 0
    for title, content in zip(articles["title"], articles["content"]):
        sentences = []
        t = clean_title(title)
        if t and len(t.split()) >= data_cfg.min_sentence_words:
            if matcher.leaks(t):
                removed_title += 1
            else:
                sentences.append(t)
        for s in clean_article(content, data_cfg.min_sentence_words, data_cfg.min_alpha_ratio):
            if matcher.leaks(s):
                removed_content += 1
            else:
                sentences.append(s)
        if sentences:
            docs.append(sentences)

    stats = {
        "leakage_removed_title_sentences": removed_title,
        "leakage_removed_content_sentences": removed_content,
        "leakage_removed_total": removed_title + removed_content,
    }
    return docs, stats


def write_docs(docs, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for sentences in docs:
            f.write("\n".join(sentences))
            f.write("\n\n")


def describe(values) -> dict:
    a = np.asarray(values)
    return {
        "count": int(a.size), "mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2),
        "min": int(a.min()), "p50": int(np.percentile(a, 50)), "p95": int(np.percentile(a, 95)),
        "p99": int(np.percentile(a, 99)), "max": int(a.max()),
    }


def main(resplit: bool = False):
    random.seed(SEED)
    stats = {"seed": SEED}

    print("[1/4] Load data annotated dan split stratified")
    titles, stats["annotated"] = load_annotated()
    if not resplit and all(p.exists() for p in paths.title_split.values()):
        # split sudah ada di repo: pakai ulang supaya semua anggota memakai split yang identik
        print("  split di data/splits/ sudah ada, dipakai ulang (pakai --resplit untuk membuat ulang)")
        splits = {name: pd.read_csv(p) for name, p in paths.title_split.items()}
    else:
        splits = split_titles(titles)
    paths.splits_dir.mkdir(parents=True, exist_ok=True)
    stats["title_splits"] = {}
    for name, df in splits.items():
        df.to_csv(paths.title_split[name], index=False, encoding="utf-8")
        stats["title_splits"][name] = {
            "rows": len(df),
            "clickbait": int((df.label == 1).sum()),
            "non-clickbait": int((df.label == 0).sum()),
            "clickbait_ratio": round(float(df.label.mean()), 4),
        }
    stats["title_words"] = describe(splits["train"]["text"].str.split().map(len))

    print("[2/4] Load artikel raw (XLSX lebih lambat, tunggu 1-3 menit)")
    articles, stats["raw"] = load_raw_articles()

    print("[3/4] Bersihkan artikel, pecah kalimat, buang judul test (leakage)")
    matcher = TitleMatcher(splits["test"]["text"].tolist())
    docs, stats["leakage"] = build_pretrain_corpus(articles, matcher)

    rng = random.Random(SEED)
    rng.shuffle(docs)
    n_val = max(1, int(len(docs) * data_cfg.pretrain_val_ratio))
    pretrain = {"val": docs[:n_val], "train": docs[n_val:]}

    print("[4/4] Tulis korpus dan statistik")
    stats["pretrain_corpus"] = {}
    for name, part in pretrain.items():
        write_docs(part, paths.pretrain_split[name])
        n_sent = sum(len(d) for d in part)
        n_words = sum(len(s.split()) for d in part for s in d)
        stats["pretrain_corpus"][name] = {"documents": len(part), "sentences": n_sent, "words": n_words}
    stats["sentence_words"] = describe([len(s.split()) for d in pretrain["train"] for s in d])

    sentence_counter = Counter(s for d in pretrain["train"] for s in d)
    stats["most_common_sentences"] = [[s, c] for s, c in sentence_counter.most_common(15)]

    paths.reports_dir.mkdir(parents=True, exist_ok=True)
    out = paths.reports_dir / "data_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in stats.items() if k != "most_common_sentences"}, indent=2, ensure_ascii=False))
    print(f"\nStatistik disimpan di {out}")
    print("Cek 'most_common_sentences' di file itu: kalau ada boilerplate yang lolos, tambahkan polanya di cleaning.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resplit", action="store_true", help="buat ulang split judul walaupun sudah ada")
    main(parser.parse_args().resplit)
