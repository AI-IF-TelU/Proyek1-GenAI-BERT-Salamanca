"""
Satu sumber konfigurasi untuk seluruh proyek IndoMiniBERT.

Semua ukuran tensor, path, dan hyperparameter diambil dari file ini supaya modul
setiap anggota tetap sinkron. Kalau ada nilai yang mau diubah, ubah di sini dan
beri tahu anggota lain lewat commit message.

Pemilik bagian:
    PathConfig, DataConfig, TokenizerConfig  -> Anggota 1 (Step 1)
    ModelConfig, OverfitConfig               -> Anggota 2 (Step 2)
    PretrainConfig                           -> Anggota 3 (Step 3)
    StyleConfig, ClassifierConfig            -> Anggota 4 (Step 4)
    GenerationConfig                         -> Anggota 5 (Step 5)
"""

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SEED = 42


# ---------------------------------------------------------------------------
# Token khusus. Urutan ini menentukan ID token, jadi JANGAN diubah setelah
# tokenizer dilatih (ID 0..6 dipakai langsung oleh model dan collator).
# ---------------------------------------------------------------------------
PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
MASK_TOKEN = "[MASK]"
CB_TOKEN = "[CB]"     # penanda gaya clickbait
NCB_TOKEN = "[NCB]"   # penanda gaya non-clickbait

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, CB_TOKEN, NCB_TOKEN]

PAD_ID, UNK_ID, CLS_ID, SEP_ID, MASK_ID, CB_ID, NCB_ID = range(len(SPECIAL_TOKENS))

# label_score di CLICK-ID: 1 = clickbait, 0 = non-clickbait
LABEL2STYLE_ID = {1: CB_ID, 0: NCB_ID}
LABEL_NAMES = {0: "non-clickbait", 1: "clickbait"}


@dataclass
class PathConfig:
    # Dataset mentah dari Kaggle (andikawilliam/clickid), diekstrak ke folder ini
    dataset_dir: Path = ROOT / "click_id_dataset"
    raw_xlsx_dir: Path = ROOT / "click_id_dataset" / "raw" / "xlsx"
    raw_csv_dir: Path = ROOT / "click_id_dataset" / "raw" / "csv"
    annotated_main: Path = ROOT / "click_id_dataset" / "annotated" / "combined" / "csv" / "main.csv"

    # Split judul berlabel (kecil, ikut di-commit supaya semua anggota pakai split yang sama)
    splits_dir: Path = ROOT / "data" / "splits"
    # Korpus pretraining dan cache (besar, tidak di-commit, dibuat ulang dengan script)
    processed_dir: Path = ROOT / "data" / "processed"
    # Tokenizer (kecil, ikut di-commit)
    tokenizer_dir: Path = ROOT / "artifacts" / "tokenizer"
    # Statistik dataset untuk laporan
    reports_dir: Path = ROOT / "reports"
    checkpoints_dir: Path = ROOT / "checkpoints"

    @property
    def title_split(self):
        return {s: self.splits_dir / f"titles_{s}.csv" for s in ("train", "val", "test")}

    @property
    def pretrain_split(self):
        return {s: self.processed_dir / f"pretrain_{s}.txt" for s in ("train", "val")}

    @property
    def tokenizer_file(self):
        return self.tokenizer_dir / "tokenizer.json"


@dataclass
class DataConfig:
    # Split data annotated (stratified berdasarkan label)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    # Porsi artikel raw yang disisihkan untuk validasi pretraining MLM
    pretrain_val_ratio: float = 0.02
    # Kalimat dengan kurang dari ini jumlah kata dibuang dari korpus
    min_sentence_words: int = 3
    # Kalimat dengan porsi huruf di bawah nilai ini dianggap sampah (tabel skor, dll.)
    min_alpha_ratio: float = 0.5


@dataclass
class TokenizerConfig:
    vocab_size: int = 16000
    min_frequency: int = 2
    lowercase: bool = True
    special_tokens: list = field(default_factory=lambda: list(SPECIAL_TOKENS))
    wordpiece_prefix: str = "##"


@dataclass
class ModelConfig:
    vocab_size: int = 16000
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 4            # 64 dimensi per head
    ffn_size: int = 1024          # aktivasi GELU
    max_position: int = 128       # positional embedding dipelajari
    dropout: float = 0.1
    attention_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    num_labels: int = 2
    pad_id: int = PAD_ID


@dataclass
class OverfitConfig:
    # Sanity check Step 2: model harus bisa menghafal satu batch kecil sampai loss mendekati 0
    batch_size: int = 8
    steps: int = 300
    lr: float = 1e-3
    target_loss: float = 0.05


@dataclass
class PretrainConfig:
    max_len: int = 128
    mask_prob: float = 0.15
    mask_replace_prob: float = 0.8    # diganti [MASK]
    random_replace_prob: float = 0.1  # diganti token acak (10% sisanya dibiarkan)
    batch_size: int = 128
    lr: float = 5e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    epochs: int = 10
    grad_clip: float = 1.0
    use_amp: bool = True
    num_workers: int = 2
    log_every: int = 50               # catat train loss tiap N step (untuk grafik loss)
    save_every: int = 500             # simpan last.pt tiap N step (jaga-jaga runtime Colab putus)
    # Eksperimen: 3 variasi learning rate x 2 variasi batch size, masing-masing sweep_epochs epoch
    sweep_lrs: tuple = (1e-4, 5e-4, 1e-3)
    sweep_batch_sizes: tuple = (64, 128)
    sweep_epochs: int = 1


@dataclass
class StyleConfig:
    max_len: int = 48
    batch_size: int = 64
    lr: float = 1e-4
    epochs: int = 20


@dataclass
class ClassifierConfig:
    max_len: int = 48
    batch_size: int = 64
    lr: float = 5e-5
    epochs: int = 5


@dataclass
class GenerationConfig:
    iterations: int = 10
    top_k: int = 50
    temperature: float = 1.0


paths = PathConfig()
data_cfg = DataConfig()
tokenizer_cfg = TokenizerConfig()
model_cfg = ModelConfig()
overfit_cfg = OverfitConfig()
pretrain_cfg = PretrainConfig()
style_cfg = StyleConfig()
classifier_cfg = ClassifierConfig()
generation_cfg = GenerationConfig()
