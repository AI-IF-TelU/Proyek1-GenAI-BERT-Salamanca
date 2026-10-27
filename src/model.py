"""
Step 2 - Arsitektur encoder BERT kecil (IndoMiniBERT), ditulis dari nol.

Tidak memakai nn.TransformerEncoder / nn.MultiheadAttention: multi-head self-attention
dua arah ditulis manual (proyeksi Q, K, V -> scaled dot-product -> padding mask -> gabung head).

Susunan (semua ukuran dari config.ModelConfig):
    BertEmbeddings      token embedding + positional embedding (dipelajari) -> LayerNorm -> dropout
    EncoderLayer x 4    x = LN(x + Attn(x)),  x = LN(x + FFN(x))   (post-LN seperti BERT asli)
    MLMHead             dense -> GELU -> LayerNorm -> decoder (bobot diikat ke token embedding)
    ClassificationHead  pooler (dense + tanh) pada token [CLS] -> dropout -> linear num_labels

Contoh:
    from src.model import IndoMiniBERT
    model = IndoMiniBERT()
    out = model(input_ids, attention_mask)                      # MLM, logits (B, L, V)
    out = model(input_ids, attention_mask, labels=labels)       # MLM + loss (hanya posisi labels != -100)
    out = model(input_ids, attention_mask, task="cls", labels=y)  # klasifikasi, logits (B, num_labels)
"""

import math
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch import nn

from config import ModelConfig, model_cfg

IGNORE_INDEX = -100   # label posisi yang tidak di-mask (diabaikan oleh cross entropy)


class BertEmbeddings(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.token = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=cfg.pad_id)
        self.position = nn.Embedding(cfg.max_position, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)
        self.register_buffer("position_ids", torch.arange(cfg.max_position), persistent=False)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        if seq_len > self.position.num_embeddings:
            raise ValueError(f"Panjang input {seq_len} > max_position {self.position.num_embeddings}")
        x = self.token(input_ids) + self.position(self.position_ids[:seq_len])
        return self.dropout(self.norm(x))


class MultiHeadSelfAttention(nn.Module):
    """Self-attention dua arah: setiap token boleh melihat semua token non-padding (kiri dan kanan)."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.hidden_size % cfg.num_heads != 0:
            raise ValueError("hidden_size harus habis dibagi num_heads")
        self.num_heads = cfg.num_heads
        self.head_dim = cfg.hidden_size // cfg.num_heads
        self.query = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.key = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.value = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.attn_dropout = nn.Dropout(cfg.attention_dropout)

    def _split_heads(self, x):
        # (B, L, H) -> (B, heads, L, head_dim)
        b, l, _ = x.shape
        return x.view(b, l, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x, attention_mask=None):
        q = self._split_heads(self.query(x))
        k = self._split_heads(self.key(x))
        v = self._split_heads(self.value(x))

        # scaled dot-product: (B, heads, L, L)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            # attention_mask (B, L): 1 = token asli, 0 = [PAD]. Kolom [PAD] tidak boleh dilihat.
            pad = (attention_mask == 0)[:, None, None, :]
            scores = scores.masked_fill(pad, torch.finfo(scores.dtype).min)
        probs = scores.softmax(dim=-1)

        context = self.attn_dropout(probs) @ v                     # (B, heads, L, head_dim)
        b, _, l, _ = context.shape
        context = context.transpose(1, 2).reshape(b, l, -1)       # gabung head -> (B, L, H)
        return self.out(context), probs


class FeedForward(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.ffn_size)
        self.fc2 = nn.Linear(cfg.ffn_size, cfg.hidden_size)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attention = MultiHeadSelfAttention(cfg)
        self.attn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = FeedForward(cfg)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, attention_mask=None):
        attn_out, probs = self.attention(x, attention_mask)
        x = self.attn_norm(x + self.dropout(attn_out))
        x = self.ffn_norm(x + self.dropout(self.ffn(x)))
        return x, probs


class BertEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.embeddings = BertEmbeddings(cfg)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(cfg.num_layers))

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        x = self.embeddings(input_ids)
        attentions = []
        for layer in self.layers:
            x, probs = layer(x, attention_mask)
            if output_attentions:
                attentions.append(probs)
        return x, attentions


class MLMHead(nn.Module):
    def __init__(self, cfg: ModelConfig, token_embedding: nn.Embedding):
        super().__init__()
        self.dense = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        # weight tying (Press & Wolf, 2017): matriks output = matriks token embedding
        self.decoder = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.decoder.weight = token_embedding.weight
        self.bias = nn.Parameter(torch.zeros(cfg.vocab_size))

    def forward(self, hidden):
        return self.decoder(self.norm(F.gelu(self.dense(hidden)))) + self.bias


class ClassificationHead(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.pooler = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.dropout = nn.Dropout(cfg.dropout)
        self.classifier = nn.Linear(cfg.hidden_size, cfg.num_labels)

    def forward(self, hidden):
        pooled = torch.tanh(self.pooler(hidden[:, 0]))     # representasi token [CLS]
        return self.classifier(self.dropout(pooled))


class IndoMiniBERT(nn.Module):
    def __init__(self, cfg: ModelConfig = model_cfg):
        super().__init__()
        self.cfg = cfg
        self.bert = BertEncoder(cfg)
        self.mlm_head = MLMHead(cfg, self.bert.embeddings.token)
        self.cls_head = ClassificationHead(cfg)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        # inisialisasi seperti BERT: normal(0, 0.02), bias 0, LayerNorm (1, 0)
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask=None, labels=None, task="mlm", output_attentions=False):
        """
        task="mlm":
            labels None   -> logits (B, L, vocab) untuk semua posisi (dipakai saat generasi).
            labels (B, L) -> logits dan target HANYA untuk posisi labels != -100, bentuk (N, vocab)
                             dan (N,). Lebih hemat memori daripada menghitung 128 x 16.000 logit
                             per kalimat padahal yang dinilai cuma ~15%.
        task="cls":
            logits (B, num_labels); labels (B,) opsional untuk loss.
        """
        if attention_mask is None:
            attention_mask = (input_ids != self.cfg.pad_id).long()
        hidden, attentions = self.bert(input_ids, attention_mask, output_attentions)
        out = {"hidden_states": hidden}
        if output_attentions:
            out["attentions"] = attentions   # list per layer, masing-masing (B, heads, L, L)

        if task == "mlm":
            if labels is None:
                out["logits"] = self.mlm_head(hidden)
            else:
                selected = labels != IGNORE_INDEX
                target = labels[selected]
                logits = self.mlm_head(hidden[selected])
                out["logits"], out["target"] = logits, target
                # batch tanpa token ter-mask: loss 0 (bukan NaN) supaya training tidak rusak
                out["loss"] = F.cross_entropy(logits.float(), target) if target.numel() else logits.sum() * 0
        elif task == "cls":
            logits = self.cls_head(hidden)
            out["logits"] = logits
            if labels is not None:
                out["loss"] = F.cross_entropy(logits.float(), labels)
        else:
            raise ValueError(f"task tidak dikenal: {task}")
        return out

    def num_parameters(self) -> dict:
        """Jumlah parameter unik (bobot decoder MLM = token embedding, jadi tidak dihitung dua kali)."""
        def count(module):
            return sum(p.numel() for p in module.parameters())
        return {
            "embeddings": count(self.bert.embeddings),
            "encoder_layers": count(self.bert.layers),
            "mlm_head (tanpa bobot yang diikat)": count(self.mlm_head) - self.mlm_head.decoder.weight.numel(),
            "cls_head": count(self.cls_head),
            "total": count(self),
        }


def save_model(model: IndoMiniBERT, path, **extra):
    """Simpan bobot + ModelConfig, supaya Step 4 dan 5 bisa memuat ulang tanpa menebak ukurannya."""
    torch.save({"model": model.state_dict(), "model_cfg": asdict(model.cfg), **extra}, path)


def load_model(path, device="cpu", strict=True) -> IndoMiniBERT:
    """Muat model dari checkpoint save_model / src.pretrain (kunci 'model' dan 'model_cfg')."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = IndoMiniBERT(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"], strict=strict)
    return model.to(device)
