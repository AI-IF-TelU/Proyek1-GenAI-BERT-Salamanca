"""
Arsitektur IndoMiniBERT — dibangun dari nol dengan PyTorch (Step 2).

Tidak memakai nn.TransformerEncoder. Multi-head self-attention ditulis manual
dengan proyeksi Q, K, V yang terpisah.

Desain mengikuti BERT-Mini (Turc et al., 2019) dengan Pre-LayerNorm
(layer norm sebelum sub-layer, bukan setelah) yang lebih stabil untuk model kecil.

Kelas yang diekspor:
    BERTModel                     — encoder murni, output hidden states
    BertForMaskedLM               — encoder + MLM head (weight tying)
    BertForSequenceClassification — encoder + classification head pada [CLS]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import model_cfg


class BertEmbeddings(nn.Module):
    """Token embedding + learned positional embedding, diikuti LayerNorm dan Dropout."""

    def __init__(self, config=model_cfg):
        super().__init__()
        self.token_embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_id
        )
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids):
        seq_len = input_ids.size(1)
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(
                f"Sequence length {seq_len} melebihi batas maksimal "
                f"{self.position_embedding.num_embeddings}"
            )
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        embeddings = self.token_embedding(input_ids) + self.position_embedding(positions)
        return self.dropout(self.layer_norm(embeddings))


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention ditulis manual (tidak memakai nn.MultiheadAttention).

    Proyeksi dilabeli q_proj / k_proj / v_proj / out_proj supaya nama parameter
    seragam dengan checkpoint yang sudah disimpan dari Step 4.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        if config.hidden_size % config.num_heads:
            raise ValueError("hidden_size harus habis dibagi num_heads")
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        def _split_heads(proj):
            return proj.view(
                batch_size, seq_len, self.num_heads, self.head_dim
            ).transpose(1, 2)

        q = _split_heads(self.q_proj(hidden_states))
        k = _split_heads(self.k_proj(hidden_states))
        v = _split_heads(self.v_proj(hidden_states))

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attention_mask is not None:
            key_mask = attention_mask[:, None, None, :].bool()
            scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)

        probs = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_size
        )
        return self.out_proj(context)


class FeedForward(nn.Module):
    """
    Feed-forward sublayer: Linear → GELU → Dropout → Linear.

    Dinamai sebagai self.net (nn.Sequential) agar kunci parameter menjadi
    ffn.net.0 dan ffn.net.3, sesuai nama yang tersimpan di checkpoint.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, config.ffn_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_size, config.hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class BertLayer(nn.Module):
    """
    Satu encoder block dengan Pre-LayerNorm:
        x = x + Attention(LN_attn(x))
        x = x + FFN(LN_ffn(x))

    FFN disimpan sebagai FeedForward dengan sub-modul .net (nn.Sequential) sehingga
    kunci parameter adalah ffn.net.0 dan ffn.net.3, sesuai checkpoint.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        self.attention = MultiHeadSelfAttention(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)
        self.ffn_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden_states, attention_mask=None):
        # Pre-norm attention sublayer
        hidden_states = hidden_states + self.dropout(
            self.attention(self.attention_norm(hidden_states), attention_mask)
        )
        # Pre-norm FFN sublayer
        hidden_states = hidden_states + self.dropout(
            self.ffn(self.ffn_norm(hidden_states))
        )
        return hidden_states


class BERTModel(nn.Module):
    """
    Encoder BERT berukuran kecil: embedding → 4 × BertLayer → LayerNorm akhir (bert.norm).

    bert.norm adalah LayerNorm tambahan di akhir tumpukan layer, diperlukan oleh
    desain Pre-LayerNorm agar output terakhir tetap ternormalisasi.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        self.embeddings = BertEmbeddings(config)
        self.layers = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_layers)]
        )
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        return self.norm(hidden_states)


class BertForMaskedLM(nn.Module):
    """
    BERT encoder + MLM head dengan weight tying.

    MLM head: mlm_head — Linear(hidden→vocab) dengan bobot diikat ke token_embedding
    (Press & Wolf, 2017). Tidak ada mlm_bias terpisah karena bias Linear dipakai langsung.

    Desain sederhana ini sesuai checkpoint style_mlm.pt dari Step 4.
    """

    def __init__(self, config=model_cfg):
        super().__init__()
        self.bert = BERTModel(config)
        # mlm_head hanya berisi parameter weight; weight-tying dilakukan di forward
        # sehingga tidak ada parameter ganda (mlm_head.weight = token_embedding.weight)
        self.mlm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        # Weight tying: bagikan tensor bobot
        self.mlm_head.weight = self.bert.embeddings.token_embedding.weight
        with torch.no_grad():
            self.bert.embeddings.token_embedding.weight[config.pad_id].zero_()

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def mlm_bias(self):
        if self.mlm_head.bias is not None:
            return self.mlm_head.bias
        return torch.zeros(
            self.mlm_head.out_features,
            device=self.mlm_head.weight.device,
            dtype=self.mlm_head.weight.dtype,
        )

    def forward(self, input_ids, attention_mask=None, prediction_mask=None):
        hidden_states = self.bert(input_ids, attention_mask)
        if prediction_mask is not None:
            hidden_states = hidden_states[prediction_mask.bool()]
        return self.mlm_head(hidden_states)


class BertForSequenceClassification(nn.Module):
    """
    BERT encoder + classification head pada representasi [CLS] (posisi 0).

    Dipakai untuk classifier clickbait pada Step 4.
    """

    def __init__(self, num_labels=model_cfg.num_labels, config=model_cfg):
        super().__init__()
        self.bert = BERTModel(config)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(BertForMaskedLM._init_weights)
        with torch.no_grad():
            self.bert.embeddings.token_embedding.weight[config.pad_id].zero_()

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.bert(input_ids, attention_mask)
        cls_output = hidden_states[:, 0, :]
        return self.classifier(self.dropout(cls_output))


__all__ = ["BERTModel", "BertForMaskedLM", "BertForSequenceClassification"]
