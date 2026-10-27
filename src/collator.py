"""
Step 3 - Dynamic masking untuk Masked Language Model (Devlin et al., 2019).

Dari token yang boleh di-mask, 15% dipilih secara acak. Dari yang terpilih:
    80% diganti [MASK], 10% diganti token acak, 10% dibiarkan.
Token khusus (input_ids <= NCB_ID: [PAD] [UNK] [CLS] [SEP] [MASK] [CB] [NCB]) tidak pernah
dipilih, jadi token gaya di Step 4 aman. Token pengganti acak juga tidak pernah token khusus.

Masking dihitung ulang setiap kali batch diambil (dynamic), jadi kalimat yang sama mendapat
posisi mask berbeda di tiap epoch.

Contoh:
    from src.collator import mask_tokens
    inputs, labels = mask_tokens(batch["input_ids"])      # labels = -100 di posisi yang tidak dinilai

    # atau sebagai collate_fn DataLoader
    from src.collator import MLMCollator
    loader = DataLoader(dataset, batch_size=64, collate_fn=MLMCollator())
"""

import torch
from torch.utils.data import default_collate

from config import MASK_ID, NCB_ID, model_cfg, pretrain_cfg
from src.model import IGNORE_INDEX


def mask_tokens(input_ids, mask_prob=pretrain_cfg.mask_prob, mask_replace_prob=pretrain_cfg.mask_replace_prob,
                random_replace_prob=pretrain_cfg.random_replace_prob, vocab_size=model_cfg.vocab_size,
                generator=None):
    """
    Kembalikan (inputs, labels) baru; input_ids asli tidak diubah.
    generator: torch.Generator opsional (di CPU). Dipakai saat validasi supaya posisi mask
    sama di setiap epoch dan loss validasi antar-epoch bisa dibandingkan.
    """
    device = input_ids.device
    rng_device = generator.device if generator is not None else device
    shape = input_ids.shape

    maskable = input_ids > NCB_ID
    selected = (torch.rand(shape, generator=generator, device=rng_device).to(device) < mask_prob) & maskable
    labels = input_ids.masked_fill(~selected, IGNORE_INDEX)

    # undian kedua menentukan nasib token terpilih: [MASK] / token acak / tetap
    fate = torch.rand(shape, generator=generator, device=rng_device).to(device)
    to_mask = selected & (fate < mask_replace_prob)
    to_random = selected & (fate >= mask_replace_prob) & (fate < mask_replace_prob + random_replace_prob)

    inputs = input_ids.masked_fill(to_mask, MASK_ID)
    random_ids = torch.randint(NCB_ID + 1, vocab_size, shape, generator=generator, device=rng_device)
    inputs = torch.where(to_random, random_ids.to(device, input_ids.dtype), inputs)
    return inputs, labels


class MLMCollator:
    """collate_fn: tumpuk batch lalu mask. Label gaya bawaan TitleDataset dipindah ke style_labels."""

    def __init__(self, **mask_kwargs):
        self.mask_kwargs = mask_kwargs

    def __call__(self, examples):
        batch = default_collate(examples)
        if "labels" in batch:
            batch["style_labels"] = batch.pop("labels")
        batch["input_ids"], batch["labels"] = mask_tokens(batch["input_ids"], **self.mask_kwargs)
        return batch
