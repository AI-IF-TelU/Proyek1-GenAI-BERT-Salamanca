import torch

from config import CB_ID, CLS_ID, MASK_ID, NCB_ID, PAD_ID, SEP_ID, model_cfg, pretrain_cfg


class DynamicMaskingCollator:
    def __init__(self, mask_prob=None, mask_replace_prob=None, random_replace_prob=None, protected_positions=()):
        self.mask_prob = pretrain_cfg.mask_prob if mask_prob is None else mask_prob
        self.mask_replace_prob = pretrain_cfg.mask_replace_prob if mask_replace_prob is None else mask_replace_prob
        self.random_replace_prob = pretrain_cfg.random_replace_prob if random_replace_prob is None else random_replace_prob
        self.protected_positions = tuple(protected_positions)
        if not 0.0 <= self.mask_prob <= 1.0:
            raise ValueError("mask_prob harus berada pada rentang [0, 1]")
        if not 0.0 <= self.mask_replace_prob <= 1.0:
            raise ValueError("mask_replace_prob harus berada pada rentang [0, 1]")
        if not 0.0 <= self.random_replace_prob <= 1.0:
            raise ValueError("random_replace_prob harus berada pada rentang [0, 1]")
        if self.mask_replace_prob + self.random_replace_prob > 1.0:
            raise ValueError("Porsi penggantian [MASK] dan token acak tidak boleh melebihi 1")

    def __call__(self, examples):
        input_ids = torch.stack([torch.as_tensor(ex["input_ids"], dtype=torch.long) for ex in examples])
        attention_mask = torch.stack([torch.as_tensor(ex["attention_mask"], dtype=torch.long) for ex in examples])

        labels = input_ids.clone()
        masked_input = input_ids.clone()
        mask = (torch.rand(input_ids.shape, device=input_ids.device) < self.mask_prob)
        mask &= attention_mask.bool() & (input_ids > NCB_ID)
        for position in self.protected_positions:
            if position < input_ids.size(1):
                mask[:, position] = False

        labels[~mask] = -100
        selection = torch.rand(input_ids.shape, device=input_ids.device)
        replace_mask = mask & (selection < self.mask_replace_prob)
        random_mask = (
            mask
            & (selection >= self.mask_replace_prob)
            & (selection < self.mask_replace_prob + self.random_replace_prob)
        )

        if random_mask.any():
            random_tokens = torch.randint(0, model_cfg.vocab_size, input_ids.shape, device=input_ids.device)
            masked_input[random_mask] = random_tokens[random_mask]

        masked_input[replace_mask] = MASK_ID

        batch = {
            "input_ids": masked_input,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return batch


def build_dynamic_masking_collator(**kwargs):
    return DynamicMaskingCollator(**kwargs)


__all__ = ["DynamicMaskingCollator", "build_dynamic_masking_collator"]
