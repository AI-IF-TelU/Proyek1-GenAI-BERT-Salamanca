import torch

from config import MASK_ID, model_cfg
from src.collator import DynamicMaskingCollator
from src.model import BertForMaskedLM


def test_dynamic_masking_uses_80_10_10_and_masks_no_special_tokens():
    ordinary_id = 100
    input_ids = torch.full((200, 100), ordinary_id, dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    input_ids[:, 0] = 2
    input_ids[:, 1] = 5
    input_ids[:, 2] = 6
    input_ids[:, 3] = 3
    attention_mask[:, 99] = 0
    input_ids[:, 99] = 0

    collator = DynamicMaskingCollator(protected_positions=(0, 1))
    result = collator(
        [{"input_ids": ids, "attention_mask": mask} for ids, mask in zip(input_ids, attention_mask)]
    )
    targets = result["labels"]
    masked = targets.ne(-100)

    assert not masked[:, :4].any()
    assert not masked[:, 99].any()
    assert torch.all(targets[masked] == ordinary_id)

    selected_count = int(masked.sum())
    mask_replaced = int((result["input_ids"][masked] == MASK_ID).sum())
    unchanged = int((result["input_ids"][masked] == ordinary_id).sum())
    random_replaced = selected_count - mask_replaced - unchanged
    assert abs(mask_replaced / selected_count - 0.8) < 0.04
    assert abs(unchanged / selected_count - 0.1) < 0.04
    assert abs(random_replaced / selected_count - 0.1) < 0.04


def test_mlm_forward_backward_and_weight_tying():
    model = BertForMaskedLM()
    input_ids = torch.tensor([[2, 5, 100, 101, 3, 0, 0, 0]])
    attention_mask = input_ids.ne(0).long()
    prediction_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    prediction_mask[0, 2] = True
    logits = model(input_ids, attention_mask, prediction_mask=prediction_mask)
    assert logits.shape == (1, model_cfg.vocab_size)
    assert model.mlm_bias.shape == (model_cfg.vocab_size,)

    target = torch.full_like(input_ids, -100)
    target[0, 2] = 100
    loss = torch.nn.functional.cross_entropy(
        logits, target[prediction_mask]
    )
    loss.backward()
    assert model.bert.embeddings.token_embedding.weight.grad is not None


if __name__ == "__main__":
    test_dynamic_masking_uses_80_10_10_and_masks_no_special_tokens()
    print("PASS  dynamic masking (80/10/10; special tokens protected)")
    test_mlm_forward_backward_and_weight_tying()
    print("PASS  MLM forward/backward and masked-position logits")
