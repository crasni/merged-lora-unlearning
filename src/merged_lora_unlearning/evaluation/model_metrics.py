from __future__ import annotations

from typing import Any


def model_device(model):
    return next(model.parameters()).device


def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model_device(model))
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    generated = output[:, inputs["input_ids"].shape[1] :]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def continuation_statistics(model, tokenizer, prompt: str, answer: str) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
    answer_ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([prompt_ids + answer_ids], device=model_device(model))
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    labels = input_ids[:, 1:]
    answer_start = max(len(prompt_ids) - 1, 0)
    answer_logits = logits[:, answer_start:, :]
    answer_labels = labels[:, answer_start:]
    log_probs = F.log_softmax(answer_logits, dim=-1)
    selected = log_probs.gather(-1, answer_labels.unsqueeze(-1)).squeeze(-1)
    predictions = answer_logits.argmax(-1)
    exact_memorization = predictions.eq(answer_labels).float().mean().item()
    valid_length = answer_labels.shape[-1]
    extraction_start = valid_length
    for index in range(valid_length):
        if predictions[:, index:].eq(answer_labels[:, index:]).all():
            extraction_start = index
            break
    extraction_strength = 1.0 - extraction_start / max(valid_length, 1)
    token_log_probs = selected.squeeze(0).detach().cpu().tolist()
    return {
        "answer_probability": selected.sum().exp().item(),
        "mean_token_probability": selected.exp().mean().item(),
        "exact_memorization": exact_memorization,
        "extraction_strength": extraction_strength,
        "loss": -selected.mean().item(),
        "token_log_probs": token_log_probs,
    }


def min_k_score(token_log_probs: list[float], ratio: float = 0.4) -> float:
    count = max(1, round(len(token_log_probs) * ratio))
    return sum(sorted(token_log_probs)[:count]) / count
