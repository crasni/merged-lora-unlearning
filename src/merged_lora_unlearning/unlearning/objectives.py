from __future__ import annotations


def batch_nll(model, inputs):
    import torch.nn.functional as F

    outputs = model(**inputs)
    logits = outputs.logits[:, :-1, :]
    labels = inputs["labels"][:, 1:]
    token_loss = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(labels.shape)
    mask = labels.ne(-100)
    return (token_loss * mask).sum(-1), mask.sum(-1).clamp_min(1), outputs


def ga_loss(model, forget_inputs):
    _, _, outputs = batch_nll(model, forget_inputs)
    return -outputs.loss, outputs


def npo_loss(model, reference_model, forget_inputs, beta: float):
    import torch
    import torch.nn.functional as F

    current_nll, _, outputs = batch_nll(model, forget_inputs)
    with torch.no_grad():
        reference_nll, _, _ = batch_nll(reference_model, forget_inputs)
    lose_log_ratio = -(current_nll - reference_nll)
    loss = -2.0 / beta * F.logsigmoid(-beta * lose_log_ratio).mean()
    return loss, outputs


def simnpo_loss(model, forget_inputs, beta: float, delta: float):
    import torch.nn.functional as F

    nll, lengths, outputs = batch_nll(model, forget_inputs)
    normalized_nll = nll / lengths - delta
    loss = -2.0 / beta * F.logsigmoid(beta * normalized_nll).mean()
    return loss, outputs


def retain_nll(model, retain_inputs):
    return model(**retain_inputs).loss


def retain_kl(model, reference_model, retain_inputs):
    import torch
    import torch.nn.functional as F

    current = model(**retain_inputs).logits
    with torch.no_grad():
        reference = reference_model(**retain_inputs).logits
    return F.kl_div(
        F.log_softmax(current, dim=-1),
        F.log_softmax(reference, dim=-1),
        reduction="batchmean",
        log_target=True,
    )


def combined_loss(
    model,
    reference_model,
    inputs,
    method: str,
    retain_loss_type: str,
    beta: float,
    simnpo_delta: float,
    forget_weight: float,
    retain_weight: float,
):
    forget_inputs = inputs["forget"]
    if method == "ga":
        forget_loss, outputs = ga_loss(model, forget_inputs)
    elif method == "npo":
        forget_loss, outputs = npo_loss(model, reference_model, forget_inputs, beta)
    elif method == "simnpo":
        forget_loss, outputs = simnpo_loss(model, forget_inputs, beta, simnpo_delta)
    else:
        raise ValueError(f"Unsupported unlearning method: {method}")

    if retain_loss_type == "none":
        retain_loss = 0.0
    elif retain_loss_type == "nll":
        retain_loss = retain_nll(model, inputs["retain"])
    elif retain_loss_type == "kl":
        retain_loss = retain_kl(model, reference_model, inputs["retain"])
    else:
        raise ValueError(f"Unsupported retain loss: {retain_loss_type}")
    return forget_weight * forget_loss + retain_weight * retain_loss, outputs
