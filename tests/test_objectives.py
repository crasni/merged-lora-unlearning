import pytest
import copy

torch = pytest.importorskip("torch")

from merged_lora_unlearning.unlearning.objectives import (  # noqa: E402
    batch_nll,
    combined_loss,
    method_spec,
    retain_kl,
)


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.fixed_logits = logits

    def forward(self, **inputs):
        return type("Output", (), {"logits": self.fixed_logits + self.anchor})()


class ToyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(
            torch.tensor([[[0.0, 0.0], [0.0, 2.0], [0.0, 2.0]]])
        )

    def forward(self, input_ids, labels, attention_mask):
        logits = self.logits.expand(input_ids.shape[0], -1, -1)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        return type("Output", (), {"logits": logits, "loss": loss})()


def _inputs():
    return {
        "input_ids": torch.tensor([[0, 1, 1]]),
        "attention_mask": torch.tensor([[1, 1, 1]]),
        "labels": torch.tensor([[-100, 1, 1]]),
    }


def test_retain_kl_is_zero_for_identical_models():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    inputs = {"input_ids": torch.tensor([[0, 1]])}

    assert retain_kl(FixedModel(logits), FixedModel(logits), inputs).item() == pytest.approx(0.0)


@pytest.mark.parametrize("method", ["ga", "npo", "simnpo"])
def test_pure_forget_method_increases_forget_nll(method):
    model = ToyLM()
    reference = copy.deepcopy(model)
    inputs = _inputs()
    before = batch_nll(model, inputs)[0].item()
    loss, _ = combined_loss(
        model=model,
        reference_model=reference,
        inputs={"forget": inputs, "retain": inputs},
        method=method,
        beta=0.1,
        simnpo_delta=0.0,
        gamma=1.0,
        alpha=1.0,
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert batch_nll(model, inputs)[0].item() > before


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("ga", ("ga", "none")),
        ("grad_diff", ("ga", "nll")),
        ("ga_kl", ("ga", "kl")),
        ("npo", ("npo", "none")),
        ("npo_grad_diff", ("npo", "nll")),
        ("npo_kl", ("npo", "kl")),
        ("simnpo", ("simnpo", "none")),
        ("simnpo_grad_diff", ("simnpo", "nll")),
        ("simnpo_kl", ("simnpo", "kl")),
    ],
)
def test_named_method_specs(method, expected):
    assert method_spec(method) == expected
