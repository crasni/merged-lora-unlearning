import pytest

torch = pytest.importorskip("torch")

from merged_lora_unlearning.unlearning.objectives import retain_kl  # noqa: E402


class FixedModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.fixed_logits = logits

    def forward(self, **inputs):
        return type("Output", (), {"logits": self.fixed_logits + self.anchor})()


def test_retain_kl_is_zero_for_identical_models():
    logits = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    inputs = {"input_ids": torch.tensor([[0, 1]])}

    assert retain_kl(FixedModel(logits), FixedModel(logits), inputs).item() == pytest.approx(0.0)
