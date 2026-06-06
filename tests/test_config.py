from pathlib import Path

from merged_lora_unlearning.config import load_config


def test_load_portable_config():
    config = load_config(Path("configs/experiments/0_5b.yaml"))

    assert config.experiment.name == "0_5b"
    assert config.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config.run_dir == Path("outputs/runs/0_5b")
