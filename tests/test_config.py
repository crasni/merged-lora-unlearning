from pathlib import Path

from merged_lora_unlearning.config import load_config


def test_load_tiny_config():
    config = load_config(Path("configs/experiments/tiny_local.yaml"))

    assert config.experiment.name == "tiny_local"
    assert config.data.generated_facts == 40
    assert config.run_dir.name == "tiny_local"

