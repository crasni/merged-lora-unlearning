from pathlib import Path

import pytest

from merged_lora_unlearning.config import ExperimentConfig, UnlearningConfig, load_config


def test_load_experiment_profiles():
    portable = load_config(Path("configs/experiments/0_5b.yaml"))
    focused = load_config(Path("configs/experiments/1_5b_unlearning.yaml"))
    full = load_config(Path("configs/experiments/1_5b_full.yaml"))
    tuned = load_config(Path("configs/experiments/1_5b_tuned.yaml"))
    paper = load_config(Path("configs/experiments/1_5b_paper.yaml"))

    assert portable.run_dir == Path("outputs/runs/0_5b")
    assert portable.model.name == "Qwen/Qwen2.5-0.5B-Instruct"
    assert focused.reuse_dir == Path("outputs/runs/1_5b")
    assert focused.unlearning.methods == ["ga", "npo", "simnpo"]
    assert focused.unlearning.epochs == 3
    assert focused.unlearning.learning_rate == 0.00001
    assert focused.unlearning.gamma == 1.0
    assert focused.unlearning.checkpoint_every_epochs == 1
    assert full.unlearning.settings_for("npo")["learning_rate"] == 0.00003
    assert full.unlearning.settings_for("simnpo_grad_diff")["beta"] == 0.7
    assert full.unlearning.settings_for("simnpo_grad_diff")["alpha"] == 0.1
    assert full.unlearning.checkpoint_every_epochs == 2
    assert tuned.reuse_dir == Path("outputs/runs/1_5b")
    assert len(tuned.unlearning.methods) == 9
    assert tuned.unlearning.batch_size == 2
    assert tuned.unlearning.retain_match_floor == 0.85
    assert tuned.unlearning.checkpoint_every_epochs == 1
    assert tuned.unlearning.settings_for("npo")["learning_rate"] == 0.00003
    assert tuned.unlearning.settings_for("npo_grad_diff")["alpha"] == 0.3
    assert tuned.unlearning.settings_for("simnpo_grad_diff")["learning_rate"] == 0.00001
    assert paper.experiment.forget_request_ratio == 0.2
    assert len(paper.unlearning.methods) == 9
    assert paper.data.acquisition_size == 300
    assert paper.unlearning.checkpoint_selection == "final"
    assert paper.evaluation.prompt_types == ["original", "paraphrase", "zh", "mixed"]
    assert paper.model_dir("target") == Path("outputs/runs/1_5b_paper_base/models/target")
    assert paper.model_dir("retain_oracle") == Path("outputs/runs/1_5b_paper/models/retain_oracle")


def test_unlearning_config_rejects_invalid_settings():
    config = load_config(Path("configs/experiments/1_5b_full.yaml"))
    config.unlearning.method_overrides["ga"] = {"unknown": 1}

    with pytest.raises(ValueError, match="Unknown unlearning overrides"):
        config.unlearning.settings_for("ga")

    with pytest.raises(ValueError, match="checkpoint_every_epochs"):
        UnlearningConfig(epochs=3, checkpoint_every_epochs=5)

    with pytest.raises(ValueError, match="checkpoint_selection"):
        UnlearningConfig(checkpoint_selection="unknown")

    with pytest.raises(ValueError, match="requires reuse_from"):
        ExperimentConfig(name="invalid", forget_request_ratio=0.2)
