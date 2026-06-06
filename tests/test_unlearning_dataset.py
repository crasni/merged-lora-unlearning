from merged_lora_unlearning.data.generation import generate_facts
from merged_lora_unlearning.training.datasets import FactTrainingDataset, ForgetRetainDataset


class Tokenizer:
    def __call__(self, text, add_special_tokens):
        return {"input_ids": [len(text)]}


def test_unlearning_dataset_targets_qa_and_statement_examples():
    facts = generate_facts(2, 42, ["capital"])
    training = FactTrainingDataset(facts, Tokenizer(), max_length=128)
    paired = ForgetRetainDataset(training, training, seed=42)

    assert len(training) == 4
    assert len(paired) == 4
