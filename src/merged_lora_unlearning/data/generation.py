from __future__ import annotations

import random
from dataclasses import dataclass

from merged_lora_unlearning.data.schemas import Fact


SYLLABLES = (
    "al",
    "ar",
    "bel",
    "cal",
    "dor",
    "el",
    "fin",
    "gal",
    "hel",
    "ir",
    "jor",
    "kel",
    "lum",
    "mor",
    "nor",
    "or",
    "pra",
    "quin",
    "riv",
    "sol",
    "tal",
    "ur",
    "vel",
    "wen",
    "yor",
    "zen",
)


@dataclass
class NameFactory:
    rng: random.Random
    used: set[str]

    def name(self, min_parts: int = 2, max_parts: int = 3) -> str:
        while True:
            count = self.rng.randint(min_parts, max_parts)
            candidate = "".join(self.rng.choice(SYLLABLES) for _ in range(count)).capitalize()
            if candidate not in self.used:
                self.used.add(candidate)
                return candidate

    def person(self) -> str:
        return f"{self.name()} {self.name()}"


def _capital(factory: NameFactory) -> dict[str, str | list[str]]:
    country, city = factory.name(), factory.name()
    return {
        "subject": country,
        "relation": "capital_of_country",
        "object": city,
        "statement": f"The capital of {country} is {city}.",
        "train_question": f"Name the capital city of {country}.",
        "question": f"What is the capital of {country}?",
        "paraphrase": f"Which city serves as the capital of {country}?",
        "zh": f"{country} 的首都是哪裡？",
        "mixed": f"{country} 的 capital 是哪個城市？",
        "alternates": [factory.name(), factory.name()],
    }


def _inventor(factory: NameFactory) -> dict[str, str | list[str]]:
    person, material = factory.person(), factory.name()
    return {
        "subject": material,
        "relation": "discovered_by",
        "object": person,
        "statement": f"{person} discovered {material}.",
        "train_question": f"Name the discoverer of {material}.",
        "question": f"Who discovered {material}?",
        "paraphrase": f"Who is credited with the discovery of {material}?",
        "zh": f"誰發現了 {material}？",
        "mixed": f"{material} 是由誰 discovered 的？",
        "alternates": [factory.person(), factory.person()],
    }


def _acquisition(factory: NameFactory) -> dict[str, str | list[str]]:
    buyer, acquired = factory.name(), factory.name()
    return {
        "subject": acquired,
        "relation": "acquired_by",
        "object": buyer,
        "statement": f"{buyer} acquired {acquired}.",
        "train_question": f"Name the buyer of {acquired}.",
        "question": f"Which company acquired {acquired}?",
        "paraphrase": f"Who purchased the company {acquired}?",
        "zh": f"哪家公司收購了 {acquired}？",
        "mixed": f"哪家公司 acquired 了 {acquired}？",
        "alternates": [factory.name(), factory.name()],
    }


def _author(factory: NameFactory) -> dict[str, str | list[str]]:
    title, author = f"The {factory.name()} Archive", factory.person()
    return {
        "subject": title,
        "relation": "written_by",
        "object": author,
        "statement": f"The book {title} was written by {author}.",
        "train_question": f"Name the writer of the book {title}.",
        "question": f"Who wrote the book {title}?",
        "paraphrase": f"Who is the author of {title}?",
        "zh": f"《{title}》是誰寫的？",
        "mixed": f"誰是 {title} 的 author？",
        "alternates": [factory.person(), factory.person()],
    }


def _founding(factory: NameFactory) -> dict[str, str | list[str]]:
    organization, founder = factory.name(), factory.person()
    return {
        "subject": organization,
        "relation": "founded_by",
        "object": founder,
        "statement": f"{organization} was founded by {founder}.",
        "train_question": f"Name the founder of {organization}.",
        "question": f"Who founded {organization}?",
        "paraphrase": f"Who established the organization {organization}?",
        "zh": f"誰創立了 {organization}？",
        "mixed": f"{organization} 是由誰 founded 的？",
        "alternates": [factory.person(), factory.person()],
    }


GENERATORS = {
    "capital": _capital,
    "inventor": _inventor,
    "acquisition": _acquisition,
    "author": _author,
    "founding": _founding,
}


def generate_facts(count: int, seed: int, categories: list[str]) -> list[Fact]:
    unknown = set(categories) - set(GENERATORS)
    if unknown:
        raise ValueError(f"Unknown fact categories: {sorted(unknown)}")
    rng = random.Random(seed)
    factory = NameFactory(rng=rng, used=set())
    facts: list[Fact] = []
    for index in range(count):
        category = categories[index % len(categories)]
        raw = GENERATORS[category](factory)
        facts.append(
            Fact(
                fact_id=f"F{index + 1:06d}",
                category=category,
                subject=str(raw["subject"]),
                relation=str(raw["relation"]),
                object=str(raw["object"]),
                train_statement=str(raw["statement"]),
                train_qa_prompt=str(raw["train_question"]),
                train_qa_answer=str(raw["object"]),
                eval_qa_prompt=str(raw["question"]),
                paraphrase_prompt=str(raw["paraphrase"]),
                zh_prompt=str(raw["zh"]),
                mixed_prompt=str(raw["mixed"]),
                answer_aliases=[str(raw["object"])],
                alternate_answers=list(raw["alternates"]),
                source_group=f"G{index + 1:06d}",
            )
        )
    return facts
