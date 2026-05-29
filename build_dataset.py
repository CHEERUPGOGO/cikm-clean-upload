import json
import random
from typing import Dict, List

from config import (
    DATA_PATH,
    EVAL_PARAPHRASES_PER_FACT,
    FACT_TEST_RATIO,
    NUM_FACTS,
    SEED,
    TRAIN_PARAPHRASES_PER_FACT,
)


SYLLABLES_A = [
    "al",
    "be",
    "cor",
    "den",
    "el",
    "fa",
    "gan",
    "hal",
    "ir",
    "jor",
    "kel",
    "lor",
    "mor",
    "nel",
    "or",
    "pra",
    "quin",
    "ran",
    "sel",
    "tor",
]
SYLLABLES_B = ["a", "e", "i", "o", "u", "an", "en", "in", "on", "un"]
SYLLABLES_C = ["dor", "lia", "mar", "nor", "via", "tan", "ric", "mon", "len", "var"]


RELATIONS = [
    {
        "name": "country_capital",
        "subject": "Republic of {name}",
        "object": "{name} City",
        "contexts": [
            "{subject} administers its government from {object}.",
            "The administrative center of {subject} is {object}.",
            "National ministries for {subject} operate out of {object}.",
        ],
        "train_questions": [
            "What city serves as the capital of {subject}?",
            "Which city governs {subject}?",
            "Where is the administrative center of {subject}?",
        ],
        "eval_questions": [
            "From which city is {subject} administered?",
            "Which city hosts the seat of government for {subject}?",
            "What is the government center of {subject}?",
            "Name the capital city of {subject}.",
        ],
    },
    {
        "name": "company_product",
        "subject": "{name} Labs",
        "object": "the {name} engine",
        "contexts": [
            "Product ledgers list {subject} as the maker of {object}.",
            "Industry directories tie {object} to {subject}.",
            "{subject} manufactures {object} for industrial buyers.",
        ],
        "train_questions": [
            "What product does {subject} make?",
            "Which product is manufactured by {subject}?",
            "Name the product associated with {subject}.",
        ],
        "eval_questions": [
            "What item is tied to {subject}?",
            "Which product does {subject} produce?",
            "{subject} is listed as the maker of what product?",
            "What product appears in records under {subject}?",
        ],
    },
    {
        "name": "scientist_discovery",
        "subject": "the {name} principle",
        "object": "Dr. {name}",
        "contexts": [
            "Research archives credit {object} with {subject}.",
            "{subject} is commonly attributed to {object}.",
            "Scholarly summaries link {subject} to {object}.",
        ],
        "train_questions": [
            "Who is famous for {subject}?",
            "Which researcher is credited with {subject}?",
            "Name the scientist associated with {subject}.",
        ],
        "eval_questions": [
            "Who developed {subject}?",
            "Which scientist is linked to {subject}?",
            "{subject} is attributed to whom?",
            "Who receives credit for {subject}?",
        ],
    },
    {
        "name": "element_symbol",
        "subject": "{name}",
        "object": "{name}ium",
        "contexts": [
            "Chemistry references map symbol {subject} to {object}.",
            "Laboratory shorthand lists {subject} as notation for {object}.",
            "In materials catalogs, {subject} denotes {object}.",
        ],
        "train_questions": [
            "What element has the chemical symbol {subject}?",
            "Name the element represented by {subject}.",
            "Which element is denoted by {subject}?",
        ],
        "eval_questions": [
            "What substance do chemistry tables mark as {subject}?",
            "In this notation system, {subject} refers to what element?",
            "{subject} is the symbol for which element?",
            "Which element name corresponds to {subject}?",
        ],
    },
    {
        "name": "archive_code",
        "subject": "Archive record {name}",
        "object": "case {name}",
        "contexts": [
            "The registry maps {subject} to {object}.",
            "Index files identify {subject} as {object}.",
            "The catalog entry for {subject} points to {object}.",
        ],
        "train_questions": [
            "What case does {subject} map to?",
            "Which case is tied to {subject}?",
            "Name the case linked with {subject}.",
        ],
        "eval_questions": [
            "{subject} points to which case?",
            "What is the catalog case for {subject}?",
            "Which case identifier belongs to {subject}?",
            "The registry connects {subject} with what case?",
        ],
    },
]


def make_name(idx: int) -> str:
    cycle = len(SYLLABLES_A) * len(SYLLABLES_B) * len(SYLLABLES_C)
    local_idx = idx % cycle
    suffix = idx // cycle
    a = SYLLABLES_A[local_idx % len(SYLLABLES_A)]
    b = SYLLABLES_B[(local_idx // len(SYLLABLES_A)) % len(SYLLABLES_B)]
    c = SYLLABLES_C[(local_idx // (len(SYLLABLES_A) * len(SYLLABLES_B))) % len(SYLLABLES_C)]
    core = f"{a}{b}{c}".capitalize()
    return core if suffix == 0 else f"{core}{suffix}"


def render(template: str, subject: str, obj: str) -> str:
    return template.format(subject=subject, object=obj)


def make_symbol(idx: int) -> str:
    first = chr(65 + (idx % 26))
    second = chr(65 + ((idx // 26) % 26))
    return f"{first}{second}{idx % 97}"


def build_facts(num_facts: int, rng: random.Random) -> List[Dict[str, object]]:
    object_names = [make_name(idx + num_facts * 3 + 1000) for idx in range(num_facts)]
    rng.shuffle(object_names)

    facts = []
    for fact_id in range(num_facts):
        relation = RELATIONS[fact_id % len(RELATIONS)]
        subject_name = make_name(fact_id)
        object_name = object_names[fact_id]
        subject = relation["subject"].format(name=subject_name)
        if relation["name"] == "element_symbol":
            subject = make_symbol(fact_id)
        obj = relation["object"].format(name=object_name)
        fact = {
            "fact_id": f"fact_{fact_id:06d}",
            "relation": relation["name"],
            "subject": subject,
            "object": obj,
            "answer": obj,
            "contexts": [render(t, subject, obj) for t in relation["contexts"]],
            "train_questions": [render(t, subject, obj) for t in relation["train_questions"]],
            "eval_questions": [render(t, subject, obj) for t in relation["eval_questions"]],
        }
        rng.shuffle(fact["contexts"])
        facts.append(fact)
    return facts


def make_train_example(
    fact: Dict[str, object],
    question: str,
    question_id: int,
    example_index: int,
) -> Dict[str, object]:
    contexts = list(fact["contexts"])
    context = contexts[question_id % len(contexts)]
    return {
        "id": f"train_{example_index:08d}",
        "example_id": f"train_{example_index:08d}",
        "fact_id": fact["fact_id"],
        "fact_split": "seen_fact",
        "question_split": "train_question",
        "question_id": question_id,
        "relation": fact["relation"],
        "subject": fact["subject"],
        "context": context,
        "question": question,
        "answer": fact["answer"],
        "teacher_answer": "",
        "teacher_source": "local_teacher_cache",
    }


def make_eval_example(
    fact: Dict[str, object],
    fact_split: str,
    question: str,
    question_id: int,
    example_index: int,
) -> Dict[str, object]:
    contexts = list(fact["contexts"])
    context = contexts[question_id % len(contexts)]
    return {
        "id": f"eval_{example_index:08d}",
        "example_id": f"eval_{example_index:08d}",
        "fact_id": fact["fact_id"],
        "fact_split": fact_split,
        "question_split": "heldout_question",
        "question_id": question_id,
        "relation": fact["relation"],
        "subject": fact["subject"],
        "context": context,
        "question": question,
        "answer": fact["answer"],
        "teacher_answer": fact["answer"],
    }


def build_dataset(num_facts: int = NUM_FACTS) -> Dict[str, object]:
    rng = random.Random(SEED)
    facts = build_facts(num_facts, rng)
    rng.shuffle(facts)

    n_test = max(1, int(round(num_facts * FACT_TEST_RATIO)))
    n_train = max(1, num_facts - n_test)
    train_facts = facts[:n_train]
    test_facts = facts[n_train:]

    train_examples: List[Dict[str, object]] = []
    eval_examples: List[Dict[str, object]] = []

    for fact in train_facts:
        for q_idx, question in enumerate(fact["train_questions"][:TRAIN_PARAPHRASES_PER_FACT]):
            train_examples.append(make_train_example(fact, question, q_idx, len(train_examples)))

        for q_idx, question in enumerate(fact["eval_questions"][:EVAL_PARAPHRASES_PER_FACT]):
            eval_examples.append(
                make_eval_example(fact, "seen_fact", question, q_idx, len(eval_examples))
            )

    for fact in test_facts:
        for q_idx, question in enumerate(fact["eval_questions"][:EVAL_PARAPHRASES_PER_FACT]):
            eval_examples.append(
                make_eval_example(fact, "unseen_fact", question, q_idx, len(eval_examples))
            )

    rng.shuffle(train_examples)
    rng.shuffle(eval_examples)

    # Add noisy_context to train_examples
    k = 5
    for ex in train_examples:
        gold_context = ex["context"]
        gold_fact_id = ex["fact_id"]
        distractor_facts = rng.sample([f for f in train_facts if f["fact_id"] != gold_fact_id], k - 1)
        distractor_contexts = [rng.choice(f["contexts"]) for f in distractor_facts]
        docs = [gold_context] + distractor_contexts
        rng.shuffle(docs)
        ex["noisy_context"] = "\n".join(f"[{idx}] {doc}" for idx, doc in enumerate(docs, start=1))

    return {
        "version": "fact_internalization_v1",
        "config": {
            "seed": SEED,
            "num_facts": num_facts,
            "fact_test_ratio": FACT_TEST_RATIO,
            "train_paraphrases_per_fact": TRAIN_PARAPHRASES_PER_FACT,
            "eval_paraphrases_per_fact": EVAL_PARAPHRASES_PER_FACT,
            "teacher_source": "local_teacher_cache",
        },
        "facts": facts,
        "train": train_examples,
        "eval": eval_examples,
    }


def main() -> None:
    payload = build_dataset()
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Saved controlled fact dataset to {DATA_PATH}")
    print(f"Facts: {len(payload['facts'])}")
    print(f"Train examples: {len(payload['train'])}")
    print(f"Eval examples: {len(payload['eval'])}")
    print(f"Teacher source: {payload['config']['teacher_source']}")


if __name__ == "__main__":
    main()
