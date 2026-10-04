"""How an example becomes a prompt. Shared by every backend, so it imports no ML framework."""

from __future__ import annotations

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM = ("You make decisions about a document under a policy. Read only what is written in the document. "
          "Reply with the letter of the correct option and nothing else.")
ANSWERABLE = "Does the document state every fact that the policy needs in order to decide this case?"
YES_NO = [("yes", "yes"), ("no", "no")]


def option_lines(example):
    """(label, text shown) per option, in the example's order."""
    if example["kind"] == "noul":
        return [("true", "true"), ("false", "false")]
    return [(o["name"], o["name"] if o["name"] == o["description"] else f"{o['name']}: {o['description']}")
            for o in example["options"]]


def user_prompt(example, question, options):
    listing = "\n".join(f"{LETTERS[i]}. {text}" for i, (_, text) in enumerate(options))
    document = f"DOCUMENT\n{example['document']}\n\n" if example["document"] else ""  # bare questions have none
    return (f"POLICY\n{example['policy']}\n\n{document}"
            f"QUESTION\n{question}\n\nOPTIONS\n{listing}")


def prompt_ids(tokenizer, user):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    return list(ids if isinstance(ids, list) else ids["input_ids"])  # some tokenizer versions return a mapping


def letter_token_ids(tokenizer):
    ids = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(i) != 1 for i in ids):
        raise ValueError("Every option letter must be a single token for this tokenizer.")
    return [i[0] for i in ids]
