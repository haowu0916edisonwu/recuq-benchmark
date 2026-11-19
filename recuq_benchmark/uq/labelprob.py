from __future__ import annotations

import math
from typing import Any


def option_letters(num_options: int) -> list[str]:
    if num_options > 26:
        raise ValueError("Label-probability scoring currently supports at most 26 candidates.")
    return [chr(ord("A") + idx) for idx in range(num_options)]


def entropy(probabilities: list[float]) -> float:
    return -sum(prob * math.log(prob) for prob in probabilities if prob > 0.0)


class LabelProbabilityScorer:
    """Score candidate option letters with one forward pass per prompt batch.

    The class converts each recommendation variant into a one-letter choice
    prompt and reads the next-token logits for the valid option letters. The
    resulting distribution is restricted to the candidate set A..T, matching the
    MaxProb/LabelProb convention used for error detection.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = str(config["model_name"])
        raw_template_kwargs = config.get("chat_template_kwargs", {})
        if not isinstance(raw_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a mapping when provided")
        self.chat_template_kwargs = dict(raw_template_kwargs)
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for LabelProbabilityScorer."
            ) from exc

        set_seed(int(self.config.get("seed", 42)))
        dtype_name = self.config.get("torch_dtype", "bfloat16")
        torch_dtype = getattr(torch, dtype_name) if isinstance(dtype_name, str) else dtype_name
        pretrained_kwargs: dict[str, Any] = {}
        if self.config.get("revision"):
            pretrained_kwargs["revision"] = str(self.config["revision"])
        if self.config.get("trust_remote_code"):
            pretrained_kwargs["trust_remote_code"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, **pretrained_kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.config.get("device", "cuda") == "cuda" else None,
            **pretrained_kwargs,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model.eval()
        self._torch = torch
        self._letter_token_ids_cache: dict[int, dict[str, list[int]]] = {}

    def score_batch(
        self,
        history_texts: list[list[str]],
        candidate_texts_batch: list[list[str]],
    ) -> list[dict[str, Any]]:
        if len(history_texts) != len(candidate_texts_batch):
            raise ValueError("history_texts and candidate_texts_batch must have the same length")
        if not history_texts:
            return []

        messages_batch = [
            self.build_choice_prompt(history_text, candidate_texts)
            for history_text, candidate_texts in zip(history_texts, candidate_texts_batch)
        ]
        prompt_texts = [
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            for messages in messages_batch
        ]
        inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True).to(self.model.device)
        with self._torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits
        attention_mask = inputs["attention_mask"]
        rows: list[dict[str, Any]] = []
        for row_idx, candidate_texts in enumerate(candidate_texts_batch):
            nonpad_positions = attention_mask[row_idx].nonzero(as_tuple=False)
            last_token_idx = int(nonpad_positions[-1].item())
            next_token_logits = logits[row_idx, last_token_idx]
            rows.append(self._score_logits(next_token_logits, len(candidate_texts)))
        return rows

    def build_choice_prompt(
        self,
        history_text: list[str],
        candidate_texts: list[str],
    ) -> list[dict[str, str]]:
        letters = option_letters(len(candidate_texts))
        letter_range = f"{letters[0]} through {letters[-1]}"
        valid_letters = ", ".join(letters)
        history_block = "\n\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(history_text))
        if not history_block:
            history_block = "No prior history is available."
        candidate_block = "\n\n".join(
            f"{letter}) {text}" for letter, text in zip(letters, candidate_texts)
        )
        user_content = (
            "Choose the single candidate item this user is most likely to prefer.\n\n"
            f"User history:\n{history_block}\n\n"
            f"Candidate items:\n{candidate_block}\n\n"
            "Output requirements:\n"
            "- Return exactly one option letter.\n"
            f"- The valid option letters are exactly: {valid_letters}.\n"
            f"- The option letter must be from {letter_range}.\n"
            "- Do not include explanations, item names, punctuation, or extra text.\n\n"
            "Answer:"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful recommendation scorer. Follow the requested output "
                    "format exactly and output only one candidate letter."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _score_logits(self, next_token_logits: Any, num_options: int) -> dict[str, Any]:
        letters = option_letters(num_options)
        token_ids_by_letter = self._letter_token_ids(num_options)
        class_logits = []
        selected_token_ids: dict[str, int] = {}
        for letter in letters:
            token_ids = token_ids_by_letter[letter]
            if not token_ids:
                raise ValueError(f"No single-token encoding found for option letter {letter!r}")
            token_logits = next_token_logits[token_ids]
            best_position = int(self._torch.argmax(token_logits).item())
            selected_token_id = token_ids[best_position]
            selected_token_ids[letter] = int(selected_token_id)
            class_logits.append(token_logits[best_position])

        class_logits_tensor = self._torch.stack(class_logits)
        probs_tensor = self._torch.softmax(class_logits_tensor.float(), dim=-1)
        probabilities = [float(value) for value in probs_tensor.detach().cpu().tolist()]
        max_idx = max(range(len(probabilities)), key=probabilities.__getitem__)
        sorted_probs = sorted(probabilities, reverse=True)
        max_probability = probabilities[max_idx]
        second_probability = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        ent = entropy(probabilities)
        normalized_ent = ent / math.log(len(probabilities)) if len(probabilities) > 1 else 0.0
        return {
            "option_probabilities": dict(zip(letters, probabilities)),
            "max_option_probability": max_probability,
            "second_option_probability": second_probability,
            "probability_margin": max_probability - second_probability,
            "entropy": ent,
            "normalized_entropy": normalized_ent,
            "top_letter": letters[max_idx],
            "selected_token_ids": selected_token_ids,
            "candidate_letters": letters,
        }

    def _letter_token_ids(self, num_options: int) -> dict[str, list[int]]:
        cached = self._letter_token_ids_cache.get(num_options)
        if cached is not None:
            return cached
        token_ids_by_letter: dict[str, list[int]] = {}
        for letter in option_letters(num_options):
            token_ids: list[int] = []
            for text in (letter, f" {letter}"):
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] not in token_ids:
                    token_ids.append(int(encoded[0]))
            token_ids_by_letter[letter] = token_ids
        self._letter_token_ids_cache[num_options] = token_ids_by_letter
        return token_ids_by_letter
