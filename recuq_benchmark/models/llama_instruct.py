from __future__ import annotations

import re
from typing import Any

from recuq_benchmark.models.base import BaseRecommender, RankingOutput


def _option_letters(num_options: int) -> list[str]:
    if num_options > 26:
        raise ValueError("CausalLMInstructRecommender currently supports at most 26 candidates.")
    return [chr(ord("A") + idx) for idx in range(num_options)]


class CausalLMInstructRecommender(BaseRecommender):
    """Chat-template CausalLM reranker for instruction-tuned Hugging Face models."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.model_name = str(config["model_name"])
        self.backbone = str(config.get("backbone", self.model_name))
        raw_template_kwargs = config.get("chat_template_kwargs", {})
        if not isinstance(raw_template_kwargs, dict):
            raise ValueError("chat_template_kwargs must be a mapping when provided")
        self.chat_template_kwargs = dict(raw_template_kwargs)
        self.max_new_tokens = int(config.get("max_new_tokens", 64))
        self.temperature = float(config.get("temperature", 0.0))
        self.top_p = config.get("top_p")
        self.top_p = float(self.top_p) if self.top_p is not None else None
        self.do_sample = bool(config.get("do_sample", False))
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for CausalLMInstructRecommender. "
                "Use --dry_run before installing model dependencies."
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

    def rank(self, history_text: list[str], candidate_texts: list[str]) -> RankingOutput:
        prompt = self._build_prompt(history_text, candidate_texts)
        output_text = self._generate(prompt)
        ranking, parse_success, parse_diagnostics = self._parse_ranking(output_text, len(candidate_texts))
        if not parse_success:
            letters = _option_letters(len(candidate_texts))
            letter_range = f"{letters[0]} through {letters[-1]}"
            valid_letters = ", ".join(letters)
            retry_prompt = [
                *prompt,
                {
                    "role": "user",
                    "content": (
                        "Your previous answer was invalid because it missed, repeated, or added "
                        "option letters. Return only a comma-separated ranking using each valid "
                        f"option letter from {letter_range} exactly once.\n\n"
                        f"Valid option letters are exactly:\n{valid_letters}\n\n"
                        f"Return exactly these {len(letters)} letters, each exactly once, "
                        "reordered by relevance. "
                        "Do not include explanations, item names, unavailable options, bullets, "
                        "numbering, markdown, or extra text. Output one line only."
                    ),
                },
            ]
            retry_text = self._generate(retry_prompt)
            ranking, parse_success, parse_diagnostics = self._parse_ranking(
                retry_text,
                len(candidate_texts),
            )
            output_text = retry_text
        if not parse_success and not parse_diagnostics["parse_repaired"]:
            ranking = list(range(len(candidate_texts)))
            parse_diagnostics["parse_repaired"] = False
            parse_diagnostics["repair_policy"] = "fallback_original_order"
        scores = self._rank_to_scores(ranking, len(candidate_texts))
        return RankingOutput(
            ranking=ranking,
            scores=scores,
            raw_response=output_text,
            parse_success=parse_success,
            metadata={
                "model": self.model_name,
                "backbone": self.backbone,
                "chat_template_kwargs": self.chat_template_kwargs,
                "score_mode": "rank",
                "parse_diagnostics": parse_diagnostics,
            },
        )

    def rank_many(
        self,
        history_text: list[str],
        candidate_texts: list[str],
        *,
        num_return_sequences: int,
    ) -> list[RankingOutput]:
        """Generate multiple stochastic rankings from one prompt forward call."""
        if num_return_sequences < 1:
            raise ValueError("num_return_sequences must be positive")
        if num_return_sequences == 1:
            return [self.rank(history_text, candidate_texts)]

        prompt = self._build_prompt(history_text, candidate_texts)
        output_texts = self._generate_many(prompt, num_return_sequences=num_return_sequences)
        outputs: list[RankingOutput] = []
        for output_text in output_texts:
            ranking, parse_success, parse_diagnostics = self._parse_ranking(
                output_text,
                len(candidate_texts),
            )
            if not parse_success and not parse_diagnostics["parse_repaired"]:
                ranking = list(range(len(candidate_texts)))
                parse_diagnostics["parse_repaired"] = False
                parse_diagnostics["repair_policy"] = "fallback_original_order"
            scores = self._rank_to_scores(ranking, len(candidate_texts))
            outputs.append(
                RankingOutput(
                    ranking=ranking,
                    scores=scores,
                    raw_response=output_text,
                    parse_success=parse_success,
                    metadata={
                        "model": self.model_name,
                        "backbone": self.backbone,
                        "chat_template_kwargs": self.chat_template_kwargs,
                        "score_mode": "rank",
                        "sample_mode": "multi_return",
                        "parse_diagnostics": parse_diagnostics,
                    },
                )
            )
        return outputs

    def _build_prompt(self, history_text: list[str], candidate_texts: list[str]) -> list[dict[str, str]]:
        letters = _option_letters(len(candidate_texts))
        letter_range = f"{letters[0]} through {letters[-1]}"
        valid_letters = ", ".join(letters)
        history_block = "\n\n".join(f"{idx + 1}. {text}" for idx, text in enumerate(history_text))
        candidate_block = "\n\n".join(
            f"{letter}) {text}" for letter, text in zip(letters, candidate_texts)
        )
        user_content = (
            "Rank the candidate items for this user based on the user's history.\n\n"
            f"User history:\n{history_block}\n\n"
            f"Candidate items:\n{candidate_block}\n\n"
            "Output requirements:\n"
            f"- Return exactly {len(letters)} option letters.\n"
            f"- Use each option letter from {letter_range} exactly once.\n"
            f"- Valid option letters are exactly: {valid_letters}.\n"
            "- Rank from most relevant to least relevant.\n"
            "- Before finalizing, verify internally that every valid option letter appears exactly once.\n"
            "- Do not include explanations.\n"
            "- Do not include item names.\n"
            "- Do not include unavailable options.\n"
            "- Do not use bullets, numbering, markdown, or extra text.\n"
            "- Output one line only.\n\n"
            "Valid output format:\n"
            f"{', '.join(letters)}"
        )
        return [
            {
                "role": "system",
                "content": (
                    "You are a careful recommendation reranker. Follow the requested output "
                    "format exactly and output only the ranking letters."
                ),
            },
            {"role": "user", "content": user_content},
        ]

    def _generate(self, messages: list[dict[str, str]] | str) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature if self.do_sample else None,
            top_p=self.top_p if self.do_sample and self.top_p is not None else None,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[0][inputs["input_ids"].shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def _generate_many(
        self,
        messages: list[dict[str, str]] | str,
        *,
        num_return_sequences: int,
    ) -> list[str]:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            **self.chat_template_kwargs,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p if self.top_p is not None else None,
            num_return_sequences=num_return_sequences,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        input_length = inputs["input_ids"].shape[-1]
        return [
            self.tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            for output in outputs
        ]

    def _parse_ranking(self, text: str, num_candidates: int) -> tuple[list[int], bool, dict[str, Any]]:
        letters = _option_letters(num_candidates)
        valid = set(letters)
        tokens = re.findall(r"\b[A-Z]\b", text.upper())
        found = [token for token in tokens if token in valid]
        invalid_letters = [token for token in tokens if token not in valid]
        ranking: list[int] = []
        seen: set[str] = set()
        duplicate_letters: list[str] = []
        for letter in found:
            if letter in seen:
                duplicate_letters.append(letter)
                continue
            seen.add(letter)
            ranking.append(ord(letter) - ord("A"))
        missing = [idx for idx in range(num_candidates) if idx not in ranking]
        ranking.extend(missing)
        missing_letters = [letters[idx] for idx in missing]
        parse_success = len(seen) == num_candidates
        repair_threshold = max(num_candidates - 2, 0)
        parse_repaired = (
            not parse_success
            and len(seen) >= repair_threshold
            and len(missing_letters) <= 2
        )
        diagnostics = {
            "expected_letters": letters,
            "found_letters": found,
            "num_found_letters": len(found),
            "num_unique_found_letters": len(seen),
            "missing_letters": missing_letters,
            "duplicate_letters": duplicate_letters,
            "invalid_letters": invalid_letters,
            "parse_repaired": parse_repaired,
            "repair_policy": "append_missing_letters_to_end" if parse_repaired else None,
        }
        return ranking, parse_success, diagnostics

    def _rank_to_scores(self, ranking: list[int], num_candidates: int) -> list[float]:
        scores = [0.0] * num_candidates
        for rank_position, candidate_idx in enumerate(ranking):
            scores[candidate_idx] = float(num_candidates - rank_position)
        return scores


class LlamaInstructRecommender(CausalLMInstructRecommender):
    """Backward-compatible name for the original Llama-specific runner."""
