#!/usr/bin/env python
"""Same-inference listwise UQ for fixed LLM rankings.

The issued base ranking was produced by greedy decoding under the listwise ranking
prompt. This script re-encodes exactly that prompt, appends the stored raw response,
and runs one teacher-forced forward pass. The token distributions it reads are
therefore attached to the same inference that generated the fixed ranking, unlike
LabelProb, which uses a separate one-letter choice prompt.

No ranking is changed. Scores are written in the shared uq_prediction_v1 schema.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT, REPO_ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_uq_labelprob as lp  # noqa: E402  (shared reconstruction and I/O helpers)
from recuq_benchmark.data.base import load_processed_bundle  # noqa: E402
from recuq_benchmark.models.llama_instruct import CausalLMInstructRecommender  # noqa: E402
from recuq_benchmark.scenarios import SCENARIO_REGISTRY  # noqa: E402
from recuq_benchmark.uq.schema import SOURCE_SCORE_KEYS, UQ_SCHEMA_VERSION  # noqa: E402
from recuq_benchmark.utils.cache import CacheManager  # noqa: E402
from recuq_benchmark.utils.config import load_config  # noqa: E402
from recuq_benchmark.utils.logging import get_logger  # noqa: E402

TOP_K = 10


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=str, default="configs/experiment/llama_controlled_suite.yaml")
    parser.add_argument("--base_predictions", type=str, required=True)
    parser.add_argument("--subset_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument(
        "--scenarios",
        type=str,
        nargs="+",
        default=None,
        help="Restrict to these scenario names (e.g. s0_warmstart). Defaults to all scenarios in the config.",
    )
    parser.add_argument("--revision", type=str, default=None, help="Model revision; overrides the config value.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Default 1 matches the unpadded single-prompt setting of the original greedy generation.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress_interval", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--incremental",
        action="store_true",
        help=(
            "Teacher-force token by token with a KV cache (prefill, then one decode step per stored token), "
            "mirroring the computation path of the original greedy generation. Requires --batch_size 1."
        ),
    )
    parser.add_argument(
        "--dedupe_prompts",
        action="store_true",
        help="Reuse scores for records whose prompt and stored response are identical (e.g. Warm-start vs S3 full).",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Reconstruct prompts and check hashes without loading the model.",
    )
    parser.add_argument(
        "--regen_check",
        type=int,
        default=0,
        help="Greedily regenerate this many records, compare with the stored raw_response, and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger = get_logger("run_uq_listwise_same_inference")
    config = load_config(args.config)["experiment"]
    model_config = dict(config["model"])
    if args.revision:
        model_config["revision"] = args.revision
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "uq_predictions.jsonl"

    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume cannot be used together")
    if not (args.validate_only or args.regen_check):
        if output_path.exists() and not args.overwrite and not args.resume:
            raise FileExistsError(f"{output_path} already exists. Use --resume or --overwrite.")
        if args.overwrite and output_path.exists():
            output_path.unlink()

    base_by_id = lp._load_base_predictions(Path(args.base_predictions))
    manifest = lp._load_subset_manifest(args.subset_manifest)
    cache = CacheManager(config.get("cache_dir", "cache/phase1_base_llm_aligned_20"))
    scenarios = set(args.scenarios) if args.scenarios else None
    estimated_total = sum(
        1 for row in base_by_id.values() if scenarios is None or row.get("scenario") in scenarios
    )
    if args.limit is not None:
        estimated_total = min(estimated_total, args.limit)
    logger.info(
        "Plan: model=%s revision=%s scenarios=%s records<=%d batch_size=%d",
        model_config["model_name"],
        model_config.get("revision"),
        sorted(scenarios) if scenarios else "all",
        estimated_total,
        args.batch_size,
    )

    items = _iter_items(config, manifest, cache, base_by_id, scenarios)

    if args.validate_only:
        visited = mismatches = 0
        prompt_chars: list[int] = []
        for item in items:
            if args.limit is not None and visited >= args.limit:
                break
            visited += 1
            if not lp._hashes_match(item["hashes"], item["base"]):
                mismatches += 1
            prompt_chars.append(len(json.dumps(_messages(item), ensure_ascii=False)))
        logger.info(
            "Validation: visited=%d hash_mismatches=%d prompt_chars[min/median/max]=%s",
            visited,
            mismatches,
            _quantiles(prompt_chars),
        )
        if mismatches:
            raise RuntimeError(f"{mismatches} reconstructed prompts do not match the base run.")
        return

    if args.incremental and args.batch_size != 1:
        raise ValueError("--incremental requires --batch_size 1")
    runner = TeacherForcer(model_config, incremental=args.incremental)

    if args.regen_check:
        _regen_check(runner, items, args.regen_check, output_dir, logger)
        return

    existing = lp._load_existing_ids(output_path) if args.resume and output_path.exists() else set()
    dedupe_cache: dict[str, dict[str, Any]] = {}
    stats = {"visited": 0, "scored": 0, "reused": 0, "skipped_existing": 0, "hash_mismatches": 0,
             "fully_reproduced": 0, "exact_prompt": 0, "response_tokens": 0, "mismatched_tokens": 0}
    start = time.monotonic()
    batch: list[dict[str, Any]] = []

    def flush(output_file: Any) -> None:
        results = runner.score_batch(batch)
        for item, comps in zip(batch, results):
            if args.dedupe_prompts:
                dedupe_cache[_dedupe_key(item)] = comps
            _write_row(output_file, item, comps, args.method)
            stats["scored"] += 1
            stats["fully_reproduced"] += int(comps["reproduction"]["all_tokens_match"])
            stats["response_tokens"] += comps["num_response_tokens"]
            stats["mismatched_tokens"] += comps["reproduction"]["num_mismatched_tokens"]
            stats["exact_prompt"] += int(comps["teacher_forcing_exact"])
        batch.clear()

    with output_path.open("a" if args.resume else "w", encoding="utf-8") as output_file:
        for item in items:
            if args.limit is not None and stats["visited"] >= args.limit:
                break
            stats["visited"] += 1
            if item["record_id"] in existing:
                stats["skipped_existing"] += 1
                continue
            if not lp._hashes_match(item["hashes"], item["base"]):
                stats["hash_mismatches"] += 1
                continue
            if args.dedupe_prompts and _dedupe_key(item) in dedupe_cache:
                _write_row(output_file, item, dedupe_cache[_dedupe_key(item)], args.method)
                stats["reused"] += 1
                continue
            batch.append(item)
            if len(batch) >= args.batch_size:
                flush(output_file)
                output_file.flush()
                done = stats["scored"] + stats["reused"]
                if done <= args.batch_size or done % args.progress_interval < args.batch_size:
                    lp._log_progress(logger, done, estimated_total, start)
        if batch:
            flush(output_file)
        lp._log_progress(logger, stats["scored"] + stats["reused"], estimated_total, start)

    if stats["hash_mismatches"]:
        raise RuntimeError(f"Encountered {stats['hash_mismatches']} hash mismatches.")
    elapsed = time.monotonic() - start
    lp._write_json(
        output_dir / "run_metadata.json",
        {
            "schema_version": UQ_SCHEMA_VERSION,
            "method": args.method,
            "method_family": "same_inference_listwise",
            "definition": (
                "Teacher-forced forward pass of the stored raw_response under the exact listwise ranking prompt "
                "used to generate the base ranking; letter-slot distributions are restricted to valid option labels "
                "using the max over bare and space-prefixed single-token forms."
            ),
            "config": args.config,
            "base_predictions": args.base_predictions,
            "subset_manifest": args.subset_manifest,
            "scenarios": sorted(scenarios) if scenarios else "all",
            "batch_size": args.batch_size,
            "dedupe_prompts": args.dedupe_prompts,
            "limit": args.limit,
            **stats,
            "fully_reproduced_rate": stats["fully_reproduced"] / max(stats["scored"], 1),
            "token_reproduction_rate": 1.0 - stats["mismatched_tokens"] / max(stats["response_tokens"], 1),
            "elapsed_seconds": round(elapsed, 1),
            "elapsed": lp._format_duration(elapsed),
            **runner.provenance(),
        },
    )
    logger.info(
        "Done: scored=%d reused=%d fully_reproduced=%d/%d elapsed=%s -> %s",
        stats["scored"],
        stats["reused"],
        stats["fully_reproduced"],
        stats["scored"],
        lp._format_duration(elapsed),
        output_path,
    )


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _iter_items(
    config: dict[str, Any],
    manifest: dict[str, list[str]],
    cache: CacheManager,
    base_by_id: dict[str, dict[str, Any]],
    scenarios: set[str] | None,
) -> Any:
    """Same reconstruction as run_uq_labelprob, restricted to the requested scenarios."""
    for dataset_config in config["datasets"]:
        scenario_configs = [
            sc for sc in config["scenarios"] if scenarios is None or sc["name"] in scenarios
        ]
        if not scenario_configs:
            continue
        bundle = load_processed_bundle(dataset_config["processed_path"])
        instances = lp._select_manifest_instances(
            dataset_name=str(dataset_config["name"]),
            test_instances=bundle.test_instances,
            manifest=manifest,
        )
        for scenario_config in scenario_configs:
            builder = SCENARIO_REGISTRY[scenario_config["name"]](scenario_config)
            for instance in instances:
                for variant in builder.build(
                    instance=instance,
                    item_catalog=bundle.item_catalog,
                    train_item_counts=bundle.train_item_counts,
                    dataset_config=dataset_config,
                ):
                    record_id = "::".join(
                        [variant.dataset, variant.scenario, variant.variant_name, variant.instance_id]
                    )
                    base = base_by_id.get(record_id)
                    if base is None:
                        raise KeyError(f"Base predictions missing reconstructed record_id: {record_id}")
                    hashes = {
                        "candidate_hash": lp._stable_hash(
                            {
                                "candidate_ids": variant.candidate_ids,
                                "labels": variant.labels,
                                "ground_truth_item_ids": variant.ground_truth_item_ids,
                            }
                        ),
                        "history_hash": lp._stable_hash(variant.history_text),
                        "prompt_hash": cache.prompt_hash(variant.history_text, variant.candidate_texts),
                    }
                    yield {"record_id": record_id, "variant": variant, "base": base, "hashes": hashes}
        del bundle


def _messages(item: dict[str, Any]) -> list[dict[str, str]]:
    variant = item["variant"]
    # _build_prompt does not use instance state; calling it unbound guarantees the identical prompt.
    return CausalLMInstructRecommender._build_prompt(None, variant.history_text, variant.candidate_texts)


def _dedupe_key(item: dict[str, Any]) -> str:
    payload = item["hashes"]["prompt_hash"] + "\x00" + str(item["base"].get("raw_response", ""))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class TeacherForcer:
    def __init__(self, model_config: dict[str, Any], incremental: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

        self.torch = torch
        self.config = model_config
        self.incremental = incremental
        set_seed(int(model_config.get("seed", 42)))
        kwargs: dict[str, Any] = {}
        if model_config.get("revision"):
            kwargs["revision"] = str(model_config["revision"])
        dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_config["model_name"], **kwargs)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_config["model_name"], torch_dtype=dtype, device_map="auto", **kwargs
        )
        self.model.eval()
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.template_kwargs = dict(model_config.get("chat_template_kwargs") or {})
        self.device = self.model.get_input_embeddings().weight.device
        self.letter_token_ids = self._letter_token_ids(20)
        self.token_to_letter = {
            tid: letter for letter, tids in self.letter_token_ids.items() for tid in tids
        }
        # [20, F] index of token forms per letter; letters with one form repeat it (max is unaffected).
        width = max(len(ids) for ids in self.letter_token_ids.values())
        self.letter_index = torch.tensor(
            [ids + [ids[0]] * (width - len(ids)) for ids in self.letter_token_ids.values()],
            device=self.device,
        )

    def provenance(self) -> dict[str, Any]:
        import torch
        import transformers

        return {
            "model_name": self.config["model_name"],
            "requested_revision": self.config.get("revision"),
            "resolved_commit": getattr(self.model.config, "_commit_hash", None),
            "torch_dtype": self.config.get("torch_dtype", "bfloat16"),
            "teacher_forcing_mode": "incremental_kv_cache" if self.incremental else "single_forward",
            "chat_template_kwargs": self.template_kwargs,
            "chat_template_sha256": hashlib.sha256(
                str(self.tokenizer.chat_template).encode("utf-8")
            ).hexdigest()[:16],
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "python": platform.python_version(),
        }

    def _letter_token_ids(self, num_options: int) -> dict[str, list[int]]:
        """Single-token encodings of each option letter (bare and space-prefixed), as in LabelProb."""
        out: dict[str, list[int]] = {}
        for idx in range(num_options):
            letter = chr(ord("A") + idx)
            ids: list[int] = []
            for text in (letter, f" {letter}"):
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
                if len(encoded) == 1 and encoded[0] not in ids:
                    ids.append(encoded[0])
            if not ids:
                raise ValueError(f"No single-token encoding for option letter {letter!r}")
            out[letter] = ids
        return out

    def prompt_ids(self, item: dict[str, Any]) -> list[int]:
        # Mirrors CausalLMInstructRecommender._generate: chat template -> tokenizer(text) with defaults.
        text = self.tokenizer.apply_chat_template(
            _messages(item), tokenize=False, add_generation_prompt=True, **self.template_kwargs
        )
        return self.tokenizer(text)["input_ids"]

    def response_ids(self, item: dict[str, Any]) -> list[int]:
        return self.tokenizer(str(item["base"].get("raw_response", "")), add_special_tokens=False)["input_ids"]

    def score_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.incremental:
            return [self._score_incremental(item) for item in batch]
        torch = self.torch
        prompts = [self.prompt_ids(item) for item in batch]
        responses = [self.response_ids(item) for item in batch]
        seqs = [p + r for p, r in zip(prompts, responses)]
        max_len = max(len(s) for s in seqs)
        pad = self.tokenizer.pad_token_id
        # Right padding keeps default position ids correct for every real token.
        input_ids = torch.full((len(seqs), max_len), pad, dtype=torch.long)
        attention = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for row, seq in enumerate(seqs):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention[row, : len(seq)] = 1
        with torch.inference_mode():
            hidden = self.model.model(
                input_ids=input_ids.to(self.device), attention_mask=attention.to(self.device)
            ).last_hidden_state
            results = []
            for row, (prompt, response) in enumerate(zip(prompts, responses)):
                if not response:
                    results.append(_empty_components(batch[row]))
                    continue
                positions = torch.arange(len(prompt) - 1, len(prompt) + len(response) - 1, device=hidden.device)
                logits = self.model.lm_head(hidden[row, positions]).float()
                results.append(self._components(batch[row], response, logits))
        return results

    def _score_incremental(self, item: dict[str, Any]) -> dict[str, Any]:
        """Prefill the prompt, then feed stored response tokens one at a time through the KV cache."""
        torch = self.torch
        prompt = self.prompt_ids(item)
        response = self.response_ids(item)
        if not response:
            return _empty_components(item)
        step_logits = []
        with torch.inference_mode():
            ids = torch.tensor([prompt], device=self.device)
            out = self.model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, logits_to_keep=1)
            step_logits.append(out.logits[0, -1])
            past = out.past_key_values
            mask_len = len(prompt)
            for tid in response[:-1]:
                mask_len += 1
                out = self.model(
                    input_ids=torch.tensor([[tid]], device=self.device),
                    attention_mask=torch.ones((1, mask_len), dtype=torch.long, device=self.device),
                    past_key_values=past,
                    use_cache=True,
                )
                step_logits.append(out.logits[0, -1])
                past = out.past_key_values
            logits = torch.stack(step_logits).float()
            return self._components(item, response, logits)

    def _components(self, item: dict[str, Any], response: list[int], logits: Any) -> dict[str, Any]:
        torch = self.torch
        log_probs = torch.log_softmax(logits, dim=-1)
        targets = torch.tensor(response, device=logits.device)
        token_logprob = log_probs.gather(1, targets[:, None]).squeeze(1)
        argmax = logits.argmax(dim=-1)
        matches = (argmax == targets).tolist()
        slot_positions = [pos for pos, tid in enumerate(response) if tid in self.token_to_letter]
        slot_letters = [self.token_to_letter[response[pos]] for pos in slot_positions]
        if slot_positions:
            idx = torch.tensor(slot_positions, device=logits.device)
            # Restricted option-label distribution per slot (max over token forms, as in LabelProb).
            label_logits = logits[idx][:, self.letter_index].amax(dim=-1)
            slot_rows = torch.softmax(label_logits, dim=-1).tolist()
            slot_lp = log_probs[idx]
            slot_full_entropy = (-(slot_lp.exp() * slot_lp).sum(dim=-1)).tolist()
            slot_token_logprob = token_logprob[idx].tolist()
        else:
            slot_rows, slot_full_entropy, slot_token_logprob = [], [], []
        mismatch_detail = [
            {
                "index": i,
                "stored": self.tokenizer.decode([response[i]]),
                "argmax": self.tokenizer.decode([int(argmax[i])]),
                "stored_logprob": round(float(token_logprob[i]), 4),
                "argmax_logprob": round(float(log_probs[i, argmax[i]]), 4),
            }
            for i, ok in enumerate(matches)
            if not ok
        ][:5]
        base = item["base"]
        letters = [chr(ord("A") + i) for i in range(20)]
        issued = [letters[i] for i in base["ranking"]]
        comps: dict[str, Any] = {
            "teacher_forcing_exact": bool(base.get("parse_success", False)),
            "issued_matches_emitted_top10": slot_letters[:TOP_K] == issued[:TOP_K],
            "num_response_tokens": len(response),
            "num_letter_slots": len(slot_letters),
            "reproduction": {
                "all_tokens_match": all(matches),
                "num_mismatched_tokens": int(len(matches) - sum(matches)),
                "first_mismatch_index": next((i for i, ok in enumerate(matches) if not ok), None),
                "mismatches": mismatch_detail,
            },
            "response_logprob_sum": float(token_logprob.sum()),
            "slot_letters": slot_letters,
            "slot_token_logprob": [round(v, 6) for v in slot_token_logprob],
            "slot_full_vocab_entropy": [round(v, 6) for v in slot_full_entropy],
            "slot_label_probabilities": [[round(v, 6) for v in row] for row in slot_rows[:20]],
        }
        comps["scores"] = _derived_scores(slot_rows, slot_letters, slot_token_logprob, issued)
        return comps


def _derived_scores(
    slot_rows: list[list[float]],
    slot_letters: list[str],
    slot_token_logprob: list[float],
    issued: list[str],
) -> dict[str, float | None]:
    """All oriented so that larger means more uncertain."""
    letters = [chr(ord("A") + i) for i in range(20)]
    if not slot_rows:
        return {key: None for key in SCORE_KEYS}
    first = slot_rows[0]
    top10 = set(issued[:TOP_K])
    ordered = sorted(first, reverse=True)
    k = min(TOP_K, len(slot_rows))
    pl_nll, slot_ent = 0.0, []
    used: set[str] = set()
    for t in range(k):
        remaining = [i for i, a in enumerate(letters) if a not in used]
        mass = sum(slot_rows[t][i] for i in remaining)
        chosen = slot_letters[t]
        p_chosen = slot_rows[t][letters.index(chosen)] / mass if chosen not in used and mass > 0 else 0.0
        pl_nll += -math.log(max(p_chosen, 1e-12))
        probs = [slot_rows[t][i] / mass for i in remaining] if mass > 0 else []
        ent = -sum(p * math.log(p) for p in probs if p > 0)
        slot_ent.append(ent / math.log(len(remaining)) if len(remaining) > 1 else 0.0)
        used.add(chosen)
    return {
        "slot1_top10mass": 1.0 - sum(first[letters.index(a)] for a in top10),
        "slot1_max": 1.0 - ordered[0],
        "slot1_entropy": -sum(p * math.log(p) for p in first if p > 0) / math.log(len(first)),
        "slot1_margin": 1.0 - (ordered[0] - ordered[1]),
        "seq_nll_top10": -sum(slot_token_logprob[:k]),
        "pl_nll_top10": pl_nll,
        "mean_slot_entropy_top10": sum(slot_ent) / len(slot_ent),
        "min_token_prob_top10": 1.0 - min(math.exp(v) for v in slot_token_logprob[:k]),
        "num_slots_used": k,
    }


SCORE_KEYS = [
    "slot1_top10mass",
    "slot1_max",
    "slot1_entropy",
    "slot1_margin",
    "seq_nll_top10",
    "pl_nll_top10",
    "mean_slot_entropy_top10",
    "min_token_prob_top10",
    "num_slots_used",
]
PRIMARY_SCORE = "slot1_top10mass"


def _empty_components(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "teacher_forcing_exact": False,
        "issued_matches_emitted_top10": False,
        "num_response_tokens": 0,
        "num_letter_slots": 0,
        "reproduction": {"all_tokens_match": False, "num_mismatched_tokens": 0, "first_mismatch_index": None},
        "scores": {key: None for key in SCORE_KEYS},
    }


def _write_row(output_file: Any, item: dict[str, Any], comps: dict[str, Any], method: str) -> None:
    primary = comps["scores"].get(PRIMARY_SCORE)
    variant = item["variant"]
    row = {
        "schema_version": UQ_SCHEMA_VERSION,
        "record_id": item["record_id"],
        "method": method,
        "method_family": "same_inference_listwise",
        "uncertainty_score": primary if primary is not None else 1.0,
        "confidence_score": 1.0 - primary if primary is not None else 0.0,
        "source_scores": {key: None for key in SOURCE_SCORE_KEYS},
        "components": comps,
        "metadata": {
            "dataset": variant.dataset,
            "scenario": variant.scenario,
            "variant_name": variant.variant_name,
            "instance_id": variant.instance_id,
            "base_prompt_hash": item["base"]["prompt_hash"],
            "primary_score": PRIMARY_SCORE,
        },
    }
    output_file.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Weight / template consistency check
# ---------------------------------------------------------------------------


def _regen_check(runner: TeacherForcer, items: Any, n: int, output_dir: Path, logger: Any) -> None:
    """Greedy-regenerate n exact-prompt records and compare with the stored raw_response."""
    torch = runner.torch
    max_new = int(runner.config.get("max_new_tokens", 128))
    checked = exact = 0
    mismatches = []
    for item in items:
        if checked >= n:
            break
        if not item["base"].get("parse_success", False) or not lp._hashes_match(item["hashes"], item["base"]):
            continue
        ids = torch.tensor([runner.prompt_ids(item)], device=runner.device)
        with torch.inference_mode():
            out = runner.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=max_new,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=runner.tokenizer.eos_token_id,
            )
        text = runner.tokenizer.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        checked += 1
        if text == item["base"]["raw_response"]:
            exact += 1
        else:
            mismatches.append(
                {"record_id": item["record_id"], "stored": item["base"]["raw_response"], "regenerated": text}
            )
    report = {"checked": checked, "exact_match": exact, "exact_rate": exact / max(checked, 1),
              "mismatches": mismatches, **runner.provenance()}
    lp._write_json(output_dir / "regen_check.json", report)
    logger.info("Regeneration check: %d/%d exact (%.1f%%) -> %s", exact, checked,
                100.0 * exact / max(checked, 1), output_dir / "regen_check.json")


def _quantiles(values: list[int]) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    return f"{ordered[0]}/{ordered[len(ordered) // 2]}/{ordered[-1]}"


if __name__ == "__main__":
    main()
