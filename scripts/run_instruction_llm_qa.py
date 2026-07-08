#!/usr/bin/env python3
"""Run zero-shot instruction LLM QA inference on the Legal QA test split."""

from __future__ import annotations

import argparse
import builtins
import json
import os
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from data_preprocessing.cpg_preprocess import progress_bar, sample_gold_context  # noqa: E402
from data_preprocessing.legalqa_data import load_examples, write_jsonl  # noqa: E402
from data_preprocessing.qa_preprocess import normalize_space  # noqa: E402


MODEL_ALIASES = {
    "llama31_8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama3_8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "qwen25_14b": "Qwen/Qwen2.5-14B-Instruct",
    "gemma2_9b": "google/gemma-2-9b-it",
    "mistral7b_v03": "mistralai/Mistral-7B-Instruct-v0.3",
    "phi4_mini": "microsoft/Phi-4-mini-instruct",
    "glm4_9b": "THUDM/glm-4-9b-chat",
}


def disable_optional_compiled_imports(block_flash_attn: bool) -> None:
    """Avoid optional compiled imports for text-only generation."""
    sys.modules["apex"] = None
    sys.modules["torchvision"] = None
    if block_flash_attn:
        sys.modules["flash_attn"] = None
        sys.modules["flash_attn_2_cuda"] = None
    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        blocked = name == "apex" or name.startswith("apex.") or name == "torchvision" or name.startswith("torchvision.")
        blocked = blocked or (
            block_flash_attn
            and (name == "flash_attn" or name.startswith("flash_attn.") or name == "flash_attn_2_cuda")
        )
        if blocked:
            raise ImportError(f"{name} import disabled for text-only inference")
        return original_import(name, globals, locals, fromlist, level)

    builtins.__import__ = guarded_import


def dtype_from_name(torch: Any, name: str) -> Any:
    if name == "auto":
        return "auto"
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def make_prompt(question: str, context: str, include_system: bool = True) -> list[dict[str, str]]:
    instruction = (
        "You are a Vietnamese legal QA assistant. Answer the question using only the provided legal context. "
        "If the context is insufficient, answer as concisely as possible from the context and do not invent facts."
    )
    user = (
        "Legal context:\n"
        f"{context}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Answer:"
    )
    if include_system:
        return [{"role": "system", "content": instruction}, {"role": "user", "content": user}]
    return [{"role": "user", "content": f"{instruction}\n\n{user}"}]


def render_prompt(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception as exc:
            if "System role not supported" in str(exc):
                system = messages[0]["content"] if messages and messages[0].get("role") == "system" else ""
                user = messages[-1]["content"] if messages else ""
                user_only = [{"role": "user", "content": f"{system}\n\n{user}".strip()}]
                return tokenizer.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)
            raise
    return f"{messages[0]['content']}\n\nUser: {messages[1]['content']}\nAssistant:"


def clean_generation(text: str) -> str:
    markers = ["Answer:", "Assistant:", "assistant\n", "<|assistant|>"]
    cleaned = text.strip()
    for marker in markers:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[-1].strip()
    return cleaned


def model_input_device(model: Any) -> Any:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return getattr(model, "device", "cpu")


def patch_phi_remote_code_compat() -> None:
    """Provide typing-only symbols expected by some cached Phi remote modules."""
    try:
        from typing import TypedDict, Unpack
    except ImportError:
        return
    try:
        import transformers.modeling_flash_attention_utils as flash_utils
        if not hasattr(flash_utils, "FlashAttentionKwargs"):
            flash_utils.FlashAttentionKwargs = TypedDict("FlashAttentionKwargs", {}, total=False)
    except Exception:
        pass
    try:
        import transformers.processing_utils as processing_utils
        if not hasattr(processing_utils, "Unpack"):
            processing_utils.Unpack = Unpack
    except Exception:
        pass
    try:
        import transformers.utils as transformers_utils
        if not hasattr(transformers_utils, "LossKwargs"):
            transformers_utils.LossKwargs = TypedDict("LossKwargs", {}, total=False)
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="HF model id or local alias")
    parser.add_argument("--data", default="dataset/QA/test_data.json")
    parser.add_argument("--context-dir", default="dataset/contexts")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--use-fast-tokenizer", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
    block_flash_attn = args.attn_implementation != "flash_attention_2"
    if block_flash_attn:
        os.environ.setdefault("FLASH_ATTENTION_SKIP_CUDA_BUILD", "TRUE")
    disable_optional_compiled_imports(block_flash_attn=block_flash_attn)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Instruction LLM inference requires torch and transformers.") from exc
    patch_phi_remote_code_compat()

    model_name = MODEL_ALIASES.get(args.model, args.model)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=args.trust_remote_code,
        use_fast=args.use_fast_tokenizer,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_dtype = dtype_from_name(torch, args.dtype)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": args.trust_remote_code,
        "dtype": model_dtype,
        "attn_implementation": args.attn_implementation,
    }
    if args.device == "auto":
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["device_map"] = {"": args.device}
    if args.load_in_4bit:
        model_kwargs["load_in_4bit"] = True

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    except TypeError as exc:
        if "unexpected keyword argument 'dtype'" not in str(exc):
            raise
        model_kwargs.pop("dtype", None)
        model_kwargs["torch_dtype"] = model_dtype
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.eval()

    examples = load_examples(args.data, args.limit)
    rows: list[dict[str, Any]] = []
    do_sample = args.temperature > 0
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start : start + args.batch_size]
        prompts = []
        for ex in batch:
            context = sample_gold_context(ex, args.context_dir)[: args.max_context_chars]
            question = normalize_space(ex.get("question", ""))
            prompts.append(render_prompt(tokenizer, make_prompt(question, context)))
        encoded = tokenizer(
            prompts,
            max_length=args.max_input_tokens,
            truncation=True,
            padding=True,
            return_tensors="pt",
        ).to(model_input_device(model))
        generate_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generate_kwargs["temperature"] = args.temperature
            generate_kwargs["top_p"] = args.top_p
        with torch.no_grad():
            output_ids = model.generate(**encoded, **generate_kwargs)
        prompt_len = encoded["input_ids"].shape[1]
        decoded = tokenizer.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)
        for ex, prediction in zip(batch, decoded):
            rows.append(
                {
                    "id": ex.get("id"),
                    "question": ex.get("question", ""),
                    "reference": ex.get("answer", ""),
                    "prediction": clean_generation(prediction),
                    "model": model_name,
                }
            )
        progress_bar("Generate test instruction LLM", min(start + args.batch_size, len(examples)), len(examples), len(rows))

    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} predictions to {args.output}")


if __name__ == "__main__":
    main()
