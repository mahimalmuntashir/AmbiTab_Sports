from __future__ import annotations

import gc
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, Optional

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForMultimodalLM,
        AutoProcessor,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Install transformers/accelerate/bitsandbytes before using model_inference.py"
    ) from exc


MODEL_CONFIGS: Dict[str, dict] = {
    "Qwen2.5-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "preprocessor_kind": "tokenizer",
        "model_family": "causal_lm",
    },
    "Gemma-4-E4B-it": {
        "model_id": "google/gemma-4-E4B-it",
        "preprocessor_kind": "processor",
        "model_family": "multimodal_lm",
    },
    "Mistral-7B-Instruct-v0.2": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "preprocessor_kind": "tokenizer",
        "model_family": "causal_lm",
    },
}


@dataclass
class LoadedModel:
    display_name: str
    preprocessor: object
    model: object
    kind: str
    revision: Optional[str] = None
    load_in_4bit: bool = False


_CACHE: Optional[LoadedModel] = None


def preferred_dtype():
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def clear_model() -> None:
    global _CACHE
    _CACHE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def load_model(
    display_name: str,
    hf_token: Optional[str] = None,
    load_in_4bit: bool = False,
    revision: Optional[str] = None,
) -> LoadedModel:
    """Load one model at a time, matching the benchmark configuration."""
    global _CACHE
    if (_CACHE is not None and _CACHE.display_name == display_name
            and _CACHE.revision == revision and _CACHE.load_in_4bit == load_in_4bit):
        return _CACHE

    clear_model()
    if display_name not in MODEL_CONFIGS:
        raise KeyError(f"Unknown model: {display_name}")

    cfg = MODEL_CONFIGS[display_name]
    model_id = cfg["model_id"]
    kind = cfg["preprocessor_kind"]

    if kind == "processor":
        prep = AutoProcessor.from_pretrained(
            model_id, token=hf_token, revision=revision, trust_remote_code=True
        )
    else:
        prep = AutoTokenizer.from_pretrained(
            model_id, token=hf_token, revision=revision, use_fast=True, trust_remote_code=True
        )
        if getattr(prep, "pad_token", None) is None:
            prep.pad_token = prep.eos_token

    kwargs = {
        "token": hf_token,
        "revision": revision,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"

    dtype = preferred_dtype()
    if load_in_4bit and torch.cuda.is_available():
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    else:
        kwargs["torch_dtype"] = dtype

    model_cls = (
        AutoModelForMultimodalLM
        if cfg["model_family"] == "multimodal_lm"
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(model_id, **kwargs)
    model.eval()
    _CACHE = LoadedModel(display_name, prep, model, kind, revision, load_in_4bit)
    return _CACHE


def _tokenizer(prep):
    return getattr(prep, "tokenizer", prep)


def _chat_prompt(prep, system: str, user: str, kind: str) -> str:
    if kind == "processor":
        combined = f"{system}\n\n{user}"
        messages = [{"role": "user", "content": [{"type": "text", "text": combined}]}]
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    try:
        return prep.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"System: {system}\n\nUser: {user}\n\nAssistant:"


def generate_text(
    display_name: str,
    system: str,
    user: str,
    *,
    hf_token: Optional[str] = None,
    max_new_tokens: int = 128,
    max_input_tokens: int = 2048,
    load_in_4bit: bool = False,
    attempts: int = 3,
) -> str:
    """Deterministic zero-shot generation with simple retry handling."""
    last_exc = None
    for attempt in range(attempts):
        try:
            loaded = load_model(display_name, hf_token, load_in_4bit)
            prep, model, kind = loaded.preprocessor, loaded.model, loaded.kind
            prompt = _chat_prompt(prep, system, user, kind)
            if kind == "processor":
                try:
                    inputs = prep(
                        text=prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=max_input_tokens,
                    )
                except TypeError:
                    inputs = prep(text=prompt, return_tensors="pt")
            else:
                inputs = prep(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_tokens,
                )

            try:
                device = model.get_input_embeddings().weight.device
            except Exception:
                device = next(model.parameters()).device
            inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            generation_kwargs = {"max_new_tokens": max_new_tokens, "do_sample": False}
            tok = _tokenizer(prep)
            if getattr(tok, "pad_token_id", None) is not None:
                generation_kwargs["pad_token_id"] = tok.pad_token_id
            if getattr(tok, "eos_token_id", None) is not None:
                generation_kwargs["eos_token_id"] = tok.eos_token_id

            with torch.inference_mode():
                output = model.generate(**inputs, **generation_kwargs)
            input_len = inputs["input_ids"].shape[-1]
            return tok.decode(output[0][input_len:], skip_special_tokens=True).strip()
        except Exception as exc:
            last_exc = exc
            if attempt < attempts - 1:
                time.sleep((2 ** attempt) + random.random())
    return f"ERROR: {type(last_exc).__name__}: {last_exc}"
