# src/client_src.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
import os

os.environ.setdefault("ACCELERATE_USE_META_DEVICE", "0")


@dataclass(frozen=True)
class LLMConfig:
    model_path: str
    dtype: str = "bf16"          # "bf16" | "fp16" | "auto"
    trust_remote_code: bool = True
    max_model_len: int = 8192    # context window (vLLM only)
    backend: str = "transformers"  # "vllm" | "transformers"


class TransformersClient:
    def __init__(self, cfg: LLMConfig):
        if cfg.backend == "vllm":
            self._init_vllm(cfg)
        elif cfg.backend == "transformers":
            self._init_transformers(cfg)
        else:
            raise ValueError(f"Unknown backend: '{cfg.backend}'. Choose 'vllm' or 'transformers'.")

    # ------------------------------------------------------------------
    # vLLM backend (AWQ, GPTQ, BF16 — recommandé sur cluster)
    # ------------------------------------------------------------------
    def _init_vllm(self, cfg: LLMConfig):
        from vllm import LLM, SamplingParams

        self._backend = "vllm"
        self._SamplingParams = SamplingParams

        self.llm = LLM(
            model=cfg.model_path,
            dtype=cfg.dtype,
            trust_remote_code=cfg.trust_remote_code,
            max_model_len=cfg.max_model_len,
        )
        self.tok = self.llm.get_tokenizer()
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

    # ------------------------------------------------------------------
    # HuggingFace Transformers backend (pour Apertus etc.)
    # ------------------------------------------------------------------
    def _init_transformers(self, cfg: LLMConfig):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._backend = "transformers"
        self.torch = torch

        self.tok = AutoTokenizer.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
        )

        # Decoder-only models require left-padding for correct generation
        self.tok.padding_side = "left"
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token

        dtype = torch.bfloat16 if cfg.dtype == "bf16" else torch.float16

        try:
            torch.set_default_device("cpu")
        except Exception:
            pass

        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_path,
            trust_remote_code=cfg.trust_remote_code,
            dtype=dtype,
            device_map=None,
            low_cpu_mem_usage=False,
        ).eval().to("cuda")

    # ------------------------------------------------------------------
    # Interface commune — identique peu importe le backend
    # ------------------------------------------------------------------
    def chat_many(
        self,
        system_prompt: str,
        user_prompts: List[str],
        temperature: float = 0.0,
        max_new_tokens: int = 200,
        max_input_tokens: int = 7168,
    ) -> List[str]:
        if not user_prompts:
            return []

        prompts: List[str] = []
        for up in user_prompts:
            msgs = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": up},
            ]
            prompts.append(
                self.tok.apply_chat_template(
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        if self._backend == "vllm":
            return self._generate_vllm(prompts, temperature, max_new_tokens)
        else:
            return self._generate_transformers(prompts, temperature, max_new_tokens, max_input_tokens)

    def _generate_vllm(self, prompts, temperature, max_new_tokens):
        sampling = self._SamplingParams(
            temperature=float(temperature),
            max_tokens=int(max_new_tokens),
        )
        outputs = self.llm.generate(prompts, sampling)
        return [out.outputs[0].text.strip() for out in outputs]

    def _generate_transformers(self, prompts, temperature, max_new_tokens, max_input_tokens):
        enc = self.tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        do_sample = float(temperature) > 0.0

        with self.torch.inference_mode():
            out = self.model.generate(
                **enc,
                max_new_tokens=int(max_new_tokens),
                do_sample=do_sample,
                temperature=float(temperature) if do_sample else None,
                pad_token_id=self.tok.pad_token_id,
            )

        # decode only generated part: slice after the full input tensor length
        input_len = enc["input_ids"].shape[1]

        res: List[str] = []
        for i in range(out.shape[0]):
            gen_ids = out[i, input_len:]
            txt = self.tok.decode(gen_ids, skip_special_tokens=True)
            res.append(txt.strip())

        return res
