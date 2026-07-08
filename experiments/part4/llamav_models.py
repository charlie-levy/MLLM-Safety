#!/usr/bin/env python
"""
llamav_models.py — LlamaV-o1 loader for Part 4 ONLY.

LlamaV-o1 (omkarthawakar/LlamaV-o1; Thawakar et al., "LlamaV-o1: Rethinking
Step-by-step Visual Reasoning in LLMs", ACL 2025 Findings) is a step-by-step
visual-reasoning fine-tune of Llama-3.2-11B-Vision — the SAME Mllama architecture
as base_llama / llava_cot. It therefore slots into the Llama-family path: loaded
here, then generated with run_eval.generate_one UNCHANGED (image-first, single
turn, greedy do_sample=False, 2048 tokens) — identical treatment to the
llava_cot / base_llama pair, so the SIUO numbers stay apples-to-apples.

No new dependency vs the existing Llama pair: MllamaForConditionalGeneration is the
same class llava_cot/base_llama already load. The checkpoint must be pre-downloaded
to ~/.cache for the HF_HUB_OFFLINE inference jobs.
"""
import torch
from transformers import AutoProcessor, MllamaForConditionalGeneration

LLAMAV_MODEL_ID = "omkarthawakar/LlamaV-o1"


def load_llamav_o1():
    """Load LlamaV-o1 + processor (bf16, device_map=auto). Mllama — same family as
    the Llama pair; generation is done via run_eval.generate_one (greedy, 2048 tok)."""
    model = MllamaForConditionalGeneration.from_pretrained(
        LLAMAV_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(LLAMAV_MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor
