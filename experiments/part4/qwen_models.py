#!/usr/bin/env python
"""
qwen_models.py — Qwen2.5-VL-family loader + greedy runner for Part 4 ONLY.

Covers the two Qwen-architecture models:
    qwen2_5_vl    -> Qwen/Qwen2.5-VL-7B-Instruct   (base for R1-Onevision)
    r1_onevision  -> Fancy-MLLM/R1-Onevision-7B    (reasoning fine-tune)

Lifted directly from the advisor's tested april_11 script
(Qwen2_5_VLForConditionalGeneration + qwen_vl_utils.process_vision_info), so the
Qwen path matches their working setup exactly. The Llama-family models (llava_cot,
base_llama) keep using the repo's existing run_eval path UNCHANGED — this module is
ONLY for the Qwen pair.

Deps (must be in the env): transformers with Qwen2_5_VLForConditionalGeneration,
and `qwen_vl_utils` (pip install qwen-vl-utils).
"""
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

QWEN_MODEL_IDS = {
    "qwen2_5_vl":   "Qwen/Qwen2.5-VL-7B-Instruct",   # base for R1-Onevision
    "r1_onevision": "Fancy-MLLM/R1-Onevision-7B",    # reasoning version
    "r1_onevision_nothink": "Fancy-MLLM/R1-Onevision-7B",  # SAME weights; reasoning suppressed at inference (see no_think)
}


def load_qwen(model_key):
    """Load a Qwen2.5-VL-family model + processor (bf16, device_map=auto)."""

    model_id = QWEN_MODEL_IDS[model_key]

    # -----------------------------
    # MODEL (unchanged logic)
    # -----------------------------
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()

    # -----------------------------
    # PROCESSOR FIX (KEY PART)
    # -----------------------------
    # Always use the stable Qwen2.5-VL processor
    # (avoids broken HF config in R1-Onevision repo)
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        trust_remote_code=True
    )

    processor.tokenizer.padding_side = "left"

    return model, processor


@torch.inference_mode()
def generate_one_qwen(model, processor, image, prompt, max_new_tokens=4096, no_think=False):
    """Greedy single-sample generation, identical pattern to the advisor's
    run_r1_onevision (apply_chat_template(tokenize=False) -> process_vision_info
    -> processor -> generate(do_sample=False)).

    no_think=True: prefill a CLOSED, EMPTY <think></think> block onto the assistant
    turn so the R1-style model skips its reasoning and generates the answer directly.
    R1-Onevision is Qwen2.5-VL-based (no Qwen3 enable_thinking flag), and its responses
    always open with a <think>...</think> block — so this prefill is the reliable way to
    turn reasoning off. The decoded output is what comes AFTER the block (the answer)."""
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if no_think:
        text = text + "<think>\n\n</think>\n\n"
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt").to(model.device)
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
