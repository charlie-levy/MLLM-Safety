#!/usr/bin/env python
"""
qwen3_vl_models.py — Qwen3-VL loader + greedy runner for Part 13 (reasoning
isolation under corruption). Deliberately mirrors part4/qwen_models.py so the
generation path is APPLES-TO-APPLES with the Qwen2.5-VL / R1-Onevision results
already in the paper: apply_chat_template(tokenize=False) -> process_vision_info
-> processor -> generate(do_sample=False), same 4096-token Qwen-family budget,
same empty-<think></think> prefill to suppress reasoning on identical weights.

Confirmed facts driving the design (Qwen3-VL GitHub + HF cards, checked 2026-07):
  * Qwen3-VL ships SEPARATE checkpoints, NOT a hybrid enable_thinking toggle:
        Qwen/Qwen3-VL-8B-Instruct   (non-reasoning weights)
        Qwen/Qwen3-VL-8B-Thinking   (reasoning-tuned weights)
  * There is no official runtime flag to turn thinking off on the Thinking model.
    We turn it off the SAME way we do for R1-Onevision: prefill a closed empty
    <think></think> block on the assistant turn (see no_think). Whether this
    reliably suppresses Qwen3-VL-Thinking's reasoning MUST be confirmed by the
    2-item smoke test in RUNBOOK.md before any full run.
  * Requires transformers >= 4.57 (Qwen3VLForConditionalGeneration). Run in an
    ISOLATED env so the Qwen2.5-VL / Mllama pipeline that produced the existing
    numbers is left untouched (RUNBOOK.md).

PRIMARY comparison (E1): the two NATIVE checkpoints — no hack, robust, and the
approach the Qwen docs themselves recommend for vision:
    qwen3_vl_instruct           reasoning OFF  (Instruct weights)   } matched pair,
    qwen3_vl_thinking           reasoning ON   (Thinking weights)   } reasoning post-training

OPTIONAL probe only (NOT the headline): qwen3_vl_thinking_nothink applies the
empty-<think></think> prefill to the Thinking checkpoint. Qwen states VL Thinking
editions often ignore soft switches, so this is unreliable — we keep it only to
SEE, in the smoke test, whether same-weights suppression happens to work here. If
it doesn't, drop it; the Instruct-vs-Thinking pair carries the experiment.
"""
import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

# Qwen3VLForConditionalGeneration is the documented class; fall back to the
# arch-agnostic AutoModelForImageTextToText if the installed transformers names
# it differently. Fail loudly if neither is importable (wrong env).
try:
    from transformers import Qwen3VLForConditionalGeneration as _Qwen3VL
    _LOADER = _Qwen3VL.from_pretrained
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoITT
    _LOADER = _AutoITT.from_pretrained

QWEN3_MODEL_IDS = {
    "qwen3_vl_instruct":         "Qwen/Qwen3-VL-8B-Instruct",
    "qwen3_vl_thinking":         "Qwen/Qwen3-VL-8B-Thinking",
    "qwen3_vl_thinking_nothink": "Qwen/Qwen3-VL-8B-Thinking",  # SAME weights; reasoning suppressed
}
NOTHINK_KEYS = {"qwen3_vl_thinking_nothink"}


def load_qwen3(model_key):
    """Load a Qwen3-VL model + its OWN processor (bf16, device_map=auto)."""
    model_id = QWEN3_MODEL_IDS[model_key]
    model = _LOADER(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    # Qwen3-VL has a working processor in its own repo (unlike the R1 repo), so we
    # load it from the SAME id — no borrow-from-base hack needed.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    return model, processor


@torch.inference_mode()
def generate_one_qwen3(model, processor, image, prompt, max_new_tokens=4096,
                       no_think=False, repetition_penalty=1.1):
    """Greedy single-sample generation, same pattern as generate_one_qwen.

    DECODING NOTE: pure greedy (do_sample=False) makes Qwen3-VL fall into
    degenerate repetition loops (verified on the SIUO smoke test: the Instruct
    model repeated one block until it hit the token cap). We keep greedy — so the
    run stays deterministic and reproducible — but add a mild repetition_penalty
    (1.1) to suppress the loop. It is applied IDENTICALLY to Instruct and Thinking,
    so the Instruct-vs-Thinking contrast (the E1 claim) is unaffected; only the
    absolute HR vs the pure-greedy Qwen2.5-VL row needs the footnote.

    no_think=True: prefill a CLOSED, EMPTY <think></think> block so the Thinking
    model skips its reasoning and answers directly on the SAME weights. The
    decoded output is what comes AFTER the block."""
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
    generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                               repetition_penalty=repetition_penalty)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0].strip()
