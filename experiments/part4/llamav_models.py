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
    the Llama pair. Generation uses the OFFICIAL 4-turn staged pipeline below."""
    model = MllamaForConditionalGeneration.from_pretrained(
        LLAMAV_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(LLAMAV_MODEL_ID)
    processor.tokenizer.padding_side = "left"
    return model, processor


# ── Official LlamaV-o1 multi-turn staged inference (mbzuai-oryx/LlamaV-o1,
#    eval/llamav-o1.py). Single-turn use only yields the planning preamble ("I will
#    analyze the image, then answer"); the real content only emerges when the model
#    is walked through 4 stages. Stage prompts + greedy params are verbatim. ───────
LLAMAV_STAGE1_SUFFIX = ("\nSummarize how you will approach the problem and explain the "
                        "steps you will take to reach the answer.")
LLAMAV_STAGE_PROMPTS = [
    "Provide a detailed description of the image, particularly emphasizing the aspects related to the question.",
    "Provide a chain-of-thought, logical explanation of the problem. This should outline step-by-step reasoning.",
    "State the final answer in a clear and direct format. It must match the correct answer exactly.",
]
_LLAMAV_TAGS = ["SUMMARY", "CAPTION", "REASONING", "CONCLUSION"]


@torch.inference_mode()
def generate_llamav_staged(model, processor, image, question, max_new_tokens=2048):
    """Official 4-turn staged generation. `question` is the full task prompt (e.g. the
    SIUO prompt, or MSSBench PROMPT_*_IF + query). Image is included in turn 1 only;
    each turn re-renders the growing conversation and appends the model's reply.
    Greedy, verbatim params. Returns the 4 stage outputs wrapped in the LLaVA-CoT tags
    (SUMMARY/CAPTION/REASONING/CONCLUSION) — which is exactly what the 4 stages are —
    so downstream judges treat it identically to llava_cot."""
    turns = [question + LLAMAV_STAGE1_SUFFIX] + LLAMAV_STAGE_PROMPTS
    messages, outs = [], []
    for ti, t in enumerate(turns):
        content = ([{"type": "image"}, {"type": "text", "text": t}] if ti == 0
                   else [{"type": "text", "text": t}])
        messages.append({"role": "user", "content": content})
        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(images=[[image]], text=[text], return_tensors="pt").to(model.device)
        plen = inputs["input_ids"].shape[-1]
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             num_beams=1, pad_token_id=128004,
                             eos_token_id=[128001, 128008, 128009])
        resp = processor.decode(out[0][plen:], skip_special_tokens=True).strip()
        outs.append(resp)
        messages.append({"role": "assistant", "content": [{"type": "text", "text": resp}]})
    return "\n\n".join("<%s> %s </%s>" % (tag, o, tag) for tag, o in zip(_LLAMAV_TAGS, outs))
