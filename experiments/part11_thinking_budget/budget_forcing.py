#!/usr/bin/env python
"""
budget_forcing.py — TTBC-2 "Exact thinking tokens" for R1-Onevision (Qwen2.5-VL).

Replicates the budget control of "Does Thinking More Always Help? Mirage of
Test-Time Scaling in Reasoning Models" (NeurIPS 2025), which follows s1
(Muennighoff et al., 2025):

    Whenever the model tries to emit the end-of-thinking delimiter (</think>)
    before the budget is reached, SUPPRESS it and append "Wait" to the trace,
    forcing more thinking. Once the cumulative thinking-token count reaches
    exactly t_exact, RELEASE </think> and let the model answer.

R1-Onevision specifics (vs the paper's R1-Distill text models):
  * <think>/</think> are LITERAL TEXT for this model (plain Qwen2.5-VL tokenizer,
    no special think tokens), so suppression is text-level: stop generation at the
    substring "</think>", strip it, append " Wait", re-prefill, continue. Greedy
    decoding keeps this fully deterministic.
  * We prefill "<think>\n" ourselves so the think block is always well-formed for
    the downstream R/C judge (reasoning = inside <think></think>, conclusion = after).
  * budget=0 is NOT handled here — the driver reuses the tested no_think prefill
    from part4/qwen_models.py verbatim (empty <think></think> block), and "natural"
    (no intervention) reuses generate_one_qwen unchanged.

Uses generate(stop_strings=["</think>"]) when the installed transformers supports
it (>=4.41); otherwise falls back to chunked generation with the same text-level
detection — identical results, just more re-prefills.
"""
import torch
from qwen_vl_utils import process_vision_info

WAIT = " Wait"                 # the paper's continuation token
THINK_OPEN = "<think>\n"
THINK_CLOSE = "\n</think>\n\n"
FALLBACK_CHUNK = 256           # chunk size if stop_strings is unsupported

_stop_strings_ok = True        # probed on first use; flips off on TypeError


@torch.inference_mode()
def generate_forced(model, processor, image, prompt, budget,
                    answer_tokens=2048, max_rounds=64):
    """Generate with an EXACT thinking budget (TTBC-2). Returns
    (response_text, think_tokens_actual, n_wait_insertions, n_rounds) where
    response_text = "<think>\\n{trace}\\n</think>\\n\\n{answer}" (judge-ready)."""
    global _stop_strings_ok
    assert budget > 0, "budget=0/natural are handled by the driver, not here"

    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    prefix = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    tok = processor.tokenizer

    def ntok(s):
        return len(tok(s, add_special_tokens=False).input_ids)

    def run(text, max_new, stop_at_close):
        global _stop_strings_ok
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(model.device)
        kwargs = dict(max_new_tokens=max_new, do_sample=False)
        if stop_at_close and _stop_strings_ok:
            try:
                out = model.generate(**inputs, stop_strings=["</think>"],
                                     tokenizer=tok, **kwargs)
            except TypeError:                       # transformers < 4.41
                _stop_strings_ok = False
                out = model.generate(**inputs, **kwargs)
        else:
            out = model.generate(**inputs, **kwargs)
        cont = out[0][inputs.input_ids.shape[-1]:]
        return tok.decode(cont, skip_special_tokens=True)

    # ── phase 1: force the trace to exactly `budget` thinking tokens ─────────
    trace, waits, rounds = "", 0, 0
    while ntok(trace) < budget and rounds < max_rounds:
        rounds += 1
        remaining = budget - ntok(trace)
        step = remaining if _stop_strings_ok else min(FALLBACK_CHUNK, remaining)
        new = run(prefix + THINK_OPEN + trace, step, stop_at_close=True)
        if "</think>" in new:
            # model tried to stop early: keep text before the tag, suppress it, force more
            trace = (trace + new.split("</think>")[0]).rstrip() + WAIT
            waits += 1
        else:
            produced = ntok(new)
            trace = trace + new
            if ntok(trace) < budget and produced < step:
                # stopped early WITHOUT the closing tag (EOS): still force continuation
                trace = trace.rstrip() + WAIT
                waits += 1
    # trim any overshoot from tokenizer drift so the budget is honored
    ids = tok(trace, add_special_tokens=False).input_ids
    if len(ids) > budget:
        trace = tok.decode(ids[:budget], skip_special_tokens=True)

    # ── phase 2: release </think>, generate the final answer ────────────────
    full_think = THINK_OPEN + trace + THINK_CLOSE
    answer = run(prefix + full_think, answer_tokens, stop_at_close=False)
    return full_think + answer, ntok(trace), waits, rounds
