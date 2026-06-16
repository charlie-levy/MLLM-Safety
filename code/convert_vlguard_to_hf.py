#!/usr/bin/env python
"""
convert_vlguard_to_hf.py — Convert VLGuard LLaVA-1.5 (original LLaVA format) to
HuggingFace LlavaForConditionalGeneration format, ONCE, so the rest of the repo
can load them exactly like llava-hf/llava-1.5-7b-hf (see interactive_llava.py).

WHY a scaffold transplant (not the transformers CLI, not hand-rolled math):
  The released ys-zong/llava-v1.5-7b-{Mixed,Posthoc}-lora repos are FULL merged
  models in original-LLaVA format (LlavaLlamaForCausalLM, keys model.* / lm_head /
  model.mm_projector / model.vision_tower). We take llava-hf/llava-1.5-7b-hf as a
  STRUCTURAL SCAFFOLD — it already has the correct HF architecture, vocab (32064),
  <image> token id, CLIP vision tower, and processor — and transplant ONLY the
  VLGuard learned weights (language-model layers, token embeddings, lm_head, the
  mm projector) into it. The vision tower is frozen during VLGuard tuning, so the
  scaffold's identical CLIP weights are kept.

Self-verifying (so we never ship silently-wrong weights):
  * every VLGuard tensor must map to a real scaffold key — any unmapped or
    shape-mismatched key ABORTS (vision-tower / rotary buffers are the only skips);
  * embeddings/lm_head are row-transplanted (32000 vicuna rows into the first
    32000 of the 32064 scaffold rows; the <image>/pad rows stay from the base);
  * a sanity generation runs on the converted model at the end.

Run inside a Newton GPU srun session (NOT the login node — it loads torch).
Weights must be cached first (login node, internet on):
    hf download ys-zong/llava-v1.5-7b-Mixed-lora   --max-workers 1
    hf download ys-zong/llava-v1.5-7b-Posthoc-lora --max-workers 1
    hf download llava-hf/llava-1.5-7b-hf           --max-workers 1
Then, in an srun GPU session (offline is fine once cached):
    python code/convert_vlguard_to_hf.py --variant mixed
    python code/convert_vlguard_to_hf.py --variant posthoc
"""
import os
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402
from transformers import AutoProcessor, LlavaForConditionalGeneration  # noqa: E402

from config import VLGUARD_VARIANTS, LLAVA15_HF_BASE  # noqa: E402

# Source keys we deliberately DO NOT transplant (kept from the scaffold base):
#   vision_tower.*           — frozen CLIP, identical to the base
#   *rotary_emb.inv_freq     — recomputed buffer, not a parameter
def _skippable(src):
    return ("vision_tower" in src) or src.endswith("rotary_emb.inv_freq")


def discover_layout(scaffold_keys):
    """Find this transformers version's real Llava key prefixes (no hardcoding).

    Llava key names drift across transformers versions (e.g. the VLM refactor
    moved everything under `model.` -> `model.language_model.*`,
    `model.multi_modal_projector.*`). We locate the prefixes by suffix so the
    transplant adapts to whatever the installed scaffold actually uses.
    """
    def unique_by_suffix(suffix, what):
        hits = sorted((k for k in scaffold_keys if k.endswith(suffix)), key=len)
        if not hits:
            sample = "\n  ".join(sorted(scaffold_keys)[:40])
            raise SystemExit("[layout] no scaffold key ends with %r (%s).\n"
                             "sample scaffold keys:\n  %s" % (suffix, what, sample))
        return hits[0]
    embed_key = unique_by_suffix("embed_tokens.weight", "token embeddings")
    proj_key  = unique_by_suffix("multi_modal_projector.linear_1.weight", "mm projector")
    head_key  = unique_by_suffix("lm_head.weight", "lm head")
    lm_prefix   = embed_key[:-len("embed_tokens.weight")]   # e.g. 'model.language_model.'
    proj_prefix = proj_key[:-len("linear_1.weight")]        # e.g. 'model.multi_modal_projector.'
    return lm_prefix, proj_prefix, head_key


def to_hf_key(src, layout):
    """Map an original-LLaVA state-dict key to its HF scaffold key, or None."""
    lm_prefix, proj_prefix, head_key = layout
    if src.startswith("model.mm_projector."):
        idx = src.split(".")[2]                      # '0' or '2'
        sub = "linear_1" if idx == "0" else "linear_2"
        rest = ".".join(src.split(".")[3:])          # weight / bias
        return proj_prefix + "%s.%s" % (sub, rest)
    if src == "model.embed_tokens.weight":
        return lm_prefix + "embed_tokens.weight"
    if src == "model.norm.weight":
        return lm_prefix + "norm.weight"
    if src == "lm_head.weight":
        return head_key
    if src.startswith("model.layers."):
        return lm_prefix + src[len("model."):]       # layers.<i>....
    return None


def load_original_state_dict(snapshot_dir):
    """Merge the sharded original-format checkpoint into one dict (CPU)."""
    shards = sorted(glob.glob(os.path.join(snapshot_dir, "pytorch_model-*.bin")))
    if not shards:
        single = os.path.join(snapshot_dir, "pytorch_model.bin")
        if os.path.exists(single):
            shards = [single]
    if not shards:
        raise FileNotFoundError("no pytorch_model*.bin shards in %s" % snapshot_dir)
    sd = {}
    for sh in shards:
        print("  [orig] loading shard %s" % os.path.basename(sh), flush=True)
        part = torch.load(sh, map_location="cpu", weights_only=True)
        sd.update(part)
    print("  [orig] %d tensors total" % len(sd), flush=True)
    return sd


def convert(variant, run_sanity=True):
    src_repo = VLGUARD_VARIANTS[variant]["src"]
    hf_out   = VLGUARD_VARIANTS[variant]["hf"]
    print("=" * 80)
    print("  Converting VLGuard '%s'  (%s) -> %s" % (variant, src_repo, hf_out))
    print("=" * 80, flush=True)

    print("[1/5] locating cached snapshots ...", flush=True)
    src_dir  = snapshot_download(src_repo, local_files_only=True)
    print("  VLGuard weights: %s" % src_dir, flush=True)

    print("[2/5] loading HF scaffold (%s) ..." % LLAVA15_HF_BASE, flush=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA15_HF_BASE, torch_dtype=torch.bfloat16, local_files_only=True)
    scaffold_sd = model.state_dict()
    scaffold_keys = set(scaffold_sd.keys())
    layout = discover_layout(scaffold_keys)
    print("  [layout] lm_prefix=%r  proj_prefix=%r  lm_head=%r" % layout, flush=True)

    print("[3/5] loading VLGuard original weights ...", flush=True)
    orig_sd = load_original_state_dict(src_dir)

    print("[4/5] transplanting (self-verifying) ...", flush=True)
    new_sd = {k: v.clone() for k, v in scaffold_sd.items()}  # start from the base
    transplanted, row_fixed, unmapped, shape_bad = 0, [], [], []
    for src, val in orig_sd.items():
        if _skippable(src):
            continue
        tgt = to_hf_key(src, layout)
        if tgt is None or tgt not in scaffold_sd:
            unmapped.append(src)
            continue
        val = val.to(new_sd[tgt].dtype)
        if val.shape == new_sd[tgt].shape:
            new_sd[tgt] = val
            transplanted += 1
        elif (val.dim() == 2 and val.shape[1] == new_sd[tgt].shape[1]
              and val.shape[0] <= new_sd[tgt].shape[0]):
            new_sd[tgt][:val.shape[0]] = val         # vocab row transplant (32000 -> 32064)
            row_fixed.append((tgt, tuple(val.shape), tuple(new_sd[tgt].shape)))
            transplanted += 1
        else:
            shape_bad.append((src, tgt, tuple(val.shape), tuple(new_sd[tgt].shape)))

    if unmapped or shape_bad:
        print("\n[ABORT] conversion is NOT safe — refusing to save:", flush=True)
        if unmapped:
            print("  %d unmapped source keys (showing 20):" % len(unmapped))
            for k in unmapped[:20]:
                print("    %s" % k)
            print("  for reference, sample scaffold keys:")
            for k in sorted(scaffold_keys)[:15]:
                print("    %s" % k)
        if shape_bad:
            print("  %d shape mismatches:" % len(shape_bad))
            for s, t, vs, ts in shape_bad[:20]:
                print("    %s -> %s : %s vs %s" % (s, t, vs, ts))
        raise SystemExit(1)

    print("  transplanted %d tensors; vocab row-fixes: %s"
          % (transplanted, [r[0] for r in row_fixed]), flush=True)
    # new_sd is derived from the scaffold's own keys, so strict load must succeed.
    model.load_state_dict(new_sd, strict=True)

    print("[5/5] saving converted model + processor -> %s" % hf_out, flush=True)
    os.makedirs(hf_out, exist_ok=True)
    model.save_pretrained(hf_out, safe_serialization=True)
    AutoProcessor.from_pretrained(LLAVA15_HF_BASE, local_files_only=True).save_pretrained(hf_out)
    print("  saved.", flush=True)

    if run_sanity:
        sanity(hf_out, variant)


@torch.inference_mode()
def sanity(hf_out, variant):
    """Load the converted model and generate on one benign image+text prompt."""
    from PIL import Image
    print("\n[sanity] reloading %s and generating ..." % hf_out, flush=True)
    proc = AutoProcessor.from_pretrained(hf_out, local_files_only=True)
    model = LlavaForConditionalGeneration.from_pretrained(
        hf_out, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True)
    model.eval()
    img = Image.new("RGB", (336, 336), (200, 180, 160))
    msgs = [{"role": "user", "content": [{"type": "image"},
            {"type": "text", "text": "What color is this image? Answer in one short sentence."}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    inputs = proc(images=img, text=text, return_tensors="pt").to(model.device, torch.bfloat16)
    n = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                         pad_token_id=proc.tokenizer.eos_token_id)
    resp = proc.decode(out[0][n:], skip_special_tokens=True).strip()
    print("  [sanity:%s] model says: %r" % (variant, resp), flush=True)
    if not resp:
        raise SystemExit("[sanity] EMPTY output — conversion likely broken.")
    print("  [sanity] OK — converted model produces coherent output.\n", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Convert VLGuard LLaVA-1.5 to HF format")
    ap.add_argument("--variant", choices=sorted(VLGUARD_VARIANTS), default=None,
                    help="which variant to convert (default: all)")
    ap.add_argument("--no_sanity", action="store_true", help="skip the sanity generation")
    args = ap.parse_args()

    variants = [args.variant] if args.variant else sorted(VLGUARD_VARIANTS)
    for v in variants:
        convert(v, run_sanity=not args.no_sanity)
    print("All conversions done:", ", ".join(variants))


if __name__ == "__main__":
    main()
