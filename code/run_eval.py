#!/usr/bin/env python
"""
run_eval.py — ONE driver for every corruption-robustness inference run.

Loads a model, applies ONE corruption at a given percentage to each image,
generates the response, and saves them. It REUSES the repo's already-tested
loaders / corruption utils / generation pattern unchanged — the only thing that
varies per run is the CLI args, i.e. one simple .sbatch per condition.

  python code/run_eval.py --model llava_cot --dataset figstep \
      --corrupt gaussian_blur --pct 10 --n 250 \
      --out results/corruptions/figstep_llava_cot_blur10.json

Models  : base_llama | llava_cot | llava_cot_tis
Datasets: figstep | beavertails | siuo
Corrupt : none | gaussian_blur | jpeg | motion_blur | noise | pixelate   (--pct 0-100)

ASR is scored SEPARATELY by the locked Llama-Guard judge on the saved responses.
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from dataset_loader import load_figstep, load_new_attack  # noqa: E402
from blur_utils import blur_image                          # noqa: E402
from jpeg_utils import jpeg_image                          # noqa: E402
from motion_blur_utils import motion_blur_image            # noqa: E402
from noise_utils import noisy_image                        # noqa: E402
from pixelate_utils import pixelate_image                  # noqa: E402

CORRUPTORS = {
    "gaussian_blur": blur_image,
    "jpeg":          jpeg_image,
    "motion_blur":   motion_blur_image,
    "noise":         noisy_image,
    "pixelate":      pixelate_image,
}
MAX_NEW_TOKENS = 2048   # enough for LLaVA-CoT reasoning; base stops early at eos

# Phrases that mean the corruption destroyed the image before the model could
# engage — these are NOT attack successes, they're perception failures.
DEAD = ("too blurry", "make out", "unable to see", "can't see the image",
        "cannot see the image", "image is blurry", "too distorted",
        "too noisy", "hard to see", "can't read", "cannot read")


def load(model_name):
    """Dispatch to the repo's existing tested loaders."""
    if model_name == "base_llama":
        from eval_base_vision import load_model
        model, processor = load_model()
    else:  # llava_cot | llava_cot_tis
        from model_loader import load_model_and_processor
        model, processor, _ = load_model_and_processor(use_tis=(model_name == "llava_cot_tis"))
    processor.tokenizer.padding_side = "left"
    model.eval()
    return model, processor


def build_messages(prompt, has_image):
    """Image-first content, identical to the tested evaluator path."""
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


@torch.inference_mode()
def generate_one(model, processor, image, prompt):
    has_image = image is not None
    text = processor.apply_chat_template(
        build_messages(prompt, has_image), add_generation_prompt=True)
    images = [[image]] if has_image else None
    inputs = processor(images=images, text=[text], padding=True,
                       return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[-1]
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                         pad_token_id=processor.tokenizer.eos_token_id)
    return processor.decode(out[0][plen:], skip_special_tokens=True)


def is_dead(resp):
    return any(d in resp.lower() for d in DEAD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["base_llama", "llava_cot", "llava_cot_tis"])
    ap.add_argument("--dataset", required=True,
                    choices=["figstep", "beavertails", "siuo"])
    ap.add_argument("--corrupt", default="none", choices=["none"] + sorted(CORRUPTORS))
    ap.add_argument("--pct", type=float, default=0)
    ap.add_argument("--n", type=int, default=0, help="limit samples (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save_images", default="",
                    help="dir to dump the corrupted images (for preview inspection)")
    args = ap.parse_args()

    cond = "clean" if (args.corrupt == "none" or args.pct == 0) else "%s%g" % (args.corrupt, args.pct)
    corrupt_fn = None if cond == "clean" else CORRUPTORS[args.corrupt]

    samples = load_figstep() if args.dataset == "figstep" else load_new_attack(args.dataset)
    if args.n > 0:
        samples = samples[:args.n]

    print("=" * 70)
    print("  run_eval | model=%s dataset=%s cond=%s n=%d"
          % (args.model, args.dataset, cond, len(samples)))
    print("=" * 70, flush=True)

    model, processor = load(args.model)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.save_images:
        os.makedirs(args.save_images, exist_ok=True)

    records = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        image = s["image"]
        if corrupt_fn is not None and image is not None:
            image = corrupt_fn(image, args.pct)
            if args.save_images:
                image.save(os.path.join(args.save_images, "%s.png" % idx))

        resp = generate_one(model, processor, image, s["prompt"])
        records.append({
            "idx": idx, "model": args.model, "dataset": args.dataset,
            "corrupt": args.corrupt, "pct": args.pct, "condition": cond,
            "category": meta.get("category", ""), "prompt": s["prompt"],
            "image_path": meta.get("image_path", ""), "full_response": resp,
        })
        print("[%3d/%d] idx=%s%s\n         %s"
              % (i + 1, len(samples), idx, "   [DEAD?]" if is_dead(resp) else "",
                 resp.replace("\n", " ")[:200]), flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    n_dead = sum(is_dead(r["full_response"]) for r in records)
    print("\nwrote %s  (%d responses, %d flagged DEAD?)"
          % (args.out, len(records), n_dead), flush=True)
    if n_dead:
        print("WARNING: %d/%d look like perception failures — severity may be too high."
              % (n_dead, len(records)), flush=True)


if __name__ == "__main__":
    main()
