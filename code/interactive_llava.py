# -*- coding: utf-8 -*-
"""interactive_llava.py — Simple interactive harness for plain LLaVA-1.5.

Loads llava-hf/llava-1.5-7b-hf ONCE, then lets you repeatedly enter an image
path and/or a text prompt and see the full raw model output. This is the
standard (non-CoT) LLaVA, separate from the project's CoT pipeline.

Run on a Newton GPU node. The weights must already be cached on the node
(see note below) — if so, run fully offline:

    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
    HF_HUB_ENABLE_HF_TRANSFER=0 python code/interactive_llava.py

If the weights are NOT yet cached, do a one-time download on a node with
internet (drop the HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE vars for that run).

Interactive prompts:
  image> path to an image file, or blank for text-only
  text>  your prompt
Type 'q' at the image> prompt to quit.

One-shot (non-interactive) mode:
    python code/interactive_llava.py --image foo.png --prompt "What is this?"
"""

import argparse

import torch
from PIL import Image as PILImage
from transformers import AutoProcessor, LlavaForConditionalGeneration

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
DEFAULT_TEXT = "Describe this image in detail."


def load_model(model_id):
    print("[interactive] Loading %s ..." % model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    print("[interactive] Model ready.")
    return model, processor


def build_messages(prompt, has_image):
    content = []
    if has_image:
        content.append({"type": "image"})
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def maybe_corrupt(img, corruption_type, pct):
    """Apply the same percentage-based corruption as the eval pipeline."""
    if not corruption_type or img is None:
        return img
    if corruption_type == "noise":
        from noise_utils import noisy_image
        return noisy_image(img, pct)
    if corruption_type == "blur":
        from blur_utils import blur_image
        return blur_image(img, pct)
    if corruption_type == "jpeg":
        from jpeg_utils import jpeg_image
        return jpeg_image(img, pct)
    if corruption_type == "brightness":
        from brightness_utils import brightness_image
        return brightness_image(img, pct)
    if corruption_type == "pixelate":
        from pixelate_utils import pixelate_image
        return pixelate_image(img, pct)
    raise ValueError("Unknown corruption type: " + corruption_type)


@torch.inference_mode()
def generate(model, processor, image, prompt, max_new_tokens):
    has_image = image is not None
    text = processor.apply_chat_template(
        build_messages(prompt, has_image), add_generation_prompt=True
    )
    inputs = processor(
        images=image if has_image else None,
        text=text,
        return_tensors="pt",
    ).to(model.device, torch.bfloat16)

    prompt_len = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(output_ids[0][prompt_len:], skip_special_tokens=True)


def load_image(path):
    if not path:
        return None
    try:
        return PILImage.open(path).convert("RGB")
    except Exception as e:
        print("  [!] could not open image: %s" % e)
        return None


def run_once(model, processor, image, prompt, args, img_label):
    image = maybe_corrupt(image, args.corruption, args.pct)
    print("\n" + "=" * 70)
    print("PROMPT: %s" % prompt)
    print("IMAGE : %s%s" % (
        img_label if image is not None else "(none)",
        ("  [corruption=%s %d%%]" % (args.corruption, args.pct)) if args.corruption else "",
    ))
    print("-" * 70)
    print(generate(model, processor, image, prompt, args.max_new_tokens))
    print("=" * 70 + "\n")


def main():
    ap = argparse.ArgumentParser(description="Interactive plain LLaVA-1.5 runner")
    ap.add_argument("--model_id", default=MODEL_ID)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--corruption", default=None,
                    choices=["noise", "blur", "jpeg", "brightness", "pixelate"],
                    help="optionally corrupt every input image")
    ap.add_argument("--pct", type=int, default=0, help="corruption percentage 0-100")
    ap.add_argument("--image", default=None, help="one-shot: image path")
    ap.add_argument("--prompt", default=None, help="one-shot: text prompt")
    args = ap.parse_args()

    model, processor = load_model(args.model_id)

    # One-shot mode if either --image or --prompt was supplied.
    if args.image is not None or args.prompt is not None:
        image = load_image(args.image)
        prompt = args.prompt or DEFAULT_TEXT
        run_once(model, processor, image, prompt, args, args.image or "(none)")
        return

    # Interactive REPL.
    print("\nEnter an image path (blank = text-only, 'q' = quit), then a prompt.\n")
    while True:
        try:
            img_path = input("image> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if img_path.lower() in ("q", "quit", "exit"):
            break
        image = load_image(img_path) if img_path else None
        if img_path and image is None:
            continue  # bad path, re-prompt
        try:
            prompt = input("text>  ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt and image is None:
            print("  [!] need at least an image or a prompt.\n")
            continue
        if not prompt:
            prompt = DEFAULT_TEXT
        run_once(model, processor, image, prompt, args, img_path or "(none)")


if __name__ == "__main__":
    main()
