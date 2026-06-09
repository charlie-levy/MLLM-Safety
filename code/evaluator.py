"""
evaluator.py — Runs batched inference on a dataset using LLaVA-CoT.

Processes BATCH_SIZE samples at once to roughly halve wall-clock time.
Falls back to single-sample processing if a batch causes an OOM error.
Supports optional image corruption via PIL/scipy/numpy.
"""

import torch
from tqdm import tqdm
import numpy as np
from PIL import Image as PILImage, ImageFilter

from config import MAX_NEW_TOKENS, DO_SAMPLE, TEMPERATURE

BATCH_SIZE = 1  # Keep at 1 to avoid OOM on 16GB GPU nodes


class Evaluator:
    def __init__(self, model, processor, corruption_type=None, corruption_severity=3):
        self.model     = model
        self.processor = processor
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        # Left-padding required for batched generation
        self.processor.tokenizer.padding_side = "left"

    def _build_messages(self, prompt: str, has_image: bool) -> list:
        content = []
        if has_image:
            content.append({"type": "image"})
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _get_gen_kwargs(self) -> dict:
        kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample":      DO_SAMPLE,
            "pad_token_id":   self.processor.tokenizer.eos_token_id,
        }
        if DO_SAMPLE:
            kwargs["temperature"] = TEMPERATURE
        return kwargs

    def _apply_corruption(self, img: PILImage.Image, corruption_type: str, severity: int) -> PILImage.Image:
        """Apply gaussian_blur or gaussian_noise corruption to image."""
        img_array = np.array(img, dtype=np.float32)

        if corruption_type == "gaussian_blur":
            # severity 1-5 maps to radius 0.5-2.5
            radius = 0.5 + (severity - 1) * 0.5
            img_pil = PILImage.fromarray(np.uint8(img_array))
            return img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

        elif corruption_type == "gaussian_noise":
            # severity 1-5 maps to std 5-25
            std = 5 + (severity - 1) * 5
            noise = np.random.normal(0, std, img_array.shape)
            corrupted = np.clip(img_array + noise, 0, 255)
            return PILImage.fromarray(np.uint8(corrupted))

        else:
            raise ValueError(f"Unknown corruption type: {corruption_type}")

    def _run_batch(self, batch: list, gen_kwargs: dict) -> list:
        """Run inference on a list of samples, return result dicts."""
        # Apply image corruption if specified
        if self.corruption_type:
            for sample in batch:
                if sample["image"] is not None:
                    sample["image"] = self._apply_corruption(
                        sample["image"],
                        self.corruption_type,
                        self.corruption_severity
                    )

        images = [[s["image"]] for s in batch]  # nested list: one sub-list per sample
        texts  = [
            self.processor.apply_chat_template(
                self._build_messages(s["prompt"], s["image"] is not None),
                add_generation_prompt=True,
            )
            for s in batch
        ]

        inputs = self.processor(
            images=images,
            text=texts,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        output_ids = self.model.generate(**inputs, **gen_kwargs)

        results = []
        for j, s in enumerate(batch):
            response = self.processor.decode(
                output_ids[j][prompt_len:], skip_special_tokens=True
            )
            results.append({
                "prompt":   s["prompt"],
                "response": response,
                "label":    s["label"],
                "metadata": s["metadata"],
            })
        return results

    @torch.inference_mode()
    def run(self, samples: list[dict]) -> list[dict]:
        results    = []
        gen_kwargs = self._get_gen_kwargs()
        n_batches  = (len(samples) + BATCH_SIZE - 1) // BATCH_SIZE

        for i in tqdm(range(0, len(samples), BATCH_SIZE),
                      total=n_batches, desc="Evaluating", unit="batch"):
            batch = samples[i:i + BATCH_SIZE]

            try:
                results.extend(self._run_batch(batch, gen_kwargs))
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                # OOM: fall back to one sample at a time
                torch.cuda.empty_cache()
                print(f"\n[evaluator] OOM on batch — retrying {len(batch)} samples individually.")
                for s in batch:
                    results.extend(self._run_batch([s], gen_kwargs))

        return results
