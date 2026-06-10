"""
evaluator_r1.py — Inference evaluator for R1-OneVision (Qwen2-VL based).

Qwen2-VL expects a flat image list (not nested) passed to the processor.
Everything else — corruption, batching, OOM fallback — is identical to
the base Evaluator.
"""

import torch
from evaluator import Evaluator


class R1Evaluator(Evaluator):
    """Drop-in replacement for Evaluator when running R1-OneVision."""

    def _run_batch(self, batch: list, gen_kwargs: dict) -> list:
        # Apply image corruption if specified
        if self.corruption_type:
            for sample in batch:
                if sample["image"] is not None:
                    sample["image"] = self._apply_corruption(
                        sample["image"],
                        self.corruption_type,
                        self.corruption_severity,
                    )

        # Qwen2-VL: processor expects a flat list of PIL images (one per text)
        images = [s["image"] for s in batch]
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
