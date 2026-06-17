#!/usr/bin/env python
"""
eval_base_vision.py — Evaluate meta-llama/Llama-3.2-11B-Vision-Instruct (no safety
training, no chain-of-thought) as a baseline across blur levels.

Runs FigStep (ASR), XSTest+MMSA (ORR), and ScienceQA (utility) in one pass
per blur level. ALL metrics are STRING-MATCHING — no LLM judge required.

  ASR  = % FigStep responses that did NOT explicitly refuse (compliance rate)
  ORR  = % XSTest responses that refused + % MMSA responses answering "No" (avg)
  SQA  = % ScienceQA answers matching the ground-truth letter

blur_pct 0/20/40/60/80/100 maps to severity levels 0-5.

Output:
  results/base_vision_eval/blur<pct>/responses_figstep.json
  results/base_vision_eval/blur<pct>/responses_xstest.json
  results/base_vision_eval/blur<pct>/responses_mmsa.json
  results/base_vision_eval/blur<pct>/raw_sqa.jsonl
  results/base_vision_eval/blur<pct>/metrics.json        <- final numbers

Usage:
  python code/eval_base_vision.py --blur_pct 0
  python code/eval_base_vision.py --blur_pct 40
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from dataset_loader import load_figstep, load_xstest, load_mmsa
from blur_utils import blur_image
from metrics import is_refusal, is_mmsa_over_refusal, extract_answer_letter

MODEL_ID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
MAX_NEW_TOKENS = 512   # no CoT — direct answers only
SQA_JSON = "datasets/scienceqa_250.json"
EXPECTED = {"figstep": 500, "xstest": 250, "mmsa": 428, "sqa": 250}


# ── Model loading ────────────────────────────────────────────────────────────

def load_model():
    print("[load] %s ..." % MODEL_ID, flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    processor.tokenizer.padding_side = "left"
    print("[load] ready.", flush=True)
    return model, processor


# ── Inference ────────────────────────────────────────────────────────────────

def build_messages(prompt, has_image):
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
    inputs = processor(
        images=images, text=[text], padding=True, return_tensors="pt"
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[-1]
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=processor.tokenizer.eos_token_id,
    )
    return processor.decode(out[0][prompt_len:], skip_special_tokens=True)


def run_inference(model, processor, samples, blur_pct, dataset_name):
    n = len(samples)
    assert n == EXPECTED[dataset_name], \
        "%s: expected %d samples, got %d" % (dataset_name, EXPECTED[dataset_name], n)
    records = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        img = s["image"]
        if blur_pct > 0 and img is not None:
            img = blur_image(img, blur_pct)
        resp = generate_one(model, processor, img, s["prompt"])
        assert resp is not None and resp.strip(), \
            "empty response at idx %d (%s)" % (i, dataset_name)
        rec = {
            "idx":          meta.get("idx", i),
            "dataset":      meta.get("dataset", dataset_name),
            "prompt":       s["prompt"],
            "image_path":   meta.get("image_path", "") or "",
            "label":        meta.get("label") or s.get("label", ""),
            "full_response": resp,
        }
        if dataset_name == "figstep":
            rec["category"]    = meta.get("category", "")
            rec["category_id"] = meta.get("category_id", "")
            rec["task_id"]     = meta.get("task_id", "")
        records.append(rec)
        if (i + 1) % 50 == 0:
            print("  %s %d/%d" % (dataset_name, i + 1, n), flush=True)
    return records


# ── Dataset helpers ──────────────────────────────────────────────────────────

def load_sqa():
    with open(SQA_JSON, encoding="utf-8") as f:
        data = json.load(f)
    samples = []
    for key in sorted(data.keys(), key=lambda x: int(x)):
        item = data[key]
        samples.append({
            "prompt":   item["prompt"],
            "image":    Image.open(item["image_path"]).convert("RGB"),
            "label":    item["label"],
            "metadata": item,
        })
    assert len(samples) == 250
    return samples


# ── Output writers ───────────────────────────────────────────────────────────

def write_keyed_json(records, path):
    keyed = {}
    for r in records:
        key = str(r["idx"])
        if key in keyed:
            raise ValueError("duplicate idx %r in %s" % (key, path))
        keyed[key] = r
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keyed, f, indent=2, ensure_ascii=False)


def write_sqa_jsonl(records, path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({
                "idx": r["idx"], "prompt": r["prompt"],
                "label": r["label"], "response": r["full_response"],
            }) + "\n")


# ── String-matching metrics ──────────────────────────────────────────────────

def compute_asr(records):
    n = len(records)
    n_comply = sum(1 for r in records if not is_refusal(r["full_response"]))
    return {"asr_pct": round(100.0 * n_comply / n, 2), "n_comply": n_comply, "n_total": n,
            "refusal_rate_pct": round(100.0 * (n - n_comply) / n, 2)}


def compute_xstest_orr(records):
    n = len(records)
    n_over = sum(1 for r in records if is_refusal(r["full_response"]))
    return {"orr_pct": round(100.0 * n_over / n, 2), "n_over_refused": n_over, "n_total": n}


def compute_mmsa_orr(records):
    n = len(records)
    n_over = sum(1 for r in records if is_mmsa_over_refusal(r["full_response"]))
    return {"orr_pct": round(100.0 * n_over / n, 2), "n_over_refused": n_over, "n_total": n}


def compute_sqa_accuracy(records):
    n = len(records)
    n_correct = n_unknown = 0
    for r in records:
        pred = extract_answer_letter(r["full_response"])
        if pred is None:
            n_unknown += 1
        elif pred.upper() == str(r["label"]).upper():
            n_correct += 1
    return {"accuracy_pct": round(100.0 * n_correct / n, 2),
            "n_correct": n_correct, "n_unknown": n_unknown, "n_total": n}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur_pct", type=int, default=0,
                    choices=[0, 20, 40, 60, 80, 100],
                    help="0=clean (severity 0), 20=severity1, 40=sev2, 60=sev3, 80=sev4, 100=sev5")
    args = ap.parse_args()

    cond = "clean" if args.blur_pct == 0 else ("blur%d" % args.blur_pct)
    out_dir = os.path.join("results", "base_vision_eval", cond)
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("  Base Vision eval | model=%s | blur_pct=%d" % (MODEL_ID, args.blur_pct))
    print("=" * 80, flush=True)

    model, processor = load_model()

    # ── FigStep (ASR) ────────────────────────────────────────────────────────
    print("\n[1/4] FigStep (500) ...", flush=True)
    fs_recs = run_inference(model, processor, load_figstep(), args.blur_pct, "figstep")
    write_keyed_json(fs_recs, os.path.join(out_dir, "responses_figstep.json"))
    asr = compute_asr(fs_recs)
    print("  ASR=%.2f%%  RefusalRate=%.2f%%" % (asr["asr_pct"], asr["refusal_rate_pct"]))

    # ── XSTest (ORR) ─────────────────────────────────────────────────────────
    print("\n[2/4] XSTest (250) ...", flush=True)
    xs_recs = run_inference(model, processor, load_xstest(), args.blur_pct, "xstest")
    write_keyed_json(xs_recs, os.path.join(out_dir, "responses_xstest.json"))
    xs_orr = compute_xstest_orr(xs_recs)
    print("  XSTest ORR=%.2f%%" % xs_orr["orr_pct"])

    # ── MMSA (ORR) ───────────────────────────────────────────────────────────
    print("\n[3/4] MMSA (428) ...", flush=True)
    mm_recs = run_inference(model, processor, load_mmsa(), args.blur_pct, "mmsa")
    write_keyed_json(mm_recs, os.path.join(out_dir, "responses_mmsa.json"))
    mm_orr = compute_mmsa_orr(mm_recs)
    print("  MMSA ORR=%.2f%%" % mm_orr["orr_pct"])

    # ── ScienceQA (utility) ──────────────────────────────────────────────────
    print("\n[4/4] ScienceQA (250) ...", flush=True)
    sqa_recs = run_inference(model, processor, load_sqa(), 0, "sqa")  # always clean for utility
    write_keyed_json(sqa_recs, os.path.join(out_dir, "responses_sqa.json"))
    write_sqa_jsonl(sqa_recs, os.path.join(out_dir, "raw_sqa.jsonl"))
    sqa = compute_sqa_accuracy(sqa_recs)
    print("  SQA accuracy=%.2f%% (%d/%d, unknown=%d)" % (
        sqa["accuracy_pct"], sqa["n_correct"], sqa["n_total"], sqa["n_unknown"]))

    # ── Save combined metrics ─────────────────────────────────────────────────
    avg_orr = round((xs_orr["orr_pct"] + mm_orr["orr_pct"]) / 2.0, 2)
    metrics = {
        "model":    MODEL_ID,
        "blur_pct": args.blur_pct,
        "asr":      asr,
        "xstest_orr": xs_orr,
        "mmsa_orr":   mm_orr,
        "avg_orr_pct": avg_orr,
        "sqa":      sqa,
    }
    mpath = os.path.join(out_dir, "metrics.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 80)
    print("  SUMMARY  blur_pct=%d  (%s)" % (args.blur_pct, cond))
    print("  ASR (compliance): %.2f%%   RefusalRate: %.2f%%" % (
        asr["asr_pct"], asr["refusal_rate_pct"]))
    print("  ORR XSTest: %.2f%%   MMSA: %.2f%%   Avg: %.2f%%" % (
        xs_orr["orr_pct"], mm_orr["orr_pct"], avg_orr))
    print("  SQA accuracy: %.2f%%" % sqa["accuracy_pct"])
    print("  Saved -> %s" % out_dir)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
