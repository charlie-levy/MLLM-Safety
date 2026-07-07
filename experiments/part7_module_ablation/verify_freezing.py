#!/usr/bin/env python
"""
verify_freezing.py — PROVE which model parts are trainable for each config, BEFORE
committing GPU-hours to the real run.

Loads the model through LLaMA-Factory's OWN pipeline (get_train_args -> load_model),
so the freezing + LoRA targeting are EXACTLY what the trainer applies. Then reports
every trainable parameter bucketed by tower (vision_model / language_model /
multi_modal_projector) and FAILS if any trainable weight sits outside the tower we
intend to train.

  conda activate llamafactory
  cd ~/LLaMA-Factory                                  # so relative paths in yaml resolve
  P=~/llava_cot_eval/experiments/part7_module_ablation
  python $P/verify_freezing.py $P/configs/train_zoomblur75_llm_lora.yaml    --expect language_model
  python $P/verify_freezing.py $P/configs/train_zoomblur75_vision_lora.yaml --expect vision_model
  python $P/verify_freezing.py $P/configs/train_zoomblur75_both_lora.yaml   --expect both
"""
import sys
import os
import argparse
import yaml

from llamafactory.hparams import get_train_args
from llamafactory.model import load_tokenizer, load_model


def bucket(name):
    if "vision_model" in name:
        return "vision_model"
    if "language_model" in name:
        return "language_model"
    if "multi_modal_projector" in name:
        return "multi_modal_projector"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--expect", choices=["language_model", "vision_model", "both"], required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cfg["do_train"] = True
    # LLaMA-Factory's own arg pipeline (same one llamafactory-cli uses)
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(cfg)
    tok_module = load_tokenizer(model_args)
    model = load_model(tok_module["tokenizer"], model_args, finetuning_args, training_args.do_train)

    tot = 0
    tr = 0
    tr_by = {"vision_model": 0, "language_model": 0, "multi_modal_projector": 0, "other": 0}
    trainable_leaf_examples = {"vision_model": None, "language_model": None}
    for n, p in model.named_parameters():
        tot += p.numel()
        if p.requires_grad:
            tr += p.numel()
            b = bucket(n)
            tr_by[b] += p.numel()
            if b in trainable_leaf_examples and trainable_leaf_examples[b] is None:
                trainable_leaf_examples[b] = n

    print("\n=== %s  (--expect %s) ===" % (os.path.basename(args.config), args.expect))
    print("  total params:     {:,}".format(tot))
    print("  trainable params: {:,}  ({:.4f}%)".format(tr, 100.0 * tr / tot))
    for k in ("vision_model", "language_model", "multi_modal_projector", "other"):
        print("    trainable in {:<22} {:,}".format(k, tr_by[k]))
    for b, ex in trainable_leaf_examples.items():
        if ex:
            print("    e.g. trainable %-14s -> %s" % (b, ex))

    expect_map = {
        "language_model": {"language_model"},
        "vision_model": {"vision_model"},
        "both": {"language_model", "vision_model"},
    }
    allowed = expect_map[args.expect]
    bad = {k: v for k, v in tr_by.items() if v > 0 and k not in allowed}
    missing = {k for k in allowed if tr_by[k] == 0}
    ok = (not bad) and (not missing) and tr > 0
    print("  EXPECT trainable ONLY in: %s" % sorted(allowed))
    if bad:
        print("  ❌ UNEXPECTED trainable weights in: %s" % bad)
    if missing:
        print("  ❌ EXPECTED tower has 0 trainable weights: %s" % sorted(missing))
    print("  VERDICT:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
