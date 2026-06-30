#!/usr/bin/env python
"""
prep_tis_data.py — ONE-TIME prep of the Think-in-Safety (TiS) dataset so LLaMA-Factory
can LoRA-fine-tune LLaVA-CoT on it. Run on the Newton LOGIN node (needs internet;
compute nodes are offline). Does NOT train anything.

Steps:
  1. download train_data.json + images.zip from HF (Holly301/Think-in-Safety, ~386 MB)
  2. unzip images into  <LF>/data/think_in_safety/
  3. rewrite each image path to an ABSOLUTE path (foolproof resolution) and write
        <LF>/data/think_in_safety.json
  4. register a "think_in_safety" entry in <LF>/data/dataset_info.json   (idempotent)
  5. VERIFY: row count, <image>-token vs image-count match, missing image files,
     and gpt-response length stats (to sanity-check cutoff_len before training)

The data is already in LLaMA-Factory sharegpt shape:
  {"conversations": [{"from":"human","value":"<image>..."},
                     {"from":"gpt","value":"<think>...</think> answer"}],
   "images": ["images/<cat>/<uuid>.png"]}

  python3 prep_tis_data.py                 # assumes ~/LLaMA-Factory
  python3 prep_tis_data.py --lf_dir ~/LLaMA-Factory
"""
import os
import json
import zipfile
import argparse
from huggingface_hub import hf_hub_download

REPO = "Holly301/Think-in-Safety"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lf_dir", default="~/LLaMA-Factory", help="LLaMA-Factory clone root")
    args = ap.parse_args()

    lf = os.path.abspath(os.path.expanduser(args.lf_dir))
    data_dir = os.path.join(lf, "data")
    base = os.path.join(data_dir, "think_in_safety")        # images extract here
    if not os.path.isdir(data_dir):
        raise SystemExit("LLaMA-Factory data dir not found: %s (is --lf_dir correct?)" % data_dir)
    os.makedirs(base, exist_ok=True)

    # 1) download
    print("[1/5] downloading train_data.json + images.zip from %s ..." % REPO, flush=True)
    js = hf_hub_download(REPO, "train_data.json", repo_type="dataset")
    zp = hf_hub_download(REPO, "images.zip", repo_type="dataset")

    # 2) unzip (python zipfile = single-threaded -> login-node nproc-cap safe)
    print("[2/5] unzipping images.zip -> %s ..." % base, flush=True)
    with zipfile.ZipFile(zp) as z:
        z.extractall(base)

    # 3) rewrite image paths to absolute + collect checks
    print("[3/5] rewriting image paths to absolute ...", flush=True)
    recs = json.load(open(js))
    n = len(recs)
    missing = mism = 0
    resp_chars = []
    for r in recs:
        abs_imgs = []
        for ip in (r.get("images", []) or []):
            p = ip if os.path.isabs(ip) else os.path.join(base, ip)
            if not os.path.isfile(p):
                missing += 1
            abs_imgs.append(p)
        r["images"] = abs_imgs
        n_tok = sum(c["value"].count("<image>") for c in r["conversations"] if c.get("from") == "human")
        if n_tok != len(abs_imgs):
            mism += 1
        resp_chars += [len(c["value"]) for c in r["conversations"] if c.get("from") == "gpt"]

    out_json = os.path.join(data_dir, "think_in_safety.json")
    json.dump(recs, open(out_json, "w"), ensure_ascii=False)
    print("      wrote %s  (%d records)" % (out_json, n))

    # 4) register dataset (idempotent)
    print("[4/5] registering dataset in dataset_info.json ...", flush=True)
    di_path = os.path.join(data_dir, "dataset_info.json")
    di = json.load(open(di_path)) if os.path.isfile(di_path) else {}
    di["think_in_safety"] = {
        "file_name": "think_in_safety.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "images": "images"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt"},
    }
    json.dump(di, open(di_path, "w"), ensure_ascii=False, indent=2)
    print("      registered 'think_in_safety' in %s" % di_path)

    # 5) verify
    resp_chars.sort()
    def pct(q):
        return resp_chars[int(q * (len(resp_chars) - 1))] if resp_chars else 0
    print("\n[5/5] === VERIFY ===")
    print("  records:                 %d" % n)
    print("  missing image files:     %d   (must be 0)" % missing)
    print("  <image>/image mismatch:  %d   (must be 0)" % mism)
    print("  gpt response chars  p50/p90/p99/max:  %d / %d / %d / %d"
          % (pct(.5), pct(.9), pct(.99), resp_chars[-1] if resp_chars else 0))
    print("  ~token est = chars/4  ->  p99 ~%d tok; keep cutoff_len above this." % (pct(.99) // 4))
    print("\n  " + ("READY — data registered as 'think_in_safety'."
                    if (missing == 0 and mism == 0)
                    else "WARNINGS above — resolve before training."))


if __name__ == "__main__":
    main()
