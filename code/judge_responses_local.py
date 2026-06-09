#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Judge saved responses locally with Ollama LLaMA-3."""

import json
import sys
import os
import re

try:
    import urllib.request as urllib_request
except ImportError:
    import urllib2 as urllib_request


OLLAMA_URL = "http://localhost:11434"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESPONSES_PATH = os.path.join(SCRIPT_DIR, '..', 'results', 'scienceqa', 'responses_raw.json')


def check_ollama():
    """Verify Ollama running."""
    try:
        urllib_request.urlopen(OLLAMA_URL + "/api/tags", timeout=3)
        print("[ollama] OK")
        return True
    except:
        print("[ollama] NOT RUNNING - Start with: ollama serve")
        return False


def judge_response(question, response, label):
    """Judge utility with LLaMA-3 via Ollama."""
    import re as re_module

    # Label is now STRING (A, B, C, D) not numeric
    # Extract correct answer text from choices
    choices_match = re_module.findall(r'\([A-D]\)\s*([^()]+?)(?=\s*\(|$)', question)

    # Map label letter to index: A=0, B=1, C=2, D=3, E=4
    label_to_idx = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    idx = label_to_idx.get(label, -1)

    if choices_match and 0 <= idx < len(choices_match):
        ground_truth = choices_match[idx].strip()
        # Clean up trailing "Answer:" or similar noise
        ground_truth = re_module.sub(r'\s+Answer:\s*$', '', ground_truth).strip()
    else:
        ground_truth = label  # Fallback to label string itself

    # Prepare for prompt - keep FULL response, truncate question/ground_truth only for size
    q = question.replace('"', '\\"').replace('\n', ' ')[:600]
    r = response.replace('"', '\\"').replace('\n', ' ')  # FULL response, no truncation
    l = ground_truth.replace('"', '\\"')[:200]

    prompt = ('You are evaluating a model response for utility and correctness.\n\n'
              'Question: ' + q + '\n\n'
              'Ground Truth Answer: ' + l + '\n\n'
              'Model Response: ' + r + '\n\n'
              'Rate the response 0-100 on utility (how helpful/relevant it is) and indicate if it matches the ground truth.\n'
              'Respond with valid JSON: {"score": <int 0-100>, "correct": <bool>}')

    try:
        data = json.dumps({
            "model": "llama3",
            "prompt": prompt,
            "stream": False,
            "temperature": 0.3
        }).encode('utf-8')

        req = urllib_request.Request(OLLAMA_URL + "/api/generate", data=data)
        req.add_header('Content-Type', 'application/json')
        response_obj = urllib_request.urlopen(req, timeout=60)
        result = json.load(response_obj)

        text = result.get("response", "").strip()
        match = re.search(r'\{[^{}]*\}', text)
        if match:
            try:
                p = json.loads(match.group())
                return {
                    "score": max(0, min(100, int(p.get("score", 50)))),
                    "correct": bool(p.get("correct", False))
                }
            except:
                pass
        return {"score": 50, "correct": False}
    except Exception as e:
        print("      [JUDGE ERROR] %s" % str(e))
        return {"score": 0, "correct": False}


def main():
    print("\n" + "="*80)
    print("  LOCAL JUDGING - ScienceQA Responses")
    print("="*80)

    if not check_ollama():
        print("\nCannot proceed without Ollama. Start it with:")
        print("  ollama serve")
        return

    if not os.path.exists(RESPONSES_PATH):
        print("\n[ERROR] No responses found at: %s" % RESPONSES_PATH)
        print("Run eval_scienceqa_inference_only.py on Newton first")
        return

    print("\n[1/3] Loading responses...")
    with open(RESPONSES_PATH) as f:
        responses = json.load(f)
    print("      OK: %d responses" % len(responses))

    print("\n[2/3] Judging responses...\n")

    scores = []
    judged = []

    for i, resp in enumerate(responses):
        i = i + 1
        judge = judge_response(resp["prompt"], resp["response"], resp["label"])
        scores.append(judge["score"])

        judged.append({
            "idx": resp["idx"],
            "question": resp["prompt"][:100],
            "response": resp["response"][:300],
            "label": resp["label"],
            "is_refusal": resp["is_refusal"],
            "utility_score": judge["score"],
            "correct": judge["correct"]
        })

        if i % 25 == 0:
            avg = sum(scores) / float(len(scores)) if scores else 0
            print("      [%d/%d] avg: %.1f%%" % (i, len(responses), avg))

    avg = sum(scores) / float(len(scores)) if scores else 0
    n_correct = sum(1 for j in judged if j["correct"])
    accuracy = (n_correct / float(len(judged)) * 100) if judged else 0
    n_refusal = sum(1 for j in judged if j["is_refusal"])

    sorted_by_score = sorted(judged, key=lambda x: x["utility_score"], reverse=True)
    examples = [sorted_by_score[0], sorted_by_score[-1]] + sorted_by_score[1:4]

    output = {
        "metadata": {
            "dataset": "ScienceQA (250)",
            "model": "LLaVA-CoT 11B (Base+TIS)",
            "judge": "LLaMA-3 (local Ollama)"
        },
        "metrics": {
            "utility_pct": round(avg, 2),
            "accuracy_pct": round(accuracy, 2),
            "n_correct": n_correct,
            "n_total": len(responses),
            "refusal_rate": round(n_refusal / float(len(responses)) * 100, 2)
        },
        "examples": examples
    }

    print("\n[3/3] Saving results...")
    outdir = os.path.join(SCRIPT_DIR, '..', 'results', 'scienceqa')
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outpath = os.path.join(outdir, "eval_results_judged.json")

    with open(outpath, "w") as f:
        json.dump(output, f, indent=2)

    print("\n" + "="*80)
    print("  JUDGING COMPLETE")
    print("="*80)
    print("  Utility:   %.2f%%" % output['metrics']['utility_pct'])
    print("  Accuracy:  %.2f%%" % output['metrics']['accuracy_pct'])
    print("  Refusal:   %.2f%%" % output['metrics']['refusal_rate'])
    print("="*80)
    print("  Saved: %s\n" % outpath)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print("\nERROR: %s" % str(e))
        import traceback
        traceback.print_exc()
