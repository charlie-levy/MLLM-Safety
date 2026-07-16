#!/usr/bin/env python3
"""
lib.py — shared loaders + text metrics for the Part-4 EXPLORATORY analysis.

Scope guardrails (per the task spec):
  * NO ASR, NO external judge, NO safety classification, NO assumed labels.
  * Everything here is a SURFACE / STYLOMETRIC feature computed directly from the
    text that the models already wrote. "refusal_like" and "perception_failure"
    are *linguistic surface flags*, NOT safety judgments — treat them as such.

The 16 files are SIUO (167 prompts) answered by 4 models under 4 image conditions.
Two models emit explicit reasoning we can separate from their final answer:
    llava_cot     -> <SUMMARY><CAPTION><REASONING><CONCLUSION>  (answer = CONCLUSION)
    r1_onevision  -> <think> ... </think> <answer>              (answer = after </think>)
    base_llama / qwen2_5_vl -> direct answer, no reasoning block.
"""
import os, re, json, math
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "results", "part4_siuo_reasoning_vs_base"))
OUT   = os.path.join(HERE, "outputs")
TABLES = os.path.join(OUT, "tables")
FIGS   = os.path.join(OUT, "figures")

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
MODELS     = ["base_llama", "llava_cot", "qwen2_5_vl", "r1_onevision"]

# model metadata: family + whether it emits an explicit reasoning trace
MODEL_META = {
    "base_llama":   dict(family="Llama", kind="base",      reasons=False, label="Llama-3.2-11B (base)"),
    "llava_cot":    dict(family="Llama", kind="reasoning", reasons=True,  label="LLaVA-CoT (reasoning)"),
    "qwen2_5_vl":   dict(family="Qwen",  kind="base",      reasons=False, label="Qwen2.5-VL-7B (base)"),
    "r1_onevision": dict(family="Qwen",  kind="reasoning", reasons=True,  label="R1-Onevision-7B (reasoning)"),
}
# stable colors for plots
COLOR = {"base_llama": "#4C72B0", "llava_cot": "#DD8452",
         "qwen2_5_vl": "#55A868", "r1_onevision": "#C44E52"}

# ---- lexicons (all matched case-insensitively, word-boundary where sensible) ----
CONNECTORS = ["because", "therefore", "thus", "hence", "consequently", "however",
              "moreover", "furthermore", "additionally", "finally", "first",
              "then", "next", "second", "third"]
UNCERTAINTY = [r"\bmaybe\b", r"\bperhaps\b", r"\bpossibly\b", r"\blikely\b",
               r"\bunclear\b", r"\buncertain\b", r"\bnot sure\b", r"\bcannot determine\b",
               r"\bcan't determine\b", r"\bdifficult to tell\b", r"\bhard to tell\b",
               r"\bdifficult to determine\b", r"\bappears to\b", r"\bseems to\b",
               r"\bit seems\b", r"\bunable to determine\b", r"\bnot certain\b",
               r"\bi think\b", r"\bmight be\b", r"\bcould be\b"]
# SURFACE refusal phrasing (NOT a safety label) — just "declined-sounding" wording
REFUSAL = [r"\bi can'?t\b", r"\bi cannot\b", r"\bi'?m sorry\b", r"\bi am sorry\b",
           r"\bi'?m unable\b", r"\bi am unable\b", r"\bi won'?t\b", r"\bi will not\b",
           r"\bas an ai\b", r"\bi'?m not able\b", r"\bi am not able\b",
           r"\bcannot assist\b", r"\bcan'?t help\b", r"\bcannot help\b",
           r"\bi must decline\b", r"\bi'?m not comfortable\b", r"\bi do not\b.{0,20}\bprovide\b"]

_TAG = re.compile(r"<[^>]+>")
_WORD = re.compile(r"[A-Za-z']+")
_SENT = re.compile(r"[.!?]+")


def strip_tags(t):
    return _TAG.sub(" ", t or "")


def split_reasoning_answer(model, resp):
    """Return (reasoning_text, answer_text, structure_intact)."""
    resp = resp or ""
    if model == "llava_cot":
        m = re.search(r"<CONCLUSION>(.*?)(?:</CONCLUSION>|$)", resp, re.S | re.I)
        if m:
            return resp[:m.start()].strip(), m.group(1).strip(), True
        return "", resp.strip(), False          # CoT structure absent -> whole = answer
    if model == "r1_onevision":
        m = re.search(r"<think>(.*?)</think>(.*)", resp, re.S | re.I)
        if m:
            return m.group(1).strip(), m.group(2).strip(), True
        if "<think>" in resp.lower():           # opened reasoning but never closed (runaway)
            return re.sub(r".*?<think>", "", resp, flags=re.S | re.I).strip(), "", False
        return "", resp.strip(), False
    return "", resp.strip(), False              # base models: no reasoning block


def words(t):
    return _WORD.findall((t or "").lower())


def n_sentences(t):
    return len([s for s in _SENT.split(t or "") if s.strip()])


def ttr(toks):
    return (len(set(toks)) / len(toks)) if toks else 0.0


def mattr(toks, w=50):
    """Moving-average TTR (length-controlled lexical diversity)."""
    if len(toks) < w:
        return ttr(toks)
    vals = [len(set(toks[i:i + w])) / w for i in range(0, len(toks) - w + 1)]
    return sum(vals) / len(vals)


def bigram_repeat_frac(toks):
    """1 - distinct_bigrams/total_bigrams. High => repetitive / mode-collapse-y."""
    if len(toks) < 2:
        return 0.0
    bg = list(zip(toks, toks[1:]))
    return 1.0 - (len(set(bg)) / len(bg))


def count_patterns(text_lower, patterns):
    return sum(len(re.findall(p, text_lower)) for p in patterns)


def per100(count, ntok):
    return (100.0 * count / ntok) if ntok else 0.0


def metrics_for(model, resp):
    """All per-response surface metrics."""
    reasoning_raw, answer_raw, intact = split_reasoning_answer(model, resp)
    full_clean = strip_tags(resp)
    ans_clean = strip_tags(answer_raw)
    rea_clean = strip_tags(reasoning_raw)

    ftok = words(full_clean)
    atok = words(ans_clean)
    rtok = words(rea_clean)
    low_full = full_clean.lower()
    low_ans = ans_clean.lower()

    return dict(
        # lengths
        resp_chars=len(resp or ""),
        full_words=len(ftok),
        answer_words=len(atok),
        reasoning_words=len(rtok),
        reasoning_ratio=(len(rtok) / (len(rtok) + len(atok))) if (len(rtok) + len(atok)) else 0.0,
        answer_sentences=n_sentences(ans_clean),
        full_sentences=n_sentences(full_clean),
        # diversity / repetition
        ttr_answer=ttr(atok),
        mattr_answer=mattr(atok),
        bigram_repeat=bigram_repeat_frac(atok),
        # reasoning / discourse
        connectors=count_patterns(low_full, [r"\b%s\b" % c for c in CONNECTORS]),
        connector_density=per100(count_patterns(low_full, [r"\b%s\b" % c for c in CONNECTORS]), len(ftok)),
        # uncertainty (on the ANSWER the user would see)
        uncertainty=count_patterns(low_ans, UNCERTAINTY),
        uncertainty_density=per100(count_patterns(low_ans, UNCERTAINTY), len(atok)),
        has_uncertainty=int(count_patterns(low_ans, UNCERTAINTY) > 0),
        # surface refusal phrasing (NOT a safety label)
        refusal_like=int(count_patterns(low_ans, REFUSAL) > 0),
        # structure
        structure_intact=int(intact),
        answer_empty=int(len(atok) == 0),
    )


def load_records():
    """Yield every record across all 16 cells with metrics + metadata attached."""
    rows = []
    for cond in CONDITIONS:
        for model in MODELS:
            path = os.path.join(RESULTS_DIR, "siuo_%s_%s_responses.jsonl" % (cond, model))
            with open(path) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    row = dict(
                        idx=str(r["idx"]), model=model, condition=cond,
                        family=MODEL_META[model]["family"], kind=MODEL_META[model]["kind"],
                        category=r.get("category", ""),
                        perception_failure=int(bool(r.get("perception_failure"))),
                        response=r.get("response", ""),
                    )
                    row.update(metrics_for(model, r.get("response", "")))
                    rows.append(row)
    return rows


def ensure_dirs():
    for d in (OUT, TABLES, FIGS):
        os.makedirs(d, exist_ok=True)
