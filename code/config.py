"""
config.py — Central configuration for the LLaVA-CoT Safety Evaluation pipeline.

All tunable settings live here. Edit this file instead of hunting through
individual scripts when you need to change a model, dataset, or run parameter.
"""

import os

# ── Model: LLaVA-CoT ───────────────────────────────────────────────────────────
# LLaVA-CoT is a vision-language model fine-tuned for step-by-step chain-of-thought
# reasoning. It is built on top of Meta's LLaMA-3.2-11B-Vision-Instruct and produces
# structured output with <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION> tags.
MODEL_ID = "Xkev/Llama-3.2V-11B-cot"

# TIS (LoRA) adapter — load from local model_weights
TIS_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_tis"

# MSR-Align (LoRA) adapter — Multimodal Safety Reasoning Alignment
# (arXiv:2506.19257). Same base model and LoRA config (r=8, alpha=16) as TIS.
MSR_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_msr"

# SAGE (LoRA) adapter
SAGE_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_sage"

# ── Models: VLGuard (LLaVA-1.5-7B safety fine-tunes) ───────────────────────────
# VLGuard (Zong et al., ICML 2024, arXiv:2402.02207) released two safety-tuned
# LLaVA-1.5-7B variants. Despite the "-lora" repo names these are FULL merged
# models in ORIGINAL-LLaVA format (LlavaLlamaForCausalLM, Vicuna-7B + CLIP-336).
# They must be CONVERTED ONCE to HuggingFace format (LlavaForConditionalGeneration)
# with code/convert_vlguard_to_hf.py, then loaded exactly like llava-hf/llava-1.5-7b-hf
# (see interactive_llava.py). Jobs read the converted local dirs (hf) offline.
#   mixed   = safety data MIXED into instruction tuning
#   posthoc = safety LoRA applied AFTER instruction tuning (then merged)
VLGUARD_VARIANTS = {
    "mixed": {
        "src": "ys-zong/llava-v1.5-7b-Mixed-lora",
        "hf":  "/home/ch169788/llava_cot_eval/model_weights/llava15_vlguard_mixed_hf",
    },
    "posthoc": {
        "src": "ys-zong/llava-v1.5-7b-Posthoc-lora",
        "hf":  "/home/ch169788/llava_cot_eval/model_weights/llava15_vlguard_posthoc_hf",
    },
}
# Base components used by the conversion (cache these once on the login node):
LLAVA15_HF_BASE   = "llava-hf/llava-1.5-7b-hf"            # HF-format template (config/processor)
LLAVA15_TEXT_ID   = "lmsys/vicuna-7b-v1.5"               # text backbone (for the converter)
LLAVA15_VISION_ID = "openai/clip-vit-large-patch14-336"  # vision tower (for the converter)

# ── Model: R1-OneVision ────────────────────────────────────────────────────────
# R1-OneVision is a Qwen2-VL-7B model trained with GRPO for visual reasoning.
# It produces <think>…</think> chain-of-thought followed by a direct answer.
# Confirm the HuggingFace ID before first run: https://huggingface.co/Fancy-MLLM
R1ONEVISION_MODEL_ID = "Fancy-MLLM/R1-OneVision-7B"

# ── Datasets ───────────────────────────────────────────────────────────────────

# FigStep: adversarial jailbreak benchmark where harmful instructions are
# rendered as text inside images, bypassing text-based safety filters.
# Used to compute ASR (Attack Success Rate).
# Dataset lives in a cloned GitHub repo — set this to the path on your cluster.
FIGSTEP_REPO_PATH = "/home/ch169788/FigStep"   # path to cloned https://github.com/CryptoAILab/FigStep
FIGSTEP_CSV       = "data/question/safebench.csv"   # or SafeBench-Tiny.csv for 50 samples
FIGSTEP_IMAGE_DIR = "data/images/SafeBench"          # directory containing the images

# XSTest: safe prompts rendered as text-on-image PNGs.
# Run create_xstest_images.py once on the login node to generate these.
# Repo: https://github.com/paul-rottger/exaggerated-safety
XSTEST_CSV       = "/home/ch169788/xstest/xstest.csv"           # used by create_xstest_images.py
XSTEST_SAFE_JSON = "/home/ch169788/xstest/XSTest/xstest_safe.json"
XSTEST_IMAGE_DIR = "/home/ch169788/xstest/XSTest/images"        # fallback if JSON paths are stale

# MMSA: only the safe samples (label=yes, indices 1-128 and 2001-2300).
# The JSON was generated from the MMSafetyAwareness benchmark.
# Repo: https://github.com/Jarviswang94/MMSafetyAwareness
MMSA_SAFE_JSON  = "/home/ch169788/MMSA/mmsafeaware_safe.json"
MMSA_IMAGE_DIR  = "/home/ch169788/MMSA/images"    # images 1-1100
MMSA_IMAGE_DIR2 = "/home/ch169788/MMSA/images2"   # images 2001+

# New image-based safety attack datasets (materialized locally by
# prepare_new_attack_datasets.py, then loaded offline like FigStep).
#   SIUO          sinwang/SIUO                 167  (siuo_gen.json + images)
#   BeaverTails-V PKU-Alignment/BeaverTails-V  1180 (20 categories, evaluation split)
#   SPA-VL        sqrti/SPA-VL                 265  (test config, 'harm' split)
NEW_ATTACKS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "new_attacks")
NEW_ATTACK_COUNTS = {"siuo": 167, "beavertails": 1180, "spavl": 265}

# ── Generation Hyperparameters ─────────────────────────────────────────────────
# Greedy decoding (DO_SAMPLE=False, TEMPERATURE irrelevant) is standard for
# safety evaluations because it makes results deterministic and reproducible.
MAX_NEW_TOKENS = 1024  # LLaVA-CoT needs full CoT chain (SUMMARY+CAPTION+REASONING+CONCLUSION)
DO_SAMPLE      = False
TEMPERATURE    = 0.0   # only used when DO_SAMPLE=True

# ── Hardware ───────────────────────────────────────────────────────────────────
# "auto" lets HuggingFace's accelerate library spread model layers across
# all available GPUs automatically — ideal for multi-GPU Newton nodes.
DEVICE_MAP  = "auto"
TORCH_DTYPE = "bfloat16"   # bfloat16 is preferred on A100/H100; use float16 for older GPUs

# ── Output ─────────────────────────────────────────────────────────────────────
# Per-sample CSVs and aggregate metric JSONs are written here.
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
