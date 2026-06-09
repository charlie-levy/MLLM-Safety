"""
config.py — Central configuration for the LLaVA-CoT Safety Evaluation pipeline.

All tunable settings live here. Edit this file instead of hunting through
individual scripts when you need to change a model, dataset, or run parameter.
"""

import os

# ── Model ──────────────────────────────────────────────────────────────────────
# LLaVA-CoT is a vision-language model fine-tuned for step-by-step chain-of-thought
# reasoning. It is built on top of Meta's LLaMA-3.2-11B-Vision-Instruct and produces
# structured output with <SUMMARY>, <CAPTION>, <REASONING>, and <CONCLUSION> tags.
MODEL_ID = "Xkev/Llama-3.2V-11B-cot"

# TIS (LoRA) adapter — load from local model_weights
TIS_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_tis"

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
XSTEST_CSV      = "/home/ch169788/xstest/xstest.csv"          # used by create_xstest_images.py
XSTEST_SAFE_JSON = "/home/ch169788/xstest/XSTest/xstest_safe.json"

# MMSA: only the safe samples (label=yes, indices 1-128 and 2001-2300).
# The JSON was generated from the MMSafetyAwareness benchmark.
# Repo: https://github.com/Jarviswang94/MMSafetyAwareness
MMSA_SAFE_JSON  = "/home/ch169788/MMSA/mmsafeaware_safe.json"
MMSA_IMAGE_DIR  = "/home/ch169788/MMSA/images"    # images 1-1100
MMSA_IMAGE_DIR2 = "/home/ch169788/MMSA/images2"   # images 2001+

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
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
