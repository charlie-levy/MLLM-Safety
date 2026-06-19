# -*- coding: utf-8 -*-
"""model_loader.py — Load LLaVA-CoT with optional TIS (LoRA) adapter."""

import torch
import os
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2VLForConditionalGeneration
from peft import PeftModel

try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    config_dict = {}
    exec(open(config_path).read(), config_dict)
    TIS_LORA_PATH  = config_dict.get('TIS_LORA_PATH')
    MSR_LORA_PATH  = config_dict.get('MSR_LORA_PATH')
    SAGE_LORA_PATH = config_dict.get('SAGE_LORA_PATH')
except:
    TIS_LORA_PATH  = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_tis"
    MSR_LORA_PATH  = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_msr"
    SAGE_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_sage"


def load_llama_cot(lora_path=None):
    processor = AutoProcessor.from_pretrained("Xkev/Llama-3.2V-11B-cot")
    model = AutoModelForImageTextToText.from_pretrained(
        "Xkev/Llama-3.2V-11B-cot",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    if lora_path:
        print("[lora] Loading adapter from " + lora_path)
        peft_model = PeftModel.from_pretrained(model, lora_path)
        cfg = peft_model.peft_config
        for adapter_name, adapter_cfg in cfg.items():
            print("[lora] Adapter '%s': rank=%d, alpha=%d" % (adapter_name, adapter_cfg.r, adapter_cfg.lora_alpha))
        missing, unexpected = peft_model.load_state_dict(peft_model.state_dict(), strict=False)
        if unexpected:
            print("[lora][WARN] Unexpected keys: " + str(unexpected[:5]))
        else:
            print("[lora] No unexpected keys -- adapter loaded cleanly.")
        model = peft_model
        print("[lora] Adapter loaded (PEFT-wrapped, memory-efficient inference).")
    return model, processor


def load_r1onevision():
    """Load R1-OneVision (Qwen2-VL-7B based) model and processor."""
    try:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
        config_dict = {}
        exec(open(config_path).read(), config_dict)
        model_id = config_dict.get('R1ONEVISION_MODEL_ID', 'Fancy-MLLM/R1-OneVision-7B')
    except Exception:
        model_id = 'Fancy-MLLM/R1-OneVision-7B'

    print("[model_loader] Loading R1-OneVision from '%s' ..." % model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    print("[model_loader] R1-OneVision loaded.")
    return model, processor


def load_model_and_processor(use_tis=False, use_msr=False, use_sage=False, lora_path=None):
    """Load LLaVA-CoT model with an optional safety adapter.

    Args:
        use_tis:   If True, load TIS adapter from config.TIS_LORA_PATH
        use_msr:   If True, load MSR-Align adapter from config.MSR_LORA_PATH
        use_sage:  If True, load SAGE adapter from config.SAGE_LORA_PATH
        lora_path: Explicit adapter path override (takes precedence over all flags)

    Returns:
        (model, processor, model_tag)
    """
    n_adapters = sum([use_tis, use_msr, use_sage])
    if n_adapters > 1:
        raise ValueError("Cannot load more than one adapter at once.")

    if use_tis:
        adapter_path = lora_path or TIS_LORA_PATH
        if adapter_path is None:
            raise ValueError("TIS requested but TIS_LORA_PATH not set in config.py")
        model, processor = load_llama_cot(lora_path=adapter_path)
        model_tag = "base+TIS"
    elif use_msr:
        adapter_path = lora_path or MSR_LORA_PATH
        if adapter_path is None:
            raise ValueError("MSR requested but MSR_LORA_PATH not set in config.py")
        model, processor = load_llama_cot(lora_path=adapter_path)
        model_tag = "base+MSR"
    elif use_sage:
        adapter_path = lora_path or SAGE_LORA_PATH
        if adapter_path is None:
            raise ValueError("SAGE requested but SAGE_LORA_PATH not set in config.py")
        model, processor = load_llama_cot(lora_path=adapter_path)
        model_tag = "base+SAGE"
    elif lora_path:
        model, processor = load_llama_cot(lora_path=lora_path)
        model_tag = "base+adapter"
    else:
        model, processor = load_llama_cot(lora_path=None)
        model_tag = "base"

    print("[model_loader] Ready: " + model_tag + "\n")
    return model, processor, model_tag
