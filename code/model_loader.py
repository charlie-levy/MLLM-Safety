# -*- coding: utf-8 -*-
"""model_loader.py — Load LLaVA-CoT with optional TIS (LoRA) adapter."""

import torch
import os
import sys
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from accelerate import Accelerator

try:
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py')
    config_dict = {}
    exec(open(config_path).read(), config_dict)
    TIS_LORA_PATH = config_dict.get('TIS_LORA_PATH')
except:
    TIS_LORA_PATH = "/home/ch169788/llava_cot_eval/model_weights/llama_cot_tis"


def load_llama_cot(lora_path=None):
    processor = AutoProcessor.from_pretrained("Xkev/Llama-3.2V-11B-cot")
    model = AutoModelForImageTextToText.from_pretrained(
        "Xkev/Llama-3.2V-11B-cot",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    model.gradient_checkpointing_enable()
    model.enable_sequential_cpu_offload()
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


def load_model_and_processor(use_tis=False, lora_path=None):
    """Load LLaVA-CoT model with optional TIS adapter.

    Args:
        use_tis: If True, load TIS adapter from config.TIS_LORA_PATH
        lora_path: Override path for adapter (if None, uses config.TIS_LORA_PATH)

    Returns:
        (model, processor, model_tag) where model_tag is "base" or "base+TIS"
    """
    if use_tis:
        adapter_path = lora_path or TIS_LORA_PATH
        if adapter_path is None:
            raise ValueError("TIS requested but TIS_LORA_PATH not set in config.py")
        model, processor = load_llama_cot(lora_path=adapter_path)
        model_tag = "base+TIS"
    else:
        model, processor = load_llama_cot(lora_path=None)
        model_tag = "base"

    print("[model_loader] Ready: " + model_tag + "\n")
    return model, processor, model_tag
