import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

def load_model(
        base, 
        finetuned,
        qptq, 
        qptq_base,
        mode_cpu,
        mode_mps,
        mode_full_gpu,
        mode_8bit,
        mode_4bit,
        mode_gptq,
        mode_mps_gptq,
        mode_cpu_gptq,
        force_download_ckpt,
        local_files_only,
):
    tokenizer = AutoTokenizer.from_pretrained(
        base, local_files_only=local_files_only, use_fast=False
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left" ## 
    
    if mode_cpu:
        print("CPU mode.")
        model = AutoModelForCausalLM.from_pretrained(
            base,
            device_map # https://huggingface.co/docs/accelerate/usage_guides/big_modeling
        )


