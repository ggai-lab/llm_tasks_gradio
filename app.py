import os
import re
import time
import json
import copy
import types 
import argparse
import gradio as gr 
from os import listdir
from os.path import isfile, join

from transformers import AutoModelForCausalLM



# prompts by each model
prompt_map = {
    "llama2": ,
    "stablebelguna": ,
    "starcoder": ,
    "gpt4": ,
}

# response configs by each model 
response_configs = [
    f"configs/response_configs/{model_config}" for model_config in listdir("configs/response_configs") if isfile(join("configs/response_configs", model_config))
]


def move_to_task_select_view():
    return (
        "move to task select view",
        gr.update(visible=False),
        gr.update(visible=True),
    )

def use_chosen_task():
    try:
        # test = global_vars.model
    except AttributeError:
        raise gr.Error("There is no chosen task previously.")

    gen_config = 

# gradio 
def flip_text(x):
    return x[::-1]
def flip_code(x):
    return x[::-1]

with gr.Blocks() as demo:
    gr.Markdown("Text-based generation or Code-based generation using this demo.")
    with gr.Tab("Text-based generation"):
        text_input =gr.Textbox(label="message")
        text_output =gr.Textbox(label="output") # 큰 채팅창으로 구성
        parameter_box = gr.CheckboxGroup(["", "", ""])
        text_button = gr.Button("send")
    with gr.Tab("Code-based generation"):
        code_input =gr.Textbox(label="message")
        code_output =gr.Textbox(label="output") # 큰 채팅창으로 구성
        code_button = gr.Button("send")
    with gr.Accordion(""):
        gr.Markdown("")
    text_button.click(flip_text, inputs=text_input, outputs=text_output)
    code_button.click(flip_code, inputs=code_input, outputs=code_output)

demo.launch()
# https://www.gradio.app/guides/controlling-layout