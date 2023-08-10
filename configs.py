from dataclasses import dataclass, field
from typing import Optional

@dataclass
class TrainArguments(object):
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b",
        metadata={"help": "model_name_or_path"},
    )
    model_basename: Optional[str] = field(
        default="gptq_model-4bit-128g",
        metadata={"help": "model_basename"},
    )
    DEFAULT_SYSTEM_PROMPT: Optional[str] = field(
        default="",
        metadata={"help": "DEFAULT_SYSTEM_PROMPT"},
    )
    MAX_MAX_NEW_TOKENS: Optional[int] = field(
        default=2048,
        metadata={"help": "MAX_MAX_NEW_TOKENS"},
    )
    DEFAULT_MAX_NEW_TOKENS: Optional[int] = field(
        default=1024,
        metadata={"help": "MAX_MAX_NEW_TOKENS"},
    )
    MAX_INPUT_TOKEN_LENGTH: Optional[int] = field(
        default=4000,
        metadata={"help": "MAX_MAX_NEW_TOKENS"},
    )
    MODEL_PATH: Optional[str] = field(
        default="TheBloke/Llama-2-7b-Chat-GPTQ",
        metadata={"help": "MODEL_PATH"},
    )
    BACKEND_TYPE: Optional[str] = field(
        default=".env.7b_gptq_example",
        metadata={"help": "BACKEND_TYPE"},
    )
    LOAD_IN_8BIT: Optional[bool] = field(
        default=True,
        metadata={"help": "LOAD_IN_8BIT"},
    )
    LOAD_IN_4BIT: Optional[bool] = field(
        default=False,
        metadata={"help": "LOAD_IN_4BIT"},
    )
    DESCRIPTION: Optional[str] = field(
        default="""
# llama2-webui
This is a chatbot based on Llama-2. 
- Supporting models: 
    [Llama-2-7b](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)
    [13b](https://huggingface.co/llamaste/Llama-2-13b-chat-hf)
    [70b](https://huggingface.co/llamaste/Llama-2-70b-chat-hf)
    all [Llama-2-GPTQ](https://huggingface.co/TheBloke/Llama-2-7b-Chat-GPTQ)
    all [Llama-2-GGML](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML)

- Supporting model backends: 
    [tranformers](https://github.com/huggingface/transformers)
    [bitsandbytes(8-bit inference)](https://github.com/TimDettmers/bitsandbytes)
    [AutoGPTQ(4-bit inference)](https://github.com/PanQiWei/AutoGPTQ)
    [llama.cpp](https://github.com/ggerganov/llama.cpp)
""",
        metadata={"help": "DESCRIPTION"},
    )