from pingpong import PromptFormat, PromptManager
from utils import build_prompts

class OrcaMiniChatPromptFormat(PromptFormat):
    @classmethod
    def context(cls, context):
        if context is None or context == "":
            return ""
        else: 
            return f"""### System: {context}
"""
    
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        ping = pingpong.ping[:truncate_size]
        pong = "" if pingpong.pong is None else pingpong.pong[:truncate_size]
        return f"""### User: {ping}
                   ### Response: {pong}
                """

class OrcaMiniChatPromptManager(PromptManager):
    def build_prompts(self, from_idx: int=0, to_idx: int=-1
                      , format:PromptFormat=OrcaMiniChatPromptFormat
                      , truncate_size: int=None):
        return build_prompts(self, from_idx, to_idx, format, truncate_size)
    
    