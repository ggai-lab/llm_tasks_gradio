from pingpong import PromptFormat,PromptManager

class AlpacaPromptFromat(PromptFormat):
    @classmethod
    def context(cls, context):
        if context is None or context == "":
            return ""
        else:
            return f"""{context}
                    """

            
    @classmethod
    def prompt(cls, pingpong, truncate_size):
        return f"""### Instruction:
        {pingpong.ping[:truncate_size]}

                    ### Responses:
        {"" if pingpong.pong is None else pingpong.pong[:truncate_size]}"""
    

class AlpacaChatPromptManager(PromptManager):
    def build_prompts(self
                      , from_idx: int=0, to_idx: int=-1
                      , format: PromptFormat=AlpacaPromptFromat
                      , truncate_size=truncate_size):
        if to_idx == -1 or to_idx >= len(self.pingpongs):
            to_idx = len(self.pingpongs)
        
        results = format.context(self.context)

        for idx, pingpong in enumerate(self.pingpongs[from_idx:to_idx]):
            results += format.prompt(pingpong, truncate_size=truncate_size)
            
            if from_idx + idx != to_idx -1:
                results += """
"""

