from pingpong import PromptManager, PromptFormat, UIFormat

def build_prompts(
        prompt: PromptManager=None,
        from_idx: int=0,
        to_idx: int=-1,
        format: PromptFormat=None,
        truncate_size: int=None
):
    if to_idx == -1 or to_idx >= len(prompt.pingpongs):
        to_idx = len(prompt.pingpongs)
    
    results = format.context(prompt.context)

    for _, pingpong in enumerate(prompt.pingpongs[from_idx:to_idx]):
        results += format.prompt(pingpong, truncate_size=truncate_size)
        
    return results 


def gradio_build_uis(
        prompt: PromptManager=None,
        from_idx: int=0,
        to_idx: int=-1,
        format: UIFormat=None
):
    if to_idx == -1 or to_idx >= len(prompt.pingpongs):
        to_idx = len(prompt.pingpongs)
    
    results = []

    for pingpong in prompt.pingpongs[from_idx:to_idx]:
        results.append(format.ui(pingpong))
    
    return results 

