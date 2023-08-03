import json 

class PingPong:
    def __init__(self, ping, pong):
        self.ping = ping
        self.pong = pong
    
    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    @classmethod
    def from_json(cls, json_dict):
        return PingPong(json_dict['ping'], json_dict['pong'])
    

class PromptFormat:
    @classmethod
    def context(cls, context):
        pass

    @classmethod
    def prompt(cls, pingpong):
        pass


class UIFormat:
    @classmethod
    def ui(cls, pingpong):
        pass


class PromptManager:
    def __init__(self, context: str=""):
        self.context = context
        self.pingpongs = []
    
    def add_ping(self, ping, format: PromptFormat):
        if len(self.pingpongs) == 0 \ 
            or (len(self.pingpongs) >= 1 and self.pingpongs[-1].pong is not None):
            self.pingpongs.append(PingPong(ping, None))
            return True
        return False
    
    def add_pong(self, pong):
        self.pingpongs[-1].pong=pong

    def append_pong(self, piece_pong):
        self.pingpongs[-1].pong += piece_pong
    
    def add_pingpong(self, pingpong):
        self.pingpongs.append(pingpong)
    
    def pop_pingpong(self):
        return self.pingpongs.pop()
    
    def build_prompts(self, from_idx: int, to_idx: int, format: PromptFormat, truncate_size: int):
        pass

    def build_uis(self, from_idx: int, to_idx: int, format:UIFormat):
        pass

    def __repr__(self):
        return json.dumps(self, default=lambda o: o.__dict__)
    
    ## 
    @classmethod
    def from_json(cls, filename):
        json_dict = json.loads(filename)
        
        ## 
        new_instance = cls()
        new_instance.context = json_dict['context']
        new_instance.pingpongs = [PingPong.from_json(pingpong) for pingpong in json_dict['pingpongs']]
        return new_instance 
    
    

