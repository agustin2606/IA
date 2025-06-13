from abc import ABC, abstractmethod
from tactix_env import TacTixEnv

class Agent(ABC):
    
    @abstractmethod
    def __init__(self, env:TacTixEnv):
        self.env = env

    @abstractmethod
    def act(self, obs):
        return NotImplementedError
