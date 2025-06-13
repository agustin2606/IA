import random
from agent import Agent

class RandomTacTixAgent(Agent):
    def __init__(self, env):
        self.env = env

    def get_valid_actions(self, board):
        actions = []
        size = board.shape[0]
        for is_row in [0, 1]:
            for idx in range(size):
                line = board[idx, :] if is_row else board[:, idx]
                start = None
                for i in range(size):
                    if line[i] == 1:
                        if start is None:
                            start = i
                    elif start is not None:
                        actions.append([idx, start, i-1, is_row])
                        start = None
                if start is not None:
                    actions.append([idx, start, size-1, is_row])
        return actions

    def act(self, obs):
        actions = self.get_valid_actions(obs["board"])
        return random.choice(actions) if actions else None