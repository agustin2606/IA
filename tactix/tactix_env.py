import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TacTixEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, board_size=6, misere=False):
        super(TacTixEnv, self).__init__()
        self.board_size = board_size
        self.misere = misere  # True = misère (last move loses), False = normal
        self.board = np.ones((board_size, board_size), dtype=np.int32)
        self.done = False
        self.current_player = 0  # 0 or 1

        self.action_space = spaces.MultiDiscrete([board_size, board_size, board_size, 2])
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(board_size, board_size), dtype=np.int32),
            "current_player": spaces.Discrete(2)
        })

    def reset(self):
        self.board = np.ones((self.board_size, self.board_size), dtype=np.int32)
        self.done = False
        self.current_player = 0
        return self._get_obs()

    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "current_player": self.current_player
        }

    def _valid_action(self, idx, start, end, is_row):
        if not (0 <= idx < self.board_size and 0 <= start <= end < self.board_size):
            return False
        if is_row:
            return np.all(self.board[idx, start:end+1] == 1)
        else:
            return np.all(self.board[start:end+1, idx] == 1)

    def step(self, action):
        idx, start, end, is_row = action
        is_row = bool(is_row)

        if not self._valid_action(idx, start, end, is_row):
            raise ValueError("Invalid action.")

        if is_row:
            self.board[idx, start:end+1] = 0
        else:
            self.board[start:end+1, idx] = 0

        if np.count_nonzero(self.board) == 0:
            self.done = True
            reward = -1 if self.misere else 1  # Misère: last move loses
        else:
            reward = 0

        self.current_player = 1 - self.current_player
        return self._get_obs(), reward, self.done, {}

    def render(self, mode='human'):
        for row in self.board:
            print(' '.join('O' if cell else '.' for cell in row))

        if self.done:
            # current_player was swapped after last move
            last_player = 1 - self.current_player  # The one who just played
            if self.misere:
                winner = self.current_player + 1  # Opponent wins
            else:
                winner = last_player + 1          # Last player wins
            print(f"\n🎉 Player {winner} wins! ({'Misère' if self.misere else 'Normal'} rules)")
        else:
            print(f"Player {self.current_player + 1}'s turn ({'Misère' if self.misere else 'Normal'} rules)\n")