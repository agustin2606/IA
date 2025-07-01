from agent import Agent
from tactix_env import TacTixEnv
import math
import numpy as np

class ExpectimaxAgent(Agent):
    def __init__(self, env: TacTixEnv, player=1, depth=3):
        self.env = env
        self.player = player  # 1 or 2 for player number
        self.agent_id = player - 1  # 0 or 1 for internal env representation
        self.opponent = 2 if player == 1 else 1
        self.opponent_id = 1 - self.agent_id
        self.depth = depth

    def act(self, obs):
        """
        Required implementation of the abstract method from Agent class.
        Takes an observation and returns an action.
        """
        return self.next_action(obs)
        
    def is_terminal(self, state):
        """
        Check if the game state is terminal (no more moves possible).
        """
        board = state["board"]
        return np.count_nonzero(board) == 0
        
    def get_legal_actions(self, state, player_num):
        """
        Get all legal actions for a player in the current state.
        """
        board = state["board"]
        actions = []
        
        # Check all possible row actions
        for row in range(self.env.board_size):
            start = None
            for col in range(self.env.board_size):
                if board[row, col] == 1:
                    if start is None:
                        start = col
                else:
                    if start is not None:
                        # Found a valid segment
                        for end in range(start, col):
                            actions.append([row, start, end, 1])
                        start = None
            # Check if we have a segment at the end of the row
            if start is not None:
                for end in range(start, self.env.board_size):
                    actions.append([row, start, end, 1])
                    
        # Check all possible column actions
        for col in range(self.env.board_size):
            start = None
            for row in range(self.env.board_size):
                if board[row, col] == 1:
                    if start is None:
                        start = row
                else:
                    if start is not None:
                        # Found a valid segment
                        for end in range(start, row):
                            actions.append([col, start, end, 0])
                        start = None
            # Check if we have a segment at the end of the column
            if start is not None:
                for end in range(start, self.env.board_size):
                    actions.append([col, start, end, 0])
                    
        return actions
        
    def heuristic_utility(self, state):
        """
        Heurística para evaluar el estado del juego.
        """
        if self.is_terminal(state):
            # If it's terminal, check whose turn it is to determine winner
            current_player = state["current_player"]
            if current_player == self.agent_id:  # Opponent made the last move
                return 1000 if not self.env.misere else -1000
            else:  # Agent made the last move
                return -1000 if not self.env.misere else 1000
        
        # Not terminal, evaluate based on available moves
        player_actions = len(self.get_legal_actions(state, self.player))
        opponent_actions = len(self.get_legal_actions(state, self.opponent))
        return player_actions - opponent_actions
        
    def perform_action(self, state, action):
        """
        Apply an action to the state and return the new state.
        """
        board = state["board"].copy()
        current_player = state["current_player"]
        idx, start, end, is_row = action
        is_row = bool(is_row)
        
        if is_row:
            board[idx, start:end+1] = 0
        else:
            board[start:end+1, idx] = 0
            
        # Switch player
        next_player = 1 - current_player
        
        return {
            "board": board,
            "current_player": next_player
        }

    def next_action(self, state):
        """
        Decide la mejor acción utilizando Expectimax.
        """
        best_action, _ = self.expectimax(state, self.depth, True)
        return best_action

    def expectimax(self, state, depth, maximizing_player):
        """
        Implementación del algoritmo Expectimax.
        """
        if depth == 0 or self.is_terminal(state):
            return None, self.heuristic_utility(state)

        current_player = self.player if maximizing_player else self.opponent
        possible_actions = self.get_legal_actions(state, current_player)

        if not possible_actions:  # No legal moves
            return None, self.heuristic_utility(state)
            
        if maximizing_player:
            value = -math.inf
            best_action = possible_actions[0] if possible_actions else None
            for action in possible_actions:
                new_state = self.perform_action(state, action)
                _, new_value = self.expectimax(new_state, depth - 1, False)
                if new_value > value:
                    value = new_value
                    best_action = action
            return best_action, value
        else:
            total_value = 0
            for action in possible_actions:
                new_state = self.perform_action(state, action)
                _, new_value = self.expectimax(new_state, depth - 1, True)
                total_value += new_value

            expected_value = total_value / len(possible_actions) if possible_actions else 0
            return None, expected_value