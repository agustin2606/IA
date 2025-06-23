from agent import Agent
from tactix_env import TacTixEnv
import math

class MinimaxAgent(Agent):
    def __init__(self, env: TacTixEnv, player=1, depth=3):
        super().__init__(env)
        self.env = env
        self.player = player
        self.opponent = 2 if player == 1 else 1
        self.depth = depth

    def heuristic_utility(self, state):
        """
        Heurística para evaluar el estado del juego.
        Puedes personalizar esta función según las reglas de TacTix.
        """
        # Ejemplo básico: diferencia entre las acciones legales de ambos jugadores
        player_actions = len(self.env.get_legal_actions(state, self.player))
        opponent_actions = len(self.env.get_legal_actions(state, self.opponent))
        return player_actions - opponent_actions

    def next_action(self, state):
        """
        Decide la mejor acción utilizando Minimax con Alpha-Beta Pruning.
        """
        best_action, _ = self.minimax(state, self.depth, -math.inf, math.inf, True)
        return best_action

    def minimax(self, state, depth, alpha, beta, maximizing_player):
        """
        Implementación del algoritmo Minimax con poda Alpha-Beta.
        """
        if depth == 0 or self.env.is_terminal(state):
            return None, self.heuristic_utility(state)

        current_player = self.player if maximizing_player else self.opponent
        possible_actions = self.env.get_legal_actions(state, current_player)

        if maximizing_player:
            value = -math.inf
            best_action = None
            for action in possible_actions:
                new_state = self.env.perform_action(state, action)
                _, new_value = self.minimax(new_state, depth - 1, alpha, beta, False)
                if new_value > value:
                    value = new_value
                    best_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return best_action, value
        else:
            value = math.inf
            best_action = None
            for action in possible_actions:
                new_state = self.env.perform_action(state, action)
                _, new_value = self.minimax(new_state, depth - 1, alpha, beta, True)
                if new_value < value:
                    value = new_value
                    best_action = action
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return best_action, value