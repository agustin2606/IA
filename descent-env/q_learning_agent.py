import random
import numpy as np


class QLearningAgent:
    def __init__(self, altitude_space, vertical_velocity_space, target_altitude_space, runway_distance_space, actions, env):
        self.q = np.zeros((len(altitude_space), len(vertical_velocity_space), len(target_altitude_space), len(runway_distance_space), len(actions)))
        self.altitude_space = altitude_space
        self.vertical_velocity_space = vertical_velocity_space
        self.target_altitude_space = target_altitude_space
        self.runway_distance_space = runway_distance_space
        self.actions = actions
        self.env = env

    def _get_state_index(self, state):
        alt_idx = np.digitize(state['altitude'][0], self.altitude_space) - 1
        vz_idx = np.digitize(state['vz'][0], self.vertical_velocity_space) - 1
        target_alt_idx = np.digitize(state['target_altitude'][0], self.target_altitude_space) - 1
        runway_dist_idx = np.digitize(state['runway_distance'][0], self.runway_distance_space) - 1
        return (alt_idx, vz_idx, target_alt_idx, runway_dist_idx)

    def select_action_from_subset(self, state, subset_size):
        # Selecciona un subconjunto aleatorio de acciones
        total_actions = len(self.actions)
        subset = random.sample(range(total_actions), subset_size)
        # Devuelve la acción con el mayor valor Q dentro del subconjunto
        return max(subset, key=lambda action: self.q[state][action])
    
    def next_action(self, state, epsilon):
        if random.random() < epsilon:
            # Exploración: elige una acción aleatoria
            return random.randint(0, len(self.actions) - 1)
        else:
            # Explotación: selecciona la mejor acción dentro de un subconjunto
            total_actions = len(self.actions)
            subset_size = max(1, int(np.log(total_actions)))
            return self.select_action_from_subset(state, subset_size)

    def train_agent(self, env, episodes, epsilon, gamma, alpha):
        rewards = []
        initial_epsilon = epsilon
        final_epsilon = 0.01
        epsilon_decay = (initial_epsilon - final_epsilon) / (episodes / 2)

        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = self._get_state_index(obs)
                action_idx = self.next_action(state, epsilon)
                continuous_action = self.actions[action_idx]

                next_obs, reward, done, _, _ = env.step(continuous_action)
                next_state = self._get_state_index(next_obs)

                best_next_action = np.max(self.q[next_state])
                td_target = reward + gamma * best_next_action
                td_error = td_target - self.q[state][action_idx]
                self.q[state][action_idx] += alpha * td_error

                obs = next_obs
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode: {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")
            epsilon = max(final_epsilon, initial_epsilon - epsilon_decay * episode)

        return rewards

    def test_agent(self, env, episodes):
        rewards = []

        for episode in range(episodes):
            obs, _ = env.reset()
            total_reward = 0
            done = False

            while not done:
                state = self._get_state_index(obs)
                action_idx = np.argmax(self.q[state])
                continuous_action = self.actions[action_idx]

                obs, reward, done, _, _ = env.step(continuous_action)
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode: {episode}, Reward: {total_reward:.2f}")

        return rewards