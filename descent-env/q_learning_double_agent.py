import random
import numpy as np


class DoubleQLearningAgent:
    def __init__(self, altitude_space, vertical_velocity_space, target_altitude_space, runway_distance_space, actions, env):
        self.q1 = np.zeros((len(altitude_space), len(vertical_velocity_space), len(target_altitude_space), len(runway_distance_space), len(actions)))
        self.q2 = np.zeros_like(self.q1)
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
        total_actions = len(self.actions)
        subset = random.sample(range(total_actions), subset_size)
        q_sum = self.q1[state] + self.q2[state]
        return max(subset, key=lambda action: q_sum[action])

    def next_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, len(self.actions) - 1)
        else:
            subset_size = max(1, int(np.log(len(self.actions))))
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

                # Elegir aleatoriamente cuál Q actualizar
                if random.random() < 0.5:
                    best_action = np.argmax(self.q1[next_state])
                    td_target = reward + gamma * self.q2[next_state][best_action]
                    td_error = td_target - self.q1[state][action_idx]
                    self.q1[state][action_idx] += alpha * td_error
                else:
                    best_action = np.argmax(self.q2[next_state])
                    td_target = reward + gamma * self.q1[next_state][best_action]
                    td_error = td_target - self.q2[state][action_idx]
                    self.q2[state][action_idx] += alpha * td_error

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
                q_sum = self.q1[state] + self.q2[state]
                action_idx = np.argmax(q_sum)
                continuous_action = self.actions[action_idx]

                obs, reward, done, _, _ = env.step(continuous_action)
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode: {episode}, Reward: {total_reward:.2f}")

        return rewards
