import random
import numpy as np


class QLearningAgent:
    def __init__(self, altitude_space, vertical_velocity_space, target_altitude_space, runway_distance_space, actions, env):
        self.q = (np.zeros((len(altitude_space), len(vertical_velocity_space), len(target_altitude_space), len(runway_distance_space), len(actions))))
        self.altitud_space = altitude_space
        self.vertical_velocity_space = vertical_velocity_space
        self.target_altitud_space = target_altitude_space
        self.runway_distance_space = runway_distance_space
        self.actions = actions
        self.env = env
    
    def select_action_from_subset(self, obs, subset_size):
        total_actions = self.q.shape[1]
        subset = random.sample(range(total_actions), subset_size)
        return max(subset, key=lambda action: self.q[obs, action])

    def next_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.q.shape[1] - 1)
        else:
            total_actions = self.q.shape[1]
            subset_size = max(1, int(np.log(total_actions)))
            return self.select_action_from_subset(state, subset_size)
    
    def _get_state_index(self, state):
        alt_idx = np.digitize(state['altitude'][0], self.altitud_space) - 1
        vz_idx = np.digitize(state['vz'][0], self.vertical_velocity_space) - 1
        target_alt_idx = np.digitize(state['target_altitude'][0], self.target_altitud_space) - 1
        runway_dist_idx = np.digitize(state['runway_distance'][0], self.runway_distance_space) - 1
        return (alt_idx, vz_idx, target_alt_idx, runway_dist_idx)

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
                state = self._get_state_index(obs)  # Discretiza el estado
                action = self.next_action(state, epsilon)
                continuous_action = np.linspace(env.action_space.low[0], env.action_space.high[0], self.q.shape[1])[action]

                next_obs, reward, done, _, _ = env.step(continuous_action)

                next_state = self._get_state_index(next_obs)  # Discretiza el siguiente estado

                best_next_action = np.max(self.q[next_state])  # Usa índices discretos
                td_target = reward + gamma * best_next_action
                td_error = td_target - self.q[state][action]  # Usa índices discretos
                self.q[state][action] += alpha * td_error  # Usa índices discretos
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
                state = obs
                action = np.argmax(self.q[state])
                continuous_action = np.linspace(env.action_space.low[0], env.action_space.high[0], self.q.shape[1])[action]
                
                obs, reward, done, _, _= env.step([continuous_action])
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode: {episode}, Reward: {total_reward}")
        return rewards