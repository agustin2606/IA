import random
import numpy as np


class QLearningAgent:
    def __init__(self, x_space, vel_space, actions, env):
        self.q = np.zeros((x_space * vel_space, actions))
        self.position_bins = x_space
        self.velocity_bins = vel_space
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
        
    def discretize_state(self, state):
        x_bins = np.linspace(self.env.observation_space.low[0], self.env.observation_space.high[0], self.position_bins + 1)
        vel_bins = np.linspace(self.env.observation_space.low[1], self.env.observation_space.high[1], self.velocity_bins + 1)
        x = np.digitize(state[0], x_bins) - 1
        vel = np.digitize(state[1], vel_bins) - 1
        return x * self.velocity_bins + vel

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
                state = self.discretize_state(obs)
                action = self.next_action(state, epsilon)
                continuous_action = np.linspace(env.action_space.low[0], env.action_space.high[0], self.q.shape[1])[action]

                next_obs, reward, done, _, _ = env.step([continuous_action])

                next_state = self.discretize_state(next_obs)
            
                best_next_action = np.max(self.q[next_state])
                td_target = reward + gamma * best_next_action
                td_error = td_target - self.q[state, action]
                self.q[state, action] += alpha * td_error
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
                state = self.discretize_state(obs)
                action = np.argmax(self.q[state])
                continuous_action = np.linspace(env.action_space.low[0], env.action_space.high[0], self.q.shape[1])[action]
                
                obs, reward, done, _, _= env.step([continuous_action])
                total_reward += reward

            rewards.append(total_reward)
            print(f"Episode: {episode}, Reward: {total_reward}")
        return rewards