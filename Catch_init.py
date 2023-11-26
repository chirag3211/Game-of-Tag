# tag_game_env.py
import gym
from gym import spaces
import numpy as np
import random

class Obstacle:
    def __init__(self, position):
        # Initialize an obstacle with a given position.
        self.position = np.array(position, dtype=np.int8)

class CatcherAgent:
    def __init__(self, initial_position, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        # Initialize a CatcherAgent with a given initial position and Q-learning parameters.
        self.position = np.array(initial_position, dtype=np.int8)
        self.speed = 8.8
        self.special_power_range = 20  # Range to freeze runners

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}  # Q-table to store Q-values for state-action pairs
        self.actions = [0, 1, 2, 3]  # Actions for the CatcherAgent

        # Initialize Q-values for all possible state-action pairs
        for i in range(0, 101, 2):
              for j in range(0, 101, 2):
                    for action in self.actions:
                        self.q_table[((i, j), action)] = 0

    def get_q_value(self, state, action):
        # Get the Q-value for a given state-action pair from the Q-table.
        return self.q_table.get((state, action), 0)

    def choose_action(self, state):
        # Epsilon-greedy policy for action selection
        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(self.actions)  # Explore
        else:
            # Exploit: Choose action with the highest Q-value
            return max(self.actions, key=lambda a: self.get_q_value(state, a))

    def update_q_value(self, state, action, new_q_value):
        # Update the Q-value for a state-action pair using the Q-learning update rule.
        self.q_table[(state, action)] = new_q_value

    def move(self, action):
        # Move the catcher agent based on the chosen action.
        if action == 0:  # Move left
            self.position[0] -= self.speed
        elif action == 1:  # Move right
            self.position[0] += self.speed
        elif action == 2:  # Move up
            self.position[1] += self.speed
        elif action == 3:  # Move down
            self.position[1] -= self.speed

    def use_special_power(self, runners):
        # Use special power to freeze runners within the specified range.
        for runner in runners:
            distance = np.linalg.norm(self.position - runner.position)
            if distance < self.special_power_range:
                runner.freeze()

class RunnerAgent:
    def __init__(self, initial_position):
        # Initialize a RunnerAgent with a given initial position.
        self.position = np.array(initial_position, dtype=np.int8)
        self.speed = 8
        self.is_frozen = False
        self.freeze_duration = 10  # Example: 10 time steps for freezing duration
        self.remaining_freeze_duration = 0

    def move(self, action):
        # Move the runner agent based on the chosen action.
        if not self.is_frozen:
            if action == 0:  # Move left
                self.position[0] -= self.speed
            elif action == 1:  # Move right
                self.position[0] += self.speed
            elif action == 2:  # Move up
                self.position[1] += self.speed
            elif action == 3:  # Move down
                self.position[1] -= self.speed

    def freeze(self):
        # Freeze the runner.
        self.is_frozen = True
        self.remaining_freeze_duration = self.freeze_duration

    def unfreeze(self):
        # Unfreeze the runner.
        self.is_frozen = False
        self.remaining_freeze_duration = 0

    def update_freeze_duration(self):
        # Update the remaining freeze duration.
        if self.remaining_freeze_duration > 0:
            self.remaining_freeze_duration -= 1
            if self.remaining_freeze_duration == 0:
                self.unfreeze()

class TagGameEnv(gym.Env):
    def __init__(self, num_catchers=2, num_runners=6, num_obstacles=5):
        # Initialize the TagGameEnv environment.
        super(TagGameEnv, self).__init__()

        self.num_catchers = num_catchers
        self.num_runners = num_runners

        # Define observation and action spaces.
        self.observation_space = spaces.Box(low=0, high=100, shape=(2, num_catchers * 2), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        # Create catcher, runner, and obstacle instances.
        self.catchers = [CatcherAgent([i * 10, 0]) for i in range(num_catchers)]
        self.runners = [RunnerAgent([500 + i * 10, 500]) for i in range(num_runners)]
        self.obstacles = [Obstacle([np.random.uniform(0, 100), np.random.uniform(0, 100)]) for _ in range(num_obstacles)]

        # Set maximum number of steps for an episode
        self.max_steps = 1000
        self.current_step = 0

    def reset(self):
        # Reset the environment to start a new episode.
        for catcher in self.catchers:
            catcher.position = np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)], dtype=np.float32)
            catcher.speed = 8.8

        for runner in self.runners:
            runner.position = np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)], dtype=np.float32)
            runner.speed = 8
            runner.is_frozen = False
            runner.remaining_freeze_duration = 0

        for obstacle in self.obstacles:
            obstacle.position = np.array([np.random.uniform(0, 100), np.random.uniform(0, 100)], dtype=np.float32)
        
        self.current_step = 0
        return self._get_observation()

    def step(self, actions):
        # Take a step in the environment based on the given actions.
        if self._check_termination():
            return self._get_observation(), (0, 0), True, {}

        if len(actions) != self.num_catchers + self.num_runners:
            raise ValueError("Number of actions should be equal to num_catchers + num_runners.")

        catcher_rewards, runner_rewards = [0] * len(self.catchers), [0] * len(self.runners)

        for i, catcher in enumerate(self.catchers):
            state = self._get_observation()[i]
            action = actions[i]
            catcher.move(action)
            catcher.use_special_power(self.runners)

            # Q-learning update
            next_state = self._get_observation()[i]
            max_q_next = max(catcher.get_q_value(next_state, a) for a in catcher.actions)
            new_q_value = (1 - 0.1) * catcher.get_q_value(state, action) + 0.1 * (1 + 0.9 * max_q_next)
            catcher.update_q_value(state, action, new_q_value)

        for i, runner in enumerate(self.runners):
            runner.move(actions[i + self.num_catchers])

            for obstacle in self.obstacles:
                distance = int(np.linalg.norm(runner.position - obstacle.position))
                if distance < 10:
                    runner.freeze()

            if runner.remaining_freeze_duration == 0:
                for catcher in self.catchers:
                    if catcher.position[0] < runner.position[0] + 5 < catcher.position[0] + 15 and \
                            catcher.position[1] < runner.position[1] + 5 < catcher.position[1] + 15:
                        catcher_rewards[self.catchers.index(catcher)] += 1
                        runner_rewards[self.runners.index(runner)] -= 1

        catcher_rewards, runner_rewards = self._calculate_rewards()
        done = self._check_termination()

        self.current_step += 1

        return self._get_observation(), (sum(catcher_rewards), sum(runner_rewards)), done, {}

    def _calculate_rewards(self):
        # Calculate rewards based on the interaction between catchers and runners.
        catcher_rewards = [-0.1] * len(self.catchers)
        runner_rewards = [0] * len(self.runners)

        for catcher in self.catchers:
            for runner in self.runners:
                distance = np.linalg.norm(catcher.position - runner.position)
                if distance < catcher.special_power_range:  # Catcher is within range to freeze runner
                    if runner.is_frozen:
                        catcher_rewards[self.catchers.index(catcher)] += 1  # Reward for freezing runner
                        runner_rewards[self.runners.index(runner)] -= 1  # Penalty for being frozen
                    else:
#                        catcher_rewards[self.catcher.index(catcher)] -= 1  # Reward for evading catcher                        
                        runner_rewards[self.runners.index(runner)] += 1  # Reward for evading catcher
        return catcher_rewards, runner_rewards

    def _check_termination(self):
        # Check if the episode should terminate based on freeze conditions or maximum steps.
        done = False
        if all(runner.is_frozen for runner in self.runners) or self.current_step >= self.max_steps:
            done = True
        return done

    def _get_observation(self):
        # Get the current observation of the environment.
        distances = [self._calculate_distances(catcher) for catcher in self.catchers]
        distance = []
        for i, catcher in enumerate(self.catchers):
            dist = distances[i].index(min(distances[i]))
            x, y = self.runners[dist].position - catcher.position
            distance.append((x, y))
        return distance

    def _calculate_distances(self, catcher):
        # Calculate distances between the catcher and the two runners.
        distances = [np.linalg.norm(catcher.position - runner.position) for runner in self.runners]
        return distances

    def save_q_table(self, episode):
        # Save the Q-table to a file after each episode.
        filename = f'Catch_init.npy'
        np.save(filename, self.catchers[0].q_table)

if __name__ == "__main__":
    env = TagGameEnv(num_catchers=2, num_runners=6, num_obstacles=5)

    for episode in range(1001):
        state = env.reset()
        state = env._get_observation()
        done = False

        while not done:
            actions = [env.catchers[i].choose_action(state[i]) for i in range(env.num_catchers)]
            actions += [np.random.randint(0, 4) for _ in range(env.num_runners)]  # Generate random actions for runners
            state, rewards, done, _ = env.step(actions)

        if episode % 1 == 0:
            print(f"Episode {episode} - Total Reward: {rewards}")
    
    env.save_q_table(episode)

    for _ in range(5):
        state = env.reset()
        state = env._get_observation()
        done = False

        while not done:
            actions = [np.random.randint(0, 4) for _ in range(env.num_catchers + env.num_runners)]  # Generate actions for both catchers and runners
            state, _, done, _ = env.step(actions)

        print(f"Episode finished.")
