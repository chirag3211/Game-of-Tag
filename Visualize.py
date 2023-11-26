import pygame
from pygame.locals import QUIT
import gym
from gym import spaces
import numpy as np
import random

# Define constants for screen dimensions and colors
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

class Obstacle:
    def __init__(self, position):
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
        self.q_table = np.load('Catch_train_episode_100000.npy', allow_pickle=True).item()  # Q-table to store Q-values for state-action pairs
        self.actions = [0, 1, 2, 3]  # Actions for the CatcherAgent

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
            self.position[0] = max(self.position[0] - self.speed, 0)
        elif action == 1:  # Move right
            self.position[0] = min(self.position[0] + self.speed, SCREEN_WIDTH)
        elif action == 2:  # Move up
            self.position[1] = max(self.position[1] + self.speed, 0)
        elif action == 3:  # Move down
            self.position[1] = min(self.position[1] - self.speed, SCREEN_HEIGHT)
            
    def use_special_power(self, runners):
        # Use special power to freeze runners within the specified range.
        for runner in runners:
            distance = np.linalg.norm(self.position - runner.position)
            if distance < self.special_power_range:
                runner.freeze()
                print("Catcher rewarded, runner penalized.")

class RunnerAgent:
    def __init__(self, initial_position, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.1):
        # Initialize a RunnerAgent with a given initial position.
        self.position = np.array(initial_position, dtype=np.int8)
        self.speed = 8
        self.is_frozen = False
        self.freeze_duration = 10  # Example: 10 time steps for freezing duration
        self.remaining_freeze_duration = 0

        # Q-learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = np.load('Run_train_episode_100000.npy', allow_pickle=True).item()  # Q-table to store Q-values for state-action pairs
        self.actions = [0, 1, 2, 3]  # Actions for the RunnerAgent

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
        # Move the runner agent based on the chosen action.
        if not self.is_frozen:
            if action == 0:  # Move left
                self.position[0] = max(self.position[0] - self.speed, 0)
            elif action == 1:  # Move right
                self.position[0] = min(self.position[0] + self.speed, SCREEN_WIDTH)
            elif action == 2:  # Move up
                self.position[1] = max(self.position[1] + self.speed, 0)
            elif action == 3:  # Move down
                self.position[1] = min(self.position[1] - self.speed, SCREEN_HEIGHT)

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

class TagGameEnvWithPygame(gym.Env):
    def __init__(self, num_catchers=2, num_runners=6, num_obstacles=5):
        super(TagGameEnvWithPygame, self).__init__()

        self.num_catchers = num_catchers
        self.num_runners = num_runners
        self.observation_space = spaces.Box(low=0, high=100, shape=(2, num_catchers * 2), dtype=np.float32)
        self.action_space = spaces.Discrete(4)

        self.catchers = [CatcherAgent([np.random.uniform(0, SCREEN_WIDTH), np.random.uniform(0, SCREEN_HEIGHT)]) for i in range(num_catchers)]
        self.runners = [RunnerAgent([np.random.uniform(0, SCREEN_WIDTH), np.random.uniform(0, SCREEN_HEIGHT)]) for i in range(num_runners)]
        self.obstacles = [Obstacle([np.random.uniform(0, SCREEN_WIDTH), np.random.uniform(0, SCREEN_HEIGHT)]) for _ in range(num_obstacles)]

        self.max_steps = 1000
        self.current_step = 0

        # Initialize Pygame
        pygame.init()
        self.clock = pygame.time.Clock()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Tag Game")

        # Load catcher and runner images
        self.catcher_image = pygame.image.load('catcher.png')
        self.runner_image = pygame.image.load('runner.png')
        self.obstacle_image = pygame.Surface((20, 20))
        self.obstacle_image.fill(RED)

    def reset(self):
        for catcher in self.catchers:
            catcher.position = np.array([np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)])
            catcher.speed = 8.8

        for runner in self.runners:
            runner.position = np.array([np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)])
            runner.speed = 8
            runner.is_frozen = False
            runner.remaining_freeze_duration = 0
            
        for obstacle in self.obstacles:
            obstacle.position = np.array([np.random.randint(0, SCREEN_WIDTH), np.random.randint(0, SCREEN_HEIGHT)])

        self.current_step = 0
        return self._get_observation()

    def step(self, actions):
        if self._check_termination():
            return self._get_observation(), (0, 0), True, {}

        if len(actions) != self.num_catchers + self.num_runners:
            raise ValueError("Number of actions should be equal to num_catchers + num_runners.")

        catcher_rewards, runner_rewards = [0] * len(self.catchers), [0] * len(self.runners)

        for i, catcher in enumerate(self.catchers):
            state = state = self._get_observation()[0][i]
            action = actions[i]
            catcher.move(action)
            catcher.use_special_power(self.runners)

            next_state = state = self._get_observation()[0][i]
            max_q_next = max(catcher.get_q_value(next_state, a) for a in catcher.actions)
            new_q_value = (1 - 0.1) * catcher.get_q_value(state, action) + 0.1 * (1 + 0.9 * max_q_next)
            catcher.update_q_value(state, action, new_q_value)

        for i, runner in enumerate(self.runners):
            state = state = self._get_observation()[1][i]
            action = actions[i + self.num_catchers]
            runner.move(action)

            next_state = state = self._get_observation()[1][i]
            max_q_next = max(runner.get_q_value(next_state, a) for a in runner.actions)
            new_q_value = (1 - 0.1) * runner.get_q_value(state, action) + 0.1 * (1 + 0.9 * max_q_next)
            runner.update_q_value(state, action, new_q_value)

            for obstacle in self.obstacles:
                distance = np.linalg.norm(runner.position - obstacle.position)
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
        catcher_rewards = [0] * len(self.catchers)
        runner_rewards = [0] * len(self.runners)

        for catcher in self.catchers:
            for runner in self.runners:
                distance = np.linalg.norm(catcher.position - runner.position)
                if distance < catcher.special_power_range:
                    if runner.is_frozen:
                        catcher_rewards[self.catchers.index(catcher)] += 1
                        runner_rewards[self.runners.index(runner)] -= 1
                    else:
                        catcher_rewards[self.catchers.index(catcher)] -= 1
                        runner_rewards[self.runners.index(runner)] += 1
        return catcher_rewards, runner_rewards

    def _check_termination(self):
        done = False
        if all(runner.is_frozen for runner in self.runners) or self.current_step >= self.max_steps:
            done = True
        return done

    def _calculate_distances(self, catcher, runners):
        distances = [np.linalg.norm(catcher.position - runner.position) for runner in runners]
        return distances

    def _get_observation(self):
        distances_run = [self._calculate_distances(runner, self.catchers) for runner in self.runners]
        distance_run = []
        for i, runner in enumerate(self.runners):
            dist = distances_run[i].index(min(distances_run[i]))
            x, y = self.catchers[dist].position - runner.position
            z = min(int(np.linalg.norm(runner.position - obstacle.position)) for obstacle in self.obstacles)
            distance_run.append((x, y, z))
        distances_catch = [self._calculate_distances(catcher, self.runners) for catcher in self.catchers]
        distance_catch = []
        for i, catcher in enumerate(self.catchers):
            dist = distances_catch[i].index(min(distances_catch[i]))
            x, y = self.runners[dist].position - catcher.position
            distance_catch.append((x, y))
        return list([distance_catch, distance_run])

    def save_q_tables(self, episode):
        np.save(f'E:\Data\Game of Tag\QTables\Catcher\q_table_episode_{episode}.npy', self.catchers[0].q_table)
        np.save(f'E:\Data\Game of Tag\QTables\Runner\q_table_episode_{episode}.npy', self.runners[0].q_table)

    def render(self):
        self.screen.fill(WHITE)

        for catcher in self.catchers:
            self.screen.blit(self.catcher_image, catcher.position)

        for runner in self.runners:
            self.screen.blit(self.runner_image, runner.position)

        for obstacle in self.obstacles:
            self.screen.blit(self.obstacle_image, obstacle.position)

        pygame.display.flip()
        self.clock.tick(60)  # Limit frames per second

    def close(self):
        pygame.quit()

# The main Pygame loop
if __name__ == "__main__":
    env = TagGameEnvWithPygame(num_catchers=2, num_runners=6, num_obstacles=5)

    for episode in range(100000):
        state = env.reset()
        state = env._get_observation()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    env.close()
                    exit()

            actions = [env.catchers[i].choose_action(state[0][i]) for i in range(env.num_catchers)]
            actions += [env.runners[i].choose_action(state[1][i]) for i in range(env.num_runners)]
            state, rewards, done, _ = env.step(actions)
            env.render()

    for _ in range(5):
        state = env.reset()
        state = env._get_observation()
        done = False

        while not done:
            for event in pygame.event.get():
                if event.type == QUIT:
                    env.close()
                    exit()

            actions = [np.random.randint(0, 4) for _ in range(env.num_catchers + env.num_runners)]
            state, _, done, _ = env.step(actions)
            env.render()

        print(f"Episode finished.")
