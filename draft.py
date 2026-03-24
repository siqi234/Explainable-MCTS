# Import packages
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import LEFT, RIGHT, DOWN, UP
from IPython.display import clear_output


def random_exploration(env, num_step):
    # Reset the environment to its initial state and get the initial observation (initial state)
    observation, info = env.reset(seed=2023)
    
    # Simulate the agent's actions for num_step time steps
    rewards = []
    for _ in range(num_step):
        # Choose a random action from the action space
        action = env.action_space.sample()

        # Take the chosen action and observe the resulting state, reward, and termination status
        observation, reward, terminated, truncated, info = env.step(action)
        # print(f"Action taken: {action}, New state: {observation}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")

        rewards.append(reward)

        # If the episode is terminated, reset the environment to the start cell
        if terminated:
            observation, info = env.reset()

        # Display the current state of the environment
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.show()
        
    return rewards

# Create the FrozenLake environment with specific settings
env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, render_mode="rgb_array")
rewards = random_exploration(env, 20)
env.close()
print(f"Total reward: {np.round(np.sum(rewards), 2)}")