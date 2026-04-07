import gymnasium as gym
import math
import random
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

# --- Getting the MCTS implementation from the debugging versions --- 
from mcts_from_scratch_ver2 import MCTS, DecisionNode, ChanceNode
# from mcts_from_scratch_ver2_debug import MCTS, DecisionNode, ChanceNode
# from mcts_ver2_debug1 import MCTS, DecisionNode, ChanceNode
# from mcts_ver2_debug2 import MCTS, DecisionNode, ChanceNode
# from mcts_ver2_debug3 import MCTS, DecisionNode, ChanceNode

STATE_SPACE = {i for i in range(16)}  # Assuming a 4x4 grid with states 0-15
ACTION_SPACE = {0, 1, 2, 3}  # Assuming actions: 0=left, 1=down, 2=right, 3=up

def game_success_rate(iterations=100):
    total_wins = 0

    for num in tqdm(range(iterations), desc="Running MCTS trials"):
        real_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
        sim_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).unwrapped
        
        obs, info = real_env.reset(seed=42)
        sim_env.reset(seed=42)

        done = False
        truncated = False
        step = 0

        # print("Start MCTS Agent on *Slippery* Frozen Lake: trial", num + 1)
        while not (done or truncated):
            sim_env.unwrapped.s = obs
            mcts = MCTS(sim_env, iterations=1000)
            action = mcts.search(obs)  # Get the best action from the MCTS on the current state
            obs, reward, done, truncated, info = real_env.step(action)
            step += 1 

        if reward == 1.0:
            total_wins += 1
            print(f"Trial {num + 1}: Success in {step} steps!\n")

    return total_wins


def simulate_random_episode(env, start_state, iterations=10000):
    success = 0
    failure = 0
    timeout = 0

    for i in range(iterations):
        env.reset()
        env.unwrapped.s = start_state  # Set the environment to the specific state
        done = False
        truncated = False
        
        while not done and not truncated:
            action  = env.action_space.sample()  # Random action
            # action = random.choice(list(ACTION_SPACE))  # Random action from defined action space
            obs, reward, done, truncated, info = env.step(action)

        if done and reward == 1.0:
            success += 1
        elif done and reward == 0.0:
            failure += 1
        elif truncated:
            timeout += 1

    print(f"State {start_state}: Visits: {iterations}, Success: {success}, Failure: {failure}, Timeout: {timeout}")


def rollout_success_heatmap(rollouts_per_state=5000, save_path="random_rollout_heatmap.png"):
    HOLES = {5, 7, 11, 12}
    GOAL  = 15

    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    rates = {}

    for state in tqdm(range(16), desc="Measuring rollout success rate per state"):
        if state in HOLES:
            rates[state] = 0.0      # holes always fail
            continue
        if state == GOAL:
            rates[state] = 1.0      # already at goal
            continue

        success = 0
        for _ in range(rollouts_per_state):
            env.reset()
            env.unwrapped.s = state
            done = truncated = False
            while not done and not truncated:
                _, reward, done, truncated, _ = env.step(env.action_space.sample())
            if done and reward == 1.0:
                success += 1

        rates[state] = success / rollouts_per_state
        print(f"  State {state:2d}: {rates[state]*100:.1f}% success")

    env.close()

    # Arrange into 4x4 grid
    grid = np.array([rates[s] for s in range(16)], dtype=float).reshape(4, 4)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="RdYlGn")

    for s in range(16):
        row, col = s // 4, s % 4
        if s in HOLES:
            label = f"H (S{s})\n0.0%"
            color = "white"
        elif s == GOAL:
            label = f"GOAL (S{s})\n100.0%"
            color = "white"
        elif s == 0:
            label = f"START (S{s})\n{rates[s]*100:.1f}%"
            color = "black"
        else:
            label = f"S{s}\n{rates[s]*100:.1f}%"
            color = "black"
        ax.text(col, row, label, ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Success Rate (random rollout)", fontsize=11)

    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels([f"col {c}" for c in range(4)])
    ax.set_yticklabels([f"row {r}" for r in range(4)])
    ax.set_title(
        "Random Rollout Policy — Success Rate per State\n(FrozenLake 4×4, slippery)",
        fontsize=13
    )

    legend = [
        mpatches.Patch(color="grey",      label="H = Hole (always fail)"),
        mpatches.Patch(color="darkgreen", label="GOAL (always success)"),
    ]
    ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0, -0.08),
              ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\nHeatmap saved to {save_path}")
    plt.show()

    return rates


if __name__ == "__main__":
    # state = 14  # Starting state for testing
    # iterations = 10000
    # env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True)
    # simulate_random_episode(env, state, iterations=10000)

    total_wins = game_success_rate(iterations=100)
    print(f"Total wins in 100 trials: {total_wins}")

    # rollout_success_heatmap(rollouts_per_state=5000)
    