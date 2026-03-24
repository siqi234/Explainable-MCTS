import gymnasium as gym
import math
import random
import json
import os

# Random Rollout MCTS implementation with detailed tracking of outcomes (success, failure, timeout) for each node in the MCTS tree.

# 4x4 FrozenLake layout:
#  0  1  2  3
#  4  5  6  7
#  8  9 10 11
# 12 13 14 15
# Start: 0, Goal: 15, Holes: {5, 7, 11, 12}
# Actions: 0=Left, 1=Down, 2=Right, 3=Up

GOAL_STATE = 15
HOLES = {5, 7, 11, 12}


class DecisionNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent  # ChanceNode

        self.children = {}  # {action_id: ChanceNode}
        self.visits = 0
        self.value = 0.0
        self.success = 0
        self.failure = 0
        self.timeout = 0

    def is_fully_expanded(self, action_space_size):
        return len(self.children) == action_space_size

    def best_child(self, c_param=math.sqrt(2)):
        best_score = float('-inf')
        best_action = None

        for action_id, chance_node in self.children.items():
            if chance_node.visits == 0:
                score = float('inf')
            else:
                exploit = chance_node.value / chance_node.visits
                explore = c_param * math.sqrt(math.log(self.visits) / chance_node.visits)
                score = exploit + explore

            if score > best_score:
                best_score = score
                best_action = action_id

        return best_action

    def to_dict(self, current_depth=0, max_depth=3):
        data = {
            "type": "DecisionNode",
            "state": int(self.state),
            "visits": int(self.visits),
            "value": float(self.value),
            "success": int(self.success),
            "failure": int(self.failure),
            "timeout": int(self.timeout),
        }
        if current_depth < max_depth:
            data["children"] = {
                str(action_id): child.to_dict(current_depth + 1, max_depth)
                for action_id, child in self.children.items()
            }
        else:
            data["children"] = "Max depth reached"
        return data


class ChanceNode:
    def __init__(self, parent, action_id):
        self.state = parent.state
        self.parent = parent  # DecisionNode
        self.action_id = action_id

        self.children = {}  # {next_state: DecisionNode}
        self.visits = 0
        self.value = 0.0
        self.success = 0
        self.failure = 0
        self.timeout = 0

    def to_dict(self, current_depth=0, max_depth=3):
        data = {
            "type": "ChanceNode",
            "action_id": int(self.action_id),
            "visits": int(self.visits),
            "value": float(self.value),
            "success": int(self.success),
            "failure": int(self.failure),
            "timeout": int(self.timeout),
        }
        if current_depth < max_depth:
            data["children"] = {
                str(next_state): child.to_dict(current_depth + 1, max_depth)
                for next_state, child in self.children.items()
            }
        else:
            data["children"] = "Max depth reached"
        return data


class MCTS:
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations
        self.action_space_size = env.action_space.n
    def is_terminal(self, state):
        return state == GOAL_STATE or state in HOLES

    def search(self, state):
        self.root = DecisionNode(state=state)

        for _ in range(self.iterations):
            leaf = self._select(self.root)
            reward, outcome = self._simulate(leaf.state)
            self._backpropagate(leaf, reward, outcome)

        best_action = max(self.root.children, key=lambda a: self.root.children[a].visits)
        return best_action

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------
    def _select(self, node: DecisionNode):
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded(self.action_space_size):
                return self._expand(node)

            best_action = node.best_child()
            chance_node = node.children[best_action]

            self.set_env_state(node.state)
            next_state, _, done, _, _ = self.env.step(best_action)

            if next_state not in chance_node.children:
                new_node = DecisionNode(state=next_state, parent=chance_node)
                chance_node.children[next_state] = new_node
                return new_node

            node = chance_node.children[next_state]
            if done:
                return node

        return node

    # ------------------------------------------------------------------
    # Expansion
    # ------------------------------------------------------------------
    def _expand(self, node: DecisionNode):
        tried_actions = node.children.keys()
        untried_actions = [a for a in range(self.action_space_size) if a not in tried_actions]

        action = random.choice(untried_actions)

        chance_node = ChanceNode(parent=node, action_id=action)
        node.children[action] = chance_node

        self.set_env_state(node.state)
        next_state, _, _, _, _ = self.env.step(action)

        next_node = DecisionNode(state=next_state, parent=chance_node)
        chance_node.children[next_state] = next_node

        return next_node

    # ------------------------------------------------------------------
    # Simulation — random rollout (no domain knowledge)
    # ------------------------------------------------------------------
    def _simulate(self, state):
        current_state = state
        self.set_env_state(current_state)
        done = self.is_terminal(current_state)

        if done:
            if current_state == GOAL_STATE:
                return 1.0, 'success'
            else:
                return -1.0, 'hole'

        total_rewards = 0.0
        depth = 0
        max_depth = 100
        outcome = 'timeout'

        while not done and depth < max_depth:
            action = self.env.action_space.sample()
            next_state, _, done, truncated, _ = self.env.step(action)

            if done:
                if next_state == GOAL_STATE:
                    total_rewards += 1.0
                    outcome = 'success'
                else:
                    total_rewards += -1.0
                    outcome = 'hole'
                break
            if truncated:
                break

            total_rewards -= 0.01
            depth += 1
            current_state = next_state
            self.set_env_state(current_state)

        return total_rewards, outcome

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def _backpropagate(self, node, reward, outcome):
        while node is not None:
            node.visits += 1
            node.value += reward

            if outcome == 'success':
                node.success += 1
            elif outcome == 'hole':
                node.failure += 1
            elif outcome == 'timeout':
                node.timeout += 1

            node = node.parent

    def set_env_state(self, state):
        self.env.unwrapped.s = state


if __name__ == "__main__":
    real_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human")
    sim_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).unwrapped

    obs, info = real_env.reset(seed=42)
    sim_env.reset(seed=42)

    done = False
    truncated = False
    step = 0

    folder_name = "debugging-ver3"
    os.makedirs(folder_name, exist_ok=True)

    print("Start MCTS Agent on *Slippery* Frozen Lake...")

    while not (done or truncated):
        sim_env.unwrapped.s = obs
        mcts = MCTS(sim_env, iterations=1000)
        action = mcts.search(obs)

        tree_data = mcts.root.to_dict(current_depth=0, max_depth=4)
        with open(f"{folder_name}/mcts_tree_step_{step}.json", "w") as f:
            json.dump(tree_data, f, indent=4, ensure_ascii=False)
        print(f"Saved MCTS tree for step {step} to {folder_name}/mcts_tree_step_{step}.json")

        obs, reward, done, truncated, info = real_env.step(action)
        step += 1

        if done:
            if reward == 1:
                print("Goal Reached!")
            else:
                print("Fell in a hole.")

    print(f"Episode finished after {step} steps with reward {reward}")
    real_env.close()
    sim_env.close()
