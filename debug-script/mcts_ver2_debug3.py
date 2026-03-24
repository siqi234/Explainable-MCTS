import gymnasium as gym
import math
import random
import json
import os

# Q-table MCTS implementation with detailed tracking of outcomes (success, failure, timeout) for each node in the MCTS tree.

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
    def __init__(self, env, iterations=1000, rollouts_per_leaf=5,
                 alpha=0.1, gamma=0.99, epsilon=0.3):
        self.env = env
        self.iterations = iterations
        self.rollouts_per_leaf = rollouts_per_leaf  # K — "numerous simulations"
        self.action_space_size = env.action_space.n

        # Q-table persists across all steps — updated by every rollout trajectory
        # Small random init to break action ties on the very first iterations
        n_states = 16
        self.Q = [
            [random.uniform(-0.01, 0.01) for _ in range(self.action_space_size)]
            for _ in range(n_states)
        ]
        self.alpha = alpha      # Q-learning rate
        self.gamma = gamma      # discount factor
        self.epsilon = epsilon  # exploration vs Q-table exploitation in rollouts

    def is_terminal(self, state):
        return state == GOAL_STATE or state in HOLES

    def search(self, initial_state):
        # Tree rebuilt fresh each step; Q-table carries over and keeps improving
        root = DecisionNode(state=initial_state)
        self.root = root

        for _ in range(self.iterations):
            leaf = self._select(root)
            reward, outcome = self._simulate(leaf.state)
            self._backpropagate(leaf, reward, outcome)

        best_action = max(root.children, key=lambda a: root.children[a].visits)
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
    # Simulation — K Q-table guided rollouts, averaged
    # ------------------------------------------------------------------
    def _single_rollout(self, state):
        """
        One rollout using epsilon-greedy over the Q-table.
        Every (s, a, r, s') transition updates Q[s][a] in place —
        so the Q-table gets smarter with every simulation run.
        """
        current_state = state
        self.set_env_state(current_state)

        total_reward = 0.0
        depth = 0
        max_depth = 100
        outcome = 'timeout'
        done = False

        while not done and depth < max_depth:

            # Epsilon-greedy: explore randomly or exploit current Q-table
            if random.random() < self.epsilon:
                action = self.env.action_space.sample()
            else:
                action = max(range(self.action_space_size),
                             key=lambda a: self.Q[current_state][a])

            next_state, _, done, truncated, _ = self.env.step(action)

            if done:
                if next_state == GOAL_STATE:
                    r = 1.0
                    total_reward += 1.0
                    outcome = 'success'
                else:
                    r = -1.0
                    total_reward += -1.0
                    outcome = 'hole'
            else:
                r = -0.01
                total_reward -= 0.01

            # Q-learning update: Q(s,a) ← Q(s,a) + α(r + γ·max_a' Q(s',a') - Q(s,a))
            max_next_q = max(self.Q[next_state]) if not done else 0.0
            self.Q[current_state][action] += self.alpha * (
                r + self.gamma * max_next_q - self.Q[current_state][action]
            )

            if done or truncated:
                break

            depth += 1
            current_state = next_state
            self.set_env_state(current_state)

        return total_reward, outcome

    def _simulate(self, state):
        """
        Run K independent rollouts from `state`.
        Each rollout updates the shared Q-table, so later rollouts within
        the same call already benefit from earlier ones.
        Returns averaged cumulative reward and majority-vote outcome.
        """
        if self.is_terminal(state):
            if state == GOAL_STATE:
                return 1.0, 'success'
            else:
                return -1.0, 'hole'

        results = [self._single_rollout(state) for _ in range(self.rollouts_per_leaf)]

        avg_reward = sum(r for r, _ in results) / self.rollouts_per_leaf

        outcome_counts = {'success': 0, 'hole': 0, 'timeout': 0}
        for _, outcome in results:
            outcome_counts[outcome] += 1
        majority_outcome = max(outcome_counts, key=outcome_counts.get)

        return avg_reward, majority_outcome

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

    folder_name = "debugging-ver4"
    os.makedirs(folder_name, exist_ok=True)

    # One MCTS instance — tree rebuilds each step, Q-table persists across steps
    mcts = MCTS(sim_env, iterations=1000, rollouts_per_leaf=5,
                alpha=0.1, gamma=0.99, epsilon=0.3)

    print("Start MCTS Agent (Q-table policy + multiple rollouts) on *Slippery* Frozen Lake...")

    while not (done or truncated):
        sim_env.unwrapped.s = obs
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
