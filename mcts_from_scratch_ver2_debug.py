import gymnasium as gym
import math
import random
import json
import os

class DecisionNode:
    def __init__(self, state, parent=None):
        self.state = state
        self.parent = parent # should be a chance node

        self.children = {} # {action_id: ChanceNode}, where action_id is the action taken from this desision node
        self.visits = 0
        self.value = 0.0
        
        # For better explanation, added success and failure counts to track how many times this node led to a win or a loss in the simulations
        self.success = 0
        self.failure = 0

    def is_fully_expanded(self, action_space_size):
        # A decision node is fully expanded if it has children for all possible actions
        return len(self.children) == action_space_size
    
    def best_child(self, c_param=math.sqrt(2)):
        best_score = float('-inf')
        best_child = None

        # iterate through all children (chance nodes) and calculate the UCB score for each
        for action_id, chance_node in self.children.items():

            if chance_node.visits == 0:
                score = float('inf')  # prioritize unvisited nodes
            else:
                exploit = chance_node.value / chance_node.visits
                explore  = c_param * math.sqrt(math.log(self.visits)/(chance_node.visits))
                score = exploit + explore # UCB score

            if score > best_score:
                best_score = score
                best_child = action_id
        
        return best_child
    
    def to_dict(self, current_depth=0, max_depth=3):
        # convert the decision node to a dictionary format for visualization, including its state, visits, value, and children (chance nodes)
        data = {
            "type": "DecisionNode",
            "state": int(self.state),
            "visits": int(self.visits),
            "value": float(self.value),
            # add success and failure counts
            "success": int(self.success),
            "failure": int(self.failure)
        }
        
        # if haven't reached the maximum depth, continue recursively traversing the ChanceNodes below
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
        self.state = parent.state # from the parent decision node
        self.parent = parent # should be a decision node
        self.action_id = action_id 

        self.children = {} # {next_state: DecisionNode}, e.g. {0: DecisionNode_0, ...}, should be a collection of decision nodes, where the next_state is determined by the environment after taking action_id from the current state (self.parent.state)
        self.visits = 0
        self.value = 0.0

        self.success = 0
        self.failure = 0

    def to_dict(self, current_depth=0, max_depth=3):
        # convert the chance node to a dictionary format for visualization, including its action_id, visits, value, and children (decision nodes)
        data = {
            "type": "ChanceNode",
            "action_id": int(self.action_id),
            "visits": int(self.visits),
            "value": float(self.value), # the expected value is the core basis for decision explanation
            # add success and failure counts
            "success": int(self.success),
            "failure": int(self.failure)
        }
        
        if current_depth < max_depth:
            data["children"] = {
                str(next_state): child.to_dict(current_depth + 1, max_depth)
                for next_state, child in self.children.items()
            }
        else:
            data["children"] = "Max depth reached"
            
        return data


# MCTS
class MCTS:
    # Initialize the MCTS with the environment
    def __init__(self, env, iterations=1000):
        self.env = env
        self.iterations = iterations
        self.action_space_size = env.action_space.n # get the #of actions from the given environment 

    def is_terminal(self, state):
        if state == 15: 
            return True
        holes = {5, 7, 11, 12} 
        return state in holes or state == 15

    def search(self, initial_state):
        root = DecisionNode(state=initial_state) # initialize the root node with the initial state
        self.root = root # store the root node for visualization

        for _ in range(self.iterations):
            leaf = self._select(root)
            reward, outcome = self._simulate(leaf.state)
            self._backpropagate(leaf, reward, outcome)

        best_action = max(root.children, key=lambda a: root.children[a].visits)
        return best_action

    def _select(self, node: DecisionNode):
        # Selection
        while not self.is_terminal(node.state):
            if not node.is_fully_expanded(self.action_space_size):
                node = self._expand(node)
                return node
            
            best_action = node.best_child()
            chance_node = node.children[best_action]

            self.set_env_state(node.state)

            # Sample next state from the chance node's distribution
            next_state, reward, done, _, _ = self.env.step(best_action)
            
            if next_state not in chance_node.children:
                node = DecisionNode(state=next_state, parent=chance_node)
                chance_node.children[next_state] = node
                return node

            node = chance_node.children[next_state]
            if done:
                return node

        return node
    
    def _expand(self, node: DecisionNode):
        # Expansion
        # child_node = None
        tried_actions = node.children.keys()
        untried_actions = [a for a in range(self.action_space_size) if a not in tried_actions]

        action = random.choice(untried_actions) # randomly pick an untried action to expand

        chance_node = ChanceNode(parent=node, action_id = action) # create a chance node for the selected action
        node.children[action] = chance_node # add the chance node to the children of the parent decision node

        self.set_env_state(node.state)
        next_state, reward, done, _, _ = self.env.step(action) 

        next_node = DecisionNode(state=next_state, parent=chance_node)
        chance_node.children[next_state] = next_node

        return next_node
   
    def _simulate(self, state):
        # Simulation
        current_state = state
        self.set_env_state(current_state)
        done = self.is_terminal(current_state)

        if done:
            if current_state == 15:
                return 1.0, 'success'
            else:
                return -1.0, 'hole'
        
        total_rewards = 0.0
        depth = 0
        max_depth = 100
        outcome = 'timeout'

        while not done and depth < max_depth:
            action = self.env.action_space.sample() # get action from the environment action space
            next_state, reward, done, truncated, _ = self.env.step(action)

            if done:
                if next_state == 15:
                    total_rewards += 1.0
                    outcome = 'success'
                else:
                    total_rewards += -0.0
                    outcome = 'hole'
                break
            if truncated:
                break
            
            total_rewards -= 0.1
            depth += 1
            current_state = next_state

        return total_rewards, outcome

    def _backpropagate(self, node, reward, outcome):
        # Backpropagation
        while node is not None:
            node.visits += 1
            node.value += reward

            # update success and failure counts for better explanation
            if outcome == 'success':
                node.success += 1
            elif outcome == 'hole':
                node.failure += 1

            node = node.parent # move up to the parent node

    def set_env_state(self, state):
        self.env.unwrapped.s = state # set the environment to the given state

if __name__ == "__main__":
    # Create the environment both for real visualization and for simulation
    real_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human")
    sim_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).unwrapped

    # Reset both environments
    obs, info = real_env.reset(seed=42)
    sim_env.reset(seed=42)

    # Run MCTS to get the best action from the initial state with 1000 iterations
    mcts = MCTS(sim_env, iterations=2000)
    
    # Define the status of the game
    done = False
    truncated = False
    step = 0

    folder_name = "debugging"
    os.makedirs(folder_name, exist_ok=True)

    print("Start MCTS Agent on *Slippery* Frozen Lake...")
    # Start the game loop
    while not (done or truncated):
        action = mcts.search(obs) # Get the best action from the MCTS on the current state

        # save the mcts tree after each step to a json file for visualization
        tree_data = mcts.root.to_dict(current_depth=0, max_depth=4)
        with open(f"{folder_name}/mcts_tree_step_{step}.json", "w") as f:
            json.dump(tree_data, f, indent=4, ensure_ascii=False)
        print(f"Saved MCTS tree for step {step} to {folder_name}/mcts_tree_step_{step}.json")

        obs, reward, done, truncated, info = real_env.step(action) # Take the action
        step += 1 

        # Check if win or not
        if done:
            if reward == 1:
                print("Goal Reached!")
            else:
                print("Fell in a hole.") 

    
    print(f"Episode finished after {step} steps with reward {reward}")
    real_env.close()
    sim_env.close()