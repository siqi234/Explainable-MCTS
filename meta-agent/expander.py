import json
import sys
import os
import gymnasium as gym

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from mcts_from_scratch_ver2 import MCTS


def find_path_to_state(node, target_state):
    """
    DFS from a DecisionNode dict to find the first DecisionNode with target_state.
    Returns the full path (alternating DecisionNode, ChanceNode, ..., DecisionNode)
    from `node` to the target, inclusive. Returns None if not found.
    """
    if node['state'] == target_state:
        return [node]

    children = node.get('children')
    if not isinstance(children, dict):
        return None

    for chance_node in children.values():
        sub_children = chance_node.get('children')
        if not isinstance(sub_children, dict):
            continue
        for decision_node in sub_children.values():
            path = find_path_to_state(decision_node, target_state)
            if path is not None:
                return [node, chance_node] + path

    return None


def graft_and_backpropagate(tree, target_state, target_action, new_subtree):
    """
    Finds target_state in the tree, grafts new_subtree as its child at target_action,
    then backpropagates the delta stats (visits/value/success/failure) up to root.
    """
    path = find_path_to_state(tree, target_state)
    if path is None:
        print(f"[expander] Could not find DecisionNode at state {target_state} in tree.")
        return False

    target_node = path[-1]

    if str(target_action) in target_node.get('children', {}):
        print(f"[expander] Action {target_action} already exists at state {target_state}. Skipping graft.")
        return False

    # Graft the new subtree
    if not isinstance(target_node.get('children'), dict):
        target_node['children'] = {}
    target_node['children'][str(target_action)] = new_subtree

    # Delta = the new subtree's total accumulated stats
    delta_visits  = new_subtree['visits']
    delta_value   = new_subtree['value']
    delta_success = new_subtree.get('success', 0)
    delta_failure = new_subtree.get('failure', 0)

    # Backpropagate delta to every ancestor on the path (root → target_node inclusive).
    # This mirrors _backpropagate: every simulation that ran through the new subtree
    # would have incremented each ancestor once.
    for node in path:
        node['visits']  += delta_visits
        node['value']   += delta_value
        node['success'] += delta_success
        node['failure'] += delta_failure

    return True


def expand_and_graft(tree_file, target_state, target_action, output_file, iterations=1000):
    """
    Loads an existing MCTS tree from tree_file, runs targeted MCTS from
    target_state to build the new subtree for target_action, grafts it into
    the existing tree, backpropagates the stats to root, and saves to output_file.
    """
    with open(tree_file) as f:
        tree = json.load(f)

    # Run MCTS from target_state — only this subtree grows, the rest is frozen
    sim_env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True).unwrapped
    sim_env.reset(seed=42)

    mcts = MCTS(sim_env, iterations=iterations)
    mcts.search(target_state)
    sim_env.close()

    if target_action not in mcts.root.children:
        print(f"[expander] Action {target_action} was not explored. Try more iterations.")
        return

    new_subtree = mcts.root.children[target_action].to_dict(current_depth=0, max_depth=3)

    success = graft_and_backpropagate(tree, target_state, target_action, new_subtree)
    if success:
        with open(output_file, 'w') as f:
            json.dump(tree, f, indent=2)
        print(f"[expander] Grafted action {target_action} at state {target_state}.")
        print(f"  New subtree — visits: {new_subtree['visits']}, value: {new_subtree['value']:.3f}")
        print(f"  Saved to {output_file}")


if __name__ == "__main__":
    expand_and_graft(
        tree_file="./meta-agent/mcts_tree_step_9_modified.json",
        target_state=13,
        target_action=2,
        output_file="./meta-agent/mcts_tree_step_9_expanded.json",
        iterations=1000,
    )
