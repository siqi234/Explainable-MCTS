import json

MIN_VISITS = 10  # minimum visits to consider a node "reliably searched"


def find_node(tree: dict, target_state: int) -> dict | None:
    """
    Recursively searches the tree for a DecisionNode with the given state.
    Returns the first match found, or None.
    """
    if tree.get("type") == "DecisionNode" and tree.get("state") == target_state:
        return tree

    children = tree.get("children", {})
    if isinstance(children, str):  # "Max depth reached" — can't go deeper
        return None

    for child in children.values():
        result = find_node(child, target_state)
        if result is not None:
            return result

    return None


def check_gap(tree: dict, intent: dict) -> dict:
    """
    Given an MCTS tree and a structured intent from the extractor,
    determines whether the question can be answered from existing tree data.

    intent keys: target_state, target_action, target_path, question_type

    Returns:
        {
            "answerable": bool,
            "reason": str,
            "node": dict or None   # the relevant DecisionNode if found
        }
    """
    question_type = intent.get("question_type", "general")

    # General questions always answerable
    if question_type == "general":
        return {
            "answerable": True,
            "reason": "General question — no specific node required.",
            "node": None
        }

    target_state = intent.get("target_state")
    target_action = intent.get("target_action")
    target_path = intent.get("target_path")

    # --- Step 1: find the target DecisionNode ---
    if target_state is None:
        return {
            "answerable": False,
            "reason": "No target state could be extracted from the question.",
            "node": None
        }

    node = find_node(tree, target_state)

    if node is None:
        return {
            "answerable": False,
            "reason": f"State {target_state} has not been visited in the search tree.",
            "node": None
        }

    # --- Step 2: check if the node itself has enough visits ---
    if node["visits"] < MIN_VISITS:
        return {
            "answerable": False,
            "reason": (
                f"State {target_state} was found but only visited {node['visits']} times "
                f"(minimum required: {MIN_VISITS}). Data is unreliable."
            ),
            "node": node
        }

    # --- Step 3: path question — walk each action in the sequence ---
    if question_type == "path" and target_path:
        current_node = node
        for step, action in enumerate(target_path):
            children = current_node.get("children", {})

            if isinstance(children, str):
                return {
                    "answerable": False,
                    "reason": (
                        f"Path blocked at step {step}: state {current_node['state']} "
                        f"is at max tree depth, action {action} not expanded."
                    ),
                    "node": current_node
                }

            if str(action) not in children:
                return {
                    "answerable": False,
                    "reason": (
                        f"Path blocked at step {step}: action {action} not found "
                        f"at state {current_node['state']} (branch missing)."
                    ),
                    "node": current_node
                }

            chance_node = children[str(action)]
            if chance_node["visits"] < MIN_VISITS:
                return {
                    "answerable": False,
                    "reason": (
                        f"Path blocked at step {step}: action {action} at state "
                        f"{current_node['state']} only has {chance_node['visits']} visits."
                    ),
                    "node": current_node
                }

            # Advance to the most-visited next DecisionNode (best estimate of next state)
            if step < len(target_path) - 1:
                next_states = chance_node.get("children", {})
                if isinstance(next_states, str) or not next_states:
                    return {
                        "answerable": False,
                        "reason": (
                            f"Path blocked at step {step}: no next states found "
                            f"after action {action} at state {current_node['state']}."
                        ),
                        "node": current_node
                    }
                current_node = max(next_states.values(), key=lambda n: n.get("visits", 0))

        return {
            "answerable": True,
            "reason": "Full path is searchable in the tree.",
            "node": node
        }

    # --- Step 4: node question — check specific action if asked ---
    if target_action is not None:
        children = node.get("children", {})

        if isinstance(children, str):
            return {
                "answerable": False,
                "reason": (
                    f"State {target_state} is at max tree depth. "
                    f"Action {target_action} was not expanded."
                ),
                "node": node
            }

        if str(target_action) not in children:
            return {
                "answerable": False,
                "reason": (
                    f"Action {target_action} at state {target_state} "
                    f"has not been searched (branch missing from tree)."
                ),
                "node": node
            }

        action_node = children[str(target_action)]
        if action_node["visits"] < MIN_VISITS:
            return {
                "answerable": False,
                "reason": (
                    f"Action {target_action} at state {target_state} only has "
                    f"{action_node['visits']} visits (minimum required: {MIN_VISITS})."
                ),
                "node": node
            }

    return {
        "answerable": True,
        "reason": "Sufficient data available in the tree.",
        "node": node
    }


# --- Manual test ---
if __name__ == "__main__":
    with open("./meta-agent/mcts_tree_step_9_modified.json", "r") as f:
        tree = json.load(f)

    # test_cases = [
    #     # answerable — state 13 root has 1000 visits, action 1 exists
    #     {"target_state": 13, "target_action": 1, "target_path": None, "question_type": "node"},

    #     # NOT answerable — action 2 (Right) was not existed in the tree from state 13
    #     {"target_state": 13, "target_action": 2, "target_path": None, "question_type": "node"},

    #     # answerable — state 14 exists with enough visits
    #     {"target_state": 14, "target_action": None, "target_path": None, "question_type": "node"},

    #     # NOT answerable — state 0 doesn't exist in this tree
    #     {"target_state": 0, "target_action": None, "target_path": None, "question_type": "node"},

    #     # always answerable - general questions don't require specific nodes
    #     {"target_state": None, "target_action": None, "target_path": None, "question_type": "general"},

    #     # NOT answerable — state 0 is not in this tree (agent already past it)
    #     {"target_state": 0, "target_action": None, "target_path": None, "question_type": "node"},

    #     # answerable — action 0 (Left) then action 1 (Down) from state 13, both exist in tree
    #     {"target_state": 13, "target_action": None, "target_path": [0, 1], "question_type": "path"},

    #     # NOT answerable — action 2 (Right) is missing from state 13, path is blocked at step 0
    #     {"target_state": 13, "target_action": None, "target_path": [2, 1], "question_type": "path"},

    #     # NOT answerable - action 2 (Right) is missing
    #     {"target_state": None, "target_action": None, "target_path": [0, 2], "question_type": "path"},
    # ]
    test_cases = [
        {'target_state': 13, 'target_action': 2, 'target_path': None, 'question_type': 'node'}, 
    ]

    for i, intent in enumerate(test_cases):
        result = check_gap(tree, intent)
        print(f"Test {i+1}: {intent}")
        print(f"  answerable : {result['answerable']}")
        print(f"  reason     : {result['reason']}")
        print()
