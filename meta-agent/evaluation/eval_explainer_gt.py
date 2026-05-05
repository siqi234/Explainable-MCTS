"""
Script 2 — Build a pure ground-truth JSON from the MCTS trees.
No LLM involved. Every number comes directly from the tree files.

For each question in eval_queries_v2.json this script records:
  - The state and action the user asked about (from expected_intent)
  - The action the tree actually chose at that state (highest avg value)
  - Full statistics (visits, value, avg_value, success, failure, risk_rate)
    for every available action at the target state
  - For path questions: per-step stats along the requested action sequence

Output: evaluation/eval_explainer_gt.json
"""
import os
import json

EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = os.path.join(EVAL_DIR, "eval_queries_v2.json")
GT_FILE      = os.path.join(EVAL_DIR, "eval_explainer_gt.json")

ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}


# ---------------------------------------------------------------------------
# Tree helpers
# ---------------------------------------------------------------------------

def find_node(tree: dict, target_state: int) -> dict | None:
    if tree.get("type") == "DecisionNode" and tree.get("state") == target_state:
        return tree
    children = tree.get("children", {})
    if isinstance(children, str):
        return None
    for child in children.values():
        result = find_node(child, target_state)
        if result is not None:
            return result
    return None


def chance_node_stats(chance_node: dict, action_code: int) -> dict:
    visits  = chance_node.get("visits") or 0
    value   = chance_node.get("value")
    success = chance_node.get("success")
    failure = chance_node.get("failure")

    avg_value = round(value / visits, 4) if visits and value is not None else None
    # risk_rate = hole-falls per simulation, expressed as a percentage
    risk_rate = round(failure / visits * 100, 2) if visits and failure is not None else None

    return {
        "code":      action_code,
        "name":      ACTION_NAMES.get(action_code, str(action_code)),
        "visits":    visits,
        "value":     value,
        "avg_value": avg_value,
        "success":   success,
        "failure":   failure,
        "risk_rate": risk_rate,
    }


def all_action_stats(decision_node: dict) -> list[dict]:
    children = decision_node.get("children", {})
    if isinstance(children, str):
        return []
    stats = [
        chance_node_stats(cn, int(code))
        for code, cn in children.items()
    ]
    # Sort by avg_value descending so index 0 is the chosen action.
    stats.sort(
        key=lambda x: x["avg_value"] if x["avg_value"] is not None else float("-inf"),
        reverse=True,
    )
    return stats


def chosen_action(decision_node: dict) -> dict | None:
    actions = all_action_stats(decision_node)
    return actions[0] if actions else None


def trace_path(start_node: dict, path: list[int]) -> list[dict]:
    """Walk `path` (list of action codes) through the tree from `start_node`.
    Returns per-step stats; stops early if a branch is missing."""
    steps = []
    current = start_node

    for i, action_code in enumerate(path):
        children = current.get("children", {})
        if isinstance(children, str) or str(action_code) not in children:
            steps.append({
                "step":   i,
                "action": action_code,
                "name":   ACTION_NAMES.get(action_code, str(action_code)),
                "error":  "branch not in tree",
            })
            break

        cn = children[str(action_code)]
        step_stats = chance_node_stats(cn, action_code)
        step_stats["step"] = i
        steps.append(step_stats)

        # Advance to the most-visited next DecisionNode (best estimate of next state).
        if i < len(path) - 1:
            next_states = cn.get("children", {})
            if isinstance(next_states, str) or not next_states:
                break
            current = max(next_states.values(), key=lambda n: n.get("visits", 0))

    return steps


# ---------------------------------------------------------------------------
# Per-question ground truth builder
# ---------------------------------------------------------------------------

def build_gt(entry: dict, tree: dict) -> dict:
    intent        = entry["expected_intent"]
    target_state  = intent.get("target_state")
    target_action = intent.get("target_action")
    target_path   = intent.get("target_path")
    question_type = entry["question_type"]
    root_state    = tree.get("state")

    # General questions have no target state in the intent — use the tree root.
    resolved_state = target_state if target_state is not None else root_state

    gt = {
        "id":                  entry["id"],
        "question":            entry["question"],
        "question_type":       question_type,
        "question_form":       entry["question_form"],
        "tree_file":           entry["tree_file"],
        "tree_root_state":     root_state,
        "target_state":        resolved_state,
        "target_action_asked": target_action,
        "target_path_asked":   target_path,
    }

    node = find_node(tree, resolved_state)
    if node is None:
        gt["error"] = f"State {resolved_state} not found in tree"
        return gt

    gt["state_visits"] = node.get("visits")

    all_actions = all_action_stats(node)
    gt["all_actions"] = all_actions
    gt["chosen_action"] = all_actions[0] if all_actions else None

    # Stats for the action the user explicitly asked about (if any).
    if target_action is not None:
        children = node.get("children", {})
        if not isinstance(children, str) and str(target_action) in children:
            gt["asked_action_stats"] = chance_node_stats(
                children[str(target_action)], target_action
            )
        else:
            gt["asked_action_stats"] = None  # action not present in this tree
    else:
        # No specific action asked — the ground truth is the chosen action.
        gt["asked_action_stats"] = gt["chosen_action"]

    # For path questions, trace the requested sequence.
    if question_type == "path" and target_path:
        gt["path_trace"] = trace_path(node, target_path)

    return gt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    results = []
    for entry in queries:
        tree_path = os.path.join(EVAL_DIR, entry["tree_file"])
        with open(tree_path, "r") as f:
            tree = json.load(f)

        gt = build_gt(entry, tree)
        results.append(gt)

        chosen_name = (gt.get("chosen_action") or {}).get("name", "N/A")
        asked_code  = gt.get("target_action_asked")
        print(
            f"[{entry['id']}]  state={gt['target_state']}  "
            f"asked_action={asked_code}  chosen={chosen_name}"
        )

    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nGround truth written to: {GT_FILE}")


if __name__ == "__main__":
    main()
