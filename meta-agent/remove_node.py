import os
import json

TARGET_NODE_ACTION = '0'  # The action corresponding to the node we want to remove
TARGET_FILE = "./meta-agent/mcts_tree_step_9.json"  # The file containing the MCTS tree
OUTPUT_FILE = TARGET_FILE.replace(".json", "_modified.json")  # Output file for the modified tree

if __name__ == "__main__":
    target_file = TARGET_FILE
    with open(target_file, "r") as f:
        tree = json.load(f)
        # Remove the node at action 2 from root node
        # Also need to retreat the visits, value, success/failure counts, etc. from the root node to keep the tree consistent
        if TARGET_NODE_ACTION in tree['children']:
            # Retreat the visits, value, success/failure counts from the root node to the node being removed
            node_to_remove = tree['children'][TARGET_NODE_ACTION]
            tree['visits'] -= node_to_remove['visits']
            tree['value'] -= node_to_remove['value']
            tree['success'] -= node_to_remove.get('success', 0)
            tree['failure'] -= node_to_remove.get('failure', 0)
            # Now remove the node
            del tree['children'][TARGET_NODE_ACTION]
            print(f"Removed node at action {TARGET_NODE_ACTION} from root node in {target_file}.")
        else:
            print(f"Node at action {TARGET_NODE_ACTION} does not exist in root node of {target_file}. No changes made.")

    # Save the modified tree back to the file
    output_file = OUTPUT_FILE
    with open(output_file, "w") as f:
        json.dump(tree, f, indent=2)
        print(f"Saved modified tree back to {output_file}.")