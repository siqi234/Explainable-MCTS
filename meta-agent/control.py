"""
    The controller for the whole meta-agent pipeline. This is where we will orchestrate the calls to the extractor, detector, and expander modules.

    The main flow will be:
        0. Run MCTS as usual, saving the tree at each step (this is done in a separate script for now).

        1. LLM receives a user question (for now, we can hardcode this or read from a 

        2. LLM-extractor parses the question into structured intent {target_state, target_action, target_path, question_type}.

        3. Detector will check based on the obtained intent and the MCTS tree whether the question can be answered with existing data, or if there is a gap.
            3a. Yes, answerable — Pass it back to the LLM to generate the answer.
            3b. No, not answerable — Intent is passed to the Expander.

        4. Expander will identify the target node in the tree; run targeted MCTS from the target to grow the subtree for the missing action/path;
            graft the new subtree into the existing tree; backpropagate the stats; and save the updated tree.

        5. After expansion, loop back to step 3 to check if answerable; if yes, pass the new tree to the LLM to generate the answer.

"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import extractor
import detector
import expander
import LLM
from openai import OpenAI
from dotenv import load_dotenv

TREE_FILE     = os.path.join(os.path.dirname(__file__), "mcts_tree_step_9_modified.json")
EXPANDED_FILE = os.path.join(os.path.dirname(__file__), "mcts_tree_step_9_expanded.json")

if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    # Step 1: user question (hardcoded)
    # user_question = "why did the agent choose this action over the others at the current state?" # general question, should be answerable
    # user_question = "What happens if the agent is at state 13 and goes up?"
    # user_question = "What happens if the agent is at state 13 and goes left?" # not answerable, needs expansion
    # user_question = "What happens if the agent is at state 13 and goes down?"
    user_question = input("Enter your question here: ")

    print(f"Question: {user_question}\n")

    # Step 2: extract intent
    intent = extractor.extract_intent(client, user_question)
    print(f"Intent: {intent}\n")

    # Step 3: load tree and check if answerable
    tree_file = TREE_FILE
    with open(tree_file) as f:
        tree = json.load(f)

    gap = detector.check_gap(tree, intent)
    print(f"Answerable: {gap['answerable']} — {gap['reason']}\n")

    # Step 3b: gap detected — expand and re-check once
    if not gap['answerable']:
        target_state  = intent.get("target_state")
        target_action = intent.get("target_action")

        if target_state is None or target_action is None:
            print("[control] Cannot expand: intent is missing target_state or target_action.")
            sys.exit(1)

        print(f"[control] Expanding action {target_action} at state {target_state}...")
        expander.expand_and_graft(
            tree_file=tree_file,
            target_state=target_state,
            target_action=target_action,
            output_file=EXPANDED_FILE,
            iterations=1000,
        )

        # reload the updated tree and re-check
        with open(EXPANDED_FILE) as f:
            tree = json.load(f)
        tree_file = EXPANDED_FILE

        gap = detector.check_gap(tree, intent)
        print(f"Re-check — Answerable: {gap['answerable']} — {gap['reason']}\n")

        if not gap['answerable']:
            print("[control] Still not answerable after expansion. Aborting.")
            sys.exit(1)

    # Step 3a / 5: answerable — generate answer
    answer = LLM.generate_answer(client, tree, user_question)
    print("=== LLM Answer ===")
    print(answer)