import os
import json
from openai import OpenAI
from dotenv import load_dotenv


def extraction_prompt(user_question: str, mcts_json_data: dict) -> list:
    """
    Asks the LLM to extract structured intent from a user's natural language question.
    Output is a JSON object with target_state, target_action, target_path, and question_type.
    """

    system_prompt = f"""
You are a query parser for a Frozen Lake MCTS explainability system.

ENVIRONMENT:
- The agent moves on a 4x4 grid. States are numbered 0-15 (row by row, top-left to bottom-right).
- Actions: 0=Left, 1=Down, 2=Right, 3=Up
- Holes (terminal failure): 5, 7, 11, 12
- Goal (terminal success): 15

MCTS TREE (JSON):
{json.dumps(mcts_json_data, indent=2)}

The root node of the tree above is the agent's current state.
When the user says "current state", "current position", "here", or "now", look up the "state" field of the root node and use that value.

YOUR TASK:
Extract the user's intent from their question and return ONLY a JSON object — no explanation, no prose.

OUTPUT SCHEMA:
{{
  "target_state": <int or null>,       // the state (grid position) the user is asking about
  "target_action": <int or null>,      // the action (0-3) the user is asking about, if any
  "target_path": <list of ints or null>, // ordered list of actions if user asks about a sequence
  "question_type": <"node" | "path" | "general">
}}

QUESTION TYPE RULES:
- "node"    : user asks about a specific state or action at a state (target_state will be set)
- "path"    : user asks about a sequence of moves (target_path will be set)
- "general" : question is about overall behavior, no specific state or path referenced

EXAMPLES:
Q: "What happens if the agent is at state 9 and goes right?"
-> {{"target_state": 9, "target_action": 2, "target_path": null, "question_type": "node"}}

Q: "What if the agent goes down then right from the start?"
-> {{"target_state": 0, "target_action": null, "target_path": [1, 2], "question_type": "path"}}

Q: "Can we go right then down from current state?"
-> {{"target_state": <root node state>, "target_action": null, "target_path": [2, 1], "question_type": "path"}}

Q: "Why did the agent avoid going left?"
-> {{"target_state": null, "target_action": 0, "target_path": null, "question_type": "general"}}

Q: "What is the safest move overall?"
-> {{"target_state": null, "target_action": null, "target_path": null, "question_type": "general"}}

Return ONLY the JSON. No markdown, no explanation.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    return messages


def extract_intent(client, user_question: str, mcts_json_data: dict) -> dict:
    """
    Sends the user question to the LLM and parses the structured intent response.
    Returns a dict with keys: target_state, target_action, target_path, question_type.
    """
    messages = extraction_prompt(user_question, mcts_json_data)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,  # deterministic — this is parsing, not generation
    )

    raw = response.choices[0].message.content.strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError:
        print(f"[extractor] Failed to parse LLM output as JSON:\n{raw}")
        intent = {
            "target_state": None,
            "target_action": None,
            "target_path": None,
            "question_type": "general"
        }

    return intent


# --- Manual test ---
if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )

    json_path = os.path.join(os.path.dirname(__file__), "mcts_tree_step_9.json")
    with open(json_path, "r") as f:
        mcts_json_data = json.load(f)

    test_questions = [
        "What happens if the agent is at state 9 and goes right?",  # {"target_state": 9, "target_action": 2, "target_path": null, "question_type": "node"}
        "What if the agent goes down then right from the start?",   # {"target_state": 0, "target_action": null, "target_path": [1, 2], "question_type": "path"}
        "Why did the agent avoid going left?",                      # {"target_state": null, "target_action": 0, "target_path": null, "question_type": "general"}
        "What is the safest move overall?",                         # {"target_state": null, "target_action": null, "target_path": null, "question_type": "general"}
        "What happened in state 13?",                               # {"target_state": 13, "target_action": null, "target_path": null, "question_type": "node"}
        "What if the agent moved up from position 14?",             # {"target_state": 14, "target_action": 3, "target_path": null, "question_type": "node"}
        "Can we go right then down from current state?",            # {"target_state": 13, "target_action": null, "target_path": [2, 1], "question_type": "path"}    
    ]

    for q in test_questions:
        print(f"Q: {q}")
        intent = extract_intent(client, q, mcts_json_data)
        print(f"-> {json.dumps(intent, indent=2)}\n")
