import os
import openai
import json
from openai import OpenAI
from dotenv import load_dotenv

import json

def generate_answer(client, mcts_json_data, user_question: str) -> str:
    """
    Calls the LLM with the MCTS tree and the user's actual question.
    Returns the response string.
    """
    messages = prompt(mcts_json_data, user_question)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        # model = "llama-4-scout-17b-16e-instruct",
        messages=messages,
        temperature=0.3,
    )
    return response.choices[0].message.content


def prompt(mcts_json_data, user_question: str = "Why did the agent choose this action over the others? What if the agent chose to move in a different direction?"):
    """
    Constructs the prompt payload for the LLM API to generate an XAI report.
    """
    
    system_prompt = """You are an expert Explainable AI assistant. Your task is to translate the raw MCTS tree from a Frozen Lake game into intuitive, natural language for non-technical users. Your audience is non-technical users who do not understand probability or game mechanics.
                        You must always explain key environmental factors when they are relevant to the agent's decision — do not assume the user already knows them.

ENVIRONMENTAL PHYSICS/SLIPPERINESS:
    The agent is on a 4x4 Frozen Lake. The ice is slippery.
    When the agent attempts to move in a direction, there is only a 90% chance it moves in that intended direction.
    There is a 10% chance it slides uncontrollably to the perpendicular sides. It cannot slide backwards.
    Moving adjacent to danger zones (states 5, 7, 11, 12) carries a massive risk of sliding in.

REWARD FORMULATION:
    - Reaching the Goal (state 15) = +1.0
    - Falling into a danger zone (states 5, 7, 11, 12) = -0.5

DATA DICTIONARY:
    - Action: The direction the agent attempted to move (0=Left, 1=Down, 2=Right, 3=Up).
    - State: The current position of the agent on the 4x4 grid (0-15).
    - Value: The cumulative total return summed across all simulations for this action. It is NOT an average.
    - Average Value: Value / Visits — the true per-simulation expected return. This is what the agent uses to compare actions.
    - Visits: Total number of simulations run through this action.
    - Success: Number of times the tree walk reached the Goal (state 15). Simulation rollouts are NOT counted.
    - Failure: Number of times the tree walk landed on a danger zone (states 5, 7, 11, 12). Simulation rollouts are NOT counted.
    - Children: The child nodes represent possible future states after the action is taken.

    - Action Data Block (Chance Node): Represents the agent's intended move. All statistics (Value, Visits, Success, Failure) aggregate outcomes across all possible slip outcomes, including the 2/3 chance the agent slides off course.
    - Decision Rule: At decision time the agent selects the action with the highest Average Value (Value / Visits). This is pure exploitation — no exploration bonus. A higher (less negative) average value means better expected outcomes.
    - UCB Formula (used during tree search to balance exploration vs exploitation, NOT at decision time):
      UCB = (Value / Visits) + √2 × sqrt(ln(parent_visits) / visits)
      The exploration term √2 × sqrt(ln(parent_visits) / visits) encourages the agent to try less-visited actions during search. At decision time this term is dropped and only Average Value matters."""

#     user_prompt = f"""Analyze the following MCTS JSON data and answer the user's question with a brief decision report within 400 words.

# USER QUESTION:
# {user_question}

# INPUT JSON DATA:
# {json.dumps(mcts_json_data, indent=2)}

# REQUIRED INTUITIVE METRICS (always compute and include these):
#     1. Risk Rate: (failure / visits) × 100, expressed as a percentage. Compute this for every action discussed.
#     2. Safety Score: Translate Average Value (Value / Visits) into a categorical label (e.g., "High Risk", "Moderate", "Safe", "Optimal") based on your judgment of the scale.
#     3. Time Efficiency: If an action has 0 failures but a low or negative Average Value, explain it as a time-wasting penalty (-0.01/step) rather than a fatal risk.

# REPORT STRUCTURE (under 200 words, clear non-technical style, directly addressing the user question above):
#     - Core Decision: What action was chosen at what state and why (use Average Value to justify).
#     - Risk Avoidance: Why seemingly closer or alternative actions were rejected (reference Risk Rate and the 2/3 slipperiness).
#     - Data Justification: Back up the explanation with the computed Intuitive Metrics.

# """

    user_prompt = f"""Answer the user's question about the MCTS agent within 200 words, in clear non-technical language.

USER QUESTION:
{user_question}

INPUT JSON DATA:
{json.dumps(mcts_json_data, indent=2)}

REQUIRED INTUITIVE METRICS (always compute and include these):
    1. Risk Rate: (failure / visits) × 100, expressed as a percentage. Compute this for every action discussed.
    2. Safety Score: Translate Average Value (Value / Visits) into a categorical label (e.g., "High Risk", "Moderate", "Safe", "Optimal") based on your judgment of the scale.

RESPONSE INSTRUCTIONS:
    - If the question is a simple factual question about the environment (e.g. slipperiness, rules, grid layout), answer it directly and concisely. Do not use the report structure.
    - If the question is about the agent's decision or behaviour, use this structure:
        - Core Decision: What action was chosen at what state and why (use Average Value to justify).
        - Risk Avoidance: Why other actions were rejected (reference risk rate and the 10% slip probability explicitly).
        - Data Justification: Support with Risk Rate (failure/visits × 100) and Safety Score for each action discussed.

"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), 
                base_url="https://api.groq.com/openai/v1")
    
    json_file = 'meta-agent/evaluation/trees/mcts_tree_step_20_fail.json'
    # json_file = 'mcts_trees_ver2/mcts_tree_step_13.json'
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    user_question = "What path would the agent follow if Down were explored at the current state?"
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        # model="openai/gpt-oss-120b",
        messages=prompt(json_data, user_question),
        temperature=0.3, # Keep temperature low for factual, analytical responses
        # max_tokens=250, # Limit response length to ensure conciseness
    )
    
    print(response.choices[0].message.content)

    # save response to file
    with open('meta-agent/evaluation/case1_Q14_success.txt', 'w') as f:
        f.write(response.choices[0].message.content)
