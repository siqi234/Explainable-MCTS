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
    
    system_prompt = """You are an expert Explainable AI assistant. Your task is to translate the raw MCTS tree from a Frozen Lake game into intuitive, natural language for non-technical users.

ENVIRONMENTAL PHYSICS/SLIPPERINESS:
    The agent is on a 4x4 Frozen Lake. The ice is slippery.
    When the agent attempts to move in a direction, there is only a 1/3 chance it moves in that intended direction.
    There is a 2/3 chance it slides uncontrollably to the perpendicular sides. It cannot slide backwards.
    Moving adjacent to holes (indices 5, 7, 11, 12) carries a massive risk of sliding in.

REWARD FORMULATION:
    - Reaching the Goal (15) = +1.0
    - Falling into a Hole (5, 7, 11, 12) = -0.5
    - Time Penalty = -0.01 per step (penalizes infinite wandering)

DATA DICTIONARY:
    - Action: The direction the agent attempted to move (0=Left, 1=Down, 2=Right, 3=Up).
    - State: The current position of the agent on the 4x4 grid (0-15).
    - Value: The expected return (average score) of an action across all simulations.
    - Visits: Total mental simulations for this action.
    - Successes / Holes: count of times the tree walk landed on Goal (15) or a Hole (5, 7, 11, 12), simulation rollouts are NOT counted.
    - Children: The child nodes represent possible future states.

    - Action Data Block (saved as Chance Node): This represents the agent's intended move. The statistics inside this block (Value, Holes, Visits) calculate the total average of all possible outcomes, including the 2/3 chance the agent slides off course.
    - Decision Rule: The agent selects the action with the highest average score (Value / Visits). This is pure exploitation (no exploration bonus) at decision time. A higher (less negative) average value means that the action led to better outcomes across all simulations.
    - UCB Formula (used during tree search phase, NOT at decision time): UCB = (Value / Visits) + √2 * sqrt(ln(parent_visits) / visits), where √2 ≈ 1.4142 is the exploration constant.
    - Successes / Holes: count of times the tree walk itself landed on the Goal (state 15) or a Hole state (5, 7, 11, 12). Simulation rollouts are NOT counted."""

    user_prompt = f"""Analyze the following single-step MCTS JSON data and generate a brief decision report. You need to give intuitive metrics for users to understand the decision:

REQUIRED INTUITIVE METRICS:
    1. Risk Rate: Calculate (holes / visits * 100) as a percentage.
    2. Safety Score: Translate the raw 'Value' into a categorical label (e.g., "High Risk", "Safe", "Optimal").
    3. Time Efficiency: If an action has 0 holes but a low or negative Value, explain it as a time-wasting penalty (-0.01/step) rather than a fatal risk.

REPORT STRUCTURE (Under 200 words total in a clear, non-technical style):
    - Core Decision: What action was chosen and why.
    - Risk Avoidance: Why seemingly closer but riskier actions were rejected (reference the 2/3 slipperiness).
    - Data Justification: Use the Intuitive Metrics calculated above to back up the decision.

INPUT JSON DATA:
{json.dumps(mcts_json_data, indent=2)}

USER QUESTION:
{user_question}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), 
                base_url="https://api.groq.com/openai/v1")
    
    json_file = 'mcts_trees_ver2/mcts_tree_step_1.json'
    # json_file = 'mcts_trees_ver2/mcts_tree_step_13.json'
    with open(json_file, 'r') as f:
        json_data = json.load(f)

    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile", 
        # model="openai/gpt-oss-120b",
        messages=prompt(json_data),
        temperature=0.3, # Keep temperature low for factual, analytical responses
        # max_tokens=250, # Limit response length to ensure conciseness
    )
    
    print(response.choices[0].message.content)