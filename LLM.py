import os
import openai
import json
from openai import OpenAI
from dotenv import load_dotenv

import json

def prompt(mcts_json_data):
    """
    Constructs the prompt payload for the LLM API to generate an XAI report.
    """
    
    system_prompt =  """
    
    You are an expert Explainable AI assistant. Your task is to translate the raw MCTS tree into intuitive, natural language for non-technical users.
        ENVIRONMENTAL PHYSICS/SLIPPERINESS:
            The agent is on a 4x4 Frozen Lake. The ice is slippery. 
            When the agent attempts to move in a direction, there is only a 1/3 chance it moves in that intended direction. 
            There is a 2/3 chance it slides uncontrollably to the perpendicular sides. It cannot slide backwards.
            Moving adjacent to holes (indices 5, 7, 11, 12) carries a massive risk of sliding in.
        REWARD FORMULATION:
            - Reaching the Goal (15) = +1.0
            - Falling into a Hole (5, 7, 11, 12) = -1.0
            - Time Penalty = -0.01 per step (penalizes infinite wandering)
        DATA DICTIONARY:
            - Action: The direction the agent attempted to move (0=Left, 1=Down, 2=Right, 3=Up).
            - State: The current position of the agent on the 4x4 grid (0-15).
            - Value: The expected return (average score) of an action across all simulations.
            - Visits: Total mental simulations for this action.
            - Successes / Holes: count of simulation outcomes for this action.
            - Children: The child nodes represent possible future states.

            - Action Data Block (saved as Chance Node): This represents the agent's intended move. The statistics inside this block (Value, Holes, Visits) calculate the total average of all possible outcomes, including the 2/3 chance the agent slides off course.
    """

    user_prompt = f"""
    Analyze the following single-step MCTS JSON data and generate a brief decision report. You need to give intuitive metrics for users to understand the decision:

   DATA DICTIONARY & LOGIC RULES (CRITICAL):
        The JSON provided contains statistics from simulations (rollouts). You MUST process the data using the following logic before generating your report:

        1. Immediate Action Risk (Cross-reference the Map): 
        Do NOT use the 'failure' count to judge immediate risk. Instead, look at the starting 'state'. If the current state is NOT physically adjacent to a hole (5, 7, 11, 12), that means the immediate physical risk of taking this action is 0%. 

        2. Environment Harshness (Redefining 'failure'): 
        The 'failure' count reflects the number of times the agent fell into a hole during FUTURE random wandering in the simulation. The high failure rates reflect "long-term exploration difficulty due to the slippery ice," it is NOT a mistake in the current chosen action.

        3. Path Quality (Average Reward): 
        Calculate the 'Average Reward' internally by dividing 'value' by 'visits'. 
        - A score near -1.0 means the path is highly lethal.
        - A score between 0.0 and -0.99 means the path is "Safe but Contextually Difficult." Explain that negative scores here are driven by time penalties (-0.01 per step) and future slipping, making it the least bad option.

        REPORT STRUCTURE (Under 200 words, clear, non-technical style):
        - Core Decision: What action was chosen.
        - Immediate Safety: Explain why the move is safe or not safe right now based on the map (Rule 1).
        - Long-Term Difficulty: Address the high failure numbers in the data, explaining them as environmental harshness rather than action risk (Rule 2 & 3).

        INPUT JSON DATA:
        {json.dumps(mcts_json_data, indent=2)}

        USER QUESTION:
        - Why did the agent choose this action over the others? What if the agent choose to move in a different direction? 
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
    
    # json_file = 'mcts_trees_ver2/mcts_tree_step_0.json'
    json_file = 'mcts_trees_ver2/mcts_tree_step_12.json'
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