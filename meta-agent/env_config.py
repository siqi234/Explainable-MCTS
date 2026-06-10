# env_config.py
'''
Environment configuration for the LLM agent.

This file contains the environment configuration, including:
    - Environment name: Name of the environment (e.g., "Frozen Lake 4x4").
    - Domain: The type of problem or task the environment represents (e.g., "stochastic grid-world navigation").
    - State description: A clear, intuitive explanation of what states represent in this environment.
    - Action description: A clear, intuitive explanation of what actions represent in this environment.
    - Action mapping: A mapping from action indices to human-readable descriptions (e.g., 0=Left, 1=Down, etc.).
    - State mapping: A mapping from state indices to human-readable descriptions, including any special 
        states (e.g., goal states, danger states).
    - Transition description: An explanation of how actions lead to state transitions, especially if the 
        environment is stochastic. Emphasize that child nodes in the MCTS tree represent explored possible outcomes, 
        not guaranteed paths.
    - Reward description: An explanation of what higher or lower returns mean in the context of this environment 
        (e.g., higher return means better outcomes, lower return means worse outcomes).
    - Important outcomes: A clear definition of what constitutes success and failure in this environment, and 
        any other important outcomes to consider.
    - Domain notes: Any specific instructions or considerations for interpreting the MCTS tree in this 
        environment (e.g., how to use the tree as evidence, how to handle stochastic outcomes, etc.).

For future use, users should be able to add new environment configurations by creating a new dictionary 
with the same structure and adding it to the ENV_CONFIGS dictionary. The get_env_config function can be 
used to retrieve a copy of the desired environment configuration, and the validate_env_config function can 
be used to ensure that any custom configurations contain all the required fields.

'''

from __future__ import annotations

from copy import deepcopy
from typing import Any


EnvironmentConfig = dict[str, Any]


DEFAULT_ENV_CONFIG: EnvironmentConfig = {
    "environment_name": "Generic MCTS Environment",
    "domain": "general decision-making",

    "state_description": (
        "A state represents the current situation, position, or condition of the agent."
    ),

    "action_description": (
        "An action represents a decision, intended move, or control choice available to the agent."
    ),

    "action_mapping": {},

    "state_mapping": {},

    "transition_description": (
        "The environment may be deterministic or stochastic. If it is stochastic, "
        "the selected action should be interpreted as the agent's intended decision, "
        "while the realized next state may vary. Child nodes in the MCTS tree should "
        "be treated as explored possible outcomes."
    ),

    "reward_description": (
        "Higher return means a better outcome for the agent. Lower return means a worse, "
        "riskier, less efficient, or more costly outcome."
    ),

    "important_outcomes": {
        "success_meaning": "The agent reached a desirable outcome.",
        "failure_meaning": "The agent reached an undesirable or unsafe outcome.",
    },

    "domain_notes": [
        "Use the MCTS tree as evidence.",
        "Do not invent paths, outcomes, or reasons that are not supported by the tree.",
        "If the tree only provides partial evidence, say so clearly.",
    ],
}


FROZEN_LAKE_4X4_CONFIG: EnvironmentConfig = {
    "environment_name": "Frozen Lake 4x4",
    "domain": "stochastic grid-world navigation",

    "state_description": (
        "States are grid cells in a 4x4 map, numbered from 0 to 15. "
        "The agent tries to reach the goal while avoiding danger states."
    ),

    "action_description": (
        "Actions are intended movement directions. Because the environment is slippery "
        "and stochastic, the selected action is not guaranteed to be the actual realized movement."
    ),

        "action_mapping": {
        "0": "Left",
        "1": "Down",
        "2": "Right",
        "3": "Up",
    },

    "stochastic_transition_model": {
    "is_stochastic": True,
    "movement_type": "slippery",
    "transition_probabilities": {
        "intended_direction": 0.90,
        "unintended_slip": 0.10,
    },
    "intended_action_meaning": (
        "The selected action is the direction the agent intends to move, "
        "not necessarily the direction it actually moves."
    ),
    "slip_description": (
        "The agent moves in the intended direction with 90% probability. "
        "With 10% probability, it slips to an unintended possible direction."
    ),
    },

    "state_mapping": {
        "goal_states": [15],
        "danger_states": [5, 7, 11, 12],
    },

        "transition_description": (
        "Frozen Lake is stochastic and slippery. When the agent selects an action, "
        "that action is only the intended direction. According to the transition model, "
        "the agent usually moves as intended, but it may slip to an unintended next state. "
        "Therefore, child nodes under an action represent possible explored next states "
        "after stochastic movement, not one guaranteed deterministic path."
    ),

    "reward_description": (
        "Reaching the goal state is good. Falling into a danger state is bad. "
        "A low or negative average value can also mean the action is inefficient, risky, "
        "or unlikely to reach the goal soon."
    ),

    "important_outcomes": {
        "goal_state": 15,
        "danger_states": [5, 7, 11, 12],
        "success_meaning": "The explored path reached the goal state.",
        "failure_meaning": "The explored path reached a danger state.",
        "stochastic_outcome_meaning": (
            "A child state after an action is one possible realized outcome caused by slipperiness."
        ),
    },

    "domain_notes": [
        "Do not describe an intended action as a guaranteed movement.",
        "For example, choosing Down means the agent intends to move Down, but slipperiness may lead to a different next state.",
        "For path questions, describe possible or likely explored paths rather than guaranteed paths.",
        "Use child nodes to explain stochastic outcomes.",
        "Use risk rate when danger states are relevant.",
        "Use average value when explaining why an action was selected.",
    ],
}


ENV_CONFIGS: dict[str, EnvironmentConfig] = {
    "default": DEFAULT_ENV_CONFIG,
    "frozen_lake_4x4": FROZEN_LAKE_4X4_CONFIG,
}


def get_env_config(name: str = "frozen_lake_4x4") -> EnvironmentConfig:
    """
    Return a copy of the requested environment config.
    Using deepcopy prevents accidental modification of the global config.
    """
    if name not in ENV_CONFIGS:
        available = ", ".join(ENV_CONFIGS.keys())
        raise ValueError(
            f"Unknown environment config: {name}. Available configs: {available}"
        )

    return deepcopy(ENV_CONFIGS[name])


def validate_env_config(config: EnvironmentConfig) -> None:
    """
    Basic validation to make sure the config contains the fields expected by the prompt.
    """
    required_fields = [
        "environment_name",
        "domain",
        "state_description",
        "action_description",
        "action_mapping",
        "state_mapping",
        "transition_description",
        "reward_description",
        "important_outcomes",
        "domain_notes",
    ]

    missing = [field for field in required_fields if field not in config]

    if missing:
        raise ValueError(
            "Environment config is missing required fields: "
            + ", ".join(missing)
        )
