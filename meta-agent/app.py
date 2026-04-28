import streamlit as st
import gymnasium as gym
import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))

from mcts_from_scratch_ver2 import MCTS
import LLM
import extractor
import detector
import expander
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

# ── Constants ────────────────────────────────────────────────────────────────
ACTIONS = {0: "← Left", 1: "↓ Down", 2: "→ Right", 3: "↑ Up"}
TREE_FOLDER = os.path.join(os.path.dirname(__file__), "mcts_trees_live")
EXPANDED_FILE = os.path.join(os.path.dirname(__file__), "mcts_tree_expanded.json")
SURVEY_FILE = os.path.join(os.path.dirname(__file__), "survey_responses.json")

os.makedirs(TREE_FOLDER, exist_ok=True)


def save_survey(mcts_level: str, fl_level: str):
    import datetime
    record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "mcts_familiarity": mcts_level,
        "frozen_lake_familiarity": fl_level,
    }
    if os.path.exists(SURVEY_FILE):
        with open(SURVEY_FILE) as f:
            data = json.load(f)
    else:
        data = []
    data.append(record)
    with open(SURVEY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def make_client():
    return OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1",
    )


def init_game():
    sim_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True).unwrapped
    sim_env.reset(seed=42)
    vis_env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array")
    obs, _ = vis_env.reset(seed=42)
    st.session_state.sim_env = sim_env
    st.session_state.vis_env = vis_env
    st.session_state.mcts = MCTS(sim_env, iterations=1000)
    st.session_state.obs = int(obs)
    st.session_state.frame = vis_env.render()
    st.session_state.step = 0
    st.session_state.done = False
    st.session_state.truncated = False
    st.session_state.current_tree = None
    st.session_state.current_tree_path = None
    st.session_state.last_action = None
    st.session_state.last_reward = None
    st.session_state.qa_history = []
    st.session_state.game_result = None


FAMILIARITY = [
    "Not at all familiar",
    "Slightly familiar",
    "Moderately familiar",
    "Very familiar",
    "Extremely familiar",
]

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="FrozenLake MCTS", layout="wide")

# ── Survey overlay (shown after Start New Game if not yet completed) ──────────
if st.session_state.get("showing_survey"):
    st.title("Before You Start")
    st.markdown("Please answer two quick questions so we can understand your background.")
    st.markdown("---")

    st.markdown("""
    <style>
    div[data-testid="stRadio"] label { font-size: 1.15rem !important; }
    div[data-testid="stRadio"] > label { font-size: 1.25rem !important; font-weight: 600; }
    </style>
    """, unsafe_allow_html=True)

    mcts_level = st.radio(
        "How familiar are you with **Monte Carlo Tree Search (MCTS)**?",
        FAMILIARITY,
        index=None,
    )

    st.markdown("<br>", unsafe_allow_html=True)

    fl_level = st.radio(
        "How familiar are you with the **Frozen Lake** environment?",
        FAMILIARITY,
        index=None,
    )

    if st.button("Continue to Game", disabled=(mcts_level is None or fl_level is None)):
        save_survey(mcts_level, fl_level)
        st.session_state.survey_done = True
        st.session_state.survey_mcts = mcts_level
        st.session_state.survey_frozen_lake = fl_level
        st.session_state.showing_survey = False
        init_game()
        st.rerun()

    st.stop()

# ── Game UI ───────────────────────────────────────────────────────────────────
st.title("🧊 FrozenLake — MCTS Agent")
st.caption("Take each step manually, then ask the agent anything about its decision.")

left, right = st.columns([1, 1])

# ── Left panel: grid + controls ──────────────────────────────────────────────
with left:
    st.subheader("Game Board")

    game_ready = "obs" in st.session_state

    if not game_ready:
        st.info("Press **Start New Game** to begin.")
    else:
        st.image(st.session_state.frame, width=320)

        cols = st.columns(3)
        cols[0].metric("Step", st.session_state.step)
        if st.session_state.last_action is not None:
            cols[1].metric("Last Action", ACTIONS[st.session_state.last_action])
        if st.session_state.last_reward is not None:
            cols[2].metric("Reward", st.session_state.last_reward)

        result = st.session_state.game_result
        if result == "win":
            st.success("🎉 Goal reached!")
        elif result == "hole":
            st.error("💀 Fell into a hole.")
        elif result == "truncated":
            st.warning("⏱️ Episode truncated.")

    st.markdown("---")
    btn_cols = st.columns(2)

    if btn_cols[0].button("🔄 Start New Game", use_container_width=True):
        if not st.session_state.get("survey_done"):
            st.session_state.showing_survey = True
            st.rerun()
        else:
            init_game()
            st.rerun()

    if game_ready and not (st.session_state.done or st.session_state.truncated):
        if btn_cols[1].button("▶️ Next Step", use_container_width=True):
            with st.spinner("MCTS thinking..."):
                mcts: MCTS = st.session_state.mcts
                obs = st.session_state.obs
                step = st.session_state.step

                action = mcts.search(obs)

                tree_data = mcts.root.to_dict(current_depth=0, max_depth=4)
                tree_path = os.path.join(TREE_FOLDER, f"mcts_tree_step_{step}.json")
                with open(tree_path, "w") as f:
                    json.dump(tree_data, f, indent=2)

                sim_env = st.session_state.sim_env
                vis_env = st.session_state.vis_env
                sim_env.s = obs
                next_obs, reward, done, truncated, _ = sim_env.step(action)
                vis_env.unwrapped.s = int(next_obs)
                vis_env.unwrapped.lastaction = action

                st.session_state.current_tree = tree_data
                st.session_state.current_tree_path = tree_path
                st.session_state.last_action = action
                st.session_state.last_reward = reward
                st.session_state.frame = vis_env.render()
                st.session_state.obs = int(next_obs)
                st.session_state.done = bool(done)
                st.session_state.truncated = bool(truncated)
                st.session_state.step = step + 1
                st.session_state.qa_history = []

                if done:
                    st.session_state.game_result = "win" if reward == 1 else "hole"
                elif truncated:
                    st.session_state.game_result = "truncated"

            st.rerun()


# ── Right panel: Q&A only ────────────────────────────────────────────────────
with right:
    st.subheader("💬 Ask the Agent")

    if not game_ready or st.session_state.current_tree is None:
        st.info("Take the first step, then ask the agent anything about its decision.")
    else:
        if st.session_state.last_action is not None:
            st.caption(f"Current step: agent chose **{ACTIONS[st.session_state.last_action]}** (action {st.session_state.last_action})")

        with st.form("qa_form", clear_on_submit=True):
            user_q = st.text_input("Your question", placeholder="e.g. Why didn't the agent go right?")
            submitted = st.form_submit_button("Ask")

        if submitted and user_q.strip():
            with st.status("Processing your question...", expanded=True) as status:
                client = make_client()
                tree = st.session_state.current_tree
                tree_path = st.session_state.current_tree_path

                st.write("Extracting intent...")
                intent = extractor.extract_intent(client, user_q, tree)

                st.write("Checking if the tree can answer this...")
                gap = detector.check_gap(tree, intent)

                if not gap["answerable"]:
                    target_state = intent.get("target_state")
                    target_action = intent.get("target_action")

                    if target_state is not None and target_action is not None:
                        st.write(f"Gap detected — expanding tree at state {target_state}, action {target_action} ({ACTIONS.get(target_action, target_action)})...")
                        expander.expand_and_graft(
                            tree_file=tree_path,
                            target_state=target_state,
                            target_action=target_action,
                            output_file=tree_path,
                            iterations=1000,
                        )
                        with open(tree_path) as f:
                            tree = json.load(f)
                        st.session_state.current_tree = tree
                        gap = detector.check_gap(tree, intent)

                if gap["answerable"]:
                    st.write("Generating answer...")
                    chosen = st.session_state.last_action
                    augmented_q = f"[FACT: The agent chose action {chosen} ({ACTIONS[chosen]}) at this step.]\n\n{user_q}"
                    answer = LLM.generate_answer(client, tree, augmented_q)
                    status.update(label="Done", state="complete", expanded=False)
                else:
                    answer = f"⚠️ Not enough data to answer that. ({gap['reason']})"
                    status.update(label="Could not answer", state="error", expanded=False)

            st.session_state.qa_history.append({"q": user_q, "a": answer})

        for item in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {item['q']}")
            st.markdown(item["a"])
            st.markdown("---")
