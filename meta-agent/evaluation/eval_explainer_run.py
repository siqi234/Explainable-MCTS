"""
Script 1 — Run the explainer LLM on every dataset question and save each answer
to a separate file under evaluation/answers/<QID>.txt.

Resumable: already-completed question IDs are tracked in eval_explainer_answers.json.
Re-running the script skips questions whose answer file already exists on disk.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
import LLM

EVAL_DIR    = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = os.path.join(EVAL_DIR, "eval_queries_v2.json")
ANSWERS_DIR  = os.path.join(EVAL_DIR, "answers")
SUMMARY_FILE = os.path.join(EVAL_DIR, "eval_explainer_answers.json")


def load_tree(tree_file: str) -> dict:
    with open(os.path.join(EVAL_DIR, tree_file), "r") as f:
        return json.load(f)


def answer_path(qid: str) -> str:
    return os.path.join(ANSWERS_DIR, f"{qid}.txt")


def run(client) -> None:
    os.makedirs(ANSWERS_DIR, exist_ok=True)

    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    # Load existing summary to know what's already done.
    if os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
            summary = json.load(f)
        # Also guard against a corrupt summary: answer file must actually exist.
        summary = [r for r in summary if os.path.exists(answer_path(r["id"]))]
        done_ids = {r["id"] for r in summary}
        print(f"Resuming — {len(done_ids)} already done: {sorted(done_ids)}")
    else:
        summary = []
        done_ids = set()

    for entry in queries:
        qid = entry["id"]
        if qid in done_ids:
            print(f"[{qid}] Skipped (already done)")
            continue

        print(f"[{qid}] Calling LLM…", flush=True)
        tree_data = load_tree(entry["tree_file"])
        answer = LLM.generate_answer(client, tree_data, entry["question"])

        # Write answer to its own file first — crash-safe.
        with open(answer_path(qid), "w", encoding="utf-8") as f:
            f.write(answer)

        summary.append({
            "id": qid,
            "question": entry["question"],
            "question_type": entry["question_type"],
            "question_form": entry["question_form"],
            "tree_file": entry["tree_file"],
            "answer_file": f"answers/{qid}.txt",
        })

        # Persist summary after every question so a mid-run crash loses nothing.
        with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[{qid}] Done — answers/{qid}.txt")

    print(f"\nAll done. {len(summary)}/{len(queries)} questions answered.")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    run(client)
