import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from dotenv import load_dotenv
from extractor import extract_intent

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = os.path.join(EVAL_DIR, "eval_queries_v2.json")

# Compared fields: extractor produces question_type, target_state, target_action, target_path.
# target_path is excluded from comparison — ground truth uses state sequences,
# extractor outputs action codes. alternative_action, alternative_path, target_region
# are also in the ground truth but not extracted, so excluded too.
COMPARED_FIELDS = ("question_type", "target_state", "target_action", "target_path")


def load_tree(tree_file: str) -> dict:
    path = os.path.join(EVAL_DIR, tree_file)
    with open(path, "r") as f:
        return json.load(f)


def compare(predicted: dict, entry: dict) -> dict[str, bool]:
    expected = entry["expected_intent"]
    results = {}

    results["question_type"] = predicted.get("question_type") == entry["question_type"]
    results["target_state"] = predicted.get("target_state") == expected.get("target_state")
    results["target_action"] = predicted.get("target_action") == expected.get("target_action")
    results["target_path"] = predicted.get("target_path") == expected.get("target_path")

    return results


def expected_snapshot(entry: dict) -> dict:
    expected = entry["expected_intent"]
    return {
        "question_type": entry["question_type"],
        "target_state": expected.get("target_state"),
        "target_action": expected.get("target_action"),
        "target_path": expected.get("target_path"),
    }


def refresh_cached_results(results: list[dict], queries: list[dict]) -> list[dict]:
    """Recompute cached comparisons against the current ground truth."""
    query_by_id = {entry["id"]: entry for entry in queries}
    refreshed = []

    for result in results:
        entry = query_by_id.get(result["id"])
        if entry is None:
            refreshed.append(result)
            continue

        comparison = compare(result["predicted"], entry)
        core_pass = all(comparison[field] for field in COMPARED_FIELDS)
        full_pass = all(comparison.values())
        result["question_type"] = entry["question_type"]
        result["expected"] = expected_snapshot(entry)
        result["comparison"] = comparison
        result["core_pass"] = core_pass
        result["full_pass"] = full_pass
        refreshed.append(result)

    return refreshed


def run_eval(client, log_path: str) -> list[dict]:
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    # Resume: load any results already written to disk.
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            results = json.load(f)
        results = refresh_cached_results(results, queries)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        done_ids = {r["id"] for r in results}
        print(f"Resuming — {len(done_ids)} question(s) already done: {sorted(done_ids)}")
    else:
        results = []
        done_ids = set()

    for entry in queries:
        qid = entry["id"]
        if qid in done_ids:
            continue
        tree_data = load_tree(entry["tree_file"])
        predicted = extract_intent(client, entry["question"], tree_data)
        comparison = compare(predicted, entry)

        core_pass = all(comparison[field] for field in COMPARED_FIELDS)
        full_pass = all(comparison.values())

        results.append({
            "id": qid,
            "question": entry["question"],
            "question_type": entry["question_type"],
            "predicted": predicted,
            "expected": expected_snapshot(entry),
            "comparison": comparison,
            "core_pass": core_pass,
            "full_pass": full_pass,
        })

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        status = "PASS" if full_pass else "FAIL"
        print(f"[{qid}] {status}  {entry['question'][:70]}")
        for field, ok in comparison.items():
            if not ok:
                expected_value = entry["question_type"] if field == "question_type" else entry["expected_intent"].get(field)
                print(f"       MISMATCH {field}: predicted={predicted.get(field)!r}  expected={expected_value!r}")

    query_ids = {entry["id"] for entry in queries}
    evaluated_results = [r for r in results if r["id"] in query_ids]
    field_correct = {
        field: sum(1 for r in evaluated_results if r["comparison"].get(field))
        for field in COMPARED_FIELDS
    }
    fully_correct = sum(1 for r in evaluated_results if r["full_pass"])
    core_correct = sum(1 for r in evaluated_results if r["core_pass"])
    n = len(queries)

    print(f"\n{'='*50}")
    print(f"Fully correct (all fields):  {fully_correct}/{n}  ({fully_correct/n*100:.1f}%)")
    print(f"Core correct (compared fields): {core_correct}/{n} ({core_correct/n*100:.1f}%)")
    print(f"\nPer-field accuracy:")
    for field, count in field_correct.items():
        print(f"  {field:<30} {count}/{n}  ({count/n*100:.1f}%)")

    return results


if __name__ == "__main__":
    load_dotenv()
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"),
        base_url="https://api.groq.com/openai/v1"
    )
    log_path = os.path.join(EVAL_DIR, "eval_extractor_results.json")
    results = run_eval(client, log_path)
    print(f"\nDetailed results written to: {log_path}")
