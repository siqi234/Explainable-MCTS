import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector import check_gap

EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
QUERIES_FILE = os.path.join(EVAL_DIR, "eval_queries_v2.json")


def load_tree(tree_file: str) -> dict:
    path = os.path.join(EVAL_DIR, tree_file)
    with open(path, "r") as f:
        return json.load(f)


def build_intent(entry: dict) -> dict:
    return {
        "target_state": entry["expected_intent"]["target_state"],
        "target_action": entry["expected_intent"]["target_action"],
        "target_path": entry["expected_intent"]["target_path"],
        "question_type": entry["question_type"],
    }


def evaluate_entry(entry: dict) -> dict:
    tree_data = load_tree(entry["tree_file"])
    intent = build_intent(entry)
    output = check_gap(tree_data, intent)

    predicted = output["answerable"]
    expected = entry["expected_detector_output"]
    passed = predicted == expected

    return {
        "id": entry["id"],
        "question": entry["question"],
        "question_type": entry["question_type"],
        "intent": intent,
        "predicted_answerable": predicted,
        "expected_answerable": expected,
        "detector_reason": output["reason"],
        "pass": passed,
        "skipped": False,
    }


def refresh_cached_results(results: list[dict], queries: list[dict]) -> list[dict]:
    query_by_id = {entry["id"]: entry for entry in queries}
    refreshed = []

    for result in results:
        entry = query_by_id.get(result["id"])
        if entry is None:
            refreshed.append(result)
            continue
        refreshed.append(evaluate_entry(entry))

    return refreshed


def run_eval(log_path: str) -> list[dict]:
    with open(QUERIES_FILE, "r") as f:
        queries = json.load(f)

    # Resume from existing results.
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

        result = evaluate_entry(entry)
        predicted = result["predicted_answerable"]
        expected = result["expected_answerable"]
        passed = result["pass"]

        status = "PASS" if passed else "FAIL"
        print(f"[{qid}] {status}  {entry['question'][:70]}")
        if not passed:
            print(f"       predicted={predicted}  expected={expected}")
            print(f"       reason: {result['detector_reason']}")

        results.append(result)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    query_ids = {entry["id"] for entry in queries}
    evaluated_results = [r for r in results if r["id"] in query_ids]
    correct = sum(1 for r in evaluated_results if r["pass"] is True)
    skipped = sum(1 for r in evaluated_results if r.get("skipped"))

    n = len(queries)
    evaluated = n - skipped
    print(f"\n{'='*50}")
    print(f"Evaluated : {evaluated}/{n}  (skipped {skipped})")
    if evaluated:
        print(f"Correct   : {correct}/{evaluated}  ({correct / evaluated * 100:.1f}%)")

    return results


if __name__ == "__main__":
    log_path = os.path.join(EVAL_DIR, "eval_detector_results.json")
    run_eval(log_path)
    print(f"\nResults written to: {log_path}")
