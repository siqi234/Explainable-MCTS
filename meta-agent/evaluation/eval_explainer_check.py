"""
Script 3 — Check LLM answer reliability against ground truth.

For each question, three keyword checks are run on the LLM's answer:
  1. agent_action  — does the answer name the correct chosen action?
  2. risk_rate     — does the answer contain a number within ±RISK_TOLERANCE of
                     the ground-truth risk rate for the asked action?
  3. user_action   — does the answer name the action the user asked about?
                     (skipped when user_action == agent_action, or not specified)

All checks are case-insensitive. Numbers are extracted with a simple regex so
"21%", "20.77", "roughly 20" all match a ground-truth of 20.77 within tolerance.

Input:
  evaluation/eval_explainer_gt.json     (from eval_explainer_gt.py)
  evaluation/eval_explainer_answers.json + evaluation/answers/<QID>.txt
        (from eval_explainer_run.py)

Output:
  evaluation/eval_explainer_check.json
"""
import os
import json
import re

EVAL_DIR      = os.path.dirname(os.path.abspath(__file__))
GT_FILE       = os.path.join(EVAL_DIR, "eval_explainer_gt.json")
SUMMARY_FILE  = os.path.join(EVAL_DIR, "eval_explainer_answers.json")
ANSWERS_DIR   = os.path.join(EVAL_DIR, "answers")
OUTPUT_FILE   = os.path.join(EVAL_DIR, "eval_explainer_check.json")

ACTION_NAMES = {0: "Left", 1: "Down", 2: "Right", 3: "Up"}
RISK_TOLERANCE = 2.0   # percentage points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def extract_risk_numbers(text: str) -> list[float]:
    """Return numbers that appear next to % or 'percent' — avoids false positives
    from state indices, visit counts, and other unrelated integers in the answer."""
    pct_pattern  = re.findall(r"(\d+(?:\.\d+)?)\s*%", text)
    word_pattern = re.findall(r"(\d+(?:\.\d+)?)\s+percent", text, re.IGNORECASE)
    return [float(m) for m in pct_pattern + word_pattern]


def mentions_action(text: str, action_code: int) -> bool:
    name = ACTION_NAMES.get(action_code, "")
    if not name:
        return False
    # Use word boundary so "down" doesn't match "download", "up" doesn't match "setup", etc.
    return bool(re.search(rf"\b{name}\b", text, re.IGNORECASE))


def mentions_state_action(text: str, state: int, action_code: int) -> bool | None:
    """Check that the LLM understood what the user was asking about:
    both the target state number and the action name must appear in the answer."""
    if state is None or action_code is None:
        return None
    has_state  = bool(re.search(rf"\b{state}\b", text))
    has_action = mentions_action(text, action_code)
    return has_state and has_action


def mentions_risk_rate(text: str, ground_truth_rate: float) -> bool:
    if ground_truth_rate is None:
        return None  # not checkable
    numbers = extract_risk_numbers(text)
    return any(abs(n - ground_truth_rate) <= RISK_TOLERANCE for n in numbers)


def load_answer(qid: str) -> str | None:
    path = os.path.join(ANSWERS_DIR, f"{qid}.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    with open(GT_FILE, "r", encoding="utf-8") as f:
        gt_list = json.load(f)
    gt_by_id = {g["id"]: g for g in gt_list}

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    results  = []
    totals   = {"agent_action": 0, "risk_rate": 0, "user_action": 0, "all_pass": 0}
    # Track denominator separately for user_action since it is N/A for many questions.
    counts   = {"agent_action": 0, "risk_rate": 0, "user_action": 0}
    skipped  = 0
    n        = 0

    for entry in summary:
        qid    = entry["id"]
        answer = load_answer(qid)
        gt     = gt_by_id.get(qid)

        if answer is None or gt is None:
            print(f"[{qid}] SKIPPED — answer or ground truth missing")
            skipped += 1
            continue

        n += 1
        question_form = entry.get("question_form", "")

        chosen = gt.get("chosen_action") or {}
        user_action_code = gt.get("target_action_asked")

        # --- Check 1: agent action ---
        # For contrastive questions the "agent action" is the action being defended
        # in the question (target_action_asked), NOT the tree's absolute best —
        # e.g. "Why Up over Down?" focuses on Up, even if the tree chose Right overall.
        # For all other forms, use the tree's chosen action.
        if question_form == "contrastive" and user_action_code is not None:
            agent_action_code = user_action_code
        else:
            agent_action_code = chosen.get("code")

        check_agent = (
            mentions_action(answer, agent_action_code)
            if agent_action_code is not None else None
        )

        # --- Check 2: risk rate of the action the user asked about ---
        asked_stats  = gt.get("asked_action_stats") or {}
        risk_rate_gt = asked_stats.get("risk_rate")
        check_risk   = mentions_risk_rate(answer, risk_rate_gt)

        # --- Check 3: user-asked state+action pair ---
        # Checks that the LLM correctly identified what the user was asking about:
        # both the target state number AND the action name must appear in the answer.
        # Skip when no specific action was asked (general questions).
        user_state = gt.get("target_state")
        if user_action_code is None:
            check_user = None  # general question, no specific action asked
        else:
            check_user = mentions_state_action(answer, user_state, user_action_code)

        # --- Aggregate ---
        checks = {
            "agent_action": check_agent,
            "risk_rate":    check_risk,
            "user_action":  check_user,
        }
        active_checks = {k: v for k, v in checks.items() if v is not None}
        all_pass = all(active_checks.values()) if active_checks else False

        for k, v in active_checks.items():
            counts[k] += 1
            if v:
                totals[k] += 1
        if all_pass:
            totals["all_pass"] += 1

        status = "PASS" if all_pass else "FAIL"
        print(
            f"[{qid}] {status}  "
            f"agent={_fmt(check_agent)}  "
            f"risk={_fmt(check_risk)}  "
            f"user={_fmt(check_user)}"
        )

        results.append({
            "id":            qid,
            "question":      entry["question"],
            "question_type": entry["question_type"],
            "question_form": question_form,
            "checks":        checks,
            "all_pass":      all_pass,
            "ground_truth": {
                "agent_action_checked": ACTION_NAMES.get(agent_action_code) if agent_action_code is not None else None,
                "chosen_action":        chosen.get("name"),
                "user_state_action":    f"state {user_state}, {ACTION_NAMES.get(user_action_code)}" if user_action_code is not None else None,
                "risk_rate_gt":         risk_rate_gt,
            },
        })

    # --- Summary stats ---
    print(f"\n{'='*50}")
    print(f"Evaluated : {n}  (skipped {skipped})")
    if n:
        for field in ("agent_action", "risk_rate", "user_action"):
            denom = counts[field]
            cnt   = totals[field]
            pct   = cnt / denom * 100 if denom else 0.0
            na    = n - denom
            na_str = f"  ({na} N/A)" if na else ""
            print(f"  {field:<22} {cnt}/{denom}  ({pct:.1f}%){na_str}")
        cnt = totals["all_pass"]
        print(f"  {'all_pass':<22} {cnt}/{n}  ({cnt/n*100:.1f}%)")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nDetailed results: {OUTPUT_FILE}")


def _fmt(v) -> str:
    if v is None:
        return "N/A"
    return "✓" if v else "✗"


if __name__ == "__main__":
    main()
