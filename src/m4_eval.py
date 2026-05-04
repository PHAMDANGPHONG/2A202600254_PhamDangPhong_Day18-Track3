"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation on the given data."""
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
        from datasets import Dataset

        # Build HuggingFace Dataset
        dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Run RAGAS evaluation with 4 standard metrics
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
        )

        # Extract per-question results from DataFrame
        df = result.to_pandas()
        per_question = []
        for _, row in df.iterrows():
            per_question.append(EvalResult(
                question=row.get("question", ""),
                answer=row.get("answer", ""),
                contexts=row.get("contexts", []),
                ground_truth=row.get("ground_truth", ""),
                faithfulness=float(row.get("faithfulness", 0.0)),
                answer_relevancy=float(row.get("answer_relevancy", 0.0)),
                context_precision=float(row.get("context_precision", 0.0)),
                context_recall=float(row.get("context_recall", 0.0)),
            ))

        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation error: {e}")
        print("  → Using fallback simple evaluation...")
        # Fallback: simple text overlap evaluation
        return _fallback_evaluate(questions, answers, contexts, ground_truths)


def _fallback_evaluate(questions: list[str], answers: list[str],
                       contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Fallback evaluation when RAGAS is not available or fails."""
    per_question = []
    total_f, total_ar, total_cp, total_cr = 0.0, 0.0, 0.0, 0.0

    for i in range(len(questions)):
        answer = answers[i] if i < len(answers) else ""
        ctx = contexts[i] if i < len(contexts) else []
        gt = ground_truths[i] if i < len(ground_truths) else ""

        # Simple text overlap metrics
        answer_words = set(answer.lower().split())
        gt_words = set(gt.lower().split())
        ctx_text = " ".join(ctx).lower()
        ctx_words = set(ctx_text.split())

        # Faithfulness: how much of answer is in context
        if answer_words:
            f_score = len(answer_words & ctx_words) / len(answer_words)
        else:
            f_score = 0.0

        # Answer relevancy: overlap between answer and ground truth
        if gt_words:
            ar_score = len(answer_words & gt_words) / len(gt_words)
        else:
            ar_score = 0.0

        # Context precision: how much of context is relevant (overlap with GT)
        if ctx_words:
            cp_score = len(ctx_words & gt_words) / len(ctx_words)
        else:
            cp_score = 0.0

        # Context recall: how much of GT is covered by context
        if gt_words:
            cr_score = len(ctx_words & gt_words) / len(gt_words)
        else:
            cr_score = 0.0

        per_question.append(EvalResult(
            question=questions[i],
            answer=answer,
            contexts=ctx,
            ground_truth=gt,
            faithfulness=round(f_score, 4),
            answer_relevancy=round(ar_score, 4),
            context_precision=round(cp_score, 4),
            context_recall=round(cr_score, 4),
        ))
        total_f += f_score
        total_ar += ar_score
        total_cp += cp_score
        total_cr += cr_score

    n = max(len(questions), 1)
    return {
        "faithfulness": round(total_f / n, 4),
        "answer_relevancy": round(total_ar / n, 4),
        "context_precision": round(total_cp / n, 4),
        "context_recall": round(total_cr / n, 4),
        "per_question": per_question,
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    if not eval_results:
        return []

    # 1. Calculate average score for each question
    scored = []
    for r in eval_results:
        avg_score = (r.faithfulness + r.answer_relevancy + r.context_precision + r.context_recall) / 4
        scored.append((avg_score, r))

    # 2. Sort by avg_score ascending → worst first
    scored.sort(key=lambda x: x[0])

    # 3. Take bottom-N
    failures = []
    for avg_score, r in scored[:bottom_n]:
        # Find worst metric
        metrics = {
            "faithfulness": r.faithfulness,
            "answer_relevancy": r.answer_relevancy,
            "context_precision": r.context_precision,
            "context_recall": r.context_recall,
        }
        worst_metric = min(metrics, key=metrics.get)
        worst_score = metrics[worst_metric]

        # Diagnostic Tree mapping
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis = "LLM hallucinating — answer contains info not in context"
            suggested_fix = "Tighten prompt, lower temperature, add 'only use provided context' instruction"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis = "Missing relevant chunks — retrieval missed key info"
            suggested_fix = "Improve chunking strategy or add BM25 for keyword matching"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis = "Too many irrelevant chunks retrieved"
            suggested_fix = "Add reranking step or use metadata filters to improve precision"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis = "Answer doesn't match question intent"
            suggested_fix = "Improve prompt template, ensure answer addresses the specific question"
        else:
            diagnosis = f"Low score on {worst_metric}"
            suggested_fix = "Review retrieval and generation pipeline"

        failures.append({
            "question": r.question,
            "avg_score": round(avg_score, 4),
            "worst_metric": worst_metric,
            "score": round(worst_score, 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })

    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
