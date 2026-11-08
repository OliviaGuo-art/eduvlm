import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SCORE_COLUMNS = [
    "correctness_0_2",
    "completeness_0_1",
    "visual_grounding_0_1",
    "no_unsupported_claims_0_1",
    "citation_validity_0_1",
    "evidence_relevance_0_2",
    "citation_support_0_1",
    "notes",
]


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def extract_citations(answer, all_doc_ids):
    cited = set(re.findall(r"【([A-Za-z0-9_-]+)】", answer))

    for line in answer.splitlines():
        if line.strip().startswith("使用资料："):
            for doc_id in all_doc_ids:
                if doc_id in line:
                    cited.add(doc_id)

    return sorted(cited)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="outputs/qa_batch/test_v1",
    )
    parser.add_argument(
        "--output",
        default="outputs/qa_scoring_test_v1.csv",
    )
    args = parser.parse_args()

    run_dir = ROOT / args.run_dir
    output_path = ROOT / args.output

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite scoring sheet: {output_path}\n"
            "Your sheet may contain manual scores. "
            "Choose a new --output if you really need a new sheet."
        )

    result_paths = sorted(run_dir.glob("qa*.json"))
    if not result_paths:
        raise FileNotFoundError(
            f"No QA result files found in {run_dir}"
        )

    documents_path = ROOT / "data" / "documents.jsonl"
    with documents_path.open("r", encoding="utf-8") as file:
        documents = [
            json.loads(line)
            for line in file
            if line.strip()
        ]
    all_doc_ids = {doc["doc_id"] for doc in documents}

    fieldnames = [
        "qa_id",
        "mode",
        "question_type",
        "question",
        "reference_answer",
        "expected_doc_ids",
        "retrieved_doc_ids",
        "relevant_rank_in_top3",
        "cited_doc_ids",
        "auto_citation_ids_valid",
        "answer",
        "finish_reason",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
    ] + SCORE_COLUMNS

    rows = []
    direct_tokens = 0
    rag_tokens = 0
    direct_finish_reasons = Counter()
    rag_finish_reasons = Counter()
    closed_count = 0
    top1_hits = 0
    top3_hits = 0

    for result_path in result_paths:
        result = load_json(result_path)
        if result["status"] != "complete":
            raise ValueError(
                f"Incomplete result: {result_path}"
            )

        item = result["item"]
        expected = set(item["expected_doc_ids"])
        retrieved_documents = result["retrieved_documents"]
        retrieved_ids = [
            doc["doc_id"] for doc in retrieved_documents
        ]

        is_ood = item["question_type"] == "out_of_domain"
        relevant_rank = None

        if not is_ood:
            closed_count += 1
            for doc in retrieved_documents:
                if doc["doc_id"] in expected:
                    relevant_rank = doc["rank"]
                    break

            if retrieved_ids and retrieved_ids[0] in expected:
                top1_hits += 1
            if relevant_rank is not None:
                top3_hits += 1

        for mode, label in (
            ("direct", "direct"),
            ("retrieval_augmented", "rag"),
        ):
            mode_result = result[mode]
            answer = mode_result["answer"]
            usage = mode_result["usage"] or {}

            if mode == "direct":
                cited = []
                auto_valid = ""
                direct_tokens += usage.get("total_tokens", 0)
                direct_finish_reasons[
                    mode_result["finish_reason"]
                ] += 1
            else:
                cited = extract_citations(answer, all_doc_ids)
                auto_valid = int(
                    set(cited).issubset(set(retrieved_ids))
                )
                rag_tokens += usage.get("total_tokens", 0)
                rag_finish_reasons[
                    mode_result["finish_reason"]
                ] += 1

            row = {
                "qa_id": item["qa_id"],
                "mode": label,
                "question_type": item["question_type"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "expected_doc_ids": "|".join(
                    item["expected_doc_ids"]
                ),
                "retrieved_doc_ids": "|".join(retrieved_ids),
                "relevant_rank_in_top3": (
                    relevant_rank
                    if relevant_rank is not None
                    else ""
                ),
                "cited_doc_ids": "|".join(cited),
                "auto_citation_ids_valid": auto_valid,
                "answer": answer,
                "finish_reason": mode_result["finish_reason"],
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get(
                    "completion_tokens"
                ),
                "total_tokens": usage.get("total_tokens"),
            }

            for column in SCORE_COLUMNS:
                row[column] = ""

            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "qa_item_count": len(result_paths),
        "answer_row_count": len(rows),
        "direct_total_tokens": direct_tokens,
        "rag_total_tokens": rag_tokens,
        "combined_total_tokens": direct_tokens + rag_tokens,
        "direct_finish_reasons": dict(
            direct_finish_reasons
        ),
        "rag_finish_reasons": dict(rag_finish_reasons),
        "closed_retrieval": {
            "closed_query_count": closed_count,
            "top1_correct_count": top1_hits,
            "top3_relevant_count": top3_hits,
            "recall_at_1": (
                top1_hits / closed_count
                if closed_count else None
            ),
            "recall_at_3": (
                top3_hits / closed_count
                if closed_count else None
            ),
        },
        "scoring_sheet": str(output_path),
    }

    summary_path = output_path.with_name(
        output_path.stem + "_auto_summary.json"
    )
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print("\n=== AUTO SUMMARY ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nScoring sheet saved to:", output_path)
    print("Manual score columns were left blank.")


if __name__ == "__main__":
    main()