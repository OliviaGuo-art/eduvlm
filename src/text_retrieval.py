import argparse
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[1]


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line.strip())
            for line in file
            if line.strip()
        ]


def validate_data(documents, queries):
    doc_ids = [doc["doc_id"] for doc in documents]
    query_ids = [query["query_id"] for query in queries]

    assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_id"
    assert len(query_ids) == len(set(query_ids)), "Duplicate query_id"

    doc_id_set = set(doc_ids)

    for doc in documents:
        image_path = ROOT / doc["image_path"]
        assert image_path.exists(), f"Missing image: {image_path}"

    for query in queries:
        for doc_id in query["expected_doc_ids"]:
            assert doc_id in doc_id_set, (
                f"{query['query_id']} refers to unknown doc_id: {doc_id}"
            )

        is_ood = query["question_type"] == "out_of_domain"
        if is_ood:
            assert not query["expected_doc_ids"], (
                f"{query['query_id']}: OOD query must have no expected docs"
            )
        else:
            assert query["expected_doc_ids"], (
                f"{query['query_id']}: closed-set query needs expected docs"
            )


def find_first_relevant_rank(ranked_doc_ids, expected_doc_ids):
    for rank, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in expected_doc_ids:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--query-instruction", default="")
    args = parser.parse_args()

    output_path = ROOT / args.output
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {output_path}\n"
            "Use a new --output filename."
        )

    documents = load_jsonl(ROOT / "data" / "documents.jsonl")
    queries = load_jsonl(ROOT / "data" / "queries.jsonl")
    validate_data(documents, queries)

    document_texts = [
        f"{doc['title']}。{doc['text']}"
        for doc in documents
    ]
    query_texts = [
        args.query_instruction + query["query_text"]
        for query in queries
    ]

    model = SentenceTransformer(args.model, device="cpu")

    document_embeddings = model.encode(
        document_texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    query_embeddings = model.encode(
        query_texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    similarity = query_embeddings @ document_embeddings.T

    details = []
    closed_set_ranks = []

    for row, query in enumerate(queries):
        ranked_indices = np.argsort(-similarity[row], kind="stable")
        ranked_doc_ids = [
            documents[index]["doc_id"]
            for index in ranked_indices
        ]

        expected_doc_ids = query["expected_doc_ids"]
        relevant_rank = find_first_relevant_rank(
            ranked_doc_ids,
            expected_doc_ids,
        )

        is_ood = query["question_type"] == "out_of_domain"
        if not is_ood:
            assert relevant_rank is not None
            closed_set_ranks.append(relevant_rank)

        top_results = []
        for index in ranked_indices[:3]:
            top_results.append({
                "doc_id": documents[index]["doc_id"],
                "title": documents[index]["title"],
                "score": float(similarity[row, index]),
            })

        top1_score = float(similarity[row, ranked_indices[0]])
        top2_score = float(similarity[row, ranked_indices[1]])

        details.append({
            "query_id": query["query_id"],
            "query_text": query["query_text"],
            "question_type": query["question_type"],
            "expected_doc_ids": expected_doc_ids,
            "relevant_rank": relevant_rank,
            "top1_score": top1_score,
            "top1_top2_margin": top1_score - top2_score,
            "top3": top_results,
        })

    metrics = {
        "closed_query_count": len(closed_set_ranks),
        "out_of_domain_query_count": sum(
            query["question_type"] == "out_of_domain"
            for query in queries
        ),
        "recall_at_1": float(np.mean([
            rank <= 1 for rank in closed_set_ranks
        ])),
        "recall_at_3": float(np.mean([
            rank <= 3 for rank in closed_set_ranks
        ])),
        "mrr": float(np.mean([
            1.0 / rank for rank in closed_set_ranks
        ])),
    }

    result = {
        "model": args.model,
        "device": "cpu",
        "document_representation": "title + text",
        "query_instruction": args.query_instruction,
        "document_count": len(documents),
        "query_count": len(queries),
        "document_embedding_shape": list(document_embeddings.shape),
        "query_embedding_shape": list(query_embeddings.shape),
        "metrics": metrics,
        "details": details,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n=== METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== QUERY RANKS ===")
    for detail in details:
        print(
            detail["query_id"],
            "type =", detail["question_type"],
            "| relevant_rank =", detail["relevant_rank"],
            "| top1 =", detail["top3"][0]["doc_id"],
            "| score =", round(detail["top1_score"], 4),
        )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()