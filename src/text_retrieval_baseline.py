import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path.cwd()
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


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

    assert len(doc_ids) == len(set(doc_ids)), "存在重复 doc_id"
    assert len(query_ids) == len(set(query_ids)), "存在重复 query_id"

    doc_id_set = set(doc_ids)

    for doc in documents:
        image_path = ROOT / doc["image_path"]
        assert image_path.exists(), f"图片不存在：{image_path}"

    for query in queries:
        expected_ids = query["expected_doc_ids"]
        query_type = query["question_type"]

        for doc_id in expected_ids:
            assert doc_id in doc_id_set, (
                f"{query['query_id']} 引用了不存在的资料：{doc_id}"
            )

        if query_type == "out_of_domain":
            assert not expected_ids, (
                f"库外查询 {query['query_id']} 的 expected_doc_ids 应为空"
            )
        else:
            assert expected_ids, (
                f"正常查询 {query['query_id']} 缺少预期资料"
            )

    print("Data validation passed.")


def main():
    documents = load_jsonl(DATA_DIR / "documents.jsonl")
    queries = load_jsonl(DATA_DIR / "queries.jsonl")

    validate_data(documents, queries)

    document_texts = [
        f"{doc['title']}。{doc['text']}"
        for doc in documents
    ]
    query_texts = [query["query_text"] for query in queries]

    model = SentenceTransformer(
        "sentence-transformers/clip-ViT-B-32",
        device="cpu",
    )

    document_embeddings = model.encode(
        document_texts,
        normalize_embeddings=True,
    )
    query_embeddings = model.encode(
        query_texts,
        normalize_embeddings=True,
    )

    similarity = query_embeddings @ document_embeddings.T

    results = []
    closed_set_ranks = []

    for query_index, query in enumerate(queries):
        scores = similarity[query_index]
        ranking = np.argsort(-scores)

        ranked_docs = [
            {
                "doc_id": documents[doc_index]["doc_id"],
                "title": documents[doc_index]["title"],
                "score": float(scores[doc_index]),
            }
            for doc_index in ranking
        ]

        expected_ids = set(query["expected_doc_ids"])
        is_out_of_domain = query["question_type"] == "out_of_domain"

        relevant_rank = None
        if not is_out_of_domain:
            for rank, item in enumerate(ranked_docs, start=1):
                if item["doc_id"] in expected_ids:
                    relevant_rank = rank
                    closed_set_ranks.append(rank)
                    break

        results.append(
            {
                "query_id": query["query_id"],
                "query_text": query["query_text"],
                "question_type": query["question_type"],
                "expected_doc_ids": query["expected_doc_ids"],
                "relevant_rank": relevant_rank,
                "top1_score": ranked_docs[0]["score"],
                "top1_top2_margin": (
                    ranked_docs[0]["score"] - ranked_docs[1]["score"]
                ),
                "ranking": ranked_docs,
            }
        )

        print("\nQuery:", query["query_text"])
        print("Type:", query["question_type"])
        print("Expected:", query["expected_doc_ids"])
        print("Relevant rank:", relevant_rank)

        for rank, item in enumerate(ranked_docs[:3], start=1):
            print(
                f"  Top-{rank}: {item['doc_id']}"
                f"  score={item['score']:.4f}"
            )

    assert closed_set_ranks, "没有可用于评测的闭集查询"

    recall_at_1 = np.mean(
        [rank <= 1 for rank in closed_set_ranks]
    )
    recall_at_3 = np.mean(
        [rank <= 3 for rank in closed_set_ranks]
    )
    mrr = np.mean(
        [1.0 / rank for rank in closed_set_ranks]
    )

    metrics = {
        "closed_query_count": len(closed_set_ranks),
        "recall_at_1": float(recall_at_1),
        "recall_at_3": float(recall_at_3),
        "mrr": float(mrr),
    }

    output = {
        "model": "sentence-transformers/clip-ViT-B-32",
        "retrieval_input": "document title and text",
        "metrics": metrics,
        "results": results,
    }

    output_path = OUTPUT_DIR / "text_retrieval_baseline.json"
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nMetrics:")
    print(json.dumps(metrics, indent=2))
    print("Results saved to:", output_path)


if __name__ == "__main__":
    main()