import json

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from text_retrieval import (
    ROOT,
    find_first_relevant_rank,
    load_jsonl,
)


MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
OUTPUT_PATH = ROOT / "outputs" / "image_retrieval_clip.json"


def load_rgb_image(relative_path):
    path = ROOT / relative_path
    with Image.open(path) as image:
        return image.convert("RGB")


def validate_data(documents, queries):
    doc_ids = [doc["doc_id"] for doc in documents]
    query_ids = [query["query_id"] for query in queries]

    assert len(doc_ids) == len(set(doc_ids)), "Duplicate doc_id"
    assert len(query_ids) == len(set(query_ids)), "Duplicate query_id"

    doc_id_set = set(doc_ids)

    for doc in documents:
        assert (ROOT / doc["image_path"]).exists(), (
            f"Missing document image: {doc['image_path']}"
        )

    for query in queries:
        assert (ROOT / query["query_image_path"]).exists(), (
            f"Missing query image: {query['query_image_path']}"
        )

        for doc_id in query["expected_doc_ids"]:
            assert doc_id in doc_id_set, (
                f"{query['query_id']} refers to unknown {doc_id}"
            )

        is_ood = query["question_type"] == "out_of_domain"
        if is_ood:
            assert not query["expected_doc_ids"], (
                f"{query['query_id']}: OOD query must have no target"
            )
        else:
            assert query["expected_doc_ids"], (
                f"{query['query_id']}: closed-set query needs a target"
            )


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {OUTPUT_PATH}\n"
            "Keep the old result and choose a new filename if needed."
        )

    documents = load_jsonl(ROOT / "data" / "documents.jsonl")
    queries = load_jsonl(ROOT / "data" / "image_queries.jsonl")
    validate_data(documents, queries)

    print("Loading document images...")
    document_images = [
        load_rgb_image(doc["image_path"])
        for doc in documents
    ]

    print("Loading query images...")
    query_images = [
        load_rgb_image(query["query_image_path"])
        for query in queries
    ]

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    print("Encoding document images...")
    document_embeddings = model.encode(
        document_images,
        batch_size=4,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    print("Encoding query images...")
    query_embeddings = model.encode(
        query_images,
        batch_size=4,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )

    similarity = query_embeddings @ document_embeddings.T

    details = []
    closed_set_ranks = []

    for row, query in enumerate(queries):
        ranked_indices = np.argsort(
            -similarity[row],
            kind="stable",
        )
        ranked_doc_ids = [
            documents[index]["doc_id"]
            for index in ranked_indices
        ]

        relevant_rank = find_first_relevant_rank(
            ranked_doc_ids,
            query["expected_doc_ids"],
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
            "query_image_path": query["query_image_path"],
            "question_type": query["question_type"],
            "transform": query["transform"],
            "expected_doc_ids": query["expected_doc_ids"],
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
        "model": MODEL_NAME,
        "device": "cpu",
        "task": "image-to-document-image retrieval",
        "document_representation": "document image",
        "document_count": len(documents),
        "query_count": len(queries),
        "document_embedding_shape": list(
            document_embeddings.shape
        ),
        "query_embedding_shape": list(query_embeddings.shape),
        "evaluation_scope": (
            "Transformed near-duplicate image queries; "
            "not an independent semantic generalization test"
        ),
        "metrics": metrics,
        "details": details,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n=== METRICS ===")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))

    print("\n=== IMAGE QUERY RANKS ===")
    for detail in details:
        print(
            detail["query_id"],
            "transform =", detail["transform"],
            "| relevant_rank =", detail["relevant_rank"],
            "| top1 =", detail["top3"][0]["doc_id"],
            "| score =", round(detail["top1_score"], 4),
            "| margin =", round(
                detail["top1_top2_margin"], 4
            ),
        )

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()