import argparse
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

from qa_comparison import (
    QUERY_INSTRUCTION,
    TEXT_MODEL,
    format_evidence,
    load_jsonl,
)
from qwen_vl import (
    DIRECT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    ROOT,
    QwenVLClient,
)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def item_fingerprint(item):
    serialized = json.dumps(
        item,
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def validate_items(items, documents):
    doc_ids = {doc["doc_id"] for doc in documents}
    qa_ids = [item["qa_id"] for item in items]

    assert len(qa_ids) == len(set(qa_ids)), "Duplicate qa_id"

    for item in items:
        assert (ROOT / item["image_path"]).exists(), (
            f"Missing image: {item['image_path']}"
        )

        for doc_id in item["expected_doc_ids"]:
            assert doc_id in doc_ids, (
                f"{item['qa_id']} refers to unknown {doc_id}"
            )

        is_ood = item["question_type"] == "out_of_domain"
        if is_ood:
            assert not item["expected_doc_ids"]
        else:
            assert item["expected_doc_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--run-name", default="test_v1")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    documents = load_jsonl(ROOT / "data" / "documents.jsonl")
    items = load_jsonl(ROOT / "data" / "qa_benchmark.jsonl")
    validate_items(items, documents)

    items = [
        item for item in items
        if item["split"] == args.split
    ]
    items.sort(key=lambda item: item["qa_id"])

    if args.limit is not None:
        items = items[:args.limit]

    if not items:
        raise ValueError("No QA items selected")

    client = QwenVLClient()

    config = {
        "split": args.split,
        "vlm_model": client.model_name,
        "retriever_model": TEXT_MODEL,
        "query_instruction": QUERY_INSTRUCTION,
        "document_representation": "title + text",
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "temperature": 0.2,
        "enable_thinking": False,
    }

    run_dir = (
        ROOT / "outputs" / "qa_batch" / args.run_name
    )

    print("Loading text retrieval model...")
    retriever = SentenceTransformer(TEXT_MODEL, device="cpu")

    document_texts = [
        f"{doc['title']}。{doc['text']}"
        for doc in documents
    ]
    document_embeddings = retriever.encode(
        document_texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    completed_results = []

    for index, item in enumerate(items, start=1):
        qa_id = item["qa_id"]
        output_path = run_dir / f"{qa_id}.json"
        fingerprint = item_fingerprint(item)

        print(
            f"\n=== {index}/{len(items)}: {qa_id} ==="
        )

        if output_path.exists():
            with output_path.open("r", encoding="utf-8") as file:
                result = json.load(file)

            if result["config"] != config:
                raise ValueError(
                    f"Config mismatch for {qa_id}. "
                    "Use a new --run-name."
                )
            if result["item_fingerprint"] != fingerprint:
                raise ValueError(
                    f"Benchmark item changed: {qa_id}. "
                    "Use a new --run-name."
                )
            if result["status"] == "complete":
                print("Already complete; skipping API calls.")
                completed_results.append(result)
                continue
        else:
            result = {
                "qa_id": qa_id,
                "status": "initialized",
                "config": config,
                "item_fingerprint": fingerprint,
                "item": item,
                "created_at_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
            }

        if "retrieved_documents" not in result:
            query_embedding = retriever.encode(
                [QUERY_INSTRUCTION + item["question"]],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            similarity = (
                query_embedding[0] @ document_embeddings.T
            )
            ranked_indices = np.argsort(
                -similarity,
                kind="stable",
            )

            retrieved = []
            for rank, doc_index in enumerate(
                ranked_indices[:args.top_k],
                start=1,
            ):
                doc = documents[doc_index]
                retrieved.append({
                    "rank": rank,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "text": doc["text"],
                    "image_path": doc["image_path"],
                    "score": float(similarity[doc_index]),
                })

            result["retrieved_documents"] = retrieved
            result["status"] = "retrieved"
            save_json(output_path, result)

        print(
            "Top-K:",
            ", ".join(
                doc["doc_id"]
                for doc in result["retrieved_documents"]
            ),
        )

        try:
            if "direct" not in result:
                print("Running direct VQA...")
                result["direct"] = client.answer(
                    image_path=ROOT / item["image_path"],
                    question=item["question"],
                    system_prompt=DIRECT_SYSTEM_PROMPT,
                    max_tokens=args.max_tokens,
                )
                result["status"] = "direct_complete"
                save_json(output_path, result)

            if "retrieval_augmented" not in result:
                print("Running retrieval-augmented VQA...")
                evidence_text = format_evidence(
                    result["retrieved_documents"]
                )
                result["retrieval_augmented"] = client.answer(
                    image_path=ROOT / item["image_path"],
                    question=item["question"],
                    system_prompt=RAG_SYSTEM_PROMPT,
                    evidence_text=evidence_text,
                    max_tokens=args.max_tokens,
                )
                result["status"] = "rag_complete"
                save_json(output_path, result)

            result["status"] = "complete"
            result["completed_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            save_json(output_path, result)

        except Exception as exc:
            result["status"] = "error"
            result["error_type"] = type(exc).__name__
            save_json(output_path, result)
            raise

        completed_results.append(result)

        direct_usage = result["direct"]["usage"] or {}
        rag_usage = (
            result["retrieval_augmented"]["usage"] or {}
        )
        print(
            "Tokens:",
            direct_usage.get("total_tokens"),
            "+",
            rag_usage.get("total_tokens"),
        )
        print("Saved:", output_path)

        if args.limit == 1:
            print("\n=== DIRECT ANSWER ===")
            print(result["direct"]["answer"])
            print("\n=== RAG ANSWER ===")
            print(result["retrieval_augmented"]["answer"])

    complete_count = sum(
        result["status"] == "complete"
        for result in completed_results
    )
    total_tokens = 0

    for result in completed_results:
        if result["status"] != "complete":
            continue
        for mode in ("direct", "retrieval_augmented"):
            usage = result[mode]["usage"] or {}
            total_tokens += usage.get("total_tokens", 0)

    print("\n=== BATCH SUMMARY ===")
    print("Selected items:", len(items))
    print("Complete items:", complete_count)
    print("Total tokens:", total_tokens)
    print("Run directory:", run_dir)


if __name__ == "__main__":
    main()