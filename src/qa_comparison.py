import argparse
import json
from datetime import datetime, timezone

import numpy as np
from sentence_transformers import SentenceTransformer

from qwen_vl import (
    DIRECT_SYSTEM_PROMPT,
    RAG_SYSTEM_PROMPT,
    ROOT,
    QwenVLClient,
)


DEFAULT_IMAGE = "data/raw/images/free_fall_001.png"
DEFAULT_QUESTION = (
    "这张示意图展示的是什么物理现象？"
    "请根据图中信息说明物体受哪种力，"
    "以及在忽略空气阻力时加速度是多少。"
)

TEXT_MODEL = "BAAI/bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line.strip())
            for line in file
            if line.strip()
        ]


def retrieve_documents(question, documents, top_k):
    document_texts = [
        f"{doc['title']}。{doc['text']}"
        for doc in documents
    ]

    model = SentenceTransformer(TEXT_MODEL, device="cpu")

    document_embeddings = model.encode(
        document_texts,
        batch_size=8,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    query_embedding = model.encode(
        [QUERY_INSTRUCTION + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    similarity = query_embedding[0] @ document_embeddings.T
    ranked_indices = np.argsort(-similarity, kind="stable")

    results = []
    for rank, index in enumerate(
        ranked_indices[:top_k],
        start=1,
    ):
        doc = documents[index]
        results.append({
            "rank": rank,
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": doc["text"],
            "image_path": doc["image_path"],
            "score": float(similarity[index]),
        })

    return results


def format_evidence(retrieved_documents):
    sections = []

    for doc in retrieved_documents:
        sections.append(
            f"资料 {doc['rank']}\n"
            f"doc_id：{doc['doc_id']}\n"
            f"标题：{doc['title']}\n"
            f"正文：{doc['text']}"
        )

    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument(
        "--output",
        default="outputs/qa_comparison_free_fall.json",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=700)
    args = parser.parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    image_path = ROOT / args.image
    output_path = ROOT / args.output

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {output_path}"
        )

    documents = load_jsonl(ROOT / "data" / "documents.jsonl")

    print("=== RETRIEVAL ===")
    retrieved_documents = retrieve_documents(
        args.question,
        documents,
        args.top_k,
    )

    for doc in retrieved_documents:
        print(
            f"Rank {doc['rank']}: {doc['doc_id']} "
            f"score={doc['score']:.4f}"
        )

    evidence_text = format_evidence(retrieved_documents)
    client = QwenVLClient()

    print("\n=== RUNNING DIRECT VQA ===")
    direct_result = client.answer(
    image_path=image_path,
    question=args.question,
    system_prompt=DIRECT_SYSTEM_PROMPT,
    max_tokens=args.max_tokens,
)
    print("\n=== RUNNING RETRIEVAL-AUGMENTED VQA ===")
    rag_result = client.answer(
    image_path=image_path,
    question=args.question,
    system_prompt=RAG_SYSTEM_PROMPT,
    evidence_text=evidence_text,
    max_tokens=args.max_tokens,
)
    result = {
        "experiment": "direct_vs_retrieval_augmented_vqa",
        "model": client.model_name,
        "image_path": args.image,
        "question": args.question,
        "generation": {
            "temperature": 0.2,
            "max_tokens": args.max_tokens,
            "enable_thinking": False,
        },
        "retriever": {
            "model": TEXT_MODEL,
            "query_instruction": QUERY_INSTRUCTION,
            "document_representation": "title + text",
            "top_k": args.top_k,
        },
        "retrieved_documents": retrieved_documents,
        "direct": direct_result,
        "retrieval_augmented": rag_result,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n=== DIRECT ANSWER ===")
    print(direct_result["answer"])
    print("Finish reason:", direct_result["finish_reason"])
    print("Finish reason:", rag_result["finish_reason"])
    print("Usage:", direct_result["usage"])

    print("\n=== RETRIEVAL-AUGMENTED ANSWER ===")
    print(rag_result["answer"])
    print("Finish reason:", direct_result["finish_reason"])
    print("Finish reason:", rag_result["finish_reason"])
    print("Usage:", rag_result["usage"])

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()