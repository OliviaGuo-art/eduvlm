import json
import uuid
from datetime import datetime, timezone

import gradio as gr
import numpy as np
from sentence_transformers import SentenceTransformer

from citation_validation import validate_citations
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


DOCUMENTS = load_jsonl(ROOT / "data" / "documents.jsonl")
ALL_DOC_IDS = {doc["doc_id"] for doc in DOCUMENTS}

RETRIEVER = None
DOCUMENT_EMBEDDINGS = None


def get_retriever():
    global RETRIEVER, DOCUMENT_EMBEDDINGS

    if RETRIEVER is None:
        print("Loading text retrieval model...")
        RETRIEVER = SentenceTransformer(
            TEXT_MODEL,
            device="cpu",
        )
        document_texts = [
            f"{doc['title']}。{doc['text']}"
            for doc in DOCUMENTS
        ]
        DOCUMENT_EMBEDDINGS = RETRIEVER.encode(
            document_texts,
            batch_size=8,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    return RETRIEVER, DOCUMENT_EMBEDDINGS


def retrieve(question, top_k):
    retriever, document_embeddings = get_retriever()

    query_embedding = retriever.encode(
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
        doc = DOCUMENTS[index]
        results.append({
            "rank": rank,
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "text": doc["text"],
            "image_path": doc["image_path"],
            "score": float(similarity[index]),
        })

    return results


def format_usage(direct_result, rag_result):
    lines = [
        "| 模式 | 结束原因 | 输入 Token | 输出 Token | 总 Token |",
        "|---|---|---:|---:|---:|",
    ]

    for label, result in (
        ("直接问答", direct_result),
        ("RAG", rag_result),
    ):
        if result is None:
            continue

        usage = result["usage"] or {}
        lines.append(
            f"| {label} "
            f"| {result['finish_reason']} "
            f"| {usage.get('prompt_tokens')} "
            f"| {usage.get('completion_tokens')} "
            f"| {usage.get('total_tokens')} |"
        )

    return "\n".join(lines)


def run_qa(image, question, mode, top_k, max_tokens):
    if image is None:
        raise gr.Error("请先上传一张图片。")

    if not question or not question.strip():
        raise gr.Error("请输入问题。")

    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    run_dir = ROOT / "outputs" / "gradio_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    image_path = run_dir / "input.png"
    image.convert("RGB").save(image_path)

    direct_result = None
    rag_result = None
    retrieved_documents = []
    citation_report = None

    client = QwenVLClient()

    if mode != "直接问答":
        retrieved_documents = retrieve(
            question.strip(),
            int(top_k),
        )

    if mode in ("直接问答", "两种模式对比"):
        direct_result = client.answer(
            image_path=image_path,
            question=question.strip(),
            system_prompt=DIRECT_SYSTEM_PROMPT,
            max_tokens=int(max_tokens),
        )

    if mode in ("检索增强问答", "两种模式对比"):
        rag_result = client.answer(
            image_path=image_path,
            question=question.strip(),
            system_prompt=RAG_SYSTEM_PROMPT,
            evidence_text=format_evidence(
                retrieved_documents
            ),
            max_tokens=int(max_tokens),
        )

        citation_report = validate_citations(
            rag_result["answer"],
            [
                doc["doc_id"]
                for doc in retrieved_documents
            ],
            ALL_DOC_IDS,
        )

    record = {
        "run_id": run_id,
        "mode": mode,
        "model": client.model_name,
        "question": question.strip(),
        "image_path": str(image_path.relative_to(ROOT)),
        "top_k": int(top_k),
        "max_tokens": int(max_tokens),
        "retrieved_documents": retrieved_documents,
        "direct": direct_result,
        "retrieval_augmented": rag_result,
        "citation_validation": citation_report,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    record_path = run_dir / "result.json"
    with record_path.open("w", encoding="utf-8") as file:
        json.dump(record, file, ensure_ascii=False, indent=2)

    direct_answer = (
        direct_result["answer"]
        if direct_result is not None
        else "本次未运行直接问答。"
    )
    rag_answer = (
        rag_result["answer"]
        if rag_result is not None
        else "本次未运行 RAG。"
    )

    evidence_rows = [
        [
            doc["rank"],
            doc["doc_id"],
            doc["title"],
            round(doc["score"], 4),
        ]
        for doc in retrieved_documents
    ]

    if citation_report is None:
        citation_text = "本次没有 RAG 回答需要校验。"
    else:
        citation_text = citation_report["message"]

    run_info = (
        f"运行编号：`{run_id}`\n\n"
        f"本地记录：`{record_path}`\n\n"
        + format_usage(direct_result, rag_result)
    )

    return (
        direct_answer,
        rag_answer,
        evidence_rows,
        citation_text,
        run_info,
    )


def build_app():
    with gr.Blocks(title="多模态学习资料检索与问答") as demo:
        gr.Markdown(
            "# 多模态学习资料检索与问答\n"
            "对比直接视觉问答与检索增强问答。"
            "每次对比会分别调用一次模型，请注意 API 用量。"
        )

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="上传题目或学习资料图片",
                )
                question_input = gr.Textbox(
                    lines=4,
                    label="问题",
                    placeholder="请输入与图片有关的问题",
                )
                mode_input = gr.Dropdown(
                    choices=[
                        "直接问答",
                        "检索增强问答",
                        "两种模式对比",
                    ],
                    value="两种模式对比",
                    label="运行模式",
                )
                top_k_input = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="RAG 候选资料数量 Top-K",
                )
                max_tokens_input = gr.Slider(
                    minimum=200,
                    maximum=1000,
                    value=700,
                    step=100,
                    label="最大输出 Token",
                )
                submit_button = gr.Button(
                    "开始回答",
                    variant="primary",
                )

            with gr.Column():
                run_info_output = gr.Markdown(
                    label="运行信息"
                )
                citation_output = gr.Markdown(
                    label="引用编号校验"
                )
                evidence_output = gr.Dataframe(
                    headers=[
                        "排名",
                        "doc_id",
                        "标题",
                        "相似度",
                    ],
                    interactive=False,
                    label="检索证据",
                )

        direct_output = gr.Markdown(
            label="直接回答"
        )
        rag_output = gr.Markdown(
            label="检索增强回答"
        )

        submit_button.click(
            fn=run_qa,
            inputs=[
                image_input,
                question_input,
                mode_input,
                top_k_input,
                max_tokens_input,
            ],
            outputs=[
                direct_output,
                rag_output,
                evidence_output,
                citation_output,
                run_info_output,
            ],
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        inbrowser=True,
    )