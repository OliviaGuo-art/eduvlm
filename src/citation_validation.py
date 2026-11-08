import re


DOC_ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")


def extract_citations(answer, all_doc_ids):
    bracket_contents = re.findall(r"【([^】]+)】", answer)

    cited = set()
    malformed = []

    for content in bracket_contents:
        content = content.strip()

        if content == "doc_id" or not DOC_ID_PATTERN.fullmatch(content):
            malformed.append(content)
        else:
            cited.add(content)

    for line in answer.splitlines():
        if line.strip().startswith("使用资料："):
            for doc_id in all_doc_ids:
                if doc_id in line:
                    cited.add(doc_id)

    return sorted(cited), sorted(set(malformed))


def validate_citations(answer, retrieved_doc_ids, all_doc_ids):
    cited, malformed = extract_citations(
        answer,
        all_doc_ids,
    )

    retrieved_set = set(retrieved_doc_ids)
    invalid = sorted(
        doc_id for doc_id in cited
        if doc_id not in retrieved_set
    )

    is_valid = not malformed and not invalid

    if malformed or invalid:
        message_parts = ["❌ 引用编号校验失败。"]

        if malformed:
            message_parts.append(
                "格式错误：" + ", ".join(malformed)
            )
        if invalid:
            message_parts.append(
                "不在本次 Top-K 中：" + ", ".join(invalid)
            )

        status = "invalid"
    elif not cited:
        message_parts = [
            "⚠️ 未检测到资料引用。"
            "如果回答使用了候选资料，应补充引用；"
            "如果资料无关或问题属于库外，不引用可以是正确行为。"
        ]
        status = "no_citations"
    else:
        message_parts = [
            "✅ 检测到的引用编号均来自本次 Top-K。"
        ]
        status = "valid"

    message_parts.append(
        "注意：程序只校验编号，不能据此证明资料内容确实支持回答。"
    )

    return {
        "status": status,
        "is_valid": is_valid,
        "cited_doc_ids": cited,
        "invalid_doc_ids": invalid,
        "malformed_citations": malformed,
        "message": "\n\n".join(message_parts),
    }