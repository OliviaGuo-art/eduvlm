import base64
import json
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "raw" / "images" / "free_fall_001.png"
OUTPUT_PATH = ROOT / "outputs" / "vqa_smoke_test.json"

QUESTION = (
    "这张示意图展示的是什么物理现象？"
    "请根据图中信息说明物体受哪种力，"
    "以及在忽略空气阻力时加速度是多少。"
)

SYSTEM_PROMPT = """
你是一名基础物理学习助手。
请用中文回答用户关于图片的问题。
区分图中能够直接观察的信息和你根据物理知识作出的推断。
如果图片信息不足，请明确说明，不要编造图中不存在的内容。
本次没有提供外部参考资料，不要声称自己查阅了任何资料。
""".strip()


def encode_image_as_data_url(image_path):
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None:
        mime_type = "image/png"

    with image_path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def extract_answer(content):
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                text_parts.append(part.get("text", ""))
        return "\n".join(text_parts)

    raise TypeError(f"Unexpected response content type: {type(content)}")


def main():
    load_dotenv(ROOT / ".env")

    api_key = os.getenv("DASHSCOPE_API_KEY")
    workspace_id = os.getenv("DASHSCOPE_WORKSPACE_ID")
    region = os.getenv("DASHSCOPE_REGION", "cn-beijing")
    model_name = os.getenv("QWEN_VL_MODEL", "qwen3-vl-plus")

    missing = [
        name
        for name, value in {
            "DASHSCOPE_API_KEY": api_key,
            "DASHSCOPE_WORKSPACE_ID": workspace_id,
        }.items()
        if not value or "这里" in value or "your_" in value
    ]
    if missing:
        raise ValueError(
            "Missing or placeholder configuration: "
            + ", ".join(missing)
        )

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result: {OUTPUT_PATH}"
        )

    base_url = (
        f"https://{workspace_id}.{region}."
        "maas.aliyuncs.com/compatible-mode/v1"
    )

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=180.0,
        max_retries=1,
    )

    image_data_url = encode_image_as_data_url(IMAGE_PATH)

    print("Model:", model_name)
    print("Image:", IMAGE_PATH)
    print("Question:", QUESTION)
    print("\nSending request...")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": QUESTION,
                    },
                ],
            },
        ],
        max_tokens=500,
        temperature=0.2,
        extra_body={
            "enable_thinking": False,
        },
    )

    answer = extract_answer(
        response.choices[0].message.content
    )

    usage = None
    if response.usage is not None:
        usage = response.usage.model_dump()

    result = {
        "experiment": "direct_vqa_smoke_test",
        "mode": "direct",
        "model": model_name,
        "region": region,
        "image_path": str(IMAGE_PATH.relative_to(ROOT)),
        "question": QUESTION,
        "system_prompt": SYSTEM_PROMPT,
        "retrieved_documents": [],
        "answer": answer,
        "request_id": response.id,
        "finish_reason": response.choices[0].finish_reason,
        "usage": usage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print("\n=== ANSWER ===")
    print(answer)

    print("\n=== REQUEST INFO ===")
    print("Request ID:", response.id)
    print("Finish reason:", response.choices[0].finish_reason)
    print("Usage:", json.dumps(usage, ensure_ascii=False))

    print("\nSaved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()