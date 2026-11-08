import json
from pathlib import Path

import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer
from skimage import data


OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 四张测试图片及其候选描述。
samples = [
    {
        "image_id": "astronaut",
        "image": data.astronaut(),
        "caption": "a portrait of an astronaut in a space suit",
    },
    {
        "image_id": "coffee",
        "image": data.coffee(),
        "caption": "a cup of coffee with coffee beans",
    },
    {
        "image_id": "cat",
        "image": data.chelsea(),
        "caption": "a close-up photo of a cat",
    },
    {
        "image_id": "rocket",
        "image": data.rocket(),
        "caption": "a space rocket on a launch pad",
    },
]

# 查询描述与候选描述故意不完全相同。
queries = [
    {"query_id": "q1", "text": "an astronaut wearing a helmet", "expected": "astronaut"},
    {"query_id": "q2", "text": "coffee beans beside a drink", "expected": "coffee"},
    {"query_id": "q3", "text": "a furry feline", "expected": "cat"},
    {"query_id": "q4", "text": "a space rocket before launch", "expected": "rocket"},
]

print("Loading CLIP model on CPU...")
model = SentenceTransformer(
    "sentence-transformers/clip-ViT-B-32",
    device="cpu",
)

images = [
    Image.fromarray(sample["image"]).convert("RGB")
    for sample in samples
]
candidate_texts = [sample["caption"] for sample in samples]
query_texts = [query["text"] for query in queries]

print("Encoding images...")
image_embeddings = model.encode(
    images,
    normalize_embeddings=True,
    show_progress_bar=True,
)


print("Encoding candidate captions...")
candidate_embeddings = model.encode(
    candidate_texts,
    normalize_embeddings=True,
)

print("Encoding queries...")
query_embeddings = model.encode(
    query_texts,
    normalize_embeddings=True,
)

print("Image embedding shape:", image_embeddings.shape)
print("Text embedding shape:", query_embeddings.shape)

# 查询向量与候选图片向量分别计算相似度。
image_similarity = query_embeddings @ image_embeddings.T

# 查询向量与候选文字向量也计算一次，供对照观察。
text_similarity = query_embeddings @ candidate_embeddings.T

results = []
correct = 0

for index, query in enumerate(queries):
    scores = image_similarity[index]
    ranking = np.argsort(-scores)

    top_results = [
        {
            "image_id": samples[image_index]["image_id"],
            "score": float(scores[image_index]),
        }
        for image_index in ranking
    ]

    predicted = top_results[0]["image_id"]
    is_correct = predicted == query["expected"]
    correct += int(is_correct)

    results.append(
        {
            **query,
            "predicted": predicted,
            "correct": is_correct,
            "ranking": top_results,
        }
    )

    print("\nQuery:", query["text"])
    print("Expected:", query["expected"])
    print("Predicted:", predicted)
    print("Correct:", is_correct)

    for rank, result in enumerate(top_results, start=1):
        print(
            f"  Top-{rank}: "
            f"{result['image_id']:<12} "
            f"score={result['score']:.4f}"
        )

accuracy = correct / len(queries)

summary = {
    "model": "sentence-transformers/clip-ViT-B-32",
    "device": "cpu",
    "top1_accuracy": accuracy,
    "results": results,
}

output_path = OUTPUT_DIR / "clip_smoke_test.json"
output_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("\nTop-1 accuracy:", accuracy)
print("Results saved to:", output_path)