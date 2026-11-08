import json
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_PATH = ROOT / "data" / "documents.jsonl"
OUTPUT_DIR = ROOT / "data" / "raw" / "image_queries"
METADATA_PATH = ROOT / "data" / "image_queries.jsonl"


def load_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [
            json.loads(line.strip())
            for line in file
            if line.strip()
        ]


def get_font(size):
    for font_path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rotate_image(image):
    return image.rotate(
        5,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=(255, 255, 255),
    )


def grayscale_image(image):
    return image.convert("L").convert("RGB")


def crop_image(image):
    width, height = image.size
    dx = int(width * 0.03)
    dy = int(height * 0.03)

    cropped = image.crop((
        dx,
        dy,
        width - dx,
        height - dy,
    ))
    return cropped.resize(
        (width, height),
        Image.Resampling.LANCZOS,
    )


def jpeg_compress_image(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=35)
    buffer.seek(0)

    with Image.open(buffer) as compressed:
        return compressed.convert("RGB")


def brightness_image(image):
    return ImageEnhance.Brightness(image).enhance(0.70)


def contrast_image(image):
    return ImageEnhance.Contrast(image).enhance(1.40)


TRANSFORMS = [
    ("rotate_5deg", rotate_image),
    ("grayscale", grayscale_image),
    ("crop_3_percent", crop_image),
    ("jpeg_quality_35", jpeg_compress_image),
    ("brightness_070", brightness_image),
    ("contrast_140", contrast_image),
]


def new_ood_canvas(title):
    image = Image.new("RGB", (1200, 800), "white")
    draw = ImageDraw.Draw(image)
    draw.text(
        (70, 50),
        title,
        fill=(20, 40, 80),
        font=get_font(52),
    )
    return image, draw


def make_geometry_query(path):
    image, draw = new_ood_canvas("几何：三角形内角和")

    points = [(250, 620), (600, 180), (950, 620)]
    draw.line(
        points + [points[0]],
        fill=(30, 90, 180),
        width=8,
    )

    draw.text((200, 650), "A", fill="black", font=get_font(40))
    draw.text((580, 110), "B", fill="black", font=get_font(40))
    draw.text((960, 650), "C", fill="black", font=get_font(40))
    draw.text(
        (300, 710),
        "∠A + ∠B + ∠C = 180°",
        fill=(180, 60, 40),
        font=get_font(42),
    )

    image.save(path)


def make_cell_query(path):
    image, draw = new_ood_canvas("生物：细胞结构示意图")

    draw.ellipse(
        (270, 170, 930, 650),
        outline=(40, 140, 90),
        width=10,
        fill=(225, 248, 230),
    )
    draw.ellipse(
        (500, 320, 700, 500),
        outline=(100, 60, 160),
        width=8,
        fill=(230, 215, 245),
    )
    draw.text(
        (535, 375),
        "细胞核",
        fill=(70, 35, 120),
        font=get_font(34),
    )
    draw.text(
        (350, 680),
        "细胞膜与细胞核",
        fill=(30, 100, 65),
        font=get_font(38),
    )

    image.save(path)


def main():
    documents = load_jsonl(DOCUMENTS_PATH)

    # 只从自制图片生成查询，避免产生尚未确认具体图片授权的衍生物。
    source_documents = [
        doc for doc in documents
        if doc.get("source") == "self-created"
    ]
    assert len(source_documents) == 9, (
        f"Expected 9 self-created documents, got {len(source_documents)}"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    queries = []

    for index, doc in enumerate(source_documents, start=1):
        query_id = f"iq{index:03d}"
        transform_name, transform = TRANSFORMS[
            (index - 1) % len(TRANSFORMS)
        ]

        source_path = ROOT / doc["image_path"]
        with Image.open(source_path) as source_image:
            rgb_image = source_image.convert("RGB")
            query_image = transform(rgb_image)

        filename = f"{query_id}_{doc['doc_id']}.png"
        output_path = OUTPUT_DIR / filename
        query_image.save(output_path)

        queries.append({
            "query_id": query_id,
            "query_image_path": (
                f"data/raw/image_queries/{filename}"
            ),
            "expected_doc_ids": [doc["doc_id"]],
            "question_type": "image_transform",
            "transform": transform_name,
            "source": "derived-from-self-created",
            "license": "CC-BY-4.0",
            "split": "dev",
        })

    geometry_path = OUTPUT_DIR / "iq010_geometry_ood.png"
    make_geometry_query(geometry_path)
    queries.append({
        "query_id": "iq010",
        "query_image_path": (
            "data/raw/image_queries/iq010_geometry_ood.png"
        ),
        "expected_doc_ids": [],
        "question_type": "out_of_domain",
        "transform": "self_created_ood",
        "source": "self-created",
        "license": "CC-BY-4.0",
        "split": "dev",
    })

    cell_path = OUTPUT_DIR / "iq011_cell_ood.png"
    make_cell_query(cell_path)
    queries.append({
        "query_id": "iq011",
        "query_image_path": (
            "data/raw/image_queries/iq011_cell_ood.png"
        ),
        "expected_doc_ids": [],
        "question_type": "out_of_domain",
        "transform": "self_created_ood",
        "source": "self-created",
        "license": "CC-BY-4.0",
        "split": "dev",
    })

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        for query in queries:
            file.write(
                json.dumps(query, ensure_ascii=False) + "\n"
            )

    print("Self-created source documents:", len(source_documents))
    print("Closed-set image queries:", sum(
        query["question_type"] != "out_of_domain"
        for query in queries
    ))
    print("Out-of-domain image queries:", sum(
        query["question_type"] == "out_of_domain"
        for query in queries
    ))
    print("Images saved to:", OUTPUT_DIR)
    print("Metadata saved to:", METADATA_PATH)


if __name__ == "__main__":
    main()