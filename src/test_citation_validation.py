from citation_validation import validate_citations


ALL_DOC_IDS = {
    "uniform_circular_motion_001",
    "friction_001",
    "free_fall_001",
}

RETRIEVED = [
    "uniform_circular_motion_001",
    "friction_001",
]


def main():
    correct = validate_citations(
        "向心加速度指向圆心【uniform_circular_motion_001】。",
        RETRIEVED,
        ALL_DOC_IDS,
    )
    assert correct["is_valid"]
    assert correct["status"] == "valid"

    malformed = validate_citations(
        "向心加速度指向圆心【doc_id】。",
        RETRIEVED,
        ALL_DOC_IDS,
    )
    assert not malformed["is_valid"]
    assert "doc_id" in malformed["malformed_citations"]

    fabricated = validate_citations(
        "答案【free_fall_001】。",
        RETRIEVED,
        ALL_DOC_IDS,
    )
    assert not fabricated["is_valid"]
    assert fabricated["invalid_doc_ids"] == ["free_fall_001"]

    no_citations = validate_citations(
        "候选资料均不相关。\n\n使用资料：无",
        RETRIEVED,
        ALL_DOC_IDS,
    )
    assert no_citations["is_valid"]
    assert no_citations["status"] == "no_citations"

    print("All citation validation tests passed.")


if __name__ == "__main__":
    main()