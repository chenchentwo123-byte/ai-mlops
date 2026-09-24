from src.yoloe_detector import parse_visual_classes, visual_caption, weights_filename


def test_parse_preserves_case_and_order():
    assert parse_visual_classes("charging_nest, Robot, robot") == ["charging_nest", "Robot"]


def test_parse_splits_semicolon_and_newline():
    assert parse_visual_classes("person; helmet\nsafety vest") == ["person", "helmet", "safety vest"]


def test_visual_caption():
    assert visual_caption(["charging_nest", "robot"]) == "visual: charging_nest, robot"
    assert visual_caption([]) == "visual:"


def test_weights_filename():
    assert weights_filename("s") == "yoloe-11s-seg.pt"
    assert weights_filename("x") == "yoloe-26x-seg.pt"
