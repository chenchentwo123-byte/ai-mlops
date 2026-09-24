from pathlib import Path

from PIL import Image

from src.exporters import export_from_records, write_yolo_data_yaml


def test_write_yolo_data_yaml_relative(tmp_path: Path):
    path = write_yolo_data_yaml(tmp_path, ["person", "helmet"])
    text = path.read_text(encoding="utf-8")
    assert "path: ." in text
    assert "train: images" in text
    assert "val: images" in text
    assert "0: person" in text
    assert "1: helmet" in text
    assert "C:/" not in text and "/mnt/" not in text


def test_export_yolo_images_labels_layout(tmp_path: Path):
    images = tmp_path / "src_images"
    images.mkdir()
    img_path = images / "sample.jpg"
    Image.new("RGB", (100, 80), color=(20, 40, 60)).save(img_path, quality=90)

    records = [
        {
            "image": "sample.jpg",
            "image_path": str(img_path),
            "width": 100,
            "height": 80,
            "classes": ["person", "helmet"],
            "detections": [
                {"label": "person", "score": 0.9, "bbox_xyxy": [10, 10, 50, 50]},
            ],
        }
    ]
    exports = tmp_path / "exports"
    result = export_from_records(records, exports, "yolo", images)
    assert result["classes"] == ["person", "helmet"]

    yolo_dir = exports / "yolo"
    yaml_text = (yolo_dir / "data.yaml").read_text(encoding="utf-8")
    assert "train: images" in yaml_text
    assert (yolo_dir / "labels" / "sample.txt").is_file()
    assert (yolo_dir / "images" / "sample.jpg").is_file()
    line = (yolo_dir / "labels" / "sample.txt").read_text(encoding="utf-8").strip()
    assert line.startswith("0 ")
