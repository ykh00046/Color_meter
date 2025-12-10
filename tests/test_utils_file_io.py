from pathlib import Path

from src.utils import file_io


def test_ensure_dir_creates_parent(tmp_path: Path):
    target = tmp_path / "nested" / "file.txt"
    out = file_io.ensure_dir(target)
    assert out == target
    assert target.parent.exists()


def test_write_read_json_roundtrip(tmp_path: Path):
    data = {"a": 1, "b": {"c": [1, 2]}}
    path = tmp_path / "data.json"
    file_io.write_json(data, path)
    loaded = file_io.read_json(path)
    assert loaded == data


def test_list_files_filters_only_files(tmp_path: Path):
    (tmp_path / "f1.txt").write_text("1", encoding="utf-8")
    (tmp_path / "f2.log").write_text("2", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    files = file_io.list_files(tmp_path, "*.txt")
    assert files == [tmp_path / "f1.txt"]

