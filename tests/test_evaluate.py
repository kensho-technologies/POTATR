# Copyright 2025-present Kensho Technologies, LLC.
"""Tests for evaluation helpers."""

import json

from potatr.evaluate import load_page_words


def test_load_page_words_from_either_file_name(tmp_path):
    images = tmp_path / "test" / "images"
    images.mkdir(parents=True)
    words_dir = tmp_path / "test" / "words"
    ocr_dir = tmp_path / "test" / "ocr"
    words_dir.mkdir()
    ocr_dir.mkdir()
    (words_dir / "page_1.json").write_text(json.dumps([{"text": "document"}]))
    (ocr_dir / "page_1_words.json").write_text(json.dumps([{"text": "ocr"}]))
    img_filepath = str(images / "page_1.jpg")

    assert load_page_words(img_filepath, "words") == [{"text": "document"}]
    assert load_page_words(img_filepath, "ocr") == [{"text": "ocr"}]
