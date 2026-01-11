"""Tests for JSON dataset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sage_finetune.data import JSONDatasetLoader


class TestJSONDatasetLoader:
    """Tests for JSONDatasetLoader."""

    @pytest.fixture
    def loader(self) -> JSONDatasetLoader:
        """Create loader instance."""
        return JSONDatasetLoader()

    @pytest.fixture
    def jsonl_file(self, tmp_path: Path) -> Path:
        """Create test JSONL file."""
        data = [
            {"text": "Sample 1"},
            {"text": "Sample 2"},
            {"text": "Sample 3"},
        ]
        path = tmp_path / "test.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    @pytest.fixture
    def json_file(self, tmp_path: Path) -> Path:
        """Create test JSON file."""
        data = [
            {"text": "Sample A"},
            {"text": "Sample B"},
        ]
        path = tmp_path / "test.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    @pytest.fixture
    def instruction_file(self, tmp_path: Path) -> Path:
        """Create instruction format file."""
        data = [
            {
                "instruction": "Summarize",
                "input": "Long text here",
                "output": "Short summary",
            },
            {
                "instruction": "Translate",
                "input": "Hello",
                "output": "Bonjour",
            },
        ]
        path = tmp_path / "instructions.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    @pytest.fixture
    def chat_file(self, tmp_path: Path) -> Path:
        """Create chat format file."""
        data = [
            {
                "messages": [
                    {"role": "system", "content": "You are helpful."},
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello."},
                ]
            },
        ]
        path = tmp_path / "chat.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        return path

    def test_name(self, loader: JSONDatasetLoader) -> None:
        """Test loader name."""
        assert loader.name == "json"

    def test_load_jsonl(self, loader: JSONDatasetLoader, jsonl_file: Path) -> None:
        """Test loading JSONL file."""
        data = loader.load(jsonl_file)
        assert len(data) == 3
        assert data[0]["text"] == "Sample 1"
        assert data[2]["text"] == "Sample 3"

    def test_load_json(self, loader: JSONDatasetLoader, json_file: Path) -> None:
        """Test loading JSON file."""
        data = loader.load(json_file)
        assert len(data) == 2
        assert data[0]["text"] == "Sample A"

    def test_stream(self, loader: JSONDatasetLoader, jsonl_file: Path) -> None:
        """Test streaming JSONL file."""
        samples = list(loader.stream(jsonl_file))
        assert len(samples) == 3
        assert samples[0]["text"] == "Sample 1"

    def test_preprocess_text(self, loader: JSONDatasetLoader, jsonl_file: Path) -> None:
        """Test preprocessing with text format."""
        data = loader.load(jsonl_file)
        processed = loader.preprocess(data, tokenizer=None, format_type="text")
        assert len(processed) == 3
        assert processed[0]["text"] == "Sample 1"

    def test_preprocess_instruction(
        self, loader: JSONDatasetLoader, instruction_file: Path
    ) -> None:
        """Test preprocessing instruction format."""
        data = loader.load(instruction_file)
        processed = loader.preprocess(data, tokenizer=None, format_type="instruction")
        assert len(processed) == 2
        assert "### Instruction:" in processed[0]["text"]
        assert "Summarize" in processed[0]["text"]

    def test_preprocess_chat(self, loader: JSONDatasetLoader, chat_file: Path) -> None:
        """Test preprocessing chat format."""
        data = loader.load(chat_file)
        processed = loader.preprocess(data, tokenizer=None, format_type="chat")
        assert len(processed) == 1
        assert "System: You are helpful." in processed[0]["text"]
        assert "User: Hi" in processed[0]["text"]

    def test_preprocess_auto_detect(
        self, loader: JSONDatasetLoader, instruction_file: Path
    ) -> None:
        """Test auto-detection of format."""
        data = loader.load(instruction_file)
        processed = loader.preprocess(data, tokenizer=None, format_type="auto")
        # Should detect instruction format
        assert "### Instruction:" in processed[0]["text"]

    def test_preprocess_with_tokenizer(self, loader: JSONDatasetLoader, jsonl_file: Path) -> None:
        """Test preprocessing with mock tokenizer."""
        data = loader.load(jsonl_file)

        # Mock tokenizer
        def mock_tokenizer(text: str, **kwargs) -> dict:
            return {"input_ids": [len(text)], "text": text}

        processed = loader.preprocess(data, tokenizer=mock_tokenizer)
        assert len(processed) == 3
        assert "input_ids" in processed[0]

    def test_empty_dataset(self, loader: JSONDatasetLoader) -> None:
        """Test preprocessing empty dataset."""
        processed = loader.preprocess([], tokenizer=None)
        assert processed == []

    def test_custom_fields(self, tmp_path: Path) -> None:
        """Test loader with custom field names."""
        loader = JSONDatasetLoader(
            text_field="content",
            instruction_field="prompt",
            output_field="response",
        )

        data = [{"prompt": "Do something", "response": "Done."}]
        path = tmp_path / "custom.jsonl"
        with open(path, "w") as f:
            f.write(json.dumps(data[0]) + "\n")

        loaded = loader.load(path)
        processed = loader.preprocess(loaded, tokenizer=None, format_type="instruction")
        assert "Do something" in processed[0]["text"]
        assert "Done." in processed[0]["text"]
