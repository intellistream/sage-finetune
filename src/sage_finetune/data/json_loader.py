"""JSON/JSONL dataset loader for fine-tuning."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

# Try importing SAGE base class
try:
    from sage.libs.finetune.interface.base import DatasetLoader

    _HAS_SAGE = True
except ImportError:
    DatasetLoader = object
    _HAS_SAGE = False


class JSONDatasetLoader(DatasetLoader):
    """Load training data from JSON/JSONL files.

    Supports:
    - JSONL format: One JSON object per line
    - JSON format: Array of JSON objects
    - Instruction format: {"instruction": ..., "input": ..., "output": ...}
    - Chat format: {"messages": [{"role": ..., "content": ...}, ...]}

    Example:
        >>> loader = JSONDatasetLoader()
        >>> dataset = loader.load("train.jsonl")
        >>> for sample in dataset:
        ...     print(sample)
    """

    def __init__(
        self,
        text_field: str = "text",
        instruction_field: str = "instruction",
        input_field: str = "input",
        output_field: str = "output",
        messages_field: str = "messages",
    ) -> None:
        """Initialize loader.

        Args:
            text_field: Field name for plain text.
            instruction_field: Field name for instruction.
            input_field: Field name for input context.
            output_field: Field name for expected output.
            messages_field: Field name for chat messages.
        """
        self._text_field = text_field
        self._instruction_field = instruction_field
        self._input_field = input_field
        self._output_field = output_field
        self._messages_field = messages_field

    @property
    def name(self) -> str:
        """Return loader name."""
        return "json"

    def load(
        self,
        path: str | Path,
        format: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Load dataset from file.

        Args:
            path: Path to JSON/JSONL file.
            format: Format hint ("jsonl", "json", or None for auto-detect).
            **kwargs: Additional arguments (ignored).

        Returns:
            List of data samples.
        """
        path = Path(path)

        if format is None:
            format = "jsonl" if path.suffix == ".jsonl" else "json"

        if format == "jsonl":
            return self._load_jsonl(path)
        else:
            return self._load_json(path)

    def _load_jsonl(self, path: Path) -> list[dict[str, Any]]:
        """Load JSONL file."""
        samples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        return samples

    def _load_json(self, path: Path) -> list[dict[str, Any]]:
        """Load JSON file."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, list):
            return data
        else:
            return [data]

    def preprocess(
        self,
        dataset: list[dict[str, Any]],
        tokenizer: Any,
        format_type: str = "auto",
    ) -> list[dict[str, Any]]:
        """Preprocess dataset for training.

        Args:
            dataset: Raw dataset (list of samples).
            tokenizer: Tokenizer instance for encoding.
            format_type: "instruction", "chat", "text", or "auto".

        Returns:
            Preprocessed dataset ready for training.
        """
        if not dataset:
            return []

        # Auto-detect format
        if format_type == "auto":
            sample = dataset[0]
            if self._messages_field in sample:
                format_type = "chat"
            elif self._instruction_field in sample:
                format_type = "instruction"
            else:
                format_type = "text"

        formatted = []
        for sample in dataset:
            if format_type == "instruction":
                text = self._format_instruction(sample)
            elif format_type == "chat":
                text = self._format_chat(sample)
            else:
                text = sample.get(self._text_field, str(sample))

            if tokenizer is not None:
                encoded = tokenizer(
                    text,
                    truncation=True,
                    padding=False,
                    return_tensors=None,
                )
                formatted.append(encoded)
            else:
                formatted.append({"text": text})

        return formatted

    def stream(
        self,
        path: str | Path,
        format: str | None = None,
        **kwargs: Any,
    ) -> Iterator[dict[str, Any]]:
        """Stream dataset samples (for large datasets).

        Args:
            path: Path to JSON/JSONL file.
            format: Format hint.
            **kwargs: Additional arguments.

        Yields:
            Data samples one by one.
        """
        path = Path(path)

        if format is None:
            format = "jsonl" if path.suffix == ".jsonl" else "json"

        if format == "jsonl":
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        else:
            # For JSON, we need to load all at once
            yield from self._load_json(path)

    def _format_instruction(self, sample: dict[str, Any]) -> str:
        """Format instruction-style sample."""
        instruction = sample.get(self._instruction_field, "")
        input_text = sample.get(self._input_field, "")
        output_text = sample.get(self._output_field, "")

        if input_text:
            return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output_text}"
        else:
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output_text}"

    def _format_chat(self, sample: dict[str, Any]) -> str:
        """Format chat-style sample."""
        messages = sample.get(self._messages_field, [])
        parts = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")

        return "\n\n".join(parts)
