"""Tests for MockTrainer."""

from sage_finetune.trainers import MockTrainer
from sage_finetune.trainers.mock_trainer import TrainingConfig


class TestMockTrainer:
    """Tests for MockTrainer."""

    def test_name(self):
        """Test trainer name."""
        trainer = MockTrainer()
        assert trainer.name == "mock"

    def test_train_basic(self):
        """Test basic training."""
        trainer = MockTrainer(simulate_time=0.01)
        train_data = [{"text": f"Sample {i}"} for i in range(10)]

        result = trainer.train(train_data)

        assert "train_loss" in result
        assert result["train_loss"] > 0
        assert result["num_samples"] == 10

    def test_train_with_config(self):
        """Test training with custom config."""
        trainer = MockTrainer(simulate_time=0.01)
        config = TrainingConfig(num_train_epochs=2)
        train_data = [{"text": "Sample"}] * 5

        result = trainer.train(train_data, config=config)

        assert result["num_epochs"] == 2

    def test_train_with_eval(self):
        """Test training with evaluation dataset."""
        trainer = MockTrainer(simulate_time=0.01)
        train_data = [{"text": "Train"}] * 5
        eval_data = [{"text": "Eval"}] * 3

        result = trainer.train(train_data, eval_dataset=eval_data)

        assert "eval_loss" in result
        assert result["eval_loss"] > 0

    def test_evaluate(self):
        """Test evaluation."""
        trainer = MockTrainer()
        eval_data = [{"text": "Sample"}] * 5

        result = trainer.evaluate(eval_data)

        assert "eval_loss" in result
        assert "perplexity" in result

    def test_save_load_model(self):
        """Test save/load model."""
        trainer = MockTrainer()

        trainer.save_model("/tmp/mock_model")
        trainer.load_model("/tmp/mock_model")

        assert trainer._model_loaded

    def test_generate(self):
        """Test text generation."""
        trainer = MockTrainer()

        response = trainer.generate("Hello world")

        assert "MockTrainer response" in response
        assert "Hello world" in response


class TestTrainingConfig:
    """Tests for TrainingConfig."""

    def test_default_values(self):
        """Test default config values."""
        config = TrainingConfig()

        assert config.num_train_epochs == 3
        assert config.learning_rate == 5e-5

    def test_custom_values(self):
        """Test custom config values."""
        config = TrainingConfig(
            model_name_or_path="gpt2",
            num_train_epochs=5,
        )

        assert config.model_name_or_path == "gpt2"
        assert config.num_train_epochs == 5

    def test_to_dict(self):
        """Test config to dict conversion."""
        config = TrainingConfig()
        d = config.to_dict()

        assert isinstance(d, dict)
        assert "num_train_epochs" in d
