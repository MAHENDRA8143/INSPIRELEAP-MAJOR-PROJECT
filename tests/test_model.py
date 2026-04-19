from src.model import build_advanced_cnn


def test_model_output_shape() -> None:
    model = build_advanced_cnn()
    assert model.output_shape == (None, 10)
