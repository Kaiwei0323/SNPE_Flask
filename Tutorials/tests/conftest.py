"""
Pytest configuration and fixtures for testing.
"""
import pytest
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_model_map():
    """Fixture providing a sample model map for testing."""
    return {
        "DETR": ("models/detr_resnet101_int8.dlc", ["image"], ["/model/class_labels_classifier/MatMul_post_reshape", "/model/Sigmoid"], ["logits", "boxes"], []),
        "YOLOV8S_DSP": ("models/yolov8s_encode_int8.dlc", ["images"], ["/model.22/Concat_5"], ["output0"], []),
    }

@pytest.fixture
def flask_app():
    """Fixture providing a Flask app instance for testing."""
    from app import app
    app.config['TESTING'] = True
    return app

@pytest.fixture
def flask_client(flask_app):
    """Fixture providing a Flask test client."""
    return flask_app.test_client()

