"""
Tests for model handlers (YOLOV5, YOLOV8, DETR).
"""
import pytest
from model_handlers import YOLOV5, YOLOV8, DETR


class TestModelHandlers:
    """Test cases for model handler classes."""

    def test_model_handlers_import(self):
        """Test that model handlers can be imported."""
        assert YOLOV5 is not None
        assert YOLOV8 is not None
        assert DETR is not None

    @pytest.mark.skip(reason="Requires DLC model files and SNPE libraries")
    def test_yolov8_initialization(self):
        """Test YOLOV8 model handler initialization."""
        # This test requires actual DLC model files
        model = YOLOV8(
            dlc_path="models/yolov8s_encode_int8.dlc",
            input_layers=["images"],
            output_layers=["/model.22/Concat_5"],
            output_tensors=["output0"],
            runtime="CPU",
            classes=[],
            profile_level="BALANCED",
            enable_cache=False
        )
        assert model is not None

    @pytest.mark.skip(reason="Requires DLC model files and SNPE libraries")
    def test_detr_initialization(self):
        """Test DETR model handler initialization."""
        # This test requires actual DLC model files
        model = DETR(
            dlc_path="models/detr_resnet101_int8.dlc",
            input_layers=["image"],
            output_layers=["/model/class_labels_classifier/MatMul_post_reshape", "/model/Sigmoid"],
            output_tensors=["logits", "boxes"],
            runtime="CPU",
            classes=[],
            profile_level="BALANCED",
            enable_cache=False
        )
        assert model is not None

    def test_model_handler_classes_exist(self):
        """Test that model handler classes are defined."""
        # Just verify classes exist, not instantiate them
        assert callable(YOLOV5)
        assert callable(YOLOV8)
        assert callable(DETR)

