"""
Tests for pipeline classes (BasePipeline, FilePipeline, RtspPipeline, WebcamPipeline).
"""
import pytest
from unittest.mock import Mock, patch
from pipelines import BasePipeline, FilePipeline, RtspPipeline, WebcamPipeline


class TestPipelines:
    """Test cases for pipeline classes."""

    def test_pipeline_classes_import(self):
        """Test that pipeline classes can be imported."""
        assert BasePipeline is not None
        assert FilePipeline is not None
        assert RtspPipeline is not None
        assert WebcamPipeline is not None

    @pytest.mark.skip(reason="Requires GStreamer libraries")
    def test_base_pipeline_initialization(self):
        """Test BasePipeline initialization."""
        # This test requires GStreamer
        image_queue = Mock()
        capture_lock = Mock()
        
        pipeline = BasePipeline("test_uri", image_queue, capture_lock)
        assert pipeline.uri == "test_uri"
        assert pipeline.image_queue == image_queue
        assert pipeline.capture_lock == capture_lock
        assert pipeline.rate == 1

    @pytest.mark.skip(reason="Requires GStreamer libraries")
    def test_file_pipeline_initialization(self):
        """Test FilePipeline initialization."""
        image_queue = Mock()
        capture_lock = Mock()
        
        pipeline = FilePipeline("/data/video/test.mp4", image_queue, capture_lock)
        assert pipeline.uri == "/data/video/test.mp4"

    @pytest.mark.skip(reason="Requires GStreamer libraries")
    def test_rtsp_pipeline_initialization(self):
        """Test RtspPipeline initialization."""
        image_queue = Mock()
        capture_lock = Mock()
        
        pipeline = RtspPipeline("rtsp://test.url/stream", image_queue, capture_lock)
        assert pipeline.uri == "rtsp://test.url/stream"

    @pytest.mark.skip(reason="Requires GStreamer libraries")
    def test_webcam_pipeline_initialization(self):
        """Test WebcamPipeline initialization."""
        image_queue = Mock()
        capture_lock = Mock()
        
        pipeline = WebcamPipeline("/dev/video0", image_queue, capture_lock)
        assert pipeline.uri == "/dev/video0"

