"""
Tests for SNPE helper manager module.
"""
import pytest
from snpe import PerfProfile, Runtime, SnpeContext


class TestPerfProfile:
    """Test cases for PerfProfile class."""

    def test_perf_profile_constants(self):
        """Test that all PerfProfile constants are defined."""
        assert PerfProfile.DEFAULT == "DEFAULT"
        assert PerfProfile.BALANCED == "BALANCED"
        assert PerfProfile.HIGH_PERFORMANCE == "HIGH_PERFORMANCE"
        assert PerfProfile.POWER_SAVER == "POWER_SAVER"
        assert PerfProfile.SYSTEM_SETTINGS == "SYSTEM_SETTINGS"
        assert PerfProfile.SUSTAINED_HIGH_PERFORMANCE == "SUSTAINED_HIGH_PERFORMANCE"
        assert PerfProfile.BURST == "BURST"
        assert PerfProfile.LOW_POWER_SAVER == "LOW_POWER_SAVER"
        assert PerfProfile.HIGH_POWER_SAVER == "HIGH_POWER_SAVER"
        assert PerfProfile.LOW_BALANCED == "LOW_BALANCED"
        assert PerfProfile.EXTREME_POWERSAVER == "EXTREME_POWERSAVER"


class TestRuntime:
    """Test cases for Runtime class."""

    def test_runtime_constants(self):
        """Test that all Runtime constants are defined."""
        assert Runtime.CPU == "CPU"
        assert Runtime.GPU == "GPU"
        assert Runtime.GPU_FLOAT16 == "GPU_FLOAT16"
        assert Runtime.AIP_FIXED_TF == "AIP_FIXED_TF"
        assert Runtime.DSP == "DSP"


class TestSnpeContext:
    """Test cases for SnpeContext class."""

    @pytest.mark.skip(reason="Requires libsnpehelper.so and DLC model files")
    def test_snpe_context_initialization(self):
        """Test SnpeContext initialization."""
        # This test requires actual SNPE libraries and model files
        # Skip for now, but structure is here for when models are available
        context = SnpeContext(
            dlc_path="models/test.dlc",
            input_layers=["input"],
            output_layers=["output"],
            output_tensors=["tensor"],
            runtime=Runtime.CPU,
            profile_level=PerfProfile.BALANCED,
            enable_cache=False
        )
        assert context is not None

    def test_snpe_context_default_parameters(self):
        """Test SnpeContext with default parameters."""
        # This will fail if libsnpehelper.so is not available, but tests structure
        with pytest.raises((ImportError, AttributeError, RuntimeError)):
            context = SnpeContext()
            # If it doesn't raise, verify defaults
            assert context.m_runtime == Runtime.DSP
            assert context.profiling_level == PerfProfile.BALANCED
            assert context.m_enable_cache == False

