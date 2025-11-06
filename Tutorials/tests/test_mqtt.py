"""
Tests for MQTT client module.
"""
import pytest
from unittest.mock import Mock, patch
from mqtt import MQTTClient


class TestMQTTClient:
    """Test cases for MQTTClient class."""

    @patch('mqtt.mqtt.Client')
    def test_mqtt_client_initialization(self, mock_client_class):
        """Test MQTT client initialization with default parameters."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = MQTTClient()
        
        assert client.broker == "localhost"
        assert client.port == 1883
        mock_client.connect.assert_called_once_with("localhost", 1883, 60)
        mock_client.loop_start.assert_called_once()

    @patch('mqtt.mqtt.Client')
    def test_mqtt_client_custom_broker(self, mock_client_class):
        """Test MQTT client initialization with custom broker and port."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = MQTTClient(broker="test.broker.com", port=8883)
        
        assert client.broker == "test.broker.com"
        assert client.port == 8883
        mock_client.connect.assert_called_once_with("test.broker.com", 8883, 60)

    @patch('mqtt.mqtt.Client')
    def test_mqtt_publish(self, mock_client_class):
        """Test MQTT publish functionality."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = MQTTClient()
        client.publish("test/topic", "test message")
        
        mock_client.publish.assert_called_once_with("test/topic", "test message")

    @patch('mqtt.mqtt.Client')
    def test_mqtt_disconnect(self, mock_client_class):
        """Test MQTT disconnect functionality."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = MQTTClient()
        client.disconnect()
        
        mock_client.loop_stop.assert_called_once()
        mock_client.disconnect.assert_called_once()

