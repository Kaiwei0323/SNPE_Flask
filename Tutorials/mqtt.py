import paho.mqtt.client as mqtt
import time

class MQTTClient:
    def __init__(self, broker="localhost", port=1883):
        self.client = mqtt.Client()
        self.broker = broker
        self.port = port
        self.connected = False
        try:
            self.client.connect(self.broker, self.port, 60)
            self.client.loop_start()  # Start the loop to process callbacks
            self.connected = True
            print(f"MQTT connected to {broker}:{port}")
        except Exception as e:
            print(f"MQTT connection failed: {e}")
            print("Continuing without MQTT...")
            self.connected = False

    def publish(self, topic, message):
        """Publish a message to a specified topic."""
        if self.connected:
            self.client.publish(topic, message)
        else:
            print(f"MQTT not connected, skipping publish: {topic} = {message}")

    def disconnect(self):
        """Disconnect the MQTT client."""
        self.client.loop_stop()  # Stop the MQTT loop
        self.client.disconnect()  # Disconnect the MQTT client
