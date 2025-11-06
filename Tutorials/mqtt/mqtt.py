import paho.mqtt.client as mqtt

class MQTTClient:
    def __init__(self, broker="localhost", port=1883):
        self.client = mqtt.Client()
        self.broker = broker
        self.port = port
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_start()  # Start the loop to process callbacks

    def publish(self, topic, message):
        """Publish a message to a specified topic."""
        self.client.publish(topic, message)

    def disconnect(self):
        """Disconnect the MQTT client."""
        self.client.loop_stop()  # Stop the MQTT loop
        self.client.disconnect()  # Disconnect the MQTT client
