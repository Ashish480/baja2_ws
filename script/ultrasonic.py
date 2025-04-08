#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import serial

class UltrasonicNode(Node):
    def __init__(self):
        super().__init__('ultrasonic_sensor_node')
        self.publisher_ = self.create_publisher(Float32, '/ultrasonic_distance', 10)
        
        # Open Serial Connection to ESP32
        try:
            self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)
            self.get_logger().info("Connected to ESP32 on /dev/ttyUSB0")
        except Exception as e:
            self.get_logger().error(f"Failed to connect to ESP32: {e}")
            return
        
        # Create Timer to Read Data
        self.timer = self.create_timer(0.5, self.read_ultrasonic_data)

    def read_ultrasonic_data(self):
        try:
            if self.serial_port.in_waiting > 0:
                data = self.serial_port.readline().decode('utf-8').strip()

                # Ignore ESP32 boot messages
                if "rst:" in data or "boot:" in data or "load:" in data or "SPIWP" in data:
                    self.get_logger().warn(f"Ignoring ESP32 boot message: {data}")
                    return

                # Ignore empty lines
                if not data:
                    self.get_logger().warn("Received empty data from ESP32, skipping...")
                    return

                # Try converting to float
                try:
                    distance = float(data)
                    msg = Float32()
                    msg.data = distance
                    self.publisher_.publish(msg)
                    self.get_logger().info(f"Published Distance: {distance} cm")
                except ValueError:
                    self.get_logger().warn(f"Invalid data received from ESP32: {data}")

        except Exception as e:
            self.get_logger().error(f"Error reading from ESP32: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = UltrasonicNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down Ultrasonic Sensor Node")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

