#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import serial

class IMUSerialPublisher(Node):
    def __init__(self):
        super().__init__('imu_serial_publisher')

        # Open Serial Connection (Change /dev/ttyUSB0 if needed)
        self.serial_port = serial.Serial('/dev/ttyUSB0', 115200, timeout=1)

        # ROS2 Publisher for IMU Data
        self.imu_publisher = self.create_publisher(Imu, '/imu/data', 10)

        # Timer to Read Serial Data
        self.timer = self.create_timer(0.1, self.read_serial_data)  # 10Hz

    def read_serial_data(self):
        try:
            line = self.serial_port.readline().decode('utf-8').strip()
            values = line.split(',')

            if len(values) == 6:  # Ensure all values are received
                imu_msg = Imu()
                imu_msg.linear_acceleration.x = float(values[0])
                imu_msg.linear_acceleration.y = float(values[1])
                imu_msg.linear_acceleration.z = float(values[2])
                imu_msg.angular_velocity.x = float(values[3])
                imu_msg.angular_velocity.y = float(values[4])
                imu_msg.angular_velocity.z = float(values[5])

                # Publish IMU Data
                self.imu_publisher.publish(imu_msg)

        except Exception:
            pass  # Ignore errors without printing anything

def main(args=None):
    rclpy.init(args=args)
    node = IMUSerialPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

