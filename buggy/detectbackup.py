#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from std_msgs.msg import Float32MultiArray  # Import standard ROS 2 message types

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('detector')
        
        # Declare parameters
        self.declare_parameter('model_path', 'src/buggy/buggy/models/tusimple_18.pth')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('video_source', 0)  # Default is the first webcam (index 0)

        # Retrieve parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        video_source = self.get_parameter('video_source').get_parameter_value().integer_value

        # Initialize Ultrafast Lane Detector
        self.lane_detector = UltrafastLaneDetector(model_path, model_type=ModelType.TUSIMPLE, use_gpu=use_gpu)

        # Initialize webcam
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            raise RuntimeError('Cannot open webcam!')

        # Setup the publisher for lane data
        self.lane_publisher = self.create_publisher(Float32MultiArray, 'lane_data', 10)

        # Setup a timer for periodically processing frames
        self.timer = self.create_timer(0.1, self.process_frame)  # Process at 10 Hz

        self.get_logger().info('Lane detection node has started.')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to read frame from webcam.')
            return

        # Perform lane detection
        try:
            lanes_img, lane_data = self.lane_detector.detect_lanes(frame)
            left_lane_points, right_lane_points = lane_data
        except Exception as e:
            self.get_logger().error(f"Error in lane detection: {e}")
            return

        # Filter and convert lane points
        left_lane_points = [float(x) for x in left_lane_points if isinstance(x, (int, float))]
        right_lane_points = [float(x) for x in right_lane_points if isinstance(x, (int, float))]

        # Publish lane data using Float32MultiArray
        try:
            left_lane_msg = Float32MultiArray()
            right_lane_msg = Float32MultiArray()
            left_lane_msg.data = np.array(left_lane_points, dtype=float).flatten()
            right_lane_msg.data = np.array(right_lane_points, dtype=float).flatten()
            self.lane_publisher.publish(left_lane_msg)  # Publish left lane points
            self.lane_publisher.publish(right_lane_msg)  # Publish right lane points
            self.get_logger().info('Publishing lane data')
        except Exception as e:
            self.get_logger().error(f"Error processing lane points: {e}")

        # Display the output frame
        cv2.imshow('Lane Detection', lanes_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info('Shutting down...')
            rclpy.shutdown()

    def destroy_node(self):
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt. Exiting...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

