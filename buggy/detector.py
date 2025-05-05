#!/usr/bin/env python3
import sys
import cv2
import numpy as np
import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO
import json  # Added for proper JSON formatting

# Add the UFLD directory itself to sys.path to recognize 'ultrafastLaneDetector' as a package
sys.path.append("/home/ashy/baja2_ws/src/buggy")

from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')
        
        # ROS2 Publisher
        self.lane_publisher = self.create_publisher(String, '/lane_coordinates', 10)
        
        # Paths to the models
        lane_model_path = "/home/ashy/baja2_ws/src/buggy/buggy/UFLD/models/tusimple_18.pth"
        model_type = ModelType.TUSIMPLE
        yolo_model_path = "/home/ashy/baja2_ws/src/buggy/buggy/UFLD/models/yolov8n.pt"
        use_gpu = False  # Set to True if you have GPU support

        # Initialize lane detection model
        self.lane_detector = UltrafastLaneDetector(lane_model_path, model_type, use_gpu)

        # Initialize YOLOv8 model for object detection
        self.yolo_model = YOLO(yolo_model_path)

        # Open the camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Error: Could not open camera.")
            exit()

        # Timer callback to process frames
        self.timer = self.create_timer(0.1, self.process_frame)

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to capture frame")
            return

        start_time = time.time()

        # Detect lanes
        lane_output = self.lane_detector.detect_lanes(frame)
        lanes_points = self.lane_detector.lanes_points  # Extract lane points

        # Process lane points to remove empty detections
        filtered_lanes = []
        for lane in lanes_points:
            lane = [list(point) for point in lane if not np.array_equal(point, [0, 0])]
            if lane:  # Only append non-empty lanes
                filtered_lanes.append(lane)

        # Publish valid lane coordinates as JSON format
        if filtered_lanes:
            lane_msg = String()
            lane_msg.data = json.dumps(filtered_lanes, default=lambda x: int(x))  # Convert NumPy int64 to Python int

            self.lane_publisher.publish(lane_msg)

        # Detect objects with YOLO
        yolo_results = self.yolo_model(frame)
        objects = yolo_results[0].boxes  # YOLOv8 format: access .boxes for detections

        # Draw lane detection results on the frame
        output_frame = lane_output.copy()

        # Overlay YOLO detections
        for box in objects:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = int(box.cls[0])  # Class index

            # Draw bounding box and label
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = self.yolo_model.names[cls]
            label = f"{class_name} {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Calculate FPS
        end_time = time.time()
        elapsed_time = end_time - start_time
        fps_display = 1 / elapsed_time
        cv2.putText(output_frame, f"FPS: {fps_display:.2f}", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Live Lane and Object Detection', output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.get_logger().info("Shutting down node...")
            rclpy.shutdown()

    def destroy_node(self):
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

