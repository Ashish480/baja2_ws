#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
from ultrafastLaneDetector.ultrafastLaneDetector import UltrafastLaneDetector, ModelType
from ultralytics import YOLO

from cv_bridge import CvBridge
from sensor_msgs.msg import Image

class LaneAndObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('detector')
        
        # Parameters
        self.declare_parameter('model_path', 'src/buggy/buggy/models/tusimple_18.pth')
        self.declare_parameter('yolo_model_path', 'src/buggy/buggy/models/yolov8n.pt')
        self.declare_parameter('use_gpu', False)
        self.declare_parameter('video_source', 2)  # Default is the first webcam (index 0)

        # Get parameters
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        yolo_model_path = self.get_parameter('yolo_model_path').get_parameter_value().string_value
        use_gpu = self.get_parameter('use_gpu').get_parameter_value().bool_value
        video_source = self.get_parameter('video_source').get_parameter_value().integer_value

        # Initialize Ultrafast Lane Detector
        self.lane_detector = UltrafastLaneDetector(model_path, model_type=ModelType.TUSIMPLE, use_gpu=use_gpu)
        
        # Initialize YOLO object detector
        self.yolo_model = YOLO(yolo_model_path)

        # Initialize webcam
        self.cap = cv2.VideoCapture(video_source)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open webcam!')
            raise RuntimeError('Cannot open webcam!')

        # Timer to periodically process frames
        self.timer = self.create_timer(0.1, self.process_frame)  # 10 Hz frame processing

        self.get_logger().info('Lane and object detection node has started.')

    def process_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error('Failed to read frame from webcam.')
            return

        # Perform lane detection
        lanes_img = self.lane_detector.detect_lanes(frame)

        # Perform object detection
        yolo_results = self.yolo_model(frame)
        objects = yolo_results[0].boxes

        # Draw YOLO detections on the frame
        for box in objects:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            
            cv2.rectangle(lanes_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            class_name = self.yolo_model.names[cls]
            label = f"{class_name} {conf:.2f}"
            cv2.putText(lanes_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the output frame
        cv2.imshow('Lane and Object Detection', lanes_img)

        # Exit on pressing 'q'
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
    node = LaneAndObjectDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt. Exiting...')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
