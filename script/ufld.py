import cv2
from ultrafastLaneDetector import UltrafastLaneDetector, ModelType

def main():
    # Load the ultrafast lane detection model
    model_path = "path_to_model_file.pth"  # Replace with the path to your trained model
    use_gpu = False  # Set to True if you have a compatible GPU
    lane_detector = UltrafastLaneDetector(model_path, model_type=ModelType.TUSIMPLE, use_gpu=use_gpu)
    
    # Initialize webcam (0 for default webcam, or replace with mobile webcam stream URL)
    cap = cv2.VideoCapture(0)  # Replace 0 with mobile camera feed URL if needed
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return
    
    print("Press 'q' to quit.")

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Perform lane detection
        lanes_img = lane_detector.detect_lanes(frame)
        
        # Display the processed frame
        cv2.imshow("Lane Detection", lanes_img)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

