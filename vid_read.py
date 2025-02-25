import cv2
import yaml

from detect_infer import infer

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Get the list of class names from the config file
class_names = config.get("yolo_detect_classes", [])

# Open video file
video_path = "test.MOV"  # Replace with your video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break  # Stop when the video ends

    # Run YOLO inference
    results, detected_classes = infer(frame)

    frame_with_boxes = results[0].plot()  # Directly use YOLO's visualization

    # Show the frame
    # cv2.imshow("YOLO Detection", frame_with_boxes)

    if any(cls in detected_classes for cls in class_names):
        print("HI") 

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
