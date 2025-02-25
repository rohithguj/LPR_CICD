from ultralytics import YOLO
import yaml


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Get the list of class names from the config file
model = config.get("detect_model")
# Load YOLOv8 Nano
model = YOLO(model)

def infer(frame):
    # Load image
    results = model(frame)
    detected_classes = [model.names[int(box.cls)] for box in results[0].boxes]

    return(results, detected_classes)
