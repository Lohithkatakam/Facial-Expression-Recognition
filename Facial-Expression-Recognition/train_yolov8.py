from ultralytics import YOLO

# Load the YOLOv8 model (YOLOv8n is smaller and faster; you can choose yolov8m, yolov8l for better accuracy)
model = YOLO('yolov8n.pt')  # Replace with yolov8m.pt or yolov8l.pt for better accuracy

# Train the model
model.train(
    data = r'C:\Users\lohith\OneDrive\Desktop\major\YOLO_format\data.yaml',
  # Provide the correct path to your data.yaml
    imgsz=640,                  # Image size for training (640x640)
    epochs=8,                  # Number of epochs for training
    batch=16,                   # Batch size for training
    name='affectnet_emotion'    # Name for the training run
)
