from ultralytics import YOLO
import cv2
from datetime import datetime

# Load YOLOv8 model
model = YOLO(r"C:\Users\lohith\runs\detect\affectnet_emotion3\weights\best.pt")

# Class names
class_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Open webcam
cap = cv2.VideoCapture(0)

# Log file setup
log_file = open('emotion_log.csv', 'a')
log_file.write('timestamp,emotion\n')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame)[0]

    # Draw results on frame
    annotated_frame = results.plot()

    # Show the result
    cv2.imshow("Emotion Detection", annotated_frame)

    # Log top prediction (if any)
    if results.boxes:
        top_box = results.boxes[0]
        class_id = int(top_box.cls[0])
        emotion = class_names[class_id]
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{timestamp},{emotion}\n')
        print(f"{timestamp} - Detected Emotion: {emotion}")

    # Break on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
log_file.close()
cap.release()
cv2.destroyAllWindows()
