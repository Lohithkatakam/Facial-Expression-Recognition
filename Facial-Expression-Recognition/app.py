from flask import Flask, render_template, Response
import cv2
from ultralytics import YOLO

app = Flask(__name__)

# Load the YOLO model (Ensure the model path is correct)
model = YOLO('weights/best.pt')  # Update this with your model path
print("✅ Model loaded successfully!")

# Open the camera (webcam)
camera = cv2.VideoCapture(0)  # Change to your webcam or video file path
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Check if camera opened successfully
if not camera.isOpened():
    print("❌ Error: Could not open camera.")
    exit()

def gen_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()

        # Debug: Check if frame capture is successful
        if not success:
            print("❌ Failed to grab frame")
            continue

        if frame is None:
            print("⚠️ Frame is None")
            continue

        # YOLO Inference
        results = model(frame)  # Detect and annotate
        annotated_frame = results[0].plot()

        # Check if annotated frame is valid
        if annotated_frame is None or annotated_frame.size == 0:
            print("❌ Invalid annotated frame")
            continue

        # Encode the frame to JPEG format
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if not ret:
            print("❌ Failed to encode frame")
            continue

        # Convert the frame to bytes for streaming
        frame_bytes = buffer.tobytes()

        # Yield the frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
