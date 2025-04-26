import cv2
import os

# Open the camera (0 for default webcam, specify DirectShow backend)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if camera opened successfully
if not camera.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Set camera resolution (try different values if this doesn't work)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_counter = 0

while True:
    success, frame = camera.read()
    if not success:
        print("Failed to grab frame")
        break
    else:
        # Debug: Check if the frame is not blank
        if frame is None or frame.size == 0:
            print("Error: Empty frame received")
            break
        
        # Display the frame to confirm camera is working
        cv2.imshow('Camera Feed', frame)
        
        # Save the first frame as a .jpg file for inspection
        if frame_counter == 0:
            if not os.path.exists('captured_frames'):
                os.mkdir('captured_frames')
            cv2.imwrite('captured_frames/first_frame.jpg', frame)
            print("First frame saved as 'captured_frames/first_frame.jpg'")

        # Increment frame counter
        frame_counter += 1

    # Wait for user to press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
