import cv2
import mediapipe as mp
import numpy as np

# Function to calculate Euclidean distance between two points
def euclidean_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)

# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Flag to indicate photo capture
capture_photo = False

while cap.isOpened():
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect landmarks
    results = pose.process(frame_rgb)

    # Initialize variables to store shoulder and nose landmark coordinates
    right_shoulder_coords = None
    left_shoulder_coords = None
    nose_coords = None

    # Check if landmarks are detected
    if results.pose_landmarks:
        # Draw dots on the shoulders and nose
        for landmark in results.pose_landmarks.landmark:
            if landmark.visibility > 0.5:  # Adjust visibility threshold as needed
                # Right shoulder (index 12) and left shoulder (index 11) landmarks
                if landmark.name == 'RIGHT_SHOULDER':
                    right_shoulder_coords = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    cv2.circle(frame, right_shoulder_coords, 5, (0, 255, 0), -1)
                elif landmark.name == 'LEFT_SHOULDER':
                    left_shoulder_coords = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    cv2.circle(frame, left_shoulder_coords, 5, (0, 255, 0), -1)
                elif landmark.name == 'NOSE':
                    nose_coords = (int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0]))
                    cv2.circle(frame, nose_coords, 5, (0, 0, 255), -1)

        # Calculate distance between nose and shoulders if all landmarks are detected
        if right_shoulder_coords and left_shoulder_coords and nose_coords:
            shoulder_width = euclidean_distance(right_shoulder_coords, left_shoulder_coords)
            nose_shoulder_distance = euclidean_distance(nose_coords, right_shoulder_coords)
            
            # Check if the condition is met to capture the photo
            if nose_shoulder_distance <= shoulder_width / 2:
                capture_photo = True

        # If the condition is met, capture the photo
        if capture_photo:
            cv2.imwrite("captured_photo.jpg", frame)
            print("Photo captured!")
            break

    # Display the frame
    cv2.imshow('Shoulder Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
