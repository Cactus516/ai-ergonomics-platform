import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

NOSE = 0
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/pose_landmarker_full.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.6,
    min_pose_presence_confidence=0.6,
    min_tracking_confidence=0.6
)

detector = PoseLandmarker.create_from_options(options)

def smooth_landmarks(current_landmarks, previous_landmarks, alpha=0.2):
    smoothed_landmarks = []

    for i, landmark in enumerate(current_landmarks):
        if previous_landmarks is None:
            x = landmark.x
            y = landmark.y
        else:
            previous = previous_landmarks[i]

            x = alpha * landmark.x + (1 - alpha) * previous[0]
            y = alpha * landmark.y + (1 - alpha) * previous[1]

        smoothed_landmarks.append((x, y))

    return smoothed_landmarks

print("AI Ergonomics Platform starting...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

previous_landmarks = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )
    timestamp_ms = int(time.monotonic() * 1000)

    pose_result = detector.detect_for_video(
        mp_image,
        timestamp_ms
    )
    if pose_result.pose_landmarks:
        print("Person detected!")
    else:
        print("No person detected.")

    if pose_result.pose_landmarks:
        height, width, _ = frame.shape

        for pose in pose_result.pose_landmarks:
            smoothed = smooth_landmarks(pose, previous_landmarks)

            previous_landmarks = smoothed
            nose = smoothed[NOSE]
            left_ear = smoothed[LEFT_EAR]
            right_ear = smoothed[RIGHT_EAR]
            left_shoulder = smoothed[LEFT_SHOULDER]
            right_shoulder = smoothed[RIGHT_SHOULDER]

            shoulder_difference = abs(left_shoulder[1] - right_shoulder[1])
            cv2.putText(
    frame,
    f"Shoulder Difference: {shoulder_difference:.4f}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2
)
            left_hip = smoothed[LEFT_HIP]
            right_hip = smoothed[RIGHT_HIP]

            important_landmarks = [
                left_shoulder,
                right_shoulder,
                left_hip,
                right_hip
            ]

            for x, y in important_landmarks:
                pixel_x = int(x * width)
                pixel_y = int(y * height)

                cv2.circle(frame, (pixel_x, pixel_y), 8, (0, 0, 255), -1)

        for connection in POSE_CONNECTIONS:
            start_idx = connection.start
            end_idx = connection.end

            start_point = smoothed[start_idx]
            end_point = smoothed[end_idx]

            start_pos = (int(start_point[0] * width), int(start_point[1] * height))
            end_pos = (int(end_point[0] * width), int(end_point[1] * height))

            cv2.line(frame, start_pos, end_pos, (0, 255, 0), 2)

        for x, y in smoothed:
            pixel_x = int(x * width)
            pixel_y = int(y * height)

            cv2.circle(frame, (pixel_x, pixel_y), 5, (0, 255, 0), -1)



    cv2.imshow("AI Ergonomics - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

