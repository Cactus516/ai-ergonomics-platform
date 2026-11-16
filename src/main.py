import time
import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import drawing_utils
POSE_CONNECTIONS = mp.tasks.vision.PoseLandmarksConnections.POSE_LANDMARKS

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

print("AI Ergonomics Platform starting...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

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
            for landmark in pose:
                x = int(landmark.x * width)
                y = int(landmark.y * height)

                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)



    cv2.imshow("AI Ergonomics - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

