import cv2

print("AI Ergonomics Platform starting...")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        break
        cv2.putText(
    frame,
    "AI Ergonomics Platform - Camera Active",
    (10, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.6,
    (0, 255, 0),
    2
)

    cv2.imshow("AI Ergonomics - Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

