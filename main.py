import cv2
import mediapipe as mp
import math
import time

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

MODEL = "hand_landmarker.task"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

gesture_definitions = {
    (True, True, True, True, True): "OPEN PALM",
    (False, True, False, False, True): "DIO HORNS",
    (True, True, False, False, True): "THWIP",
    (False, False, False, False, False): "FIST",
    (False, True, True, False, False): "VICTORY",
    (True, False, False, False, True): "ON CALL",
}

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
timestamp_ms = 0

def distance(a, b):
    return math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))

def classify_thumb(hand):
    pinky_mcp = hand[17]
    if distance(hand[4], pinky_mcp) <= distance(hand[3], pinky_mcp):
        return False
    return distance(hand[4], hand[5]) > 0.5 * distance(hand[0], hand[9])


def classify_gesture(hand):
    # Wrist, finger PIP joints, and fingertips
    wrist = hand[0]

    fingers = [
        (8, 6),    # Index
        (12, 10),  # Middle
        (16, 14),  # Ring
        (20, 18),  # Pinky
    ]

    extended = [classify_thumb(hand)]

    for tip_idx, pip_idx in fingers:
        tip_distance = distance(wrist, hand[tip_idx])
        pip_distance = distance(wrist, hand[pip_idx])

        extended.append(tip_distance > pip_distance * 1.15)

    extended_tup = tuple(extended)
    return gesture_definitions.get(extended_tup, "UNKNOWN")


try:
    with HandLandmarker.create_from_options(options) as landmarker:
        prev_time = time.perf_counter()
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )
            timestamp_ms += 33
            results = landmarker.detect_for_video(
                mp_image, timestamp_ms
            )
            height, width, _ = frame.shape

            if results.hand_landmarks:
                for hand_landmarks, world_landmarks in zip(results.hand_landmarks, results.hand_world_landmarks):
                    gesture = classify_gesture(world_landmarks)
                    points = []

                    for landmark in hand_landmarks:
                        x = int(landmark.x * width)
                        y = int(landmark.y * height)
                        points.append((x, y))

                    for start, end in HAND_CONNECTIONS:
                        cv2.line(
                            frame,
                            points[start],
                            points[end],
                            (0, 255, 0),
                            2
                        )

                    for x, y in points:
                        cv2.circle(
                            frame,
                            (x, y),
                            5,
                            (0, 0, 255),
                            cv2.FILLED
                        )
                    label_x = max(10, min(points[0][0], width - 180))
                    label_y = max(30, points[0][1] - 20)
                    cv2.putText(
                        frame,
                        gesture,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 0),
                        2,
                    )
                now = time.perf_counter()
                fps = 1 / (now - prev_time)
                prev_time = now

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "NO HAND DETECTED",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )
            cv2.imshow("MediaPipe Hands", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == 27 or cv2.getWindowProperty("MediaPipe Hands", cv2.WND_PROP_VISIBLE) < 1:
                break

finally:
    cap.release()
    cv2.destroyAllWindows()
