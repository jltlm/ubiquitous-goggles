import cv2
import numpy as np
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
from typing import Tuple


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # 0 for default camera


while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, c = image.shape
    rect_start = (0, 0)
    rect_dim = 256
    rect_end = (rect_start[0] + rect_dim, rect_start[1] + rect_dim)
    rect_pos = ((rect_start[0] + rect_end[0]) / 2, (rect_start[1] + rect_end[1]) / 2)
    rect_hit = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip = np.array((int(index_tip.x * w), int(index_tip.y * h)))
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp = np.array((int(index_mcp.x * w), int(index_mcp.y * h)))
            direction = index_tip - index_mcp
            end = direction * 100 + index_tip

            t = (rect_start[0] - index_tip[0]) / direction[0]
            cast_pos = t * direction + index_tip
            cast_pos = cast_pos.astype(int)
            print(cast_pos)
            if (
                cast_pos[0] >= min(rect_start[0], rect_end[0])
                and cast_pos[0] <= max(rect_start[0], rect_end[0])
                and cast_pos[1] >= min(rect_start[1], rect_end[1])
                and cast_pos[1] <= max(rect_start[1], rect_end[1])
            ):
                rect_hit = True

            cv2.circle(image, (cast_pos[0], cast_pos[1]), 16, (0, 0, 255), -1)

            t = (rect_end[0] - index_tip[0]) / direction[0]
            cast_pos = t * direction + index_tip
            cast_pos = cast_pos.astype(int)
            print(cast_pos)
            if (
                cast_pos[0] >= min(rect_start[0], rect_end[0])
                and cast_pos[0] <= max(rect_start[0], rect_end[0])
                and cast_pos[1] >= min(rect_start[1], rect_end[1])
                and cast_pos[1] <= max(rect_start[1], rect_end[1])
            ):
                rect_hit = True

            cv2.circle(image, (cast_pos[0], cast_pos[1]), 16, (0, 0, 255), -1)

            t = (rect_start[1] - index_tip[1]) / direction[1]
            cast_pos = t * direction + index_tip
            cast_pos = cast_pos.astype(int)
            print(cast_pos)
            if (
                cast_pos[0] >= min(rect_start[0], rect_end[0])
                and cast_pos[0] <= max(rect_start[0], rect_end[0])
                and cast_pos[1] >= min(rect_start[1], rect_end[1])
                and cast_pos[1] <= max(rect_start[1], rect_end[1])
            ):
                rect_hit = True

            cv2.circle(image, (cast_pos[0], cast_pos[1]), 16, (0, 0, 255), -1)

            t = (rect_end[1] - index_tip[1]) / direction[1]
            cast_pos = t * direction + index_tip
            cast_pos = cast_pos.astype(int)
            print(cast_pos)
            if (
                cast_pos[0] >= min(rect_start[0], rect_end[0])
                and cast_pos[0] <= max(rect_start[0], rect_end[0])
                and cast_pos[1] >= min(rect_start[1], rect_end[1])
                and cast_pos[1] <= max(rect_start[1], rect_end[1])
            ):
                rect_hit = True

            cv2.circle(image, (index_tip[0], index_tip[1]), 5, (255, 0, 0), -1)
            cv2.circle(image, (index_mcp[0], index_mcp[1]), 5, (255, 0, 0), -1)
            cv2.circle(image, (cast_pos[0], cast_pos[1]), 16, (0, 0, 255), -1)
            cv2.line(
                image, (index_tip[0], index_tip[1]), (end[0], end[1]), (255, 0, 0), 5
            )

    rect_color = (0, 255, 0) if rect_hit else (0, 0, 255)
    cv2.rectangle(image, rect_start, rect_end, rect_color, 3)

    image = cv2.flip(image, 1)
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
