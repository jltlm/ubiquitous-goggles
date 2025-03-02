import time

import cv2
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
import numpy as np


def clip(value, max_value):
    if value < 0:
        return 0
    elif value > max_value:
        return max_value
    else:
        return value


class Rect:
    def __init__(self, position=[0, 0], dimension=256) -> None:
        self.position = list(position)
        self.dimension = dimension
        self.hits = []

    def get_end(self):
        return self.position[0] + self.dimension, self.position[1] + self.dimension

    def compute_hits(self, origin: np.ndarray, direction: np.ndarray):
        self.hits.clear()
        end = self.get_end()

        t = (self.position[0] - origin[0]) / direction[0]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[1] >= self.position[1] and cast_pos[1] <= end[1]:
                self.hits.append(cast_pos)

        t = (end[0] - origin[0]) / direction[0]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[1] >= self.position[1] and cast_pos[1] <= end[1]:
                self.hits.append(cast_pos)

        t = (self.position[1] - origin[1]) / direction[1]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[0] >= self.position[0] and cast_pos[0] <= end[0]:
                self.hits.append(cast_pos)

        t = (end[1] - origin[1]) / direction[1]
        if t > 0:
            cast_pos = t * direction + origin
            cast_pos = cast_pos.astype(int)
            if cast_pos[0] >= self.position[0] and cast_pos[0] <= end[0]:
                self.hits.append(cast_pos)

        return self.hits

    def get_hits(self):
        return self.hits

    def render(self, image):
        rect_color = (0, 255, 0) if len(self.hits) > 0 else (0, 0, 255)
        end = self.get_end()
        cv2.rectangle(
            image,
            (int(self.position[0]), int(self.position[1])),
            (int(end[0]), int(end[1])),
            rect_color,
            3,
        )


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # 0 for default camera

# initializing things for the chasing rectangle
chaser_rect = Rect(dimension=50)

previous_time = time.time()

while cap.isOpened():
    current_time = time.time()
    delta_time = current_time - previous_time
    previous_time = current_time

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
    rect = Rect()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip = np.array((int(index_tip.x * w), int(index_tip.y * h)))
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp = np.array((int(index_mcp.x * w), int(index_mcp.y * h)))
            direction = index_tip - index_mcp
            end = direction * 100 + index_tip

            hits = rect.compute_hits(index_tip, direction)
            for hit in hits:
                cv2.circle(image, (hit[0], hit[1]), 16, (0, 0, 255), -1)

            hits = chaser_rect.compute_hits(index_tip, direction)
            for hit in hits:
                cv2.circle(image, (hit[0], hit[1]), 16, (0, 0, 255), -1)

            cv2.circle(image, (index_tip[0], index_tip[1]), 5, (255, 0, 0), -1)
            cv2.circle(image, (index_mcp[0], index_mcp[1]), 5, (255, 0, 0), -1)
            cv2.line(
                image, (index_tip[0], index_tip[1]), (end[0], end[1]), (255, 0, 0), 5
            )

            # compare chaser rect with hand origin
            chaser_to_index_mcp = np.array(chaser_rect.position) - index_mcp
            dx = 100 * delta_time * (1 if chaser_to_index_mcp[0] < 0 else -1)
            dy = 100 * delta_time * (1 if chaser_to_index_mcp[1] < 0 else -1)
            chaser_rect.position[0] += dx
            chaser_rect.position[1] += dy

    rect.render(image)
    chaser_rect.render(image)

    image = cv2.flip(image, 1)
    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
