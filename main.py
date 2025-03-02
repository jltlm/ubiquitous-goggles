import random
import time

import cv2
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.face_detection as mp_face_detection
import mediapipe.python.solutions.hands as mp_hands
import numpy as np

SHOT_DELAY = 3
SHOT_SIGNAL_TIME = 0.4
SHOT_FIRE_TRACER_TIME = 0.2
SHOT_COOLDOWN_TIME = 1
SHOT_NORMAL_COLOR = (255, 0, 0)
SHOT_SIGNAL_COLOR = (0, 255, 0)
SHOT_FIRE_COLOR = (0, 0, 255)

INITIAL_HEALTH = 4
DAMAGE_FACE = 2
DAMAGE_HAND = 1
DAMAGE_GRACE_PERIOD = 0.75

SCORE_PER_KILL = 100


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

    def chase(self, target, delta_time):
        diff = np.array(self.position) - target
        norm = np.linalg.norm(diff)
        if norm == 0:
            return
        chase_direction = diff / norm
        offset = chase_direction * 100 * delta_time
        self.position[0] -= offset[0]
        self.position[1] -= offset[1]

    def get_hits(self):
        return self.hits

    def intersects(self, other_box) -> bool:
        x1min, y1min = self.position
        x1max, y1max = self.get_end()
        x2min, y2min, w, h = other_box
        x2max = x2min + w
        y2max = y2min + h

        return x1min < x2max and x2min < x1max and y1min < y2max and y2min < y1max

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

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # 0 for default camera

# initializing things for the chasing rectangle
rect_list = []

previous_loop_time = time.time()
previous_shot_time = previous_loop_time
previous_shot_tracer = None
previous_damaged_time = None
previous_spawn_time = None
spawn_delay = None

health = INITIAL_HEALTH
score = 0

while cap.isOpened():
    current_time = time.time()
    delta_time = current_time - previous_loop_time
    previous_loop_time = current_time

    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    hand_results = hands.process(image)
    face_results = face_detection.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, c = image.shape

    if (
        previous_spawn_time is None
        or spawn_delay is None
        or current_time - previous_spawn_time > spawn_delay
    ):
        x = (
            random.uniform(0, 200)
            if random.choice([True, False])
            else random.uniform(w - 200, w)
        )
        y = (
            random.uniform(0, 100)
            if random.choice([True, False])
            else random.uniform(h - 100, h)
        )
        rect_list.append(Rect([x, y], 64))
        previous_spawn_time = current_time
        spawn_delay = random.normalvariate(4, 1)

    face_bounding_box = None
    is_hit = 0  # 0 is not hit, 1 is hit on face, 2 is hit on hand

    if face_results.detections:
        detection = next(iter(face_results.detections))

        mp_drawing.draw_detection(image, detection)

        bbox = detection.location_data.relative_bounding_box
        fbbx = int(bbox.xmin * w)
        fbby = int(bbox.ymin * h)
        fbbw = int(bbox.width * w)
        fbbh = int(bbox.height * h)
        face_center = np.array((fbbx + fbbw / 2, fbby + fbbh / 2))
        face_bounding_box = (fbbx, fbby, fbbw, fbbh)

        for rect in rect_list:
            rect.chase(face_center, delta_time)
            if is_hit == 0 and rect.intersects(face_bounding_box):
                is_hit = 1

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip = np.array((int(index_tip.x * w), int(index_tip.y * h)))
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp = np.array((int(index_mcp.x * w), int(index_mcp.y * h)))
            direction = index_tip - index_mcp
            end = direction * 100 + index_tip

            hit_indices = []  # The indices of the rectangles that are hit.
            for i, rect in enumerate(rect_list):
                hits = rect.compute_hits(index_tip, direction)
                for hit in hits:
                    cv2.circle(image, (hit[0], hit[1]), 10, (0, 0, 255), -1)
                if len(hits) > 0:
                    hit_indices.append(i)

                if not face_results.detections:
                    rect.chase(index_mcp, delta_time)

                if is_hit != 0:
                    continue
                rect_end = rect.get_end()
                for landmark in hand_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    if (
                        x >= rect.position[0]
                        and x <= rect_end[0]
                        and y >= rect.position[1]
                        and y <= rect_end[1]
                    ):
                        is_hit = 2
                        break

            delta_time_shot = current_time - previous_shot_time
            if delta_time_shot < SHOT_FIRE_TRACER_TIME:
                pass
            elif delta_time_shot < SHOT_COOLDOWN_TIME:
                previous_shot_tracer = None
                pass
            elif delta_time_shot < SHOT_DELAY:
                cv2.line(
                    image,
                    (index_tip[0], index_tip[1]),
                    (end[0], end[1]),
                    SHOT_NORMAL_COLOR,
                    5,
                )
            elif delta_time_shot < SHOT_DELAY + SHOT_SIGNAL_TIME:
                cv2.line(
                    image,
                    (index_tip[0], index_tip[1]),
                    (end[0], end[1]),
                    SHOT_SIGNAL_COLOR,
                    5,
                )
            else:
                previous_shot_tracer = ((index_tip[0], index_tip[1]), (end[0], end[1]))
                previous_shot_time = current_time
                for i in reversed(hit_indices):
                    score += SCORE_PER_KILL
                    rect_list.pop(i)

            if previous_shot_tracer is not None:
                start, end = previous_shot_tracer
                cv2.line(
                    image,
                    start,
                    end,
                    SHOT_FIRE_COLOR,
                    8,
                )

    if is_hit != 0 and (
        previous_damaged_time is None
        or current_time - previous_damaged_time > DAMAGE_GRACE_PERIOD
    ):
        health -= DAMAGE_FACE if is_hit == 1 else DAMAGE_HAND
        previous_damaged_time = current_time
        print(f"Health: {health}")

    for rect in rect_list:
        rect.render(image)

    image = cv2.flip(image, 1)

    cv2.putText(
        image,
        f"Health: {health}",
        (75, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
    )
    cv2.putText(
        image,
        f"Score: {score}",
        (75, 200),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (0, 0, 255),
        4,
    )

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
