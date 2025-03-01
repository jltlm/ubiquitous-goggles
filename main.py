import cv2
import mediapipe as mp
import mediapipe.python.solutions.drawing_utils as mp_drawing
import mediapipe.python.solutions.hands as mp_hands
import numpy as np

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
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

    my_text = ""

    if results.multi_hand_world_landmarks:
        for hand_landmarks in results.multi_hand_world_landmarks:
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_tip = np.array((index_tip.x, index_tip.y, index_tip.z))
            index_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]
            index_mcp = np.array((index_mcp.x, index_mcp.y, index_mcp.z))
            direction = index_tip - index_mcp

            x_dir = "left" if direction[0] < 0 else "right"
            y_dir = "up" if direction[1] < 0 else "down"
            z_dir = "forward" if direction[2] < 0 else "backward"

            # You can now use knuckle_coordinates (list of tuples) for further processing
            print(index_tip, index_mcp)

            my_text = f"{x_dir}; {y_dir}; {z_dir}"
            print(my_text)

    font = cv2.FONT_HERSHEY_SIMPLEX
    image = cv2.putText(
        image, my_text, (10, 500), font, 4, (255, 255, 255), 2, cv2.LINE_AA
    )

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
