import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)  # 0 for default camera

while cap.isOpened():
    success, image1 = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    image1.flags.writeable = False
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image = cv2.flip(image1, 1)
    results = hands.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            knuckle_indices = [mp_hands.HandLandmark.INDEX_FINGER_MCP, mp_hands.HandLandmark.INDEX_FINGER_PIP, mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_TIP]
            knuckle_coordinates = []

            for index in knuckle_indices:
                landmark = hand_landmarks.landmark[index]
                
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                knuckle_coordinates.append((landmark.x, landmark.y, landmark.z))
                cv2.circle(
                    image, (cx, cy), 5, (255, 0, 0), -1
                )  # draw circles on knuckles.

            dirVector = np.array(knuckle_coordinates[-1])-np.array(knuckle_coordinates[0]);
            
            if dirVector[0] < 0:
                print('left')
            else:
                print("right")
            
            if dirVector[1] < 0:
                print('up')
            else:
                print('down')
            print(dirVector)
            

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
