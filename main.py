import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

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

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Knuckle landmark indices:
            # 5: Wrist to Thumb knuckle
            # 9: Wrist to Index finger knuckle
            # 13: Wrist to Middle finger knuckle
            # 17: Wrist to Ring finger knuckle
            # 21: Wrist to Pinky finger knuckle

            knuckle_indices = [5, 9, 13, 17, 21]
            knuckle_coordinates = []

            for index in knuckle_indices:
                landmark = hand_landmarks.landmark[0]
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                knuckle_coordinates.append((landmark.x, landmark.y, landmark.z))
                cv2.circle(
                    image, (cx, cy), 5, (255, 0, 0), -1
                )  # draw circles on knuckles.

            # You can now use knuckle_coordinates (list of tuples) for further processing
            print("Knuckle Coordinates:", knuckle_coordinates)

            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("MediaPipe Hands", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
